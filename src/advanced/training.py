"""On-policy training, POMO grouping, elite replay, and evaluation."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.advanced.features import MARGINAL_FEATURE_INDEX, StructuredState
from src.advanced.ppo import AdvancedPPOAgent, AdvancedPPOBatch, compute_episode_gae
from src.core import CIPPInstance, evaluate_itinerary
from src.envs import CIPPEnv


@dataclass(slots=True)
class EpisodeTrajectory:
    states: list[StructuredState]
    actions: list[int]
    log_probabilities: list[float]
    values: list[float]
    scaled_rewards: list[float]
    itinerary: tuple[int, ...]
    objective: float
    visit_counts: np.ndarray
    group_id: int


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    method: str
    objective: float
    itinerary: tuple[int, ...]
    feasible: bool
    runtime_seconds: float
    idle_days: int
    unique_locations: int
    visit_hhi: float
    start_generation_seconds: float = 0.0
    improvement_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConstructionTrainingConfig:
    updates: int = 300
    episodes_per_update: int = 32
    pomo_group_size: int = 1
    validation_interval: int = 10
    validation_rollouts: int = 30
    validation_metric: Literal["deterministic", "mean_stochastic", "best_of_k"] = "best_of_k"
    elite_capacity: int = 64
    self_imitation_interval: int = 10
    self_imitation_elites: int = 4
    self_imitation_epochs: int = 1
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 1.0
    early_stopping_warmup_updates: int = 0
    history_flush_interval: int = 10
    vectorized_rollouts: bool = True


class EliteArchive:
    """Diversity-preserving archive of the best agent-generated solutions."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = max(int(capacity), 1)
        self._items: dict[tuple[int, ...], float] = {}

    def add(self, itinerary: tuple[int, ...], objective: float) -> None:
        previous = self._items.get(itinerary)
        if previous is None or objective > previous:
            self._items[itinerary] = float(objective)
        if len(self._items) > self.capacity:
            keep = sorted(self._items.items(), key=lambda item: item[1], reverse=True)[
                : self.capacity
            ]
            self._items = dict(keep)

    @property
    def best(self) -> tuple[tuple[int, ...], float] | None:
        if not self._items:
            return None
        return max(self._items.items(), key=lambda item: item[1])

    def top(self, count: int) -> list[tuple[tuple[int, ...], float]]:
        return sorted(self._items.items(), key=lambda item: item[1], reverse=True)[:count]

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {"itinerary": list(itinerary), "objective": objective}
            for itinerary, objective in self.top(self.capacity)
        ]


def _first_actions(agent: AdvancedPPOAgent, group_size: int, rng: np.random.Generator) -> list[int]:
    environment = CIPPEnv(agent.instance)
    environment.reset()
    state = agent.feature_builder.build(environment)
    valid = np.flatnonzero(state.action_mask)
    if group_size <= 1:
        return []
    visit_actions = valid[valid > 0]
    ordered_visits = sorted(
        visit_actions.tolist(),
        key=lambda action: state.locations[action - 1, MARGINAL_FEATURE_INDEX],
        reverse=True,
    )
    candidates: list[int] = []
    if bool(state.action_mask[0]):
        candidates.append(0)
    candidates.extend(ordered_visits)
    if len(candidates) < group_size:
        candidates.extend(
            int(rng.choice(valid)) for _ in range(group_size - len(candidates))
        )
    return candidates[:group_size]


def _collect_episode(
    agent: AdvancedPPOAgent,
    *,
    seed: int,
    group_id: int,
    forced_first_action: int | None,
) -> EpisodeTrajectory:
    environment = CIPPEnv(agent.instance, seed=seed)
    environment.reset(seed=seed)
    if forced_first_action is not None:
        environment.step(forced_first_action)

    states: list[StructuredState] = []
    actions: list[int] = []
    log_probabilities: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    while not environment.done:
        state = agent.feature_builder.build(environment)
        action, log_probability, value = agent.select_action(state)
        _, reward, _, _, _ = environment.step(action)
        states.append(state)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append(float(reward) * agent.reward_scale)

    evaluation = evaluate_itinerary(agent.instance, environment.itinerary)
    if not evaluation.feasible:
        raise RuntimeError("masked construction policy produced an infeasible itinerary")
    return EpisodeTrajectory(
        states=states,
        actions=actions,
        log_probabilities=log_probabilities,
        values=values,
        scaled_rewards=rewards,
        itinerary=tuple(int(value) for value in environment.itinerary),
        objective=float(evaluation.objective),
        visit_counts=evaluation.visit_counts.copy(),
        group_id=group_id,
    )


def _forced_first_actions(
    agent: AdvancedPPOAgent,
    *,
    number_of_episodes: int,
    group_size: int,
    rng: np.random.Generator,
) -> tuple[list[int | None], list[int]]:
    forced: list[int | None] = []
    group_ids: list[int] = []
    number_of_groups = int(np.ceil(number_of_episodes / group_size))
    for group_id in range(number_of_groups):
        members = min(group_size, number_of_episodes - len(forced))
        starts = _first_actions(agent, members, rng)
        for offset in range(members):
            forced.append(starts[offset] if starts else None)
            group_ids.append(group_id)
    return forced, group_ids


def _collect_episodes_vectorized(
    agent: AdvancedPPOAgent,
    *,
    number_of_episodes: int,
    group_size: int,
    seed: int,
) -> list[EpisodeTrajectory]:
    """Collect synchronized PPO trajectories with one batched forward per day."""

    rng = np.random.default_rng(seed)
    forced_actions, group_ids = _forced_first_actions(
        agent,
        number_of_episodes=number_of_episodes,
        group_size=group_size,
        rng=rng,
    )
    environments = [CIPPEnv(agent.instance, seed=seed + index) for index in range(number_of_episodes)]
    state_lists: list[list[StructuredState]] = [[] for _ in environments]
    action_lists: list[list[int]] = [[] for _ in environments]
    logp_lists: list[list[float]] = [[] for _ in environments]
    value_lists: list[list[float]] = [[] for _ in environments]
    reward_lists: list[list[float]] = [[] for _ in environments]

    for index, environment in enumerate(environments):
        environment.reset(seed=seed + index)
        if forced_actions[index] is not None:
            environment.step(int(forced_actions[index]))

    while True:
        active = [index for index, environment in enumerate(environments) if not environment.done]
        if not active:
            break
        states = [agent.feature_builder.build(environments[index]) for index in active]
        actions, log_probabilities, values = agent.batch_actions(states, deterministic=False)
        for local_index, environment_index in enumerate(active):
            environment = environments[environment_index]
            _, reward, _, _, _ = environment.step(int(actions[local_index]))
            state_lists[environment_index].append(states[local_index])
            action_lists[environment_index].append(int(actions[local_index]))
            logp_lists[environment_index].append(float(log_probabilities[local_index]))
            value_lists[environment_index].append(float(values[local_index]))
            reward_lists[environment_index].append(float(reward) * agent.reward_scale)

    trajectories: list[EpisodeTrajectory] = []
    for index, environment in enumerate(environments):
        evaluation = evaluate_itinerary(agent.instance, environment.itinerary)
        if not evaluation.feasible:
            raise RuntimeError("masked construction policy produced an infeasible itinerary")
        trajectories.append(
            EpisodeTrajectory(
                states=state_lists[index],
                actions=action_lists[index],
                log_probabilities=logp_lists[index],
                values=value_lists[index],
                scaled_rewards=reward_lists[index],
                itinerary=tuple(int(value) for value in environment.itinerary),
                objective=float(evaluation.objective),
                visit_counts=evaluation.visit_counts.copy(),
                group_id=group_ids[index],
            )
        )
    return trajectories


def collect_construction_batch(
    agent: AdvancedPPOAgent,
    *,
    number_of_episodes: int,
    pomo_group_size: int,
    seed: int,
    vectorized: bool = True,
) -> tuple[AdvancedPPOBatch, dict[str, float], list[EpisodeTrajectory]]:
    if number_of_episodes < 1:
        raise ValueError("number_of_episodes must be positive")
    group_size = max(1, min(pomo_group_size, number_of_episodes))
    if vectorized:
        trajectories = _collect_episodes_vectorized(
            agent,
            number_of_episodes=number_of_episodes,
            group_size=group_size,
            seed=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        forced_actions, group_ids = _forced_first_actions(
            agent,
            number_of_episodes=number_of_episodes,
            group_size=group_size,
            rng=rng,
        )
        trajectories = [
            _collect_episode(
                agent,
                seed=seed + episode_index,
                group_id=group_ids[episode_index],
                forced_first_action=forced_actions[episode_index],
            )
            for episode_index in range(number_of_episodes)
        ]

    gae_advantages: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    for trajectory in trajectories:
        advantages, episode_returns = compute_episode_gae(
            np.asarray(trajectory.scaled_rewards, dtype=np.float32),
            np.asarray(trajectory.values, dtype=np.float32),
            discount_factor=agent.config.discount_factor,
            gae_lambda=agent.config.gae_lambda,
        )
        gae_advantages.append(advantages)
        returns.append(episode_returns)

    group_objectives: dict[int, list[float]] = {}
    for trajectory in trajectories:
        group_objectives.setdefault(trajectory.group_id, []).append(trajectory.objective)
    group_advantages: list[np.ndarray] = []
    for trajectory in trajectories:
        members = group_objectives[trajectory.group_id]
        mean = float(np.mean(members))
        std = float(np.std(members))
        relative = (trajectory.objective - mean) / (std + 1e-8)
        group_advantages.append(np.full(len(trajectory.actions), relative, dtype=np.float32))

    resolved_advantages: list[np.ndarray] = []
    for gae, group in zip(gae_advantages, group_advantages):
        if agent.config.advantage_mode == "gae":
            resolved = gae
        elif agent.config.advantage_mode == "group_relative":
            resolved = group
        else:
            gae_normalized = (gae - gae.mean()) / (gae.std() + 1e-8)
            weight = agent.config.group_advantage_weight
            resolved = (1.0 - weight) * gae_normalized + weight * group
        resolved_advantages.append(resolved.astype(np.float32))

    all_states = [state for trajectory in trajectories for state in trajectory.states]
    batch = AdvancedPPOBatch(
        locations=np.stack([state.locations for state in all_states]),
        global_features=np.stack([state.global_features for state in all_states]),
        action_masks=np.stack([state.action_mask for state in all_states]),
        actions=np.concatenate([np.asarray(item.actions, dtype=np.int64) for item in trajectories]),
        old_log_probabilities=np.concatenate(
            [np.asarray(item.log_probabilities, dtype=np.float32) for item in trajectories]
        ),
        old_values=np.concatenate(
            [np.asarray(item.values, dtype=np.float32) for item in trajectories]
        ),
        returns=np.concatenate(returns),
        advantages=np.concatenate(resolved_advantages),
        final_visit_counts=np.concatenate(
            [
                np.repeat(item.visit_counts[None, :], len(item.actions), axis=0)
                for item in trajectories
            ],
            axis=0,
        ),
    )
    objectives = np.asarray([item.objective for item in trajectories], dtype=np.float64)
    return (
        batch,
        {
            "mean_objective": float(np.mean(objectives)),
            "best_objective": float(np.max(objectives)),
            "std_objective": float(np.std(objectives)),
            "feasible_rate": 1.0,
            "transitions": float(batch.size),
            "reward_scale": float(agent.reward_scale),
            "vectorized_rollouts": float(bool(vectorized)),
        },
        trajectories,
    )


def _visit_hhi(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    shares = counts.astype(np.float64) / total
    return float(np.sum(shares**2))


def evaluate_construction_policy(
    agent: AdvancedPPOAgent,
    *,
    deterministic: bool,
    seed: int,
    method: str,
) -> PolicyEvaluation:
    started = time.perf_counter()
    environment = CIPPEnv(agent.instance, seed=seed)
    environment.reset(seed=seed)
    rng = np.random.default_rng(seed)
    while not environment.done:
        state = agent.feature_builder.build(environment)
        if deterministic:
            action, _, _ = agent.select_action(state, deterministic=True)
        else:
            probabilities, _ = agent.probabilities_and_value(state)
            probabilities = np.where(
                state.action_mask, np.clip(probabilities, 0.0, None), 0.0
            )
            probabilities /= float(np.sum(probabilities))
            action = int(rng.choice(agent.instance.num_actions, p=probabilities))
        environment.step(action)
    evaluation = evaluate_itinerary(agent.instance, environment.itinerary)
    return PolicyEvaluation(
        method=method,
        objective=float(evaluation.objective),
        itinerary=tuple(int(value) for value in environment.itinerary),
        feasible=bool(evaluation.feasible),
        runtime_seconds=float(time.perf_counter() - started),
        idle_days=int(evaluation.idle_days),
        unique_locations=int(np.count_nonzero(evaluation.visit_counts)),
        visit_hhi=_visit_hhi(evaluation.visit_counts),
    )


def _evaluate_stochastic_rollouts(
    agent: AdvancedPPOAgent,
    *,
    rollouts: int,
    seed: int,
    method: str,
) -> list[PolicyEvaluation]:
    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    started = time.perf_counter()
    environments = [CIPPEnv(agent.instance, seed=seed + index) for index in range(rollouts)]
    rngs = [np.random.default_rng(seed + index) for index in range(rollouts)]
    for index, environment in enumerate(environments):
        environment.reset(seed=seed + index)

    while True:
        active = [index for index, environment in enumerate(environments) if not environment.done]
        if not active:
            break
        states = [agent.feature_builder.build(environments[index]) for index in active]
        probabilities, _ = agent.batch_probabilities_and_values(states)
        for local_index, environment_index in enumerate(active):
            state = states[local_index]
            row = np.where(state.action_mask, np.clip(probabilities[local_index], 0.0, None), 0.0)
            total = float(np.sum(row))
            if total <= 0.0:
                raise RuntimeError("masked policy produced zero probability on every feasible action")
            row /= total
            action = int(rngs[environment_index].choice(agent.instance.num_actions, p=row))
            environments[environment_index].step(action)

    elapsed = float(time.perf_counter() - started)
    per_rollout_time = elapsed / rollouts
    results: list[PolicyEvaluation] = []
    for environment in environments:
        evaluation = evaluate_itinerary(agent.instance, environment.itinerary)
        results.append(
            PolicyEvaluation(
                method=method,
                objective=float(evaluation.objective),
                itinerary=tuple(int(value) for value in environment.itinerary),
                feasible=bool(evaluation.feasible),
                runtime_seconds=per_rollout_time,
                idle_days=int(evaluation.idle_days),
                unique_locations=int(np.count_nonzero(evaluation.visit_counts)),
                visit_hhi=_visit_hhi(evaluation.visit_counts),
            )
        )
    return results


def evaluate_best_of_k(
    agent: AdvancedPPOAgent,
    *,
    rollouts: int,
    seed: int,
    method: str,
) -> PolicyEvaluation:
    started = time.perf_counter()
    candidates = _evaluate_stochastic_rollouts(
        agent,
        rollouts=rollouts,
        seed=seed,
        method=method,
    )
    best = max(candidates, key=lambda result: result.objective)
    return PolicyEvaluation(
        method=method,
        objective=best.objective,
        itinerary=best.itinerary,
        feasible=best.feasible,
        runtime_seconds=float(time.perf_counter() - started),
        idle_days=best.idle_days,
        unique_locations=best.unique_locations,
        visit_hhi=best.visit_hhi,
    )


def _fixed_validation(
    agent: AdvancedPPOAgent,
    *,
    rollouts: int,
    seed: int,
    method_name: str,
) -> tuple[PolicyEvaluation, PolicyEvaluation, float, float]:
    deterministic = evaluate_construction_policy(
        agent,
        deterministic=True,
        seed=seed,
        method=f"{method_name}_greedy_validation",
    )
    stochastic = _evaluate_stochastic_rollouts(
        agent,
        rollouts=rollouts,
        seed=seed + 1_000_000,
        method=f"{method_name}_validation_rollout",
    )
    objectives = np.asarray([item.objective for item in stochastic], dtype=np.float64)
    best = max(stochastic, key=lambda result: result.objective)
    best_of_k = PolicyEvaluation(
        method=f"{method_name}_best_of_{rollouts}_validation",
        objective=best.objective,
        itinerary=best.itinerary,
        feasible=best.feasible,
        runtime_seconds=float(sum(item.runtime_seconds for item in stochastic)),
        idle_days=best.idle_days,
        unique_locations=best.unique_locations,
        visit_hhi=best.visit_hhi,
    )
    return deterministic, best_of_k, float(np.mean(objectives)), float(np.std(objectives))


def _replay_elites(
    agent: AdvancedPPOAgent,
    archive: EliteArchive,
    *,
    count: int,
    epochs: int,
) -> float:
    states: list[StructuredState] = []
    actions: list[int] = []
    for itinerary, _ in archive.top(count):
        environment = CIPPEnv(agent.instance)
        environment.reset()
        for action in itinerary:
            states.append(agent.feature_builder.build(environment))
            actions.append(action)
            environment.step(action)
    return agent.self_imitation_update(states, actions, epochs=epochs)


def _selection_value(validation: dict[str, float], metric: str) -> float:
    mapping = {
        "deterministic": "deterministic_objective",
        "mean_stochastic": "mean_stochastic_objective",
        "best_of_k": "best_of_k_objective",
    }
    if metric not in mapping:
        raise ValueError(
            "validation_metric must be deterministic, mean_stochastic, or best_of_k"
        )
    return float(validation[mapping[metric]])


def _write_history_files(
    output_directory: Path,
    history: list[dict[str, Any]],
    archive: EliteArchive,
) -> None:
    (output_directory / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "elite_archive.json").write_text(
        json.dumps(archive.to_list(), indent=2) + "\n", encoding="utf-8"
    )
    flattened: list[dict[str, Any]] = []
    for record in history:
        row = {"update": record["update"], "elapsed_seconds": record["elapsed_seconds"]}
        row.update({f"rollout_{key}": value for key, value in record["rollout"].items()})
        row.update(
            {f"optimization_{key}": value for key, value in record["optimization"].items()}
        )
        if record.get("validation"):
            row.update(
                {f"validation_{key}": value for key, value in record["validation"].items()}
            )
        row["self_imitation_loss"] = record.get("self_imitation_loss")
        flattened.append(row)
    if flattened:
        columns = sorted({key for row in flattened for key in row})
        with (output_directory / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flattened)


def train_construction_agent(
    agent: AdvancedPPOAgent,
    *,
    config: ConstructionTrainingConfig,
    output_directory: str | Path,
    seed: int,
    method_name: str,
    log_every_episodes: int = 256,
) -> tuple[Path, list[dict[str, Any]], EliteArchive]:
    """Train one construction PPO with fixed validation and full early stopping."""

    if log_every_episodes < 0:
        raise ValueError("log_every_episodes must be non-negative")
    if config.updates < 1 or config.episodes_per_update < 1:
        raise ValueError("updates and episodes_per_update must be positive")
    if config.validation_interval < 1 or config.validation_rollouts < 1:
        raise ValueError("validation_interval and validation_rollouts must be positive")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if config.early_stopping_warmup_updates < 0:
        raise ValueError("early_stopping_warmup_updates must be non-negative")
    if config.history_flush_interval < 1:
        raise ValueError("history_flush_interval must be positive")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    archive = EliteArchive(config.elite_capacity)
    history: list[dict[str, Any]] = []
    best_checkpoint_objective = -np.inf
    best_checkpoint_update = 0
    best_for_patience = -np.inf
    checks_without_improvement = 0
    best_path = output_directory / "checkpoint_best.pt"
    total_episodes = config.updates * config.episodes_per_update
    training_started = time.perf_counter()
    next_log_episode = log_every_episodes
    completed_updates = 0
    stop_reason = "maximum_updates_reached"
    fixed_validation_seed = seed + 70_000_000

    for update in range(1, config.updates + 1):
        started = time.perf_counter()
        batch, rollout, trajectories = collect_construction_batch(
            agent,
            number_of_episodes=config.episodes_per_update,
            pomo_group_size=config.pomo_group_size,
            seed=seed + update * 100_003,
            vectorized=config.vectorized_rollouts,
        )
        for trajectory in trajectories:
            archive.add(trajectory.itinerary, trajectory.objective)
        progress = update / max(config.updates, 1)
        agent.set_learning_rate_fraction(max(1.0 - progress, 0.05))
        optimization = agent.update(batch, progress=progress)
        imitation_loss = None
        if (
            config.self_imitation_interval > 0
            and update % config.self_imitation_interval == 0
        ):
            imitation_loss = _replay_elites(
                agent,
                archive,
                count=config.self_imitation_elites,
                epochs=config.self_imitation_epochs,
            )

        record: dict[str, Any] = {
            "update": update,
            "rollout": rollout,
            "optimization": optimization,
            "self_imitation_loss": imitation_loss,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        should_validate = (
            update == 1
            or update == config.updates
            or update % config.validation_interval == 0
        )
        significant_improvement = False
        if should_validate:
            deterministic, best_of_k, mean_stochastic, std_stochastic = _fixed_validation(
                agent,
                rollouts=config.validation_rollouts,
                seed=fixed_validation_seed,
                method_name=method_name,
            )
            validation = {
                "deterministic_objective": deterministic.objective,
                "mean_stochastic_objective": mean_stochastic,
                "std_stochastic_objective": std_stochastic,
                "best_of_k_objective": best_of_k.objective,
                "archive_best_objective": archive.best[1] if archive.best else None,
                "fixed_validation_seed": fixed_validation_seed,
            }
            record["validation"] = validation
            selection_objective = _selection_value(validation, config.validation_metric)
            record["validation"]["selection_objective"] = selection_objective
            record["validation"]["selection_metric"] = config.validation_metric

            if selection_objective > best_checkpoint_objective + 1e-12:
                best_checkpoint_objective = selection_objective
                best_checkpoint_update = update
                agent.save(
                    best_path,
                    metadata={
                        "method": method_name,
                        "selected_update": update,
                        "selection_metric": config.validation_metric,
                        "selection_objective": selection_objective,
                        "fixed_validation_seed": fixed_validation_seed,
                        "training_config": asdict(config),
                    },
                )

            if selection_objective > best_for_patience + config.early_stopping_min_delta:
                best_for_patience = selection_objective
                checks_without_improvement = 0
                significant_improvement = True
            elif update >= config.early_stopping_warmup_updates:
                checks_without_improvement += 1

            print(
                f"[validation] phase=construction "
                f"instance={agent.instance.instance_id} method={method_name} "
                f"update={update}/{config.updates} metric={config.validation_metric} "
                f"selection={selection_objective:.3f} "
                f"deterministic={deterministic.objective:.3f} "
                f"mean_stochastic={mean_stochastic:.3f} "
                f"best_of_k={best_of_k.objective:.3f} "
                f"best_checkpoint={best_checkpoint_objective:.3f} "
                f"checks_without_improvement={checks_without_improvement}/"
                f"{config.early_stopping_patience}",
                flush=True,
            )

        history.append(record)
        completed_updates = update
        completed_episodes = update * config.episodes_per_update
        should_log = log_every_episodes > 0 and (
            update == 1
            or update == config.updates
            or completed_episodes >= next_log_episode
        )
        if should_log:
            elapsed = time.perf_counter() - training_started
            eta = (
                elapsed * (total_episodes - completed_episodes) / completed_episodes
                if completed_episodes > 0
                else 0.0
            )
            archive_best = archive.best[1] if archive.best else -np.inf
            validation = record.get("validation")
            validation_text = (
                f" validation_selection={validation['selection_objective']:.3f}"
                if validation is not None
                else ""
            )
            print(
                f"[progress] phase=construction "
                f"instance={agent.instance.instance_id} method={method_name} "
                f"update={update}/{config.updates} "
                f"episodes={completed_episodes}/{total_episodes} "
                f"mean_objective={rollout['mean_objective']:.3f} "
                f"batch_best={rollout['best_objective']:.3f} "
                f"best_seen={archive_best:.3f} "
                f"loss={optimization['loss']:.6f} "
                f"policy_loss={optimization['policy_loss']:.6f} "
                f"value_loss={optimization['value_loss']:.6f} "
                f"entropy={optimization['normalized_entropy']:.6f} "
                f"kl={optimization['approximate_kl']:.6f} "
                f"ppo_epochs={int(optimization['epochs_completed'])} "
                f"elapsed_seconds={elapsed:.1f} eta_seconds={eta:.1f}"
                f"{validation_text}",
                flush=True,
            )
            while next_log_episode <= completed_episodes:
                next_log_episode += log_every_episodes

        should_flush = (
            update == 1
            or should_validate
            or should_log
            or update % config.history_flush_interval == 0
        )
        if should_flush:
            (output_directory / "history.partial.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )

        if (
            should_validate
            and config.early_stopping_patience > 0
            and update >= config.early_stopping_warmup_updates
            and checks_without_improvement >= config.early_stopping_patience
        ):
            stop_reason = (
                f"early_stopping_after_{checks_without_improvement}_validation_checks"
            )
            print(
                f"[early-stop] phase=construction instance={agent.instance.instance_id} "
                f"method={method_name} update={update} "
                f"best_update={best_checkpoint_update} "
                f"best_selection={best_checkpoint_objective:.3f} "
                f"min_delta={config.early_stopping_min_delta}",
                flush=True,
            )
            break

    if not best_path.exists():
        best_checkpoint_update = completed_updates
        best_checkpoint_objective = float(archive.best[1] if archive.best else -np.inf)
        agent.save(
            best_path,
            metadata={
                "method": method_name,
                "selected_update": completed_updates,
                "selection_metric": "fallback_archive",
                "selection_objective": best_checkpoint_objective,
                "training_config": asdict(config),
            },
        )

    agent.save(
        output_directory / "checkpoint_last.pt",
        metadata={
            "method": method_name,
            "selected_update": completed_updates,
            "training_config": asdict(config),
            "stop_reason": stop_reason,
        },
    )
    _write_history_files(output_directory, history, archive)
    elapsed_total = float(time.perf_counter() - training_started)
    summary = {
        "instance_id": agent.instance.instance_id,
        "method": method_name,
        "planned_updates": config.updates,
        "completed_updates": completed_updates,
        "planned_episodes": total_episodes,
        "completed_episodes": completed_updates * config.episodes_per_update,
        "best_checkpoint_update": best_checkpoint_update,
        "best_selection_objective": best_checkpoint_objective,
        "selection_metric": config.validation_metric,
        "fixed_validation_seed": fixed_validation_seed,
        "early_stopping_enabled": config.early_stopping_patience > 0,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "early_stopping_warmup_updates": config.early_stopping_warmup_updates,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed_total,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(output_directory / "checkpoint_last.pt"),
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[training:done] phase=construction instance={agent.instance.instance_id} "
        f"method={method_name} completed_updates={completed_updates}/{config.updates} "
        f"best_update={best_checkpoint_update} "
        f"best_selection={best_checkpoint_objective:.3f} "
        f"stop_reason={stop_reason} elapsed_seconds={elapsed_total:.1f}",
        flush=True,
    )
    return best_path, history, archive
