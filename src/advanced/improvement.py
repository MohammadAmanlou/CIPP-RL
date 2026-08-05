"""RL-only neural neighborhood improvement trained with PPO."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.advanced.ppo import AdvancedPPOAgent, compute_episode_gae, masked_distribution
from src.advanced.training import PolicyEvaluation, evaluate_best_of_k
from src.core import CIPPInstance, evaluate_itinerary


OPERATOR_NAMES = ("stop", "replace", "swap", "window_rotate", "window_shuffle")
CANDIDATE_FEATURE_DIM = 16
IMPROVEMENT_GLOBAL_DIM = 10


@dataclass(frozen=True, slots=True)
class ImprovementPPOConfig:
    hidden_dim: int = 128
    learning_rate: float = 1e-4
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    target_kl: float = 0.02
    gradient_clip_norm: float = 0.5
    max_candidates: int = 96
    max_moves: int = 30
    discount_factor: float = 1.0
    gae_lambda: float = 1.0




@dataclass(frozen=True, slots=True)
class ImprovementTrainingConfig:
    updates: int = 100
    episodes_per_update: int = 8
    construction_rollouts: int = 4
    validation_interval: int = 10
    validation_starts: int = 4
    validation_construction_rollouts: int = 4
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 1.0
    early_stopping_warmup_updates: int = 0
    history_flush_interval: int = 10

@dataclass(slots=True)
class CandidateSet:
    itineraries: list[tuple[int, ...]]
    features: np.ndarray
    mask: np.ndarray
    objectives: np.ndarray


@dataclass(frozen=True, slots=True)
class ImprovementBatch:
    candidates: np.ndarray
    global_features: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray

    @property
    def size(self) -> int:
        return int(self.actions.size)


def _batch_objectives_and_feasibility(
    instance: CIPPInstance, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact objective and all deterministic feasibility checks."""

    actions = np.arange(1, instance.n + 1, dtype=np.int64)
    visits = candidates[:, :, None] == actions[None, None, :]
    counts = visits.sum(axis=1)
    exposures = (visits * instance.temporal_weights[None, :, None]).sum(axis=1)
    factors = 1.0 - instance.gamma * np.maximum(
        counts - instance.repeat_count_offset, 0
    )
    objectives = (instance.rewards[None, :] * factors * exposures).sum(axis=1)
    feasible = np.all(counts <= instance.q, axis=1)
    costs = np.where(
        candidates > 0,
        instance.costs[np.maximum(candidates - 1, 0)],
        0.0,
    ).sum(axis=1)
    feasible &= costs <= instance.budget + 1e-9
    for start in range(instance.num_rolling_windows):
        window = candidates[:, start : start + instance.alpha]
        feasible &= (window == 0).sum(axis=1) >= instance.idle_requirements[start]
        for action in actions:
            feasible &= (window == action).sum(axis=1) <= instance.w
    return objectives.astype(np.float64), feasible, counts.astype(np.int64)


def _proposal_pool(
    instance: CIPPInstance,
    itinerary: tuple[int, ...],
    *,
    rng: np.random.Generator,
    target_size: int,
) -> tuple[list[tuple[int, ...]], list[int], list[tuple[int, int]]]:
    proposals = [itinerary]
    operators = [0]
    positions = [(-1, -1)]
    seen = {itinerary}
    high_reward_actions = np.argsort(instance.rewards)[::-1][: min(instance.n, 12)] + 1
    array = np.asarray(itinerary, dtype=np.int64)
    attempts = max(target_size * 8, 128)
    for _ in range(attempts):
        operator = int(rng.integers(1, 5))
        candidate = array.copy()
        first = int(rng.integers(0, instance.H))
        second = -1
        if operator == 1:
            if rng.random() < 0.75:
                new_action = int(rng.choice(high_reward_actions))
            else:
                new_action = int(rng.integers(0, instance.num_actions))
            candidate[first] = new_action
        elif operator == 2:
            second = int(rng.integers(0, instance.H))
            candidate[first], candidate[second] = candidate[second], candidate[first]
        else:
            window = min(instance.alpha, instance.H)
            first = int(rng.integers(0, instance.H - window + 1))
            second = first + window - 1
            segment = candidate[first : second + 1].copy()
            if operator == 3:
                direction = 1 if rng.random() < 0.5 else -1
                candidate[first : second + 1] = np.roll(segment, direction)
            else:
                candidate[first : second + 1] = rng.permutation(segment)
        key = tuple(int(value) for value in candidate)
        if key not in seen:
            seen.add(key)
            proposals.append(key)
            operators.append(operator)
            positions.append((first, second))
        if len(proposals) >= target_size * 4:
            break
    return proposals, operators, positions


def build_candidate_set(
    instance: CIPPInstance,
    itinerary: tuple[int, ...],
    *,
    current_best: float,
    move_index: int,
    config: ImprovementPPOConfig,
    seed: int,
) -> tuple[CandidateSet, np.ndarray]:
    rng = np.random.default_rng(seed)
    proposals, operators, positions = _proposal_pool(
        instance,
        itinerary,
        rng=rng,
        target_size=config.max_candidates,
    )
    matrix = np.asarray(proposals, dtype=np.int64)
    objectives, feasible, counts = _batch_objectives_and_feasibility(instance, matrix)
    feasible[0] = True
    feasible_indices = np.flatnonzero(feasible)
    current_objective = float(objectives[0])
    if feasible_indices.size > config.max_candidates:
        non_stop = feasible_indices[feasible_indices != 0]
        ranked = non_stop[np.argsort(objectives[non_stop])[::-1]]
        elite_count = max(1, (config.max_candidates - 1) // 2)
        elite = ranked[:elite_count]
        remainder = ranked[elite_count:]
        random_count = config.max_candidates - 1 - elite.size
        random = (
            rng.choice(remainder, size=min(random_count, remainder.size), replace=False)
            if remainder.size
            else np.empty(0, dtype=np.int64)
        )
        feasible_indices = np.concatenate([[0], elite, random])
    selected_itineraries = [proposals[int(index)] for index in feasible_indices]
    selected_objectives = objectives[feasible_indices]
    scale = max(abs(current_best), abs(current_objective), 1.0)
    features = np.zeros((config.max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(config.max_candidates, dtype=np.bool_)
    for row, source_index in enumerate(feasible_indices):
        operator = operators[int(source_index)]
        first, second = positions[int(source_index)]
        old_action = int(itinerary[first]) if first >= 0 else 0
        new_action = int(proposals[int(source_index)][first]) if first >= 0 else 0
        old_reward = instance.rewards[old_action - 1] if old_action > 0 else 0.0
        new_reward = instance.rewards[new_action - 1] if new_action > 0 else 0.0
        selected_counts = counts[int(source_index)]
        total_visits = max(int(np.sum(selected_counts)), 1)
        shares = selected_counts / total_visits
        features[row] = np.asarray(
            [
                (objectives[int(source_index)] - current_objective) / scale,
                objectives[int(source_index)] / scale,
                *[float(operator == value) for value in range(5)],
                first / max(instance.H - 1, 1) if first >= 0 else 0.0,
                second / max(instance.H - 1, 1) if second >= 0 else 0.0,
                old_reward / max(float(np.max(instance.rewards)), 1.0),
                new_reward / max(float(np.max(instance.rewards)), 1.0),
                float(old_action == 0),
                float(new_action == 0),
                len(set(proposals[int(source_index)]) - {0}) / max(instance.n, 1),
                float(np.sum(shares**2)),
                (current_best - current_objective) / scale,
            ],
            dtype=np.float32,
        )
        mask[row] = True
    while len(selected_itineraries) < config.max_candidates:
        selected_itineraries.append(itinerary)
    padded_objectives = np.zeros(config.max_candidates, dtype=np.float64)
    padded_objectives[: selected_objectives.size] = selected_objectives
    visit_counts = np.bincount(np.asarray(itinerary), minlength=instance.num_actions)[1:]
    total = max(float(np.sum(visit_counts)), 1.0)
    global_features = np.asarray(
        [
            current_objective / scale,
            current_best / scale,
            move_index / max(config.max_moves, 1),
            (config.max_moves - move_index) / max(config.max_moves, 1),
            np.count_nonzero(np.asarray(itinerary) == 0) / max(instance.H, 1),
            np.count_nonzero(visit_counts) / max(instance.n, 1),
            float(np.mean(visit_counts / max(instance.q, 1))),
            float(np.max(visit_counts / max(instance.q, 1), initial=0.0)),
            float(np.sum((visit_counts / total) ** 2)),
            float(np.mean(mask)),
        ],
        dtype=np.float32,
    )
    return (
        CandidateSet(selected_itineraries, features, mask, padded_objectives),
        global_features,
    )


class ImprovementNetwork(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.global_actor = nn.Sequential(
            nn.Linear(IMPROVEMENT_GLOBAL_DIM, hidden_dim), nn.GELU()
        )
        self.scorer = nn.Linear(hidden_dim, 1)
        self.delta_scale = nn.Parameter(torch.tensor(1.0))
        self.global_critic = nn.Sequential(
            nn.Linear(IMPROVEMENT_GLOBAL_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, candidates: torch.Tensor, global_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.candidate_encoder(candidates)
        context = self.global_actor(global_features).unsqueeze(1)
        logits = self.scorer(torch.tanh(tokens + context)).squeeze(-1)
        logits = logits + F.softplus(self.delta_scale) * candidates[:, :, 0]
        values = self.global_critic(global_features).squeeze(-1)
        return logits, values


class ImprovementPPOAgent:
    model_type = "cipp_rl_improvement_ppo_v1"

    def __init__(
        self,
        instance: CIPPInstance,
        config: ImprovementPPOConfig,
        *,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.instance = instance
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.network = ImprovementNetwork(config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def select(
        self,
        candidate_set: CandidateSet,
        global_features: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[int, float, float]:
        self.network.eval()
        with torch.no_grad():
            candidates = torch.as_tensor(
                candidate_set.features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            global_tensor = torch.as_tensor(
                global_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mask = torch.as_tensor(
                candidate_set.mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            logits, values = self.network(candidates, global_tensor)
            distribution = masked_distribution(logits, mask)
            action = (
                torch.argmax(distribution.logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_probability = distribution.log_prob(action)
        self.network.train()
        return int(action.item()), float(log_probability.item()), float(values.item())

    def update(self, batch: ImprovementBatch) -> dict[str, float]:
        tensors = {
            "candidates": torch.as_tensor(batch.candidates, dtype=torch.float32, device=self.device),
            "global": torch.as_tensor(batch.global_features, dtype=torch.float32, device=self.device),
            "masks": torch.as_tensor(batch.masks, dtype=torch.bool, device=self.device),
            "actions": torch.as_tensor(batch.actions, dtype=torch.long, device=self.device),
            "old_logp": torch.as_tensor(batch.old_log_probabilities, dtype=torch.float32, device=self.device),
            "old_values": torch.as_tensor(batch.old_values, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device),
        }
        advantages = tensors["advantages"]
        tensors["advantages"] = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approximate_kl": 0.0}
        count = 0
        epochs = 0
        for _ in range(self.config.update_epochs):
            permutation = self.rng.permutation(batch.size)
            epoch_kl: list[float] = []
            for start in range(0, batch.size, self.config.minibatch_size):
                index = torch.as_tensor(
                    permutation[start : start + self.config.minibatch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                logits, values = self.network(tensors["candidates"][index], tensors["global"][index])
                distribution = masked_distribution(logits, tensors["masks"][index])
                new_logp = distribution.log_prob(tensors["actions"][index])
                log_ratio = new_logp - tensors["old_logp"][index]
                ratio = torch.exp(log_ratio)
                policy_loss = -torch.min(
                    ratio * tensors["advantages"][index],
                    torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                    * tensors["advantages"][index],
                ).mean()
                old_values = tensors["old_values"][index]
                clipped = old_values + torch.clamp(
                    values - old_values,
                    -self.config.value_clip_epsilon,
                    self.config.value_clip_epsilon,
                )
                target = tensors["returns"][index]
                value_loss = torch.maximum(
                    F.smooth_l1_loss(values, target, reduction="none"),
                    F.smooth_l1_loss(clipped, target, reduction="none"),
                ).mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + self.config.value_loss_coefficient * value_loss - self.config.entropy_coefficient * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1.0) - log_ratio).mean()
                for key, value in {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approximate_kl": kl,
                }.items():
                    totals[key] += float(value.detach().cpu().item())
                epoch_kl.append(float(kl.detach().cpu().item()))
                count += 1
            epochs += 1
            if epoch_kl and float(np.mean(epoch_kl)) > self.config.target_kl:
                break
        result = {key: value / max(count, 1) for key, value in totals.items()}
        result["epochs_completed"] = float(epochs)
        return result

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "config": asdict(self.config),
                "instance": {"n": self.instance.n, "H": self.instance.H, "id": self.instance.instance_id},
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metadata": metadata,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: str | Path, *, instance: CIPPInstance, device: str = "cpu"
    ) -> tuple["ImprovementPPOAgent", dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("model_type") != cls.model_type:
            raise ValueError("checkpoint is not a CIPP improvement PPO model")
        if payload["instance"]["n"] != instance.n or payload["instance"]["H"] != instance.H:
            raise ValueError("improvement checkpoint dimensions do not match instance")
        agent = cls(instance, ImprovementPPOConfig(**payload["config"]), device=device)
        agent.network.load_state_dict(payload["network_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return agent, payload


def _improvement_episode(
    agent: ImprovementPPOAgent,
    start: tuple[int, ...],
    *,
    seed: int,
    deterministic: bool,
) -> tuple[ImprovementBatch, tuple[int, ...], float]:
    current = start
    current_objective = float(evaluate_itinerary(agent.instance, current).objective)
    best = current
    best_objective = current_objective
    candidate_features: list[np.ndarray] = []
    global_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    log_probabilities: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    scale = max(abs(current_objective), 1.0)
    for move in range(agent.config.max_moves):
        candidate_set, global_state = build_candidate_set(
            agent.instance,
            current,
            current_best=best_objective,
            move_index=move,
            config=agent.config,
            seed=seed + move * 7919,
        )
        action, log_probability, value = agent.select(
            candidate_set, global_state, deterministic=deterministic
        )
        new = candidate_set.itineraries[action]
        new_objective = float(candidate_set.objectives[action])
        candidate_features.append(candidate_set.features)
        global_features.append(global_state)
        masks.append(candidate_set.mask)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append((new_objective - current_objective) / scale)
        current, current_objective = new, new_objective
        if new_objective > best_objective:
            best, best_objective = new, new_objective
        if action == 0:
            break
    advantages, returns = compute_episode_gae(
        np.asarray(rewards, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        discount_factor=agent.config.discount_factor,
        gae_lambda=agent.config.gae_lambda,
    )
    batch = ImprovementBatch(
        candidates=np.stack(candidate_features),
        global_features=np.stack(global_features),
        masks=np.stack(masks),
        actions=np.asarray(actions, dtype=np.int64),
        old_log_probabilities=np.asarray(log_probabilities, dtype=np.float32),
        old_values=np.asarray(values, dtype=np.float32),
        returns=returns,
        advantages=advantages,
    )
    return batch, best, best_objective


def _merge_batches(batches: list[ImprovementBatch]) -> ImprovementBatch:
    return ImprovementBatch(
        candidates=np.concatenate([batch.candidates for batch in batches]),
        global_features=np.concatenate([batch.global_features for batch in batches]),
        masks=np.concatenate([batch.masks for batch in batches]),
        actions=np.concatenate([batch.actions for batch in batches]),
        old_log_probabilities=np.concatenate([batch.old_log_probabilities for batch in batches]),
        old_values=np.concatenate([batch.old_values for batch in batches]),
        returns=np.concatenate([batch.returns for batch in batches]),
        advantages=np.concatenate([batch.advantages for batch in batches]),
    )


def _fixed_improvement_validation(
    agent: ImprovementPPOAgent,
    starts: list[PolicyEvaluation],
    *,
    seed: int,
) -> dict[str, float]:
    results = [
        evaluate_rl_improvement(
            agent,
            start,
            seed=seed + index * 10_007,
            method="hacipp_rl_improve_validation",
        )
        for index, start in enumerate(starts)
    ]
    final_objectives = np.asarray([result.objective for result in results], dtype=np.float64)
    start_objectives = np.asarray([start.objective for start in starts], dtype=np.float64)
    gains = final_objectives - start_objectives
    return {
        "mean_final_objective": float(np.mean(final_objectives)),
        "best_final_objective": float(np.max(final_objectives)),
        "std_final_objective": float(np.std(final_objectives)),
        "mean_gain": float(np.mean(gains)),
        "best_gain": float(np.max(gains)),
        "feasible_rate": float(np.mean([result.feasible for result in results])),
    }


def train_improvement_agent(
    agent: ImprovementPPOAgent,
    construction_agent: AdvancedPPOAgent,
    *,
    config: ImprovementTrainingConfig | None = None,
    updates: int | None = None,
    episodes_per_update: int | None = None,
    construction_rollouts: int | None = None,
    output_directory: str | Path,
    seed: int,
    log_every_episodes: int = 256,
) -> Path:
    """Train the RL improvement policy with fixed-start validation and early stop."""

    if config is None:
        if updates is None or episodes_per_update is None or construction_rollouts is None:
            raise ValueError("Pass config, or pass updates, episodes_per_update, and construction_rollouts")
        config = ImprovementTrainingConfig(
            updates=updates,
            episodes_per_update=episodes_per_update,
            construction_rollouts=construction_rollouts,
            validation_interval=max(1, updates),
            validation_starts=1,
            validation_construction_rollouts=construction_rollouts,
            early_stopping_patience=0,
        )
    if log_every_episodes < 0:
        raise ValueError("log_every_episodes must be non-negative")
    if config.updates < 1 or config.episodes_per_update < 1:
        raise ValueError("updates and episodes_per_update must be positive")
    if config.construction_rollouts < 1:
        raise ValueError("construction_rollouts must be positive")
    if config.validation_interval < 1 or config.validation_starts < 1:
        raise ValueError("validation_interval and validation_starts must be positive")
    if config.validation_construction_rollouts < 1:
        raise ValueError("validation_construction_rollouts must be positive")
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
    best_checkpoint_metric = -np.inf
    best_checkpoint_update = 0
    best_for_patience = -np.inf
    checks_without_improvement = 0
    best_path = output_directory / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    total_episodes = config.updates * config.episodes_per_update
    training_started = time.perf_counter()
    next_log_episode = log_every_episodes
    completed_updates = 0
    stop_reason = "maximum_updates_reached"

    validation_start_seed = seed + 80_000_000
    validation_starts = [
        evaluate_best_of_k(
            construction_agent,
            rollouts=config.validation_construction_rollouts,
            seed=validation_start_seed + index * 100_003,
            method="fixed_construction_start_for_improvement_validation",
        )
        for index in range(config.validation_starts)
    ]
    (output_directory / "fixed_validation_starts.json").write_text(
        json.dumps([start.to_dict() for start in validation_starts], indent=2) + "\n",
        encoding="utf-8",
    )

    for update in range(1, config.updates + 1):
        update_started = time.perf_counter()
        batches: list[ImprovementBatch] = []
        episode_best: list[float] = []
        start_objectives: list[float] = []
        for episode in range(config.episodes_per_update):
            start_result = evaluate_best_of_k(
                construction_agent,
                rollouts=config.construction_rollouts,
                seed=seed + update * 100_003 + episode * config.construction_rollouts,
                method="construction_for_improvement_training",
            )
            batch, _, objective = _improvement_episode(
                agent,
                start_result.itinerary,
                seed=seed + update * 1_000_003 + episode * 10_007,
                deterministic=False,
            )
            batches.append(batch)
            start_objectives.append(start_result.objective)
            episode_best.append(objective)
        metrics = agent.update(_merge_batches(batches))
        update_best = float(np.max(episode_best))
        mean_best = float(np.mean(episode_best))
        mean_gain = float(np.mean(np.asarray(episode_best) - np.asarray(start_objectives)))
        record: dict[str, Any] = {
            "update": update,
            "mean_start_objective": float(np.mean(start_objectives)),
            "mean_best_objective": mean_best,
            "best_objective": update_best,
            "mean_gain": mean_gain,
            "optimization": metrics,
            "elapsed_seconds": float(time.perf_counter() - update_started),
        }

        should_validate = (
            update == 1
            or update == config.updates
            or update % config.validation_interval == 0
        )
        if should_validate:
            validation = _fixed_improvement_validation(
                agent,
                validation_starts,
                seed=seed + 81_000_000,
            )
            record["validation"] = validation
            selection_metric = float(validation["mean_final_objective"])
            if selection_metric > best_checkpoint_metric + 1e-12:
                best_checkpoint_metric = selection_metric
                best_checkpoint_update = update
                agent.save(
                    best_path,
                    metadata={
                        "selected_update": update,
                        "selection_metric": "fixed_validation_mean_final_objective",
                        "selection_objective": selection_metric,
                        "training_config": asdict(config),
                        "fixed_validation_start_seed": validation_start_seed,
                    },
                )

            if selection_metric > best_for_patience + config.early_stopping_min_delta:
                best_for_patience = selection_metric
                checks_without_improvement = 0
            elif update >= config.early_stopping_warmup_updates:
                checks_without_improvement += 1

            print(
                f"[validation] phase=improvement "
                f"instance={agent.instance.instance_id} method=hacipp_rl_improve "
                f"update={update}/{config.updates} "
                f"mean_final={validation['mean_final_objective']:.3f} "
                f"mean_gain={validation['mean_gain']:.3f} "
                f"best_final={validation['best_final_objective']:.3f} "
                f"best_checkpoint={best_checkpoint_metric:.3f} "
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
            validation_text = (
                f" validation_mean_final={record['validation']['mean_final_objective']:.3f}"
                if record.get("validation") is not None
                else ""
            )
            print(
                f"[progress] phase=improvement "
                f"instance={agent.instance.instance_id} method=hacipp_rl_improve "
                f"update={update}/{config.updates} "
                f"episodes={completed_episodes}/{total_episodes} "
                f"mean_start={np.mean(start_objectives):.3f} "
                f"mean_objective={mean_best:.3f} "
                f"mean_gain={mean_gain:.3f} "
                f"batch_best={update_best:.3f} best_seen={best_checkpoint_metric:.3f} "
                f"loss={metrics['loss']:.6f} "
                f"policy_loss={metrics['policy_loss']:.6f} "
                f"value_loss={metrics['value_loss']:.6f} "
                f"entropy={metrics['entropy']:.6f} "
                f"kl={metrics['approximate_kl']:.6f} "
                f"ppo_epochs={int(metrics['epochs_completed'])} "
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
                f"[early-stop] phase=improvement instance={agent.instance.instance_id} "
                f"method=hacipp_rl_improve update={update} "
                f"best_update={best_checkpoint_update} "
                f"best_selection={best_checkpoint_metric:.3f} "
                f"min_delta={config.early_stopping_min_delta}",
                flush=True,
            )
            break

    if not best_path.exists():
        best_checkpoint_update = completed_updates
        best_checkpoint_metric = float(np.mean(episode_best))
        agent.save(
            best_path,
            metadata={
                "selected_update": completed_updates,
                "selection_metric": "fallback_training_mean",
                "selection_objective": best_checkpoint_metric,
                "training_config": asdict(config),
            },
        )

    agent.save(
        output_directory / "checkpoint_last.pt",
        metadata={
            "selected_update": completed_updates,
            "training_config": asdict(config),
            "stop_reason": stop_reason,
        },
    )
    (output_directory / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    elapsed_total = float(time.perf_counter() - training_started)
    summary = {
        "instance_id": agent.instance.instance_id,
        "method": "hacipp_rl_improve",
        "planned_updates": config.updates,
        "completed_updates": completed_updates,
        "planned_episodes": total_episodes,
        "completed_episodes": completed_updates * config.episodes_per_update,
        "best_checkpoint_update": best_checkpoint_update,
        "best_selection_objective": best_checkpoint_metric,
        "selection_metric": "fixed_validation_mean_final_objective",
        "fixed_validation_start_seed": validation_start_seed,
        "validation_starts": config.validation_starts,
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
        f"[training:done] phase=improvement instance={agent.instance.instance_id} "
        f"method=hacipp_rl_improve completed_updates={completed_updates}/{config.updates} "
        f"best_update={best_checkpoint_update} "
        f"best_selection={best_checkpoint_metric:.3f} "
        f"stop_reason={stop_reason} elapsed_seconds={elapsed_total:.1f}",
        flush=True,
    )
    return best_path


def evaluate_rl_improvement(
    agent: ImprovementPPOAgent,
    start: PolicyEvaluation,
    *,
    seed: int,
    method: str = "hacipp_rl_improve",
) -> PolicyEvaluation:
    started = time.perf_counter()
    _, best, best_objective = _improvement_episode(
        agent, start.itinerary, seed=seed, deterministic=True
    )
    evaluation = evaluate_itinerary(agent.instance, best)
    counts = evaluation.visit_counts
    total = max(float(np.sum(counts)), 1.0)
    hhi = float(np.sum((counts / total) ** 2))
    if best_objective + 1e-8 < start.objective:
        raise RuntimeError("best-incumbent RL improvement became worse than its start")
    return PolicyEvaluation(
        method=method,
        objective=float(best_objective),
        itinerary=best,
        feasible=bool(evaluation.feasible),
        runtime_seconds=float(time.perf_counter() - started),
        idle_days=int(evaluation.idle_days),
        unique_locations=int(np.count_nonzero(counts)),
        visit_hhi=hhi,
    )
