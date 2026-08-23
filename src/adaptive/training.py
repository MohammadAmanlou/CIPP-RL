"""Training and residual inference for context-conditioned adaptive Attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import time

import numpy as np

from src.advanced.ppo import AdvancedPPOAgent, AdvancedPPOBatch, compute_episode_gae
from src.advanced.features import StructuredFeatureBuilder
from src.envs import CIPPEnv
from src.adaptive.context import (
    AdaptiveCIPPEnv,
    AdaptiveFeatureBuilder,
    ShockScenario,
    replay_prefix,
    sample_training_scenario,
)


@dataclass(frozen=True, slots=True)
class AdaptiveTrainingConfig:
    updates: int = 300
    episodes_per_update: int = 64
    validation_interval: int = 10
    validation_scenarios: int = 8
    validation_rollouts_per_scenario: int = 16
    early_stopping_patience: int = 6
    early_stopping_warmup_updates: int = 80
    early_stopping_min_delta: float = 2.0
    learning_rate_scale: float = 0.10
    update_epochs: int = 2


def _collect_batch(
    agent: AdvancedPPOAgent,
    *,
    episodes: int,
    seed: int,
) -> tuple[AdvancedPPOBatch, dict[str, float]]:
    rng = np.random.default_rng(seed)
    envs = []
    for episode in range(episodes):
        scenario = sample_training_scenario(agent.instance, rng)
        env = AdaptiveCIPPEnv(agent.instance, scenario, seed=seed + episode)
        env.reset(seed=seed + episode)
        envs.append(env)

    states_per_episode = [[] for _ in envs]
    actions_per_episode = [[] for _ in envs]
    logp_per_episode = [[] for _ in envs]
    values_per_episode = [[] for _ in envs]
    rewards_per_episode = [[] for _ in envs]

    while True:
        active = [i for i, env in enumerate(envs) if not env.done]
        if not active:
            break
        states = [agent.feature_builder.build(envs[i]) for i in active]
        actions, logps, values = agent.batch_actions(states, deterministic=False)
        for j, idx in enumerate(active):
            env = envs[idx]
            _, reward, _, _, _ = env.step(int(actions[j]))
            states_per_episode[idx].append(states[j])
            actions_per_episode[idx].append(int(actions[j]))
            logp_per_episode[idx].append(float(logps[j]))
            values_per_episode[idx].append(float(values[j]))
            rewards_per_episode[idx].append(float(reward) * agent.reward_scale)

    all_states = []
    all_actions = []
    all_logps = []
    all_values = []
    all_returns = []
    all_advantages = []
    all_counts = []
    objectives = []

    for idx, env in enumerate(envs):
        rewards = np.asarray(rewards_per_episode[idx], dtype=np.float32)
        values = np.asarray(values_per_episode[idx], dtype=np.float32)
        advantages, returns = compute_episode_gae(
            rewards,
            values,
            discount_factor=agent.config.discount_factor,
            gae_lambda=agent.config.gae_lambda,
        )
        n_steps = len(actions_per_episode[idx])
        all_states.extend(states_per_episode[idx])
        all_actions.append(np.asarray(actions_per_episode[idx], dtype=np.int64))
        all_logps.append(np.asarray(logp_per_episode[idx], dtype=np.float32))
        all_values.append(values)
        all_returns.append(returns)
        all_advantages.append(advantages)
        all_counts.append(np.repeat(env.visit_counts[None, :], n_steps, axis=0))
        objectives.append(float(env.cumulative_reward))

    batch = AdvancedPPOBatch(
        locations=np.stack([s.locations for s in all_states]),
        global_features=np.stack([s.global_features for s in all_states]),
        action_masks=np.stack([s.action_mask for s in all_states]),
        actions=np.concatenate(all_actions),
        old_log_probabilities=np.concatenate(all_logps),
        old_values=np.concatenate(all_values),
        returns=np.concatenate(all_returns),
        advantages=np.concatenate(all_advantages),
        final_visit_counts=np.concatenate(all_counts, axis=0),
    )
    return batch, {
        "mean_objective": float(np.mean(objectives)),
        "best_objective": float(np.max(objectives)),
        "std_objective": float(np.std(objectives)),
        "transitions": float(batch.size),
    }



def _replay_static_prefix(
    instance,
    prefix: tuple[int, ...],
    *,
    seed: int,
) -> CIPPEnv:
    """Replay realized actions in a genuinely static CIPP environment.

    This environment is used ONLY for B/D policy observations.  It never sees
    shock-adjusted rewards, adaptive cumulative reward, adaptive exposure, or
    any other post-shock context.  Because shocks do not alter feasibility,
    stepping it in lockstep with AdaptiveCIPPEnv is valid.
    """
    env = CIPPEnv(instance, seed=seed)
    env.reset(seed=seed)
    for action in prefix:
        env.step(int(action))
    return env


def _assert_lockstep_feasibility(
    static_env: CIPPEnv,
    adaptive_env: AdaptiveCIPPEnv,
) -> None:
    """Fail loudly if the paired static/adaptive histories ever diverge."""
    if static_env.day != adaptive_env.day:
        raise RuntimeError(
            f"paired environments are on different days: "
            f"{static_env.day} vs {adaptive_env.day}"
        )
    if not np.array_equal(static_env.itinerary, adaptive_env.itinerary):
        raise RuntimeError("paired environments have different realized itineraries")
    if not np.array_equal(
        static_env.get_action_mask(),
        adaptive_env.get_action_mask(),
    ):
        raise RuntimeError(
            "shock unexpectedly changed feasibility/action masks; "
            "clean static baseline requires identical constraints"
        )


def residual_best_of_k(
    agent: AdvancedPPOAgent,
    *,
    scenario: ShockScenario,
    prefix: tuple[int, ...],
    rollouts: int,
    seed: int,
    context_aware: bool,
) -> dict[str, object]:
    """Sample only the suffix and select using the true adaptive objective.

    For Method C (``context_aware=True``), the policy observes AdaptiveCIPPEnv.

    For frozen baselines B/D (``context_aware=False``), policy observations are
    built from a SEPARATE static CIPPEnv replayed with exactly the same actions.
    The adaptive environment is used only for scoring.  This removes the V3
    leakage through global ``current_objective`` and any other adaptive state.
    """

    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    started = time.perf_counter()
    rngs = [np.random.default_rng(seed + i) for i in range(rollouts)]

    score_envs = [
        replay_prefix(agent.instance, scenario, prefix, seed=seed + i)
        for i in range(rollouts)
    ]

    if context_aware:
        observation_envs = score_envs
        static_builder = None
    else:
        observation_envs = [
            _replay_static_prefix(
                agent.instance,
                prefix,
                seed=seed + i,
            )
            for i in range(rollouts)
        ]
        static_builder = StructuredFeatureBuilder(agent.instance)
        for static_env, adaptive_env in zip(observation_envs, score_envs):
            _assert_lockstep_feasibility(static_env, adaptive_env)

    while True:
        active = [i for i, env in enumerate(score_envs) if not env.done]
        if not active:
            break

        if context_aware:
            states = [
                agent.feature_builder.build(observation_envs[i])
                for i in active
            ]
        else:
            states = [
                static_builder.build(observation_envs[i])
                for i in active
            ]

        probs, _ = agent.batch_probabilities_and_values(states)
        for j, idx in enumerate(active):
            state = states[j]
            row = np.where(
                state.action_mask,
                np.clip(probs[j], 0.0, None),
                0.0,
            )
            total = float(row.sum())
            if total <= 0.0:
                raise RuntimeError("zero probability on every feasible action")
            row /= total
            action = int(
                rngs[idx].choice(agent.instance.num_actions, p=row)
            )

            # Same realized action in both worlds:
            # - observation_env stays static for B/D policy input
            # - score_env uses the true shocked objective
            if not context_aware:
                observation_envs[idx].step(action)
            score_envs[idx].step(action)

            if not context_aware:
                _assert_lockstep_feasibility(
                    observation_envs[idx],
                    score_envs[idx],
                )

    best = max(score_envs, key=lambda e: e.cumulative_reward)
    return {
        "objective": float(best.cumulative_reward),
        "itinerary": best.itinerary.tolist(),
        "runtime_seconds": float(time.perf_counter() - started),
        "rollouts": int(rollouts),
        "observation_semantics": (
            "adaptive_context"
            if context_aware
            else "strict_static_parallel_env_no_shock_leakage"
        ),
    }


def residual_greedy(
    agent: AdvancedPPOAgent,
    *,
    scenario: ShockScenario,
    prefix: tuple[int, ...],
    seed: int,
    context_aware: bool,
) -> dict[str, object]:
    """Greedy suffix inference with leakage-free frozen baselines."""

    started = time.perf_counter()
    score_env = replay_prefix(
        agent.instance,
        scenario,
        prefix,
        seed=seed,
    )

    if context_aware:
        observation_env = score_env
        static_builder = None
    else:
        observation_env = _replay_static_prefix(
            agent.instance,
            prefix,
            seed=seed,
        )
        static_builder = StructuredFeatureBuilder(agent.instance)
        _assert_lockstep_feasibility(observation_env, score_env)

    while not score_env.done:
        if context_aware:
            state = agent.feature_builder.build(observation_env)
        else:
            state = static_builder.build(observation_env)

        action, _, _ = agent.select_action(
            state,
            deterministic=True,
        )

        if not context_aware:
            observation_env.step(action)
        score_env.step(action)

        if not context_aware:
            _assert_lockstep_feasibility(observation_env, score_env)

    return {
        "objective": float(score_env.cumulative_reward),
        "itinerary": score_env.itinerary.tolist(),
        "runtime_seconds": float(time.perf_counter() - started),
        "rollouts": 1,
        "observation_semantics": (
            "adaptive_context"
            if context_aware
            else "strict_static_parallel_env_no_shock_leakage"
        ),
    }



def build_anticipatory_prefix(
    agent: AdvancedPPOAgent,
    *,
    scenario: ShockScenario,
    seed: int,
    deterministic: bool = True,
) -> tuple[int, ...]:
    """Generate Method C's realized pre-shock history without future leakage.

    The adaptive policy starts at day 1 and controls every pre-shock action.
    Before ``scenario.switch_day`` the environment exposes multiplier 1 for all
    locations, so the policy cannot know which locations will later be boosted
    or suppressed.  It can only exploit anticipatory behavior learned from the
    training distribution (for example, preserving budget/visit optionality).

    No best-of-K or hindsight selection is allowed before the shock.
    """

    scenario.validate(agent.instance)
    env = AdaptiveCIPPEnv(agent.instance, scenario, seed=seed)
    env.reset(seed=seed)

    prefix: list[int] = []
    while env.day < scenario.switch_day:
        state = agent.feature_builder.build(env)
        action, _, _ = agent.select_action(
            state,
            deterministic=deterministic,
        )
        env.step(int(action))
        prefix.append(int(action))

    return tuple(prefix)


def anticipatory_then_residual_best_of_k(
    agent: AdvancedPPOAgent,
    *,
    scenario: ShockScenario,
    rollouts: int,
    seed: int,
    deterministic_prefix: bool = True,
) -> dict[str, object]:
    """Fair online E2E Method C: anticipate first, adapt after the shock.

    Phase 1 (before shock):
        One realized prefix is generated online without knowledge of the future
        shock identity.  Critically, there is NO best-of-K prefix selection.

    Phase 2 (after shock):
        The realized shock is now observable, so Method C may sample K suffixes
        from the fixed realized prefix and select the best adaptive continuation.

    This matches the information pattern of a real deployment and makes the
    comparison to two-stage Gurobi scientifically fair.
    """

    prefix = build_anticipatory_prefix(
        agent,
        scenario=scenario,
        seed=seed,
        deterministic=deterministic_prefix,
    )
    result = residual_best_of_k(
        agent,
        scenario=scenario,
        prefix=prefix,
        rollouts=rollouts,
        seed=seed + 10_000_000,
        context_aware=True,
    )
    result["prefix"] = list(prefix)
    result["prefix_source"] = "method_c_anticipatory_no_future_shock_info"
    result["pre_shock_hindsight_selection"] = False
    return result


def anticipatory_then_greedy(
    agent: AdvancedPPOAgent,
    *,
    scenario: ShockScenario,
    seed: int,
    deterministic_prefix: bool = True,
) -> dict[str, object]:
    """Fair E2E Method C with greedy post-shock continuation."""

    prefix = build_anticipatory_prefix(
        agent,
        scenario=scenario,
        seed=seed,
        deterministic=deterministic_prefix,
    )
    result = residual_greedy(
        agent,
        scenario=scenario,
        prefix=prefix,
        seed=seed + 10_000_000,
        context_aware=True,
    )
    result["prefix"] = list(prefix)
    result["prefix_source"] = "method_c_anticipatory_no_future_shock_info"
    result["pre_shock_hindsight_selection"] = False
    return result


def _fixed_validation(
    agent: AdvancedPPOAgent,
    *,
    scenario_seed: int,
    scenario_count: int,
    rollouts_per_scenario: int,
) -> float:
    """Validate the actual deployment semantics, not a clairvoyant K-sample oracle.

    Earlier versions sampled K complete trajectories from day 1 and selected the
    best after seeing the realized shock.  That can reward lucky pre-shock
    prefixes in hindsight.  V5 instead generates exactly one deterministic
    anticipatory prefix without future shock information and only applies
    best-of-K search after the shock becomes observable.
    """

    rng = np.random.default_rng(scenario_seed)
    objectives = []
    scenarios = [
        sample_training_scenario(agent.instance, rng)
        for _ in range(scenario_count)
    ]
    for k, scenario in enumerate(scenarios):
        result = anticipatory_then_residual_best_of_k(
            agent,
            scenario=scenario,
            rollouts=rollouts_per_scenario,
            seed=scenario_seed + 100_000 * (k + 1),
            deterministic_prefix=True,
        )
        objectives.append(float(result["objective"]))
    return float(np.mean(objectives))



def train_method_c(
    agent: AdvancedPPOAgent,
    *,
    config: AdaptiveTrainingConfig,
    output_directory: str | Path,
    seed: int = 42,
) -> Path:
    """Train context-conditioned Attention over a distribution of shocks."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if not 0.0 < config.learning_rate_scale <= 1.0:
        raise ValueError("learning_rate_scale must be in (0, 1]")
    if config.update_epochs < 1:
        raise ValueError("update_epochs must be >= 1")

    # Method C is a warm-start adaptation of an already-trained static policy.
    # Use a smaller learning rate and fewer PPO epochs to avoid destroying the
    # pretrained policy when the reward-context distribution changes.
    scaled_actor_lr = agent.config.actor_learning_rate * config.learning_rate_scale
    scaled_critic_lr = agent.config.critic_learning_rate * config.learning_rate_scale
    agent.config = replace(
        agent.config,
        actor_learning_rate=scaled_actor_lr,
        critic_learning_rate=scaled_critic_lr,
        update_epochs=config.update_epochs,
    )
    scaled_base_lrs = [
        float(lr) * config.learning_rate_scale for lr in agent._base_group_lrs
    ]
    agent._base_group_lrs = scaled_base_lrs
    for group, lr in zip(agent.optimizer.param_groups, scaled_base_lrs):
        group["lr"] = lr

    history = []
    best_value = -np.inf
    best_for_patience = -np.inf
    checks_without_improvement = 0
    best_path = output_directory / "checkpoint_best.pt"
    validation_seed = seed + 91_000_000
    started_all = time.perf_counter()

    for update in range(1, config.updates + 1):
        started = time.perf_counter()
        batch, rollout = _collect_batch(
            agent,
            episodes=config.episodes_per_update,
            seed=seed + update * 100_003,
        )
        progress = update / config.updates
        agent.set_learning_rate_fraction(max(1.0 - progress, 0.05))
        opt = agent.update(batch, progress=progress)

        record = {
            "update": update,
            "rollout": rollout,
            "optimization": opt,
            "elapsed_seconds": float(time.perf_counter() - started),
        }

        should_validate = (
            update == 1
            or update == config.updates
            or update % config.validation_interval == 0
        )
        if should_validate:
            value = _fixed_validation(
                agent,
                scenario_seed=validation_seed,
                scenario_count=config.validation_scenarios,
                rollouts_per_scenario=config.validation_rollouts_per_scenario,
            )
            record["validation_mean_best_of_k"] = value
            print(
                f"[adaptive:update] update={update} "
                f"train_mean={rollout['mean_objective']:.3f} "
                f"train_best={rollout['best_objective']:.3f} "
                f"validation={value:.3f} "
                f"kl={opt['approximate_kl']:.6f} "
                f"epochs={int(opt['epochs_completed'])} "
                f"kl_stop={bool(opt['early_kl_stop'])}",
                flush=True,
            )
            if value > best_value:
                best_value = value
                agent.save(
                    best_path,
                    metadata={
                        "phase": "adaptive_method_c",
                        "best_validation": best_value,
                        "best_update": update,
                    },
                )

            # A patience of 0 explicitly DISABLES early stopping. This is
            # useful for smoke tests and avoids the previous immediate stop at
            # update 1.
            if (
                config.early_stopping_patience > 0
                and update >= config.early_stopping_warmup_updates
            ):
                if value > best_for_patience + config.early_stopping_min_delta:
                    best_for_patience = value
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                if checks_without_improvement >= config.early_stopping_patience:
                    record["early_stop"] = True
                    history.append(record)
                    break

        history.append(record)
        if update % 5 == 0:
            (output_directory / "history.partial.json").write_text(
                json.dumps(history, indent=2) + "\n",
                encoding="utf-8",
            )

    agent.save(
        output_directory / "checkpoint_last.pt",
        metadata={
            "phase": "adaptive_method_c",
            "best_validation": float(best_value),
            "elapsed_seconds": float(time.perf_counter() - started_all),
        },
    )
    (output_directory / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "training_summary.json").write_text(
        json.dumps(
            {
                "best_validation": float(best_value),
                "updates_completed": len(history),
                "elapsed_seconds": float(time.perf_counter() - started_all),
                "config": asdict(config),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return best_path
