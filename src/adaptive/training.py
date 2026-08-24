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
    AdaptiveScenario,
    ShockScenario,
    RICH_CURRICULUM_DESCRIPTION,
    generate_validation_scenarios,
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


@dataclass(frozen=True, slots=True)
class ResidualRetrainingConfig:
    """Online RL retraining after a realized shock with the prefix frozen."""

    updates: int = 120
    episodes_per_update: int = 64
    validation_interval: int = 5
    validation_rollouts: int = 32
    early_stopping_patience: int = 10
    early_stopping_warmup_updates: int = 20
    early_stopping_min_delta: float = 1.0
    learning_rate_scale: float = 0.20
    update_epochs: int = 2


@dataclass(frozen=True, slots=True)
class RecourseAwareTrainingConfig:
    """Prefix-policy fine-tuning against a strong downstream recourse operator.

    The student controls ONLY the pre-shock prefix without seeing the future
    shock identity. Once the shock is revealed, a frozen teacher policy (or,
    optionally, exact Gurobi on a licensed machine) solves the suffix. The final
    recourse value is fed back as the terminal learning signal for the prefix.
    """

    updates: int = 180
    prefixes_per_update: int = 16
    teacher_rollouts: int = 32
    validation_interval: int = 10
    validation_scenarios: int = 16
    validation_rollouts: int = 64
    early_stopping_patience: int = 8
    early_stopping_warmup_updates: int = 50
    early_stopping_min_delta: float = 1.0
    learning_rate_scale: float = 0.05
    update_epochs: int = 2


def _collect_batch(
    agent: AdvancedPPOAgent,
    *,
    episodes: int,
    seed: int,
    progress: float,
) -> tuple[AdvancedPPOBatch, dict[str, float]]:
    rng = np.random.default_rng(seed)
    envs = []
    for episode in range(episodes):
        scenario = sample_training_scenario(agent.instance, rng, progress=progress)
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
    scenario: AdaptiveScenario,
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
    scenario: AdaptiveScenario,
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
    scenario: AdaptiveScenario,
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
    scenario: AdaptiveScenario,
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
    scenario: AdaptiveScenario,
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
    """Stable V6 validation under fair end-to-end deployment semantics.

    Validation uses a fixed heterogeneous set of single-shock scenarios that is
    disjoint from training RNG streams. One deterministic pre-shock prefix is
    generated per scenario, with no hindsight selection. Best-of-K is allowed
    only after the shock is realized.
    """

    scenarios = generate_validation_scenarios(
        agent.instance,
        count=scenario_count,
        seed=scenario_seed,
    )
    objectives = []
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
        progress = update / config.updates
        batch, rollout = _collect_batch(
            agent,
            episodes=config.episodes_per_update,
            seed=seed + update * 100_003,
            progress=progress,
        )
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
                        "phase": "adaptive_method_c_v6_rich_curriculum",
                        "best_validation": best_value,
                        "curriculum": RICH_CURRICULUM_DESCRIPTION,
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
            "phase": "adaptive_method_c_v6_rich_curriculum",
            "best_validation": float(best_value),
            "curriculum": RICH_CURRICULUM_DESCRIPTION,
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
                "curriculum": RICH_CURRICULUM_DESCRIPTION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return best_path

def _collect_fixed_prefix_batch(
    agent: AdvancedPPOAgent,
    *,
    scenario: AdaptiveScenario,
    prefix: tuple[int, ...],
    episodes: int,
    seed: int,
) -> tuple[AdvancedPPOBatch, dict[str, float]]:
    """Collect PPO transitions ONLY after a fixed realized prefix.

    This is the user's proposed shock-time retraining idea: the campaign follows
    the static RL plan before the shock, those actions are immutable, and a new
    RL optimization phase learns only the remaining suffix on the realized
    scenario.
    """

    if len(prefix) != scenario.switch_day:
        raise ValueError(
            "fixed-prefix retraining requires len(prefix) == scenario.switch_day"
        )
    envs = [
        replay_prefix(agent.instance, scenario, prefix, seed=seed + episode)
        for episode in range(episodes)
    ]

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


def train_fixed_prefix_residual(
    agent: AdvancedPPOAgent,
    *,
    scenario: AdaptiveScenario,
    prefix: tuple[int, ...],
    config: ResidualRetrainingConfig,
    output_directory: str | Path,
    seed: int = 42,
    label: str = "shock_time_residual_retraining",
) -> Path:
    """Train a shock-specific RL suffix with pre-shock decisions frozen.

    This is intentionally an *online optimization baseline*, not a generalist
    policy. It may specialize heavily to one realized shock and one realized
    prefix. Its online training time must therefore be reported separately.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    scenario.validate(agent.instance)
    prefix = tuple(int(x) for x in prefix)
    if len(prefix) != scenario.switch_day:
        raise ValueError("prefix length must equal the first shock day")

    if not 0.0 < config.learning_rate_scale <= 1.0:
        raise ValueError("learning_rate_scale must be in (0, 1]")
    if config.update_epochs < 1:
        raise ValueError("update_epochs must be >= 1")

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
    started_all = time.perf_counter()

    for update in range(1, config.updates + 1):
        started = time.perf_counter()
        progress = update / config.updates
        batch, rollout = _collect_fixed_prefix_batch(
            agent,
            scenario=scenario,
            prefix=prefix,
            episodes=config.episodes_per_update,
            seed=seed + update * 200_003,
        )
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
            value = float(
                residual_best_of_k(
                    agent,
                    scenario=scenario,
                    prefix=prefix,
                    rollouts=config.validation_rollouts,
                    seed=seed + 77_000_000 + update,
                    context_aware=True,
                )["objective"]
            )
            record["validation_best_of_k"] = value
            print(
                f"[residual-retrain:{label}] update={update} "
                f"train_mean={rollout['mean_objective']:.3f} "
                f"train_best={rollout['best_objective']:.3f} "
                f"validation={value:.3f} "
                f"kl={opt['approximate_kl']:.6f}",
                flush=True,
            )

            if value > best_value:
                best_value = value
                agent.save(
                    best_path,
                    metadata={
                        "phase": label,
                        "best_validation": best_value,
                        "best_update": update,
                        "fixed_prefix": list(prefix),
                    },
                )

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

    elapsed = float(time.perf_counter() - started_all)
    agent.save(
        output_directory / "checkpoint_last.pt",
        metadata={
            "phase": label,
            "best_validation": float(best_value),
            "fixed_prefix": list(prefix),
            "elapsed_seconds": elapsed,
        },
    )
    (output_directory / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "training_summary.json").write_text(
        json.dumps(
            {
                "label": label,
                "best_validation": float(best_value),
                "updates_completed": len(history),
                "elapsed_seconds": elapsed,
                "fixed_prefix": list(prefix),
                "config": asdict(config),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return best_path

def _recourse_objective(
    teacher_agent: AdvancedPPOAgent | None,
    *,
    scenario: AdaptiveScenario,
    prefix: tuple[int, ...],
    rollouts: int,
    seed: int,
    recourse_solver: str,
    gurobi_time_limit_seconds: float,
) -> tuple[float, list[int]]:
    """Evaluate the best downstream continuation from a fixed prefix."""

    mode = str(recourse_solver).lower()
    if mode == "policy":
        if teacher_agent is None:
            raise ValueError("teacher_agent is required for policy recourse")
        result = residual_best_of_k(
            teacher_agent,
            scenario=scenario,
            prefix=prefix,
            rollouts=rollouts,
            seed=seed,
            context_aware=True,
        )
        return float(result["objective"]), [int(x) for x in result["itinerary"]]

    if mode == "gurobi":
        # Optional exact recourse-aware fine-tuning for a licensed local machine.
        from src.adaptive.gurobi import solve_adaptive_gurobi

        if teacher_agent is None:
            raise ValueError(
                "teacher_agent is still required in Gurobi mode to identify "
                "the instance/action space"
            )
        result = solve_adaptive_gurobi(
            teacher_agent.instance,
            scenario,
            prefix,
            time_limit_seconds=gurobi_time_limit_seconds,
            mip_gap=0.0,
            verbose=False,
            output_directory=None,
            model_label="recourse_aware_training_oracle",
        )
        return float(result["objective"]), [int(x) for x in result["itinerary"]]

    raise ValueError(
        f"unknown recourse_solver={recourse_solver!r}; expected 'policy' or 'gurobi'"
    )


def _collect_recourse_prefix_batch(
    student: AdvancedPPOAgent,
    teacher: AdvancedPPOAgent | None,
    *,
    episodes: int,
    seed: int,
    progress: float,
    teacher_rollouts: int,
    recourse_solver: str,
    gurobi_time_limit_seconds: float,
) -> tuple[AdvancedPPOBatch, dict[str, float]]:
    """Collect PPO transitions only before the first shock.

    No future shock identity is exposed to the student. The downstream suffix
    is solved only after the prefix has been committed. Its final objective
    becomes a terminal bonus, so the prefix learns the option value it leaves
    for the second-stage optimizer.
    """

    if episodes < 1:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)

    states_all = []
    actions_all = []
    logps_all = []
    values_all = []
    returns_all = []
    advantages_all = []
    counts_all = []
    recourse_objectives = []
    prefix_objectives = []
    prefix_lengths = []

    for episode in range(episodes):
        scenario = sample_training_scenario(
            student.instance,
            rng,
            progress=progress,
            allow_multi_shock=True,
        )
        env = AdaptiveCIPPEnv(
            student.instance,
            scenario,
            seed=seed + episode,
        )
        env.reset(seed=seed + episode)

        ep_states = []
        ep_actions = []
        ep_logps = []
        ep_values = []
        ep_rewards = []

        while env.day < scenario.switch_day:
            state = student.feature_builder.build(env)
            action, logp, value = student.select_action(
                state,
                deterministic=False,
            )
            _, reward, _, _, _ = env.step(int(action))
            ep_states.append(state)
            ep_actions.append(int(action))
            ep_logps.append(float(logp))
            ep_values.append(float(value))
            ep_rewards.append(float(reward) * student.reward_scale)

        prefix = tuple(int(x) for x in env.itinerary[: scenario.switch_day])
        if not ep_rewards:
            raise RuntimeError("recourse-aware prefix contained zero decisions")

        prefix_value = float(env.cumulative_reward)
        recourse_value, full_itinerary = _recourse_objective(
            teacher,
            scenario=scenario,
            prefix=prefix,
            rollouts=teacher_rollouts,
            seed=seed + 50_000_000 + episode * 10_003,
            recourse_solver=recourse_solver,
            gurobi_time_limit_seconds=gurobi_time_limit_seconds,
        )

        # Sum(prefix immediate rewards + terminal bonus) equals final recourse
        # objective after scaling. This pushes credit for downstream flexibility
        # back into the actions that created the prefix.
        terminal_bonus = (
            recourse_value - prefix_value
        ) * student.reward_scale
        ep_rewards[-1] += float(terminal_bonus)

        rewards = np.asarray(ep_rewards, dtype=np.float32)
        values = np.asarray(ep_values, dtype=np.float32)
        advantages, returns = compute_episode_gae(
            rewards,
            values,
            discount_factor=student.config.discount_factor,
            gae_lambda=student.config.gae_lambda,
        )

        visit_counts = np.zeros(student.instance.n, dtype=np.int64)
        for action in full_itinerary:
            if int(action) > 0:
                visit_counts[int(action) - 1] += 1

        n_steps = len(ep_actions)
        states_all.extend(ep_states)
        actions_all.append(np.asarray(ep_actions, dtype=np.int64))
        logps_all.append(np.asarray(ep_logps, dtype=np.float32))
        values_all.append(values)
        returns_all.append(returns)
        advantages_all.append(advantages)
        counts_all.append(np.repeat(visit_counts[None, :], n_steps, axis=0))
        recourse_objectives.append(recourse_value)
        prefix_objectives.append(prefix_value)
        prefix_lengths.append(len(prefix))

    batch = AdvancedPPOBatch(
        locations=np.stack([s.locations for s in states_all]),
        global_features=np.stack([s.global_features for s in states_all]),
        action_masks=np.stack([s.action_mask for s in states_all]),
        actions=np.concatenate(actions_all),
        old_log_probabilities=np.concatenate(logps_all),
        old_values=np.concatenate(values_all),
        returns=np.concatenate(returns_all),
        advantages=np.concatenate(advantages_all),
        final_visit_counts=np.concatenate(counts_all, axis=0),
    )
    return batch, {
        "mean_recourse_objective": float(np.mean(recourse_objectives)),
        "best_recourse_objective": float(np.max(recourse_objectives)),
        "std_recourse_objective": float(np.std(recourse_objectives)),
        "mean_prefix_objective": float(np.mean(prefix_objectives)),
        "mean_prefix_length": float(np.mean(prefix_lengths)),
        "transitions": float(batch.size),
    }


def _validate_recourse_prefix(
    student: AdvancedPPOAgent,
    teacher: AdvancedPPOAgent | None,
    *,
    scenario_seed: int,
    scenario_count: int,
    rollouts_per_scenario: int,
    recourse_solver: str,
    gurobi_time_limit_seconds: float,
) -> float:
    scenarios = generate_validation_scenarios(
        student.instance,
        count=scenario_count,
        seed=scenario_seed,
    )
    values = []
    for k, scenario in enumerate(scenarios):
        prefix = build_anticipatory_prefix(
            student,
            scenario=scenario,
            seed=scenario_seed + 100_000 * (k + 1),
            deterministic=True,
        )
        objective, _ = _recourse_objective(
            teacher,
            scenario=scenario,
            prefix=prefix,
            rollouts=rollouts_per_scenario,
            seed=scenario_seed + 20_000_000 + k * 1009,
            recourse_solver=recourse_solver,
            gurobi_time_limit_seconds=gurobi_time_limit_seconds,
        )
        values.append(objective)
    return float(np.mean(values))


def train_recourse_aware_prefix(
    student: AdvancedPPOAgent,
    *,
    teacher: AdvancedPPOAgent | None,
    config: RecourseAwareTrainingConfig,
    output_directory: str | Path,
    seed: int = 42,
    label: str = "recourse_aware_prefix",
    recourse_solver: str = "policy",
    gurobi_time_limit_seconds: float = 120.0,
) -> Path:
    """Fine-tune a non-clairvoyant prefix policy for downstream recourse value.

    This is the V7 implementation of the key new idea:
    before the shock, RL is responsible for creating a high-option-value state;
    after the shock, a separate recourse solver takes over. During training the
    downstream result is explicitly propagated back to pre-shock decisions.

    `recourse_solver="policy"` is GPU/Kaggle friendly and uses a frozen strong
    adaptive teacher. `recourse_solver="gurobi"` is an optional high-cost exact
    fine-tuning mode for a locally licensed machine.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if not 0.0 < config.learning_rate_scale <= 1.0:
        raise ValueError("learning_rate_scale must be in (0, 1]")
    if config.update_epochs < 1:
        raise ValueError("update_epochs must be >= 1")

    scaled_actor_lr = student.config.actor_learning_rate * config.learning_rate_scale
    scaled_critic_lr = student.config.critic_learning_rate * config.learning_rate_scale
    student.config = replace(
        student.config,
        actor_learning_rate=scaled_actor_lr,
        critic_learning_rate=scaled_critic_lr,
        update_epochs=config.update_epochs,
    )
    student._base_group_lrs = [
        float(lr) * config.learning_rate_scale
        for lr in student._base_group_lrs
    ]
    for group, lr in zip(student.optimizer.param_groups, student._base_group_lrs):
        group["lr"] = lr

    history = []
    best_value = -np.inf
    best_for_patience = -np.inf
    checks_without_improvement = 0
    validation_seed = seed + 191_000_000
    best_path = output_directory / "checkpoint_best.pt"
    started_all = time.perf_counter()

    for update in range(1, config.updates + 1):
        started = time.perf_counter()
        progress = update / config.updates
        batch, rollout = _collect_recourse_prefix_batch(
            student,
            teacher,
            episodes=config.prefixes_per_update,
            seed=seed + update * 300_007,
            progress=progress,
            teacher_rollouts=config.teacher_rollouts,
            recourse_solver=recourse_solver,
            gurobi_time_limit_seconds=gurobi_time_limit_seconds,
        )
        student.set_learning_rate_fraction(max(1.0 - progress, 0.05))
        opt = student.update(batch, progress=progress)

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
            value = _validate_recourse_prefix(
                student,
                teacher,
                scenario_seed=validation_seed,
                scenario_count=config.validation_scenarios,
                rollouts_per_scenario=config.validation_rollouts,
                recourse_solver=recourse_solver,
                gurobi_time_limit_seconds=gurobi_time_limit_seconds,
            )
            record["validation_mean_recourse"] = value
            print(
                f"[recourse-prefix:{label}] update={update} "
                f"train_recourse={rollout['mean_recourse_objective']:.3f} "
                f"validation={value:.3f} "
                f"kl={opt['approximate_kl']:.6f}",
                flush=True,
            )

            if value > best_value:
                best_value = value
                student.save(
                    best_path,
                    metadata={
                        "phase": label,
                        "best_validation_recourse": best_value,
                        "best_update": update,
                        "recourse_solver": recourse_solver,
                        "teacher_rollouts": int(config.teacher_rollouts),
                    },
                )

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

    elapsed = float(time.perf_counter() - started_all)
    student.save(
        output_directory / "checkpoint_last.pt",
        metadata={
            "phase": label,
            "best_validation_recourse": float(best_value),
            "elapsed_seconds": elapsed,
            "recourse_solver": recourse_solver,
        },
    )
    (output_directory / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "training_summary.json").write_text(
        json.dumps(
            {
                "label": label,
                "best_validation_recourse": float(best_value),
                "updates_completed": len(history),
                "elapsed_seconds": elapsed,
                "recourse_solver": recourse_solver,
                "config": asdict(config),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return best_path

def _concat_ppo_batches(batches: list[AdvancedPPOBatch]) -> AdvancedPPOBatch:
    if not batches:
        raise ValueError("at least one PPO batch is required")
    return AdvancedPPOBatch(
        locations=np.concatenate([b.locations for b in batches], axis=0),
        global_features=np.concatenate(
            [b.global_features for b in batches], axis=0
        ),
        action_masks=np.concatenate([b.action_masks for b in batches], axis=0),
        actions=np.concatenate([b.actions for b in batches], axis=0),
        old_log_probabilities=np.concatenate(
            [b.old_log_probabilities for b in batches], axis=0
        ),
        old_values=np.concatenate([b.old_values for b in batches], axis=0),
        returns=np.concatenate([b.returns for b in batches], axis=0),
        advantages=np.concatenate([b.advantages for b in batches], axis=0),
        final_visit_counts=np.concatenate(
            [b.final_visit_counts for b in batches], axis=0
        ),
    )


def train_multi_instance_method_c(
    agent: AdvancedPPOAgent,
    *,
    feature_builders: list[AdaptiveFeatureBuilder],
    config: AdaptiveTrainingConfig,
    output_directory: str | Path,
    seed: int = 42,
) -> Path:
    """Train one shared Method-C network over multiple compatible instances.

    All instances must have the same number of locations/action space and the
    same feature dimensions. Horizons may differ. Feature normalization remains
    instance-specific because each batch is collected with its own builder.

    This turns Method C from "general over shocks on one instance" into a true
    compatible-instance generalist when the caller supplies multiple benchmark
    instances.
    """

    if not feature_builders:
        raise ValueError("feature_builders must not be empty")
    n = feature_builders[0].instance.n
    location_dim = feature_builders[0].location_dim
    global_dim = feature_builders[0].global_dim
    for builder in feature_builders:
        if builder.instance.n != n:
            raise ValueError(
                "multi-instance Method C requires identical location counts"
            )
        if builder.location_dim != location_dim or builder.global_dim != global_dim:
            raise ValueError(
                "multi-instance Method C requires identical feature dimensions"
            )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    scaled_actor_lr = agent.config.actor_learning_rate * config.learning_rate_scale
    scaled_critic_lr = agent.config.critic_learning_rate * config.learning_rate_scale
    agent.config = replace(
        agent.config,
        actor_learning_rate=scaled_actor_lr,
        critic_learning_rate=scaled_critic_lr,
        update_epochs=config.update_epochs,
    )
    agent._base_group_lrs = [
        float(lr) * config.learning_rate_scale for lr in agent._base_group_lrs
    ]
    for group, lr in zip(agent.optimizer.param_groups, agent._base_group_lrs):
        group["lr"] = lr

    original_builder = agent.feature_builder
    original_instance = agent.instance
    history = []
    best_value = -np.inf
    best_for_patience = -np.inf
    checks_without_improvement = 0
    best_path = output_directory / "checkpoint_best.pt"
    started_all = time.perf_counter()

    try:
        for update in range(1, config.updates + 1):
            started = time.perf_counter()
            progress = update / config.updates
            per_instance = max(
                1,
                int(np.ceil(config.episodes_per_update / len(feature_builders))),
            )
            batches = []
            rollouts = []

            for j, builder in enumerate(feature_builders):
                agent.feature_builder = builder
                agent.instance = builder.instance
                batch, rollout = _collect_batch(
                    agent,
                    episodes=per_instance,
                    seed=seed + update * 100_003 + j * 10_007,
                    progress=progress,
                )
                batches.append(batch)
                rollouts.append(
                    {
                        "instance": builder.instance.instance_id,
                        **rollout,
                    }
                )

            agent.feature_builder = original_builder
            agent.instance = original_instance
            merged = _concat_ppo_batches(batches)
            agent.set_learning_rate_fraction(max(1.0 - progress, 0.05))
            opt = agent.update(merged, progress=progress)

            record = {
                "update": update,
                "instance_rollouts": rollouts,
                "optimization": opt,
                "elapsed_seconds": float(time.perf_counter() - started),
            }

            should_validate = (
                update == 1
                or update == config.updates
                or update % config.validation_interval == 0
            )
            if should_validate:
                vals = []
                for j, builder in enumerate(feature_builders):
                    agent.feature_builder = builder
                    agent.instance = builder.instance
                    vals.append(
                        _fixed_validation(
                            agent,
                            scenario_seed=seed + 91_000_000 + j * 1_000_003,
                            scenario_count=config.validation_scenarios,
                            rollouts_per_scenario=config.validation_rollouts_per_scenario,
                        )
                    )
                agent.feature_builder = original_builder
                agent.instance = original_instance
                value = float(np.mean(vals))
                record["validation_mean_across_instances"] = value
                record["validation_by_instance"] = {
                    builder.instance.instance_id: float(v)
                    for builder, v in zip(feature_builders, vals)
                }
                mean_train = float(
                    np.mean([x["mean_objective"] for x in rollouts])
                )
                print(
                    f"[multi-instance-C] update={update} "
                    f"train_mean={mean_train:.3f} validation={value:.3f} "
                    f"instances={len(feature_builders)} "
                    f"kl={opt['approximate_kl']:.6f}",
                    flush=True,
                )

                if value > best_value:
                    best_value = value
                    agent.save(
                        best_path,
                        metadata={
                            "phase": "multi_instance_method_c_v7",
                            "best_validation": best_value,
                            "best_update": update,
                            "training_instances": [
                                b.instance.instance_id for b in feature_builders
                            ],
                            "curriculum": RICH_CURRICULUM_DESCRIPTION,
                        },
                    )

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
    finally:
        agent.feature_builder = original_builder
        agent.instance = original_instance

    elapsed = float(time.perf_counter() - started_all)
    agent.save(
        output_directory / "checkpoint_last.pt",
        metadata={
            "phase": "multi_instance_method_c_v7",
            "best_validation": float(best_value),
            "training_instances": [
                b.instance.instance_id for b in feature_builders
            ],
            "elapsed_seconds": elapsed,
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
                "elapsed_seconds": elapsed,
                "training_instances": [
                    b.instance.instance_id for b in feature_builders
                ],
                "config": asdict(config),
                "curriculum": RICH_CURRICULUM_DESCRIPTION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return best_path

