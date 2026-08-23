"""Training and residual inference for context-conditioned adaptive Attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import time

import numpy as np

from src.advanced.ppo import AdvancedPPOAgent, AdvancedPPOBatch, compute_episode_gae
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


def residual_best_of_k(
    agent: AdvancedPPOAgent,
    *,
    scenario: ShockScenario,
    prefix: tuple[int, ...],
    rollouts: int,
    seed: int,
    context_aware: bool,
) -> dict[str, object]:
    """Sample only the suffix and select with the adaptive objective."""

    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    started = time.perf_counter()
    rngs = [np.random.default_rng(seed + i) for i in range(rollouts)]
    envs = [
        replay_prefix(agent.instance, scenario, prefix, seed=seed + i)
        for i in range(rollouts)
    ]

    if context_aware:
        builders = [agent.feature_builder] * rollouts
    else:
        # Static feature builder deliberately hides the new reward context.
        static_builder = AdaptiveFeatureBuilder(agent.instance)
        # We will feed a normal CIPP-style feature state by temporarily building
        # with the parent method explicitly in _static_state below.
        builders = [static_builder] * rollouts

    while True:
        active = [i for i, env in enumerate(envs) if not env.done]
        if not active:
            break

        if context_aware:
            states = [agent.feature_builder.build(envs[i]) for i in active]
        else:
            # Build exactly the old static observation: call parent implementation.
            states = [
                super(AdaptiveFeatureBuilder, builders[i]).build(envs[i])
                for i in active
            ]

        probs, _ = agent.batch_probabilities_and_values(states)
        for j, idx in enumerate(active):
            state = states[j]
            row = np.where(state.action_mask, np.clip(probs[j], 0.0, None), 0.0)
            total = float(row.sum())
            if total <= 0.0:
                raise RuntimeError("zero probability on every feasible action")
            row /= total
            action = int(rngs[idx].choice(agent.instance.num_actions, p=row))
            envs[idx].step(action)

    best = max(envs, key=lambda e: e.cumulative_reward)
    return {
        "objective": float(best.cumulative_reward),
        "itinerary": best.itinerary.tolist(),
        "runtime_seconds": float(time.perf_counter() - started),
        "rollouts": int(rollouts),
    }


def residual_greedy(
    agent: AdvancedPPOAgent,
    *,
    scenario: ShockScenario,
    prefix: tuple[int, ...],
    seed: int,
    context_aware: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    env = replay_prefix(agent.instance, scenario, prefix, seed=seed)
    static_builder = AdaptiveFeatureBuilder(agent.instance)

    while not env.done:
        if context_aware:
            state = agent.feature_builder.build(env)
        else:
            state = super(AdaptiveFeatureBuilder, static_builder).build(env)
        action, _, _ = agent.select_action(state, deterministic=True)
        env.step(action)

    return {
        "objective": float(env.cumulative_reward),
        "itinerary": env.itinerary.tolist(),
        "runtime_seconds": float(time.perf_counter() - started),
        "rollouts": 1,
    }


def _fixed_validation(
    agent: AdvancedPPOAgent,
    *,
    scenario_seed: int,
    scenario_count: int,
    rollouts_per_scenario: int,
) -> float:
    rng = np.random.default_rng(scenario_seed)
    objectives = []
    # Fixed validation shocks are random but deterministic across updates.
    scenarios = [
        sample_training_scenario(agent.instance, rng)
        for _ in range(scenario_count)
    ]
    for k, scenario in enumerate(scenarios):
        result = residual_best_of_k(
            agent,
            scenario=scenario,
            prefix=(),
            rollouts=rollouts_per_scenario,
            seed=scenario_seed + 100_000 * (k + 1),
            context_aware=True,
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
