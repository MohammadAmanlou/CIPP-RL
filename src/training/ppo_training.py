"""Rollout collection and leakage-free evaluation for masked PPO."""

from __future__ import annotations

import time

import numpy as np

from src.baselines import PolicyResult
from src.core import (
    CIPPInstance,
    evaluate_itinerary,
)
from src.envs import CIPPEnv
from src.models import (
    DQNAgent,
    PPOAgent,
    PPOBatch,
    compute_gae,
)
from src.utils import (
    ObservationNormalizer,
)


def collect_normalizer_observations(
    instances: list[CIPPInstance],
    *,
    episodes_per_instance: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Collect training-only raw observations with random feasible policies."""

    if not instances:
        raise ValueError(
            "at least one training instance is required."
        )

    if episodes_per_instance < 1:
        raise ValueError(
            "episodes_per_instance must be positive."
        )

    observations: list[np.ndarray] = []

    for instance_index, instance in enumerate(
        instances
    ):
        viability_cache: dict[
            tuple[int, ...],
            np.ndarray,
        ] = {}

        for episode_index in range(
            episodes_per_instance
        ):
            episode_seed = (
                seed
                + 10_000
                * instance_index
                + episode_index
            )

            environment = CIPPEnv(
                instance,
                seed=episode_seed,
                viability_cache=(
                    viability_cache
                ),
            )

            observation, _ = (
                environment.reset(
                    seed=episode_seed
                )
            )

            while not environment.done:
                observations.append(
                    observation.copy()
                )

                action = (
                    environment
                    .sample_viable_action()
                )

                (
                    observation,
                    _,
                    _,
                    _,
                    _,
                ) = environment.step(
                    action
                )

    return np.stack(
        observations
    )


def fit_training_normalizer(
    instances: list[CIPPInstance],
    *,
    episodes_per_instance: int = 1,
    seed: int = 0,
) -> ObservationNormalizer:
    """Fit and freeze normalization statistics using training data only."""

    return ObservationNormalizer.fit(
        collect_normalizer_observations(
            instances,
            episodes_per_instance=(
                episodes_per_instance
            ),
            seed=seed,
        )
    )


def collect_ppo_rollouts(
    agent: PPOAgent,
    *,
    normalizer: ObservationNormalizer,
    instances: list[CIPPInstance],
    number_of_episodes: int,
    seed: int,
    viability_caches: dict[
        int,
        dict[
            tuple[int, ...],
            np.ndarray,
        ],
    ]
    | None = None,
) -> tuple[
    PPOBatch,
    dict[str, float],
]:
    """Collect complete on-policy episodes and compute GAE."""

    if not instances:
        raise ValueError(
            "at least one rollout instance is required."
        )

    if number_of_episodes < 1:
        raise ValueError(
            "number_of_episodes must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    log_probabilities: list[float] = []
    values: list[float] = []
    action_masks: list[np.ndarray] = []
    episode_objectives: list[float] = []
    feasibility: list[float] = []

    started = time.perf_counter()
    resolved_viability_caches = (
        viability_caches
        if viability_caches
        is not None
        else {
            id(instance): {}
            for instance
            in instances
        }
    )

    for instance in instances:
        resolved_viability_caches.setdefault(
            id(instance),
            {},
        )

    for episode_index in range(
        number_of_episodes
    ):
        instance = instances[
            int(
                rng.integers(
                    0,
                    len(instances),
                )
            )
        ]

        environment = CIPPEnv(
            instance,
            seed=(
                seed + episode_index
            ),
            viability_cache=(
                resolved_viability_caches[
                    id(instance)
                ]
            ),
        )

        observation, info = (
            environment.reset(
                seed=(
                    seed + episode_index
                )
            )
        )

        while not environment.done:
            normalized_observation = (
                normalizer.transform(
                    observation
                ).astype(
                    np.float32
                )
            )

            mask = np.asarray(
                info["action_mask"],
                dtype=np.bool_,
            )

            (
                action,
                log_probability,
                value,
            ) = agent.select_action(
                normalized_observation,
                mask,
                deterministic=False,
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                next_info,
            ) = environment.step(
                action
            )

            states.append(
                normalized_observation
            )

            actions.append(
                action
            )

            rewards.append(
                float(reward)
                * agent.config.reward_scale
            )

            dones.append(
                bool(
                    terminated
                    or truncated
                )
            )

            log_probabilities.append(
                log_probability
            )

            values.append(
                value
            )

            action_masks.append(
                mask.copy()
            )

            observation = (
                next_observation
            )

            info = next_info

        final_evaluation = info[
            "final_evaluation"
        ]

        episode_objectives.append(
            float(
                final_evaluation[
                    "objective"
                ]
            )
        )

        feasibility.append(
            float(
                final_evaluation[
                    "feasible"
                ]
            )
        )

    advantages, returns = compute_gae(
        np.asarray(
            rewards,
            dtype=np.float32,
        ),
        np.asarray(
            values,
            dtype=np.float32,
        ),
        np.asarray(
            dones,
            dtype=np.bool_,
        ),
        discount_factor=(
            agent.config
            .discount_factor
        ),
        gae_lambda=(
            agent.config
            .gae_lambda
        ),
        last_value=0.0,
    )

    batch = PPOBatch(
        states=np.stack(
            states
        ),
        actions=np.asarray(
            actions,
            dtype=np.int64,
        ),
        old_log_probabilities=(
            np.asarray(
                log_probabilities,
                dtype=np.float32,
            )
        ),
        returns=returns,
        advantages=advantages,
        action_masks=np.stack(
            action_masks
        ),
    )

    summary = {
        "mean_episode_objective": float(
            np.mean(
                episode_objectives
            )
        ),
        "std_episode_objective": float(
            np.std(
                episode_objectives
            )
        ),
        "feasible_rate": float(
            np.mean(
                feasibility
            )
        ),
        "transitions": float(
            batch.size
        ),
        "rollout_seconds": float(
            time.perf_counter()
            - started
        ),
        "reward_scale": float(
            agent.config.reward_scale
        ),
    }

    return batch, summary


def _policy_result(
    *,
    method: str,
    instance: CIPPInstance,
    itinerary: np.ndarray,
    runtime_seconds: float,
) -> PolicyResult:
    evaluation = evaluate_itinerary(
        instance,
        itinerary,
    )

    return PolicyResult(
        method=method,
        instance_id=instance.instance_id,
        itinerary=tuple(
            int(action)
            for action in itinerary
        ),
        objective=float(
            evaluation.objective
        ),
        total_cost=float(
            evaluation.total_cost
        ),
        idle_days=int(
            evaluation.idle_days
        ),
        unique_locations=int(
            np.count_nonzero(
                evaluation.visit_counts
            )
        ),
        feasible=bool(
            evaluation.feasible
        ),
        violations=tuple(
            evaluation.violations
        ),
        runtime_seconds=float(
            runtime_seconds
        ),
    )


def evaluate_ppo_policy(
    instance: CIPPInstance,
    agent: PPOAgent,
    normalizer: ObservationNormalizer,
    *,
    deterministic: bool,
    seed: int = 0,
    method: str | None = None,
    viability_cache: dict[
        tuple[int, ...],
        np.ndarray,
    ]
    | None = None,
) -> PolicyResult:
    """Evaluate one frozen PPO checkpoint without any parameter update."""

    environment = CIPPEnv(
        instance,
        seed=seed,
        viability_cache=(
            viability_cache
        ),
    )

    observation, info = (
        environment.reset(
            seed=seed
        )
    )

    rng = np.random.default_rng(
        seed
    )

    started = time.perf_counter()

    while not environment.done:
        normalized_observation = (
            normalizer.transform(
                observation
            ).astype(
                np.float32
            )
        )

        mask = np.asarray(
            info["action_mask"],
            dtype=np.bool_,
        )

        if deterministic:
            action, _, _ = (
                agent.select_action(
                    normalized_observation,
                    mask,
                    deterministic=True,
                )
            )
        else:
            probabilities = (
                agent.action_probabilities(
                    normalized_observation,
                    mask,
                )
            )

            probabilities = np.where(
                mask,
                np.clip(
                    probabilities,
                    0.0,
                    None,
                ),
                0.0,
            )

            probabilities /= float(
                probabilities.sum()
            )

            action = int(
                rng.choice(
                    instance.num_actions,
                    p=probabilities,
                )
            )

        (
            observation,
            _,
            _,
            _,
            info,
        ) = environment.step(
            action
        )

    runtime_seconds = (
        time.perf_counter() - started
    )

    return _policy_result(
        method=(
            method
            or (
                "ppo_greedy"
                if deterministic
                else "ppo_stochastic"
            )
        ),
        instance=instance,
        itinerary=environment.itinerary,
        runtime_seconds=runtime_seconds,
    )


def evaluate_ppo_best_of_rollouts(
    instance: CIPPInstance,
    agent: PPOAgent,
    normalizer: ObservationNormalizer,
    *,
    number_of_rollouts: int = 30,
    seed: int = 0,
    method: str = "ppo_best_of_30",
) -> PolicyResult:
    """Sample complete feasible itineraries and return the best exact objective."""

    if number_of_rollouts < 1:
        raise ValueError(
            "number_of_rollouts must be positive."
        )

    started = time.perf_counter()
    candidates: list[PolicyResult] = []
    viability_cache: dict[
        tuple[int, ...],
        np.ndarray,
    ] = {}

    for rollout_index in range(
        number_of_rollouts
    ):
        candidates.append(
            evaluate_ppo_policy(
                instance,
                agent,
                normalizer,
                deterministic=False,
                seed=(
                    seed + rollout_index
                ),
                method=method,
                viability_cache=(
                    viability_cache
                ),
            )
        )

    best = max(
        candidates,
        key=lambda result: (
            result.objective
        ),
    )

    return PolicyResult(
        method=method,
        instance_id=best.instance_id,
        itinerary=best.itinerary,
        objective=best.objective,
        total_cost=best.total_cost,
        idle_days=best.idle_days,
        unique_locations=(
            best.unique_locations
        ),
        feasible=best.feasible,
        violations=best.violations,
        runtime_seconds=float(
            time.perf_counter()
            - started
        ),
    )


def evaluate_dqn_policy(
    instance: CIPPInstance,
    agent: DQNAgent,
    normalizer: ObservationNormalizer,
    *,
    method: str = "dqn_greedy",
    exploration_epsilon: float = 0.0,
    seed: int = 0,
    viability_cache: dict[
        tuple[int, ...],
        np.ndarray,
    ]
    | None = None,
) -> PolicyResult:
    """Evaluate a frozen DQN checkpoint with optional local exploration."""

    if not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError(
            "exploration_epsilon must be between zero and one."
        )

    environment = CIPPEnv(
        instance,
        seed=seed,
        viability_cache=(
            viability_cache
        ),
    )

    observation, info = (
        environment.reset(
            seed=seed
        )
    )

    rng = np.random.default_rng(
        seed
    )

    started = time.perf_counter()

    while not environment.done:
        normalized_observation = (
            normalizer.transform(
                observation
            ).astype(
                np.float32
            )
        )

        mask = np.asarray(
            info["action_mask"],
            dtype=np.bool_,
        )

        if (
            exploration_epsilon > 0.0
            and rng.random()
            < exploration_epsilon
        ):
            action = int(
                rng.choice(
                    np.flatnonzero(
                        mask
                    )
                )
            )
        else:
            action = agent.select_action(
                normalized_observation,
                mask,
                epsilon=0.0,
            )

        (
            observation,
            _,
            _,
            _,
            info,
        ) = environment.step(
            action
        )

    return _policy_result(
        method=method,
        instance=instance,
        itinerary=environment.itinerary,
        runtime_seconds=(
            time.perf_counter()
            - started
        ),
    )


def evaluate_dqn_best_of_rollouts(
    instance: CIPPInstance,
    agent: DQNAgent,
    normalizer: ObservationNormalizer,
    *,
    number_of_rollouts: int = 30,
    exploration_epsilon: float = 0.10,
    seed: int = 0,
    method: str = "dqn_best_of_30",
) -> PolicyResult:
    """Return the best exact objective from independent epsilon-greedy paths."""

    if number_of_rollouts < 1:
        raise ValueError(
            "number_of_rollouts must be positive."
        )

    if not 0.0 < exploration_epsilon <= 1.0:
        raise ValueError(
            "exploration_epsilon must be in (0, 1]."
        )

    started = time.perf_counter()
    viability_cache: dict[
        tuple[int, ...],
        np.ndarray,
    ] = {}

    candidates = [
        evaluate_dqn_policy(
            instance,
            agent,
            normalizer,
            method=method,
            exploration_epsilon=(
                exploration_epsilon
            ),
            seed=seed + rollout_index,
            viability_cache=(
                viability_cache
            ),
        )
        for rollout_index in range(
            number_of_rollouts
        )
    ]

    best = max(
        candidates,
        key=lambda result: (
            result.objective
        ),
    )

    return PolicyResult(
        method=method,
        instance_id=best.instance_id,
        itinerary=best.itinerary,
        objective=best.objective,
        total_cost=best.total_cost,
        idle_days=best.idle_days,
        unique_locations=(
            best.unique_locations
        ),
        feasible=best.feasible,
        violations=best.violations,
        runtime_seconds=float(
            time.perf_counter()
            - started
        ),
    )
