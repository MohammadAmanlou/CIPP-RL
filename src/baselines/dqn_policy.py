"""Frozen-checkpoint DQN inference for deterministic CIPP.

Two evaluation modes are provided:

* ``run_dqn_greedy_policy``: one deterministic greedy trajectory.
* ``run_dqn_backtracking_rollouts``: the same frozen network performs a
  Q-guided best-first search.  It stores alternative actions at previously
  visited states, reconstructs that prefix, and greedily completes it.  Thus a
  budget of 30 means at most 30 complete feasible trajectories, not 30 gradient
  updates and not test-time training.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass

import numpy as np

from src.baselines.common import PolicyResult
from src.core import CIPPInstance, evaluate_itinerary
from src.envs import CIPPEnv
from src.models import DQNAgent
from src.utils import ObservationNormalizer


@dataclass(frozen=True, slots=True)
class _CompletedRollout:
    itinerary: tuple[int, ...]
    objective: float
    branches: tuple[tuple[float, tuple[int, ...]], ...]


def _normalized_observation(
    normalizer: ObservationNormalizer,
    observation: np.ndarray,
) -> np.ndarray:
    return normalizer.transform(observation).astype(np.float32)


def _ranked_viable_actions(
    agent: DQNAgent,
    normalizer: ObservationNormalizer,
    observation: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[int, float]]:
    normalized = _normalized_observation(normalizer, observation)
    q_values = agent.predict_q_values(normalized, mask)
    viable = np.flatnonzero(mask)
    ranked = sorted(
        ((int(action), float(q_values[action])) for action in viable),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        raise RuntimeError("No viable action is available.")
    return ranked


def _complete_prefix_greedily(
    instance: CIPPInstance,
    agent: DQNAgent,
    normalizer: ObservationNormalizer,
    prefix: tuple[int, ...],
    *,
    alternatives_per_state: int | None,
    reward_scale: float,
) -> _CompletedRollout:
    """Replay a prefix and greedily complete it while recording alternatives."""

    if reward_scale <= 0 or not np.isfinite(reward_scale):
        raise ValueError("reward_scale must be positive and finite.")

    env = CIPPEnv(instance)
    observation, info = env.reset()

    for action in prefix:
        mask = info["action_mask"]
        if action < 0 or action >= mask.size or not bool(mask[action]):
            raise ValueError(f"Non-viable backtracking prefix action: {action}")
        observation, _, _, _, info = env.step(action)

    branches: list[tuple[float, tuple[int, ...]]] = []

    while not env.done:
        ranked = _ranked_viable_actions(
            agent,
            normalizer,
            observation,
            info["action_mask"],
        )
        chosen_action = ranked[0][0]
        current_prefix = tuple(int(x) for x in env.itinerary)

        alternatives = ranked[1:]
        if alternatives_per_state is not None:
            alternatives = alternatives[:alternatives_per_state]

        # Q-values are trained on reward/reward_scale.  Prefixes at different
        # depths are therefore comparable only after adding the exact reward
        # already collected before the alternative action.
        prefix_return = env.cumulative_reward / reward_scale
        for alternative_action, q_value in alternatives:
            root_value_estimate = prefix_return + q_value
            branches.append(
                (root_value_estimate, current_prefix + (alternative_action,))
            )

        observation, _, _, _, info = env.step(chosen_action)

    evaluation = evaluate_itinerary(instance, env.itinerary)
    if not evaluation.feasible:
        raise RuntimeError(
            "Masked DQN produced an infeasible itinerary: "
            + "; ".join(evaluation.violations)
        )

    return _CompletedRollout(
        itinerary=tuple(int(x) for x in env.itinerary),
        objective=float(evaluation.objective),
        branches=tuple(branches),
    )


def _policy_result(
    *,
    method: str,
    instance: CIPPInstance,
    itinerary: tuple[int, ...],
    runtime_seconds: float,
    rollouts: int,
    metadata: dict[str, object],
) -> PolicyResult:
    evaluation = evaluate_itinerary(instance, itinerary)
    if not evaluation.feasible:
        raise RuntimeError("DQN result is infeasible.")

    return PolicyResult(
        method=method,
        instance_id=instance.instance_id,
        itinerary=itinerary,
        objective=float(evaluation.objective),
        total_cost=float(evaluation.total_cost),
        idle_days=int(evaluation.idle_days),
        unique_locations=int(np.count_nonzero(evaluation.visit_counts)),
        feasible=True,
        violations=(),
        runtime_seconds=float(runtime_seconds),
        rollouts=int(rollouts),
        metadata=metadata,
    )


def run_dqn_greedy_policy(
    instance: CIPPInstance,
    *,
    agent: DQNAgent,
    normalizer: ObservationNormalizer,
) -> PolicyResult:
    """Run one frozen, deterministic, epsilon-zero DQN trajectory."""

    started = time.perf_counter()
    completed = _complete_prefix_greedily(
        instance,
        agent,
        normalizer,
        (),
        alternatives_per_state=0,
        reward_scale=1.0,
    )
    runtime = time.perf_counter() - started

    return _policy_result(
        method="dqn_greedy",
        instance=instance,
        itinerary=completed.itinerary,
        runtime_seconds=runtime,
        rollouts=1,
        metadata={"search": "single frozen greedy trajectory"},
    )


def run_dqn_backtracking_rollouts(
    instance: CIPPInstance,
    *,
    agent: DQNAgent,
    normalizer: ObservationNormalizer,
    rollout_budget: int = 30,
    alternatives_per_state: int | None = 3,
    reward_scale: float = 1.0,
) -> PolicyResult:
    """Run Q-guided best-first backtracking with a complete-rollout budget.

    The first rollout is exactly the greedy DQN result.  Later rollouts return
    to a stored prefix, replace one decision with the next promising viable
    action, and greedily complete the remainder.  Network weights stay frozen.
    """

    if rollout_budget < 1:
        raise ValueError("rollout_budget must be positive.")
    if alternatives_per_state is not None and alternatives_per_state < 1:
        raise ValueError("alternatives_per_state must be positive or None.")
    if reward_scale <= 0 or not np.isfinite(reward_scale):
        raise ValueError("reward_scale must be positive and finite.")

    started = time.perf_counter()
    counter = itertools.count()
    heap: list[tuple[float, int, tuple[int, ...]]] = []
    queued_prefixes: set[tuple[int, ...]] = set()
    evaluated_prefixes: set[tuple[int, ...]] = set()
    seen_itineraries: set[tuple[int, ...]] = set()
    rollout_objectives: list[float] = []

    def enqueue(branches: tuple[tuple[float, tuple[int, ...]], ...]) -> None:
        for priority, candidate_prefix in branches:
            if (
                candidate_prefix in queued_prefixes
                or candidate_prefix in evaluated_prefixes
            ):
                continue
            queued_prefixes.add(candidate_prefix)
            # heapq is a min-heap, so negate Q priority.
            heapq.heappush(
                heap,
                (-float(priority), next(counter), candidate_prefix),
            )

    first = _complete_prefix_greedily(
        instance,
        agent,
        normalizer,
        (),
        alternatives_per_state=alternatives_per_state,
        reward_scale=reward_scale,
    )
    evaluated_prefixes.add(())
    seen_itineraries.add(first.itinerary)
    rollout_objectives.append(first.objective)
    best = first
    enqueue(first.branches)

    while len(rollout_objectives) < rollout_budget and heap:
        _, _, prefix = heapq.heappop(heap)
        queued_prefixes.discard(prefix)
        if prefix in evaluated_prefixes:
            continue
        evaluated_prefixes.add(prefix)

        completed = _complete_prefix_greedily(
            instance,
            agent,
            normalizer,
            prefix,
            alternatives_per_state=alternatives_per_state,
            reward_scale=reward_scale,
        )
        enqueue(completed.branches)

        # A different prefix can occasionally converge to an already seen full
        # itinerary.  It does not consume the complete-rollout budget twice.
        if completed.itinerary in seen_itineraries:
            continue

        seen_itineraries.add(completed.itinerary)
        rollout_objectives.append(completed.objective)
        if completed.objective > best.objective:
            best = completed

    runtime = time.perf_counter() - started

    return _policy_result(
        method=f"dqn_backtracking_{rollout_budget}",
        instance=instance,
        itinerary=best.itinerary,
        runtime_seconds=runtime,
        rollouts=len(rollout_objectives),
        metadata={
            "requested_rollout_budget": rollout_budget,
            "unique_rollouts": len(rollout_objectives),
            "best_rollout_index": int(np.argmax(rollout_objectives)),
            "rollout_objectives": rollout_objectives,
            "alternatives_per_state": alternatives_per_state,
            "reward_scale": reward_scale,
            "branch_priority": "exact_scaled_prefix_return_plus_Q",
            "weights_updated_during_test": False,
        },
    )
