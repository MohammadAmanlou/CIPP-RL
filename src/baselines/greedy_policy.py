"""Exact one-step greedy construction baseline for deterministic CIPP."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from src.baselines.common import PolicyResult
from src.core import CIPPInstance, evaluate_itinerary
from src.envs import CIPPEnv


FloatArray = NDArray[np.float64]


def viable_action_rewards(
    environment: CIPPEnv,
) -> FloatArray:
    """Return exact immediate rewards, using -inf for masked actions."""

    mask = environment.get_action_mask()

    values = np.full(
        mask.shape,
        -np.inf,
        dtype=np.float64,
    )

    if not bool(mask.any()):
        return values

    # Idle has zero objective contribution.
    if bool(mask[0]):
        values[0] = 0.0

    counts = environment.visit_counts
    exposures = environment.temporal_exposures

    current_lambda = float(
        environment.instance.temporal_weights[
            environment.day
        ]
    )

    viable_visit_actions = (
        np.flatnonzero(mask[1:]) + 1
    )

    for action in viable_visit_actions:
        location_index = int(action) - 1

        old_count = int(
            counts[location_index]
        )

        new_count = old_count + 1

        old_exposure = float(
            exposures[location_index]
        )

        new_exposure = (
            old_exposure
            + current_lambda
        )

        base_reward = float(
            environment.instance.rewards[
                location_index
            ]
        )

        old_contribution = (
            base_reward
            * environment.instance.repeat_factor(
                old_count
            )
            * old_exposure
        )

        new_contribution = (
            base_reward
            * environment.instance.repeat_factor(
                new_count
            )
            * new_exposure
        )

        values[int(action)] = (
            new_contribution
            - old_contribution
        )

    return values


def select_greedy_action(
    environment: CIPPEnv,
) -> int:
    """Choose the viable action with the largest exact increment.

    Ties are resolved using the smallest action identifier.
    Action 0 is Idle.
    """

    values = viable_action_rewards(
        environment
    )

    if not np.isfinite(values).any():
        raise RuntimeError(
            "No viable action is available."
        )

    best_value = float(
        np.max(values)
    )

    tied_actions = np.flatnonzero(
        np.isclose(
            values,
            best_value,
            rtol=1e-12,
            atol=1e-12,
        )
    )

    if tied_actions.size == 0:
        raise RuntimeError(
            "Unable to select a greedy action."
        )

    return int(
        tied_actions[0]
    )


def run_greedy_policy(
    instance: CIPPInstance,
) -> PolicyResult:
    """Construct one itinerary using exact one-step greedy decisions."""

    environment = CIPPEnv(
        instance
    )

    environment.reset()

    started = time.perf_counter()

    while not environment.done:
        action = select_greedy_action(
            environment
        )

        environment.step(
            action
        )

    runtime_seconds = (
        time.perf_counter() - started
    )

    evaluation = evaluate_itinerary(
        instance,
        environment.itinerary,
    )

    if not evaluation.feasible:
        raise RuntimeError(
            "Greedy policy produced an infeasible itinerary: "
            + "; ".join(evaluation.violations)
        )

    if not np.isclose(
        environment.cumulative_reward,
        evaluation.objective,
        rtol=1e-10,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Greedy policy reward does not match "
            "the exact objective."
        )

    unique_locations = int(
        np.count_nonzero(
            evaluation.visit_counts
        )
    )

    return PolicyResult(
        method="greedy_exact_increment",
        instance_id=instance.instance_id,
        itinerary=tuple(
            int(action)
            for action in environment.itinerary
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
        unique_locations=unique_locations,
        feasible=True,
        violations=(),
        runtime_seconds=float(
            runtime_seconds
        ),
    )