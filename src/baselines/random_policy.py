"""Random feasible construction baseline for deterministic CIPP."""

from __future__ import annotations

import time

import numpy as np

from src.baselines.common import PolicyResult
from src.core import CIPPInstance, evaluate_itinerary
from src.envs import CIPPEnv


def run_random_feasible_policy(
    instance: CIPPInstance,
    *,
    seed: int = 0,
) -> PolicyResult:
    """Construct one itinerary by sampling uniformly from viable actions."""

    environment = CIPPEnv(
        instance,
        seed=seed,
    )

    environment.reset(
        seed=seed,
    )

    started = time.perf_counter()

    while not environment.done:
        action = environment.sample_viable_action()

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
            "Random feasible policy produced an infeasible itinerary: "
            + "; ".join(evaluation.violations)
        )

    if not np.isclose(
        environment.cumulative_reward,
        evaluation.objective,
        rtol=1e-10,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Random feasible policy reward does not match "
            "the exact objective."
        )

    unique_locations = int(
        np.count_nonzero(
            evaluation.visit_counts
        )
    )

    return PolicyResult(
        method="random_feasible",
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