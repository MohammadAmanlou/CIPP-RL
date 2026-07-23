"""Exact objective and feasibility evaluation for deterministic CIPP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.core.instance import CIPPInstance


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class EvaluationResult:
    """Complete evaluation of one campaign itinerary."""

    objective: float
    total_cost: float
    idle_days: int
    visit_counts: IntArray
    feasible: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into JSON-compatible values."""

        return {
            "objective": self.objective,
            "total_cost": self.total_cost,
            "idle_days": self.idle_days,
            "visit_counts": self.visit_counts.tolist(),
            "feasible": self.feasible,
            "violations": list(self.violations),
        }


def _validated_complete_itinerary(
    instance: CIPPInstance,
    itinerary: ArrayLike,
) -> IntArray:
    """Return a validated complete action sequence."""

    try:
        numeric = np.asarray(
            itinerary,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "itinerary must contain numeric action identifiers."
        ) from error

    if numeric.ndim != 1:
        raise ValueError(
            "itinerary must be one-dimensional."
        )

    if numeric.shape != (instance.H,):
        raise ValueError(
            f"itinerary must have length H={instance.H}; "
            f"got shape {numeric.shape}."
        )

    if not np.all(np.isfinite(numeric)):
        raise ValueError(
            "itinerary must contain only finite values."
        )

    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(
            "itinerary actions must be integers."
        )

    actions = numeric.astype(np.int64)

    if np.any(actions < 0) or np.any(actions > instance.n):
        raise ValueError(
            "itinerary actions must be between "
            f"0 and n={instance.n}."
        )

    return actions


def visit_counts(
    instance: CIPPInstance,
    itinerary: ArrayLike,
) -> IntArray:
    """Count the total visits assigned to every location."""

    actions = _validated_complete_itinerary(
        instance,
        itinerary,
    )

    counts = np.bincount(
        actions,
        minlength=instance.n + 1,
    )[1 : instance.n + 1]

    return counts.astype(np.int64)


def temporal_exposures(
    instance: CIPPInstance,
    itinerary: ArrayLike,
) -> FloatArray:
    """Compute the temporal exposure of every location."""

    actions = _validated_complete_itinerary(
        instance,
        itinerary,
    )

    exposures = np.zeros(
        instance.n,
        dtype=np.float64,
    )

    for location in range(1, instance.n + 1):
        mask = actions == location

        exposures[location - 1] = float(
            instance.temporal_weights[mask].sum()
        )

    return exposures


def deterministic_objective(
    instance: CIPPInstance,
    itinerary: ArrayLike,
) -> float:
    """Compute the exact campaign-wide deterministic objective."""

    counts = visit_counts(
        instance,
        itinerary,
    )

    exposures = temporal_exposures(
        instance,
        itinerary,
    )

    repeat_factors = (
        1.0 - instance.gamma * counts
    )

    contributions = (
        instance.rewards
        * repeat_factors
        * exposures
    )

    return float(contributions.sum())


def total_cost(
    instance: CIPPInstance,
    itinerary: ArrayLike,
) -> float:
    """Compute the total visit cost of an itinerary."""

    actions = _validated_complete_itinerary(
        instance,
        itinerary,
    )

    visit_actions = actions[actions > 0]

    if visit_actions.size == 0:
        return 0.0

    indices = visit_actions - 1

    return float(
        instance.costs[indices].sum()
    )


def evaluate_itinerary(
    instance: CIPPInstance,
    itinerary: ArrayLike,
) -> EvaluationResult:
    """Evaluate objective value and every deterministic constraint."""

    try:
        actions = _validated_complete_itinerary(
            instance,
            itinerary,
        )
    except ValueError as error:
        return EvaluationResult(
            objective=float("nan"),
            total_cost=float("nan"),
            idle_days=0,
            visit_counts=np.zeros(
                instance.n,
                dtype=np.int64,
            ),
            feasible=False,
            violations=(
                f"invalid_itinerary: {error}",
            ),
        )

    counts = visit_counts(
        instance,
        actions,
    )

    cost = total_cost(
        instance,
        actions,
    )

    objective = deterministic_objective(
        instance,
        actions,
    )

    idle_days = int(
        np.count_nonzero(actions == 0)
    )

    violations: list[str] = []

    # Total visit cap: sum_t Z_it <= q
    for location_index, count in enumerate(
        counts,
        start=1,
    ):
        if count > instance.q:
            violations.append(
                "total_visit_cap: "
                f"location={location_index}, "
                f"visits={int(count)}, "
                f"q={instance.q}"
            )

    # Rolling idle and location visit constraints.
    for start in range(instance.num_rolling_windows):
        end = start + instance.alpha
        window = actions[start:end]

        idle_count = int(
            np.count_nonzero(window == 0)
        )

        required_idle = int(
            instance.idle_requirements[start]
        )

        if idle_count < required_idle:
            violations.append(
                "idle_requirement: "
                f"window_start={start + 1}, "
                f"idle={idle_count}, "
                f"required={required_idle}"
            )

        for location in range(1, instance.n + 1):
            window_visits = int(
                np.count_nonzero(
                    window == location
                )
            )

            if window_visits > instance.w:
                violations.append(
                    "rolling_visit_cap: "
                    f"location={location}, "
                    f"window_start={start + 1}, "
                    f"visits={window_visits}, "
                    f"w={instance.w}"
                )

    # Budget constraint.
    if cost > instance.budget + 1e-9:
        violations.append(
            "budget_exceeded: "
            f"cost={cost}, "
            f"budget={instance.budget}"
        )

    return EvaluationResult(
        objective=objective,
        total_cost=cost,
        idle_days=idle_days,
        visit_counts=counts,
        feasible=len(violations) == 0,
        violations=tuple(violations),
    )