"""Exact viability masking for the deterministic base CIPP model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.core import CIPPInstance


IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def validate_partial_itinerary(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> IntArray:
    """Validate a prefix with actions 0..n and length at most H."""

    try:
        values = np.asarray(
            itinerary_prefix,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "itinerary_prefix must contain numeric actions."
        ) from error

    if values.ndim != 1:
        raise ValueError(
            "itinerary_prefix must be one-dimensional."
        )

    if values.size > instance.H:
        raise ValueError(
            f"prefix length cannot exceed H={instance.H}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "prefix actions must be finite."
        )

    if not np.all(values == np.floor(values)):
        raise ValueError(
            "prefix actions must be integers."
        )

    actions = values.astype(np.int64)

    if np.any(actions < 0) or np.any(actions > instance.n):
        raise ValueError(
            f"prefix actions must be between 0 and n={instance.n}."
        )

    return actions


def partial_visit_counts(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> IntArray:
    """Count visits made so far to every location."""

    actions = validate_partial_itinerary(
        instance,
        itinerary_prefix,
    )

    return np.bincount(
        actions,
        minlength=instance.n + 1,
    )[1:].astype(np.int64)


def partial_cost(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> float:
    """Compute the cost committed by a partial itinerary."""

    actions = validate_partial_itinerary(
        instance,
        itinerary_prefix,
    )

    visits = actions[actions > 0]

    if visits.size == 0:
        return 0.0

    return float(
        instance.costs[visits - 1].sum()
    )


def partial_temporal_exposures(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> NDArray[np.float64]:
    """Compute accumulated temporal exposure for every location."""

    actions = validate_partial_itinerary(
        instance,
        itinerary_prefix,
    )

    exposures = np.zeros(
        instance.n,
        dtype=np.float64,
    )

    for day, action in enumerate(actions):
        if action > 0:
            exposures[action - 1] += (
                instance.temporal_weights[day]
            )

    return exposures


def prefix_has_feasible_completion(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> bool:
    """Check whether the prefix can be completed using Idle actions.

    In the deterministic base CIPP, future Idle actions:

    - add no cost;
    - add no location visits;
    - maximize idle counts in future windows.

    Therefore, if the all-idle completion is feasible, the prefix has
    at least one feasible completion.
    """

    prefix = validate_partial_itinerary(
        instance,
        itinerary_prefix,
    )

    prefix_length = int(prefix.size)

    counts = partial_visit_counts(
        instance,
        prefix,
    )

    if np.any(counts > instance.q):
        return False

    if (
        partial_cost(instance, prefix)
        > instance.budget + 1e-9
    ):
        return False

    for start in range(
        instance.num_rolling_windows
    ):
        end = start + instance.alpha

        assigned_end = min(
            prefix_length,
            end,
        )

        if assigned_end > start:
            fixed_actions = prefix[
                start:assigned_end
            ]
        else:
            fixed_actions = np.empty(
                0,
                dtype=np.int64,
            )

        fixed_idle = int(
            np.count_nonzero(
                fixed_actions == 0
            )
        )

        unassigned_positions = (
            end - assigned_end
        )

        required_idle = int(
            instance.idle_requirements[start]
        )

        if (
            fixed_idle + unassigned_positions
            < required_idle
        ):
            return False

        if fixed_actions.size > 0:
            window_counts = np.bincount(
                fixed_actions,
                minlength=instance.n + 1,
            )[1:]

            if np.any(
                window_counts > instance.w
            ):
                return False

    return True


def is_action_viable(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
    action: int,
) -> bool:
    """Check whether one action preserves a feasible completion."""

    prefix = validate_partial_itinerary(
        instance,
        itinerary_prefix,
    )

    if prefix.size >= instance.H:
        return False

    if isinstance(action, bool) or not isinstance(
        action,
        (int, np.integer),
    ):
        return False

    normalized_action = int(action)

    if (
        normalized_action < 0
        or normalized_action > instance.n
    ):
        return False

    candidate_prefix = np.append(
        prefix,
        normalized_action,
    )

    return prefix_has_feasible_completion(
        instance,
        candidate_prefix,
    )


def get_viability_mask(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> BoolArray:
    """Return a mask of all viable actions.

    Index 0 corresponds to Idle.
    Indices 1..n correspond to location visits.
    """

    prefix = validate_partial_itinerary(
        instance,
        itinerary_prefix,
    )

    mask = np.zeros(
        instance.num_actions,
        dtype=np.bool_,
    )

    if prefix.size >= instance.H:
        return mask

    if not prefix_has_feasible_completion(
        instance,
        prefix,
    ):
        return mask

    for action in range(
        instance.num_actions
    ):
        mask[action] = is_action_viable(
            instance,
            prefix,
            action,
        )

    return mask


def all_idle_completion(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> IntArray:
    """Complete a viable prefix using only Idle actions."""

    prefix = validate_partial_itinerary(
        instance,
        itinerary_prefix,
    )

    if not prefix_has_feasible_completion(
        instance,
        prefix,
    ):
        raise ValueError(
            "The prefix has no feasible all-idle completion."
        )

    remaining_periods = (
        instance.H - prefix.size
    )

    return np.concatenate(
        [
            prefix,
            np.zeros(
                remaining_periods,
                dtype=np.int64,
            ),
        ]
    )