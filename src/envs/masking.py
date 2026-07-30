"""Exact and efficient viability masking for deterministic CIPP.

For the base CIPP model, a prefix has a feasible completion if completing every
remaining day with ``Idle`` is feasible.  Future idle actions add no visits and
no cost and maximize the number of idle periods, so this is an exact
certificate for the current constraint set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.core import CIPPInstance


IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def validate_partial_itinerary(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> IntArray:
    """Validate a one-dimensional prefix with actions in ``0..n``."""

    try:
        values = np.asarray(itinerary_prefix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "itinerary_prefix must contain numeric actions."
        ) from error

    if values.ndim != 1:
        raise ValueError("itinerary_prefix must be one-dimensional.")
    if values.size > instance.H:
        raise ValueError(f"prefix length cannot exceed H={instance.H}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("prefix actions must be finite.")
    if not np.all(values == np.floor(values)):
        raise ValueError("prefix actions must be integers.")

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

    actions = validate_partial_itinerary(instance, itinerary_prefix)
    return np.bincount(
        actions,
        minlength=instance.n + 1,
    )[1:].astype(np.int64)


def partial_cost(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> float:
    """Compute cost committed by a partial itinerary."""

    actions = validate_partial_itinerary(instance, itinerary_prefix)
    visits = actions[actions > 0]
    if visits.size == 0:
        return 0.0
    return float(instance.costs[visits - 1].sum())


def partial_temporal_exposures(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> FloatArray:
    """Compute accumulated temporal exposure for every location."""

    actions = validate_partial_itinerary(instance, itinerary_prefix)
    exposures = np.zeros(instance.n, dtype=np.float64)
    for day, action in enumerate(actions):
        if action > 0:
            exposures[action - 1] += instance.temporal_weights[day]
    return exposures


def _active_window_starts(instance: CIPPInstance, day: int) -> range:
    """Return rolling-window starts whose windows contain ``day``.

    ``day`` is zero based and identifies the action that is about to be placed.
    """

    first = max(0, day - instance.alpha + 1)
    last = min(day, instance.num_rolling_windows - 1)
    return range(first, last + 1)


def prefix_has_feasible_completion(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> bool:
    """Return whether the prefix has an all-idle feasible completion."""

    prefix = validate_partial_itinerary(instance, itinerary_prefix)
    prefix_length = int(prefix.size)

    counts = np.bincount(prefix, minlength=instance.n + 1)[1:]
    if np.any(counts > instance.q):
        return False

    visits = prefix[prefix > 0]
    committed_cost = (
        0.0 if visits.size == 0 else float(instance.costs[visits - 1].sum())
    )
    if committed_cost > instance.budget + 1e-9:
        return False

    # Only windows that have at least one assigned position need inspection.
    last_relevant_start = min(prefix_length - 1, instance.num_rolling_windows - 1)
    if last_relevant_start < 0:
        return True

    for start in range(last_relevant_start + 1):
        end = start + instance.alpha
        assigned_end = min(prefix_length, end)
        fixed = prefix[start:assigned_end]

        fixed_idle = int(np.count_nonzero(fixed == 0))
        future_positions = end - assigned_end
        if fixed_idle + future_positions < int(instance.idle_requirements[start]):
            return False

        if fixed.size:
            window_counts = np.bincount(
                fixed,
                minlength=instance.n + 1,
            )[1:]
            if np.any(window_counts > instance.w):
                return False

    return True


def is_action_viable(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
    action: int,
) -> bool:
    """Check whether one action preserves an all-idle completion."""

    prefix = validate_partial_itinerary(instance, itinerary_prefix)
    if prefix.size >= instance.H:
        return False
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        return False
    normalized = int(action)
    if normalized < 0 or normalized > instance.n:
        return False

    return bool(get_viability_mask(instance, prefix)[normalized])


def get_viability_mask(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> BoolArray:
    """Return the exact viability mask for the next decision.

    The implementation validates the prefix once and checks only rolling
    windows that contain the next day.  This is equivalent to testing every
    candidate with a full all-idle completion, but is much faster during RL
    training.
    """

    prefix = validate_partial_itinerary(instance, itinerary_prefix)
    day = int(prefix.size)
    mask = np.zeros(instance.num_actions, dtype=np.bool_)

    if day >= instance.H or not prefix_has_feasible_completion(instance, prefix):
        return mask

    # Idle never adds cost or visits and can only help idle requirements.
    mask[0] = True

    counts = np.bincount(prefix, minlength=instance.n + 1)[1:].astype(np.int64)
    visits = prefix[prefix > 0]
    spent = 0.0 if visits.size == 0 else float(instance.costs[visits - 1].sum())

    # Conditions shared by all visit actions: choosing a visit instead of idle
    # must still leave enough unassigned periods in every active window to meet
    # the minimum-idle requirement.
    active_starts = tuple(_active_window_starts(instance, day))
    visit_preserves_idle = True
    for start in active_starts:
        end = start + instance.alpha
        fixed_before = prefix[start:day]
        fixed_idle = int(np.count_nonzero(fixed_before == 0))
        future_positions = max(0, end - (day + 1))
        if fixed_idle + future_positions < int(instance.idle_requirements[start]):
            visit_preserves_idle = False
            break

    if not visit_preserves_idle:
        return mask

    for action in range(1, instance.num_actions):
        location = action - 1
        if counts[location] >= instance.q:
            continue
        if spent + float(instance.costs[location]) > instance.budget + 1e-9:
            continue

        rolling_ok = True
        for start in active_starts:
            prior_window_visits = int(np.count_nonzero(prefix[start:day] == action))
            if prior_window_visits + 1 > instance.w:
                rolling_ok = False
                break
        if rolling_ok:
            mask[action] = True

    return mask


def all_idle_completion(
    instance: CIPPInstance,
    itinerary_prefix: ArrayLike,
) -> IntArray:
    """Complete a viable prefix using only Idle actions."""

    prefix = validate_partial_itinerary(instance, itinerary_prefix)
    if not prefix_has_feasible_completion(instance, prefix):
        raise ValueError("The prefix has no feasible all-idle completion.")

    return np.concatenate(
        [prefix, np.zeros(instance.H - prefix.size, dtype=np.int64)]
    )
