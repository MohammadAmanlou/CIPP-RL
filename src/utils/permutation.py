"""Location relabelling helpers used in invariance tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.core import CIPPInstance


IntArray = NDArray[np.int64]


def _validated_permutation(
    permutation: Sequence[int],
    n: int,
) -> IntArray:
    """Return a zero-based permutation of range(n)."""

    values = np.asarray(
        permutation,
        dtype=np.int64,
    )

    if values.shape != (n,):
        raise ValueError(
            f"permutation must have shape ({n},)."
        )

    expected = np.arange(
        n,
        dtype=np.int64,
    )

    if not np.array_equal(
        np.sort(values),
        expected,
    ):
        raise ValueError(
            "permutation must contain every integer "
            "from 0 to n-1 once."
        )

    return values


def permute_instance(
    instance: CIPPInstance,
    permutation: Sequence[int],
    *,
    instance_id: str | None = None,
) -> CIPPInstance:
    """Return an equivalent instance with relabelled locations."""

    order = _validated_permutation(
        permutation,
        instance.n,
    )

    return CIPPInstance(
        n=instance.n,
        H=instance.H,
        rewards=instance.rewards[order],
        costs=instance.costs[order],
        budget=instance.budget,
        alpha=instance.alpha,
        idle_requirements=instance.idle_requirements,
        q=instance.q,
        w=instance.w,
        temporal_weights=instance.temporal_weights,
        gamma=instance.gamma,
        p=instance.p,
        instance_id=(
            instance_id
            or f"{instance.instance_id}-permuted"
        ),
    )


def relabel_itinerary(
    itinerary: ArrayLike,
    permutation: Sequence[int],
) -> IntArray:
    """Relabel an itinerary consistently with permute_instance."""

    actions = np.asarray(
        itinerary,
        dtype=np.int64,
    )

    n = len(permutation)

    order = _validated_permutation(
        permutation,
        n,
    )

    if actions.ndim != 1:
        raise ValueError(
            "itinerary must be one-dimensional."
        )

    if np.any(actions < 0) or np.any(actions > n):
        raise ValueError(
            f"itinerary actions must be between 0 and {n}."
        )

    old_to_new = np.empty(
        n,
        dtype=np.int64,
    )

    old_to_new[order] = np.arange(
        n,
        dtype=np.int64,
    )

    relabelled = actions.copy()

    visit_mask = relabelled > 0

    relabelled[visit_mask] = (
        old_to_new[
            relabelled[visit_mask] - 1
        ]
        + 1
    )

    return relabelled