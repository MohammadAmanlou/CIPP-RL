"""Data model for deterministic CIPP instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _require_integer(
    name: str,
    value: Any,
    minimum: int,
) -> int:
    """Return a validated integer scalar."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    normalized = int(value)

    if normalized < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}; "
            f"got {normalized}."
        )

    return normalized


def _require_nonnegative_float(
    name: str,
    value: Any,
) -> float:
    """Return a finite, nonnegative float scalar."""

    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be numeric, not bool."
        )

    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be numeric."
        ) from error

    if not np.isfinite(normalized):
        raise ValueError(
            f"{name} must be finite."
        )

    if normalized < 0:
        raise ValueError(
            f"{name} must be nonnegative; "
            f"got {normalized}."
        )

    return normalized


def _float_vector(
    name: str,
    values: ArrayLike,
    expected_length: int,
) -> FloatArray:
    """Create a validated nonnegative float vector."""

    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        ).copy()
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must contain numeric values."
        ) from error

    expected_shape = (expected_length,)

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; "
            f"got {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    if np.any(array < 0):
        raise ValueError(
            f"{name} must contain only nonnegative values."
        )

    array.setflags(write=False)

    return array


def _integer_vector(
    name: str,
    values: ArrayLike,
    expected_length: int,
    minimum: int,
    maximum: int,
) -> IntArray:
    """Create a validated bounded integer vector."""

    try:
        numeric_array = np.asarray(
            values,
            dtype=np.float64,
        ).copy()
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must contain numeric values."
        ) from error

    expected_shape = (expected_length,)

    if numeric_array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; "
            f"got {numeric_array.shape}."
        )

    if not np.all(np.isfinite(numeric_array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    if not np.all(numeric_array == np.floor(numeric_array)):
        raise ValueError(
            f"{name} must contain only integer values."
        )

    array = numeric_array.astype(np.int64)

    if np.any(array < minimum):
        raise ValueError(
            f"{name} values must be at least {minimum}."
        )

    if np.any(array > maximum):
        raise ValueError(
            f"{name} values must not exceed {maximum}."
        )

    array.setflags(write=False)

    return array


@dataclass(slots=True)
class CIPPInstance:
    """Parameters of one deterministic CIPP instance.

    Action encoding:

    - 0 means Idle.
    - Actions 1 through n represent location visits.
    """

    n: int
    H: int

    rewards: ArrayLike = field(repr=False)
    costs: ArrayLike = field(repr=False)

    budget: float
    alpha: int

    idle_requirements: ArrayLike = field(repr=False)

    q: int
    w: int

    temporal_weights: ArrayLike = field(repr=False)
    gamma: float

    # 0 reproduces the original project formulation mu_j = 1-gamma*j.
    # 1 reproduces the professor MILP formulation 1-gamma*(j-1).
    repeat_count_offset: int = 0

    p: int = 1
    instance_id: str = "unnamed"

    def __post_init__(self) -> None:
        """Normalize and validate all model parameters."""

        self.n = _require_integer(
            "n",
            self.n,
            minimum=1,
        )

        self.H = _require_integer(
            "H",
            self.H,
            minimum=1,
        )

        self.alpha = _require_integer(
            "alpha",
            self.alpha,
            minimum=1,
        )

        self.q = _require_integer(
            "q",
            self.q,
            minimum=1,
        )

        self.w = _require_integer(
            "w",
            self.w,
            minimum=0,
        )

        self.p = _require_integer(
            "p",
            self.p,
            minimum=1,
        )

        self.repeat_count_offset = _require_integer(
            "repeat_count_offset",
            self.repeat_count_offset,
            minimum=0,
        )

        if self.repeat_count_offset not in (0, 1):
            raise ValueError(
                "repeat_count_offset must be either 0 or 1."
            )

        if self.p != 1:
            raise ValueError(
                "The base implementation requires p == 1."
            )

        if self.alpha > self.H:
            raise ValueError(
                "alpha must not exceed H; "
                f"got alpha={self.alpha}, H={self.H}."
            )

        if self.w > self.alpha:
            raise ValueError(
                "w must not exceed alpha; "
                f"got w={self.w}, alpha={self.alpha}."
            )

        self.budget = _require_nonnegative_float(
            "budget",
            self.budget,
        )

        self.gamma = _require_nonnegative_float(
            "gamma",
            self.gamma,
        )

        maximum_discounted_count = max(
            self.q - self.repeat_count_offset,
            0,
        )
        maximum_gamma = (
            float("inf")
            if maximum_discounted_count == 0
            else 1.0 / maximum_discounted_count
        )

        if self.gamma > maximum_gamma + 1e-12:
            raise ValueError(
                "gamma must satisfy nonnegative repeat factors up to q; "
                f"got gamma={self.gamma}, q={self.q}, "
                f"repeat_count_offset={self.repeat_count_offset}."
            )

        self.rewards = _float_vector(
            name="rewards",
            values=self.rewards,
            expected_length=self.n,
        )

        self.costs = _float_vector(
            name="costs",
            values=self.costs,
            expected_length=self.n,
        )

        self.temporal_weights = _float_vector(
            name="temporal_weights",
            values=self.temporal_weights,
            expected_length=self.H,
        )

        number_of_windows = self.H - self.alpha + 1

        self.idle_requirements = _integer_vector(
            name="idle_requirements",
            values=self.idle_requirements,
            expected_length=number_of_windows,
            minimum=0,
            maximum=self.alpha,
        )

        if not isinstance(self.instance_id, str):
            raise TypeError(
                "instance_id must be a string."
            )

        self.instance_id = self.instance_id.strip()

        if not self.instance_id:
            raise ValueError(
                "instance_id must not be empty."
            )

    @property
    def tmax(self) -> int:
        """Paper-compatible alias for H."""
        return self.H

    @property
    def num_actions(self) -> int:
        """Number of actions before masking."""
        return self.n + 1

    @property
    def num_rolling_windows(self) -> int:
        """Number of complete alpha-period windows."""
        return self.H - self.alpha + 1

    def repeat_factor(
        self,
        visit_count: int,
    ) -> float:
        """Return the configured repeat factor for ``visit_count``.

        ``repeat_count_offset=0`` gives ``1-gamma*j``.
        ``repeat_count_offset=1`` gives the professor MILP factor
        ``1-gamma*(j-1)`` for positive visit counts.
        """

        count = _require_integer(
            "visit_count",
            visit_count,
            minimum=0,
        )

        # The evaluator also calls this method for infeasible itineraries so it
        # can report the visit-cap violation together with a diagnostic
        # objective value.  Therefore counts above q are accepted here.
        discounted_count = max(
            count - self.repeat_count_offset,
            0,
        )

        return 1.0 - self.gamma * discounted_count