"""Reproducible synthetic instance generation for deterministic CIPP."""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.core import CIPPInstance


TemporalProfile = Literal[
    "random",
    "flat",
    "increasing",
    "decreasing",
    "u_shaped",
]


def _validate_range(
    name: str,
    value_range: tuple[float, float],
) -> tuple[float, float]:
    """Validate and normalize a positive numeric interval."""

    if len(value_range) != 2:
        raise ValueError(f"{name} must contain exactly two values.")

    low = float(value_range[0])
    high = float(value_range[1])

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{name} values must be finite.")

    if low <= 0 or high <= 0:
        raise ValueError(f"{name} values must be positive.")

    if low > high:
        raise ValueError(
            f"{name} lower bound must not exceed its upper bound."
        )

    return low, high


def _temporal_weights(
    horizon: int,
    profile: TemporalProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a positive temporal profile with mean exactly one."""

    allowed_profiles = {
        "random",
        "flat",
        "increasing",
        "decreasing",
        "u_shaped",
    }

    if profile not in allowed_profiles:
        raise ValueError(
            "temporal_profile must be one of: "
            + ", ".join(sorted(allowed_profiles))
        )

    selected_profile = profile

    if selected_profile == "random":
        selected_profile = str(
            rng.choice(
                [
                    "flat",
                    "increasing",
                    "decreasing",
                    "u_shaped",
                ]
            )
        )

    x = np.linspace(
        0.0,
        1.0,
        horizon,
        dtype=np.float64,
    )

    if selected_profile == "flat":
        weights = np.ones(
            horizon,
            dtype=np.float64,
        )

    elif selected_profile == "increasing":
        weights = 0.75 + 0.50 * x

    elif selected_profile == "decreasing":
        weights = 1.25 - 0.50 * x

    else:
        weights = 0.75 + np.abs(x - 0.5)

    noise = rng.normal(
        loc=0.0,
        scale=0.025,
        size=horizon,
    )

    weights = np.clip(
        weights + noise,
        0.05,
        None,
    )

    weights /= float(weights.mean())

    return weights.astype(np.float64)


def _idle_schedule(
    horizon: int,
    alpha: int,
    rng: np.random.Generator,
    max_idle_requirement: int | None,
) -> np.ndarray:
    """Generate a piecewise-constant nonincreasing idle schedule."""

    number_of_windows = horizon - alpha + 1

    if max_idle_requirement is None:
        max_idle_requirement = max(
            1,
            int(np.ceil(0.4 * alpha)),
        )

    if isinstance(max_idle_requirement, bool) or not isinstance(
        max_idle_requirement,
        (int, np.integer),
    ):
        raise TypeError(
            "max_idle_requirement must be an integer."
        )

    maximum = int(max_idle_requirement)

    if maximum < 0 or maximum > alpha:
        raise ValueError(
            "max_idle_requirement must satisfy "
            f"0 <= value <= alpha={alpha}."
        )

    number_of_segments = min(
        3,
        number_of_windows,
    )

    if number_of_segments == 1:
        return np.array(
            [
                int(
                    rng.integers(
                        0,
                        maximum + 1,
                    )
                )
            ],
            dtype=np.int64,
        )

    candidate_breaks = np.arange(
        1,
        number_of_windows,
    )

    break_count = min(
        number_of_segments - 1,
        candidate_breaks.size,
    )

    if break_count > 0:
        breakpoints = np.sort(
            rng.choice(
                candidate_breaks,
                size=break_count,
                replace=False,
            )
        )
    else:
        breakpoints = np.empty(
            0,
            dtype=np.int64,
        )

    segment_bounds = np.concatenate(
        (
            np.array(
                [0],
                dtype=np.int64,
            ),
            breakpoints.astype(np.int64),
            np.array(
                [number_of_windows],
                dtype=np.int64,
            ),
        )
    )

    segment_count = len(segment_bounds) - 1

    levels = np.sort(
        rng.integers(
            0,
            maximum + 1,
            size=segment_count,
        )
    )[::-1]

    schedule = np.empty(
        number_of_windows,
        dtype=np.int64,
    )

    for index in range(segment_count):
        start = int(segment_bounds[index])
        end = int(segment_bounds[index + 1])

        schedule[start:end] = int(
            levels[index]
        )

    return schedule


def generate_cipp_instance(
    *,
    n: int = 14,
    H: int = 30,
    seed: int = 42,
    alpha: int = 5,
    q: int = 6,
    w: int = 2,
    gamma: float | None = None,
    reward_range: tuple[float, float] = (
        50.0,
        150.0,
    ),
    cost_range: tuple[float, float] = (
        5.0,
        25.0,
    ),
    budget_tightness: float = 0.65,
    max_idle_requirement: int | None = None,
    temporal_profile: TemporalProfile = "random",
    instance_id: str | None = None,
) -> CIPPInstance:
    """Generate one deterministic synthetic CIPP instance.

    The same arguments and seed always produce exactly
    the same instance.
    """

    if isinstance(seed, bool) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise TypeError("seed must be an integer.")

    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or n < 1
    ):
        raise ValueError(
            "n must be a positive integer."
        )

    if (
        isinstance(H, bool)
        or not isinstance(H, (int, np.integer))
        or H < 1
    ):
        raise ValueError(
            "H must be a positive integer."
        )

    if isinstance(alpha, bool) or not isinstance(
        alpha,
        (int, np.integer),
    ):
        raise TypeError(
            "alpha must be an integer."
        )

    if alpha < 1 or alpha > H:
        raise ValueError(
            "alpha must satisfy 1 <= alpha <= H."
        )

    if (
        isinstance(q, bool)
        or not isinstance(q, (int, np.integer))
        or q < 1
    ):
        raise ValueError(
            "q must be a positive integer."
        )

    if (
        isinstance(w, bool)
        or not isinstance(w, (int, np.integer))
        or w < 0
    ):
        raise ValueError(
            "w must be a nonnegative integer."
        )

    if w > alpha:
        raise ValueError(
            "w must not exceed alpha."
        )

    tightness = float(budget_tightness)

    if (
        not np.isfinite(tightness)
        or not 0.0 < tightness <= 1.0
    ):
        raise ValueError(
            "budget_tightness must be in the interval (0, 1]."
        )

    reward_low, reward_high = _validate_range(
        "reward_range",
        reward_range,
    )

    cost_low, cost_high = _validate_range(
        "cost_range",
        cost_range,
    )

    rng = np.random.default_rng(
        int(seed)
    )

    latent = rng.uniform(
        0.0,
        1.0,
        size=int(n),
    )

    reward_noise = rng.normal(
        0.0,
        0.08,
        size=int(n),
    )

    cost_noise = rng.normal(
        0.0,
        0.08,
        size=int(n),
    )

    reward_scale = np.clip(
        latent + reward_noise,
        0.0,
        1.0,
    )

    cost_scale = np.clip(
        latent + cost_noise,
        0.0,
        1.0,
    )

    rewards = (
        reward_low
        + (reward_high - reward_low)
        * reward_scale
    )

    costs = (
        cost_low
        + (cost_high - cost_low)
        * cost_scale
    )

    temporal_weights = _temporal_weights(
        int(H),
        temporal_profile,
        rng,
    )

    idle_requirements = _idle_schedule(
        int(H),
        int(alpha),
        rng,
        max_idle_requirement,
    )

    if gamma is None:
        gamma_value = float(
            rng.uniform(
                0.0,
                1.0 / int(q),
            )
        )
    else:
        gamma_value = float(gamma)

    if (
        not np.isfinite(gamma_value)
        or not 0.0 <= gamma_value <= 1.0 / int(q)
    ):
        raise ValueError(
            "gamma must satisfy 0 <= gamma <= 1/q."
        )

    budget = (
        tightness
        * int(H)
        * float(costs.mean())
    )

    resolved_id = (
        instance_id
        or f"synthetic-n{n}-H{H}-seed{int(seed)}"
    )

    return CIPPInstance(
        n=int(n),
        H=int(H),
        rewards=rewards,
        costs=costs,
        budget=float(budget),
        alpha=int(alpha),
        idle_requirements=idle_requirements,
        q=int(q),
        w=int(w),
        temporal_weights=temporal_weights,
        gamma=gamma_value,
        p=1,
        instance_id=resolved_id,
    )