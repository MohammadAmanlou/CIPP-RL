"""Adapters for the real CIPP data and the supplied professor MILP.

Two small-instance interpretations are supported because the supplied script
and the revised paper are not identical:

``supplied-code``
    Reproduces ``Cities=16`` from the supplied Python script.  Since row/index
    zero is ``Rest``, this contains 15 real visit locations.

``paper-14``
    Uses 14 real locations, matching the smallest class described in the
    revised paper.  The exact subset should be confirmed with the PI before
    comparing against a published table value.

All methods in one benchmark use the same canonical :class:`CIPPInstance`, so
RL, greedy, random and Gurobi are always evaluated under identical semantics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from src.core import CIPPInstance


PROFESSOR_GAMMA = 0.04
PROFESSOR_MAX_MEETINGS = 12
PROFESSOR_ROLLING_VISIT_CAP = 2
PROFESSOR_WINDOW_LENGTH = 7
PROFESSOR_BETA = (2, 2, 2, 0)
PROFESSOR_SMALLEST_CITIES_PARAMETER = 16
PAPER_SMALLEST_REAL_LOCATIONS = 14


def professor_temporal_weights(horizon: int) -> np.ndarray:
    """Return the exact U/convex coefficients used in the supplied MILP."""

    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")

    # The supplied code has T={0,...,H}; day 0 is fictitious and has no visit.
    # For actual days 1..H, len(T)=H+1.
    total_period_count = horizon + 1
    center = total_period_count / 2.0
    constant = ((total_period_count - 2) / 4.0) ** 2
    actual_days = np.arange(1, horizon + 1, dtype=np.float64)
    return (actual_days - center) ** 2 + constant


def professor_idle_requirements(
    horizon: int,
    *,
    window_length: int = PROFESSOR_WINDOW_LENGTH,
    beta: tuple[int, int, int, int] = PROFESSOR_BETA,
) -> np.ndarray:
    """Return the paper-consistent piecewise idle requirements.

    The supplied code writes ``alpha=6`` and sums from ``t`` through
    ``t+alpha`` inclusively, so the effective complete-window length is seven.
    We use only complete seven-day windows, matching the mathematical model.
    """

    if window_length < 1 or window_length > horizon:
        raise ValueError("window_length must satisfy 1 <= value <= horizon.")
    if len(beta) != 4 or any(value < 0 for value in beta):
        raise ValueError("beta must contain four nonnegative integers.")

    number_of_windows = horizon - window_length + 1
    requirements = np.empty(number_of_windows, dtype=np.int64)
    quarter = horizon // 4
    half = horizon // 2
    three_quarters = (3 * horizon) // 4

    for index in range(number_of_windows):
        start_day = index + 1
        if start_day <= quarter:
            requirements[index] = beta[0]
        elif start_day <= half:
            requirements[index] = beta[1]
        elif start_day <= three_quarters:
            requirements[index] = beta[2]
        else:
            requirements[index] = beta[3]

    if np.any(requirements > window_length):
        raise ValueError("idle requirement cannot exceed the window length.")
    return requirements


def _read_flat_xls_sheet(path: str | Path, sheet_index: int) -> list[object]:
    """Read one legacy XLS sheet in the same column-major order as the script."""

    try:
        import xlrd  # type: ignore
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading legacy .xls files requires xlrd. Install dependencies "
            "with: python -m pip install -r requirements.txt. Alternatively "
            "use the processed CSV files included in data/processed/."
        ) from error

    workbook = xlrd.open_workbook(str(path))
    sheet = workbook.sheet_by_index(sheet_index)
    return [
        sheet.cell_value(row, column)
        for column in range(sheet.ncols)
        for row in range(sheet.nrows)
    ]


def _read_processed_csv(path: str | Path) -> tuple[list[float], list[str]]:
    rewards: list[float] = []
    identifiers: list[str] = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"location_id", "reward"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Processed professor CSV must contain location_id and reward columns."
            )
        for row in reader:
            identifiers.append(str(row["location_id"]).strip())
            rewards.append(float(row["reward"]))
    return rewards, identifiers


def load_professor_dataset(
    path: str | Path,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load all source rows, including the leading ``Rest`` row."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        reward_raw, id_raw = _read_processed_csv(source)
    elif suffix == ".xls":
        reward_raw = _read_flat_xls_sheet(source, 0)
        id_raw = _read_flat_xls_sheet(source, 4)
    else:
        raise ValueError("Professor data must be a .xls or processed .csv file.")

    if len(reward_raw) != len(id_raw):
        raise ValueError("Reward and ID sheets have different lengths.")
    rewards = np.asarray(reward_raw, dtype=np.float64)
    identifiers = tuple(str(value).strip() for value in id_raw)

    if rewards.ndim != 1 or rewards.size < 2:
        raise ValueError("Professor data must contain Rest and at least one location.")
    if not np.all(np.isfinite(rewards)) or np.any(rewards < 0):
        raise ValueError("BRP rewards must be finite and nonnegative.")
    if identifiers[0].lower() not in {"rest", "idle"}:
        raise ValueError(
            "The first source row must be the dummy Rest/Idle entry; "
            f"got {identifiers[0]!r}."
        )
    return rewards, identifiers


def resolve_real_location_count(
    *,
    cities_parameter: int = PROFESSOR_SMALLEST_CITIES_PARAMETER,
    instance_mode: str = "supplied-code",
    real_location_count: int | None = None,
) -> int:
    """Resolve the number of non-dummy locations for one benchmark."""

    if cities_parameter < 2:
        raise ValueError("cities_parameter must include Rest and one location.")
    if real_location_count is not None:
        count = int(real_location_count)
    elif instance_mode == "supplied-code":
        count = cities_parameter - 1
    elif instance_mode == "paper-14":
        count = PAPER_SMALLEST_REAL_LOCATIONS
    else:
        raise ValueError("instance_mode must be 'supplied-code' or 'paper-14'.")

    if count < 1 or count > cities_parameter - 1:
        raise ValueError(
            "real_location_count must be between 1 and cities_parameter-1."
        )
    return count


def load_professor_arrays(
    data_path: str | Path,
    *,
    cities_parameter: int = PROFESSOR_SMALLEST_CITIES_PARAMETER,
    instance_mode: str = "supplied-code",
    real_location_count: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Select real rewards/IDs after removing the dummy Rest row."""

    all_rewards, all_ids = load_professor_dataset(data_path)
    count = resolve_real_location_count(
        cities_parameter=cities_parameter,
        instance_mode=instance_mode,
        real_location_count=real_location_count,
    )
    required_rows = 1 + count
    if all_rewards.size < required_rows:
        raise ValueError(
            f"Data contains {all_rewards.size - 1} real rows but {count} are required."
        )

    rewards = all_rewards[1:required_rows].astype(np.float64, copy=True)
    identifiers = tuple(all_ids[1:required_rows])
    return rewards, identifiers


def professor_instance_label(
    party: str,
    real_location_count: int,
    horizon: int,
    *,
    instance_mode: str,
) -> str:
    """Create an honest label that does not hide the code/paper discrepancy."""

    normalized_party = party.upper()
    if normalized_party not in {"D", "R"}:
        raise ValueError("party must be D or R.")
    suffix = "CODE" if instance_mode == "supplied-code" else "PAPER_SHAPE"
    return f"{normalized_party}_{real_location_count}S_{horizon}P_{suffix}"


def published_instance_label(party: str, cities_parameter: int, horizon: int) -> str:
    """Backward-compatible label helper for the supplied-code interpretation."""

    count = cities_parameter - 1
    return professor_instance_label(
        party,
        count,
        horizon,
        instance_mode="supplied-code",
    )


def load_professor_instance(
    data_path: str | Path,
    *,
    party: str,
    cities_parameter: int = PROFESSOR_SMALLEST_CITIES_PARAMETER,
    horizon: int = 30,
    instance_mode: str = "supplied-code",
    real_location_count: int | None = None,
) -> tuple[CIPPInstance, tuple[str, ...]]:
    """Create a canonical RL/Gurobi instance from the professor data."""

    rewards, location_ids = load_professor_arrays(
        data_path,
        cities_parameter=cities_parameter,
        instance_mode=instance_mode,
        real_location_count=real_location_count,
    )

    instance = CIPPInstance(
        n=int(rewards.size),
        H=horizon,
        rewards=rewards,
        # The supplied Python script comments out the budget constraint.
        costs=np.zeros(rewards.size, dtype=np.float64),
        budget=0.0,
        alpha=PROFESSOR_WINDOW_LENGTH,
        idle_requirements=professor_idle_requirements(horizon),
        q=PROFESSOR_MAX_MEETINGS,
        w=PROFESSOR_ROLLING_VISIT_CAP,
        temporal_weights=professor_temporal_weights(horizon),
        gamma=PROFESSOR_GAMMA,
        repeat_count_offset=1,
        p=1,
        instance_id=professor_instance_label(
            party,
            int(rewards.size),
            horizon,
            instance_mode=instance_mode,
        ),
    )
    return instance, location_ids


def reward_profiles_from_data(
    paths: Iterable[str | Path],
    *,
    cities_parameter: int = PROFESSOR_SMALLEST_CITIES_PARAMETER,
    instance_mode: str = "supplied-code",
    real_location_count: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Load aligned real reward profiles for calibrating synthetic training."""

    profiles = tuple(
        load_professor_arrays(
            path,
            cities_parameter=cities_parameter,
            instance_mode=instance_mode,
            real_location_count=real_location_count,
        )[0]
        for path in paths
    )
    if not profiles:
        raise ValueError("At least one data path is required.")
    expected = profiles[0].shape
    if any(profile.shape != expected for profile in profiles):
        raise ValueError("All calibration profiles must have the same shape.")
    return profiles


def reward_range_from_xls(
    paths: Iterable[str | Path],
    *,
    cities_parameter: int = PROFESSOR_SMALLEST_CITIES_PARAMETER,
    instance_mode: str = "supplied-code",
    real_location_count: int | None = None,
    margin_fraction: float = 0.10,
) -> tuple[float, float]:
    """Backward-compatible reward-range estimator for XLS or CSV data."""

    profiles = reward_profiles_from_data(
        paths,
        cities_parameter=cities_parameter,
        instance_mode=instance_mode,
        real_location_count=real_location_count,
    )
    values = np.concatenate(profiles)
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, max(abs(low), abs(high), 1.0) * 0.05)
    margin = margin_fraction * span
    return max(0.0, low - margin), high + margin


def generate_professor_matched_instance(
    *,
    n: int,
    horizon: int,
    seed: int,
    reward_range: tuple[float, float] | None = None,
    reward_profiles: Sequence[ArrayLike] | None = None,
    jitter_fraction: float = 0.10,
    scale_range: tuple[float, float] = (0.85, 1.15),
    instance_id: str | None = None,
) -> CIPPInstance:
    """Generate one domain-matched synthetic training instance.

    When empirical profiles are supplied, the generator blends two profiles,
    applies global scaling and per-location jitter, then randomly permutes the
    locations.  This preserves realistic reward geometry without replaying the
    exact ordered test instance.  ``reward_range`` remains available as a
    fallback for older callers.
    """

    if n < 1 or horizon < PROFESSOR_WINDOW_LENGTH:
        raise ValueError("n must be positive and horizon must be at least 7.")
    if jitter_fraction < 0 or not np.isfinite(jitter_fraction):
        raise ValueError("jitter_fraction must be finite and nonnegative.")
    scale_low, scale_high = map(float, scale_range)
    if not (0 < scale_low <= scale_high and np.isfinite(scale_high)):
        raise ValueError("scale_range must contain positive finite bounds.")

    rng = np.random.default_rng(seed)

    if reward_profiles:
        profiles = [np.asarray(profile, dtype=np.float64) for profile in reward_profiles]
        if any(profile.shape != (n,) for profile in profiles):
            raise ValueError(f"Every reward profile must have shape ({n},).")
        if any(np.any(profile < 0) or not np.all(np.isfinite(profile)) for profile in profiles):
            raise ValueError("Reward profiles must be finite and nonnegative.")

        left = profiles[int(rng.integers(0, len(profiles)))]
        right = profiles[int(rng.integers(0, len(profiles)))]
        blend = float(rng.uniform(0.0, 1.0))
        rewards = blend * left + (1.0 - blend) * right
        rewards = rewards * float(rng.uniform(scale_low, scale_high))
        if jitter_fraction > 0:
            rewards = rewards * np.clip(
                1.0 + rng.normal(0.0, jitter_fraction, size=n),
                0.25,
                2.0,
            )
        rewards = np.maximum(rewards, 0.0)
    else:
        if reward_range is None:
            raise ValueError("Provide reward_profiles or reward_range.")
        low, high = map(float, reward_range)
        if not (np.isfinite(low) and np.isfinite(high) and 0 <= low < high):
            raise ValueError("reward_range must satisfy 0 <= low < high.")
        latent = rng.beta(1.5, 1.5, size=n)
        rewards = low + (high - low) * latent
        rewards += rng.normal(0.0, 0.025 * (high - low), size=n)
        rewards = np.clip(rewards, low, high)

    rewards = rewards[rng.permutation(n)]

    return CIPPInstance(
        n=n,
        H=horizon,
        rewards=rewards.astype(np.float64),
        costs=np.zeros(n, dtype=np.float64),
        budget=0.0,
        alpha=PROFESSOR_WINDOW_LENGTH,
        idle_requirements=professor_idle_requirements(horizon),
        q=PROFESSOR_MAX_MEETINGS,
        w=PROFESSOR_ROLLING_VISIT_CAP,
        temporal_weights=professor_temporal_weights(horizon),
        gamma=PROFESSOR_GAMMA,
        repeat_count_offset=1,
        p=1,
        instance_id=instance_id or f"professor-matched-seed-{seed}",
    )
