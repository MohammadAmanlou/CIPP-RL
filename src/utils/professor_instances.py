"""Professor benchmark loading and leakage-free paper-like generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xlrd

from src.core import CIPPInstance


ObjectiveVariant = Literal[
    "paper_equation",
    "professor_code",
]


@dataclass(frozen=True, slots=True)
class Table5Reference:
    """Published Gurobi result for one deterministic CIPP instance."""

    bfs: float
    optimality_gap_percent: float
    runtime_seconds: float


TABLE5_REFERENCES: dict[str, Table5Reference] = {
    "D_14S_30P": Table5Reference(15_611.6, 0.0, 15.3),
    "D_14S_45P": Table5Reference(47_854.3, 0.0, 77.8),
    "D_14S_60P": Table5Reference(103_419.1, 0.0, 89.8),
    "D_14S_75P": Table5Reference(188_144.5, 0.0, 301.4),
    "D_14S_90P": Table5Reference(295_916.9, 0.8, 3_600.0),
    "D_30S_30P": Table5Reference(23_412.1, 0.0, 63.2),
    "D_30S_45P": Table5Reference(71_210.3, 0.0, 125.1),
    "D_30S_60P": Table5Reference(153_132.5, 0.7, 3_600.0),
    "D_30S_75P": Table5Reference(271_750.1, 116.4, 3_600.0),
    "D_30S_90P": Table5Reference(430_819.3, 188.5, 3_600.0),
    "D_51S_30P": Table5Reference(35_221.0, 0.0, 1_699.6),
    "D_51S_45P": Table5Reference(107_930.6, 6.5, 3_600.0),
    "D_51S_60P": Table5Reference(233_039.8, 72.5, 3_600.0),
    "D_51S_75P": Table5Reference(320_635.2, 189.3, 3_600.0),
    "D_51S_90P": Table5Reference(537_845.6, 214.5, 3_600.0),
    "R_14S_30P": Table5Reference(14_745.4, 0.0, 9.5),
    "R_14S_45P": Table5Reference(44_936.9, 0.0, 48.2),
    "R_14S_60P": Table5Reference(96_617.1, 0.0, 183.8),
    "R_14S_75P": Table5Reference(174_877.1, 0.2, 3_600.0),
    "R_14S_90P": Table5Reference(274_054.0, 0.0, 1_311.6),
    "R_30S_30P": Table5Reference(22_831.7, 0.0, 1_532.5),
    "R_30S_45P": Table5Reference(68_970.3, 0.0, 131.6),
    "R_30S_60P": Table5Reference(147_776.9, 0.0, 3_021.1),
    "R_30S_75P": Table5Reference(263_310.9, 70.6, 3_600.0),
    "R_30S_90P": Table5Reference(414_059.7, 148.2, 3_600.0),
    "R_51S_30P": Table5Reference(34_855.0, 0.0, 1_385.8),
    "R_51S_45P": Table5Reference(106_520.7, 3.8, 3_600.0),
    "R_51S_60P": Table5Reference(229_379.0, 42.7, 3_600.0),
    "R_51S_75P": Table5Reference(344_366.0, 125.8, 3_600.0),
    "R_51S_90P": Table5Reference(458_126.8, 293.2, 3_600.0),
}


@dataclass(frozen=True, slots=True)
class ProfessorBenchmark:
    """One Excel-backed benchmark plus publication metadata."""

    instance: CIPPInstance
    location_names: tuple[str, ...]
    party: str
    objective_variant: ObjectiveVariant
    table5_reference: Table5Reference | None


def paper_temporal_weights(
    horizon: int,
) -> np.ndarray:
    """Return the U-shaped weights used by the provided Gurobi script."""

    if horizon < 1:
        raise ValueError(
            "horizon must be positive."
        )

    periods = np.arange(
        1,
        horizon + 1,
        dtype=np.float64,
    )

    midpoint = (
        horizon + 1
    ) / 2.0

    epsilon = (
        (horizon - 1) / 4.0
    ) ** 2

    return (
        (periods - midpoint) ** 2
        + epsilon
    )


def paper_idle_requirements(
    horizon: int,
    *,
    alpha: int = 7,
    betas: tuple[int, int, int, int] = (
        2,
        2,
        2,
        0,
    ),
) -> np.ndarray:
    """Return the paper's piecewise rolling-window idle requirements."""

    if not 1 <= alpha <= horizon:
        raise ValueError(
            "alpha must satisfy 1 <= alpha <= horizon."
        )

    if len(betas) != 4:
        raise ValueError(
            "betas must contain four values."
        )

    number_of_windows = (
        horizon - alpha + 1
    )

    requirements: list[int] = []

    for one_based_start in range(
        1,
        number_of_windows + 1,
    ):
        if one_based_start <= horizon / 4:
            value = betas[0]
        elif one_based_start <= horizon / 2:
            value = betas[1]
        elif one_based_start <= 3 * horizon / 4:
            value = betas[2]
        else:
            value = betas[3]

        requirements.append(
            int(value)
        )

    return np.asarray(
        requirements,
        dtype=np.int64,
    )


def _variant_parameters(
    variant: ObjectiveVariant,
    *,
    horizon: int,
) -> tuple[int, float]:
    """Return repeat offset and budget for an objective variant."""

    if variant == "paper_equation":
        return 0, 3_000_000.0 * horizon / 60.0

    if variant == "professor_code":
        # The provided July 13 script uses (j - 1) and has its budget
        # constraint commented out.  A large finite budget reproduces
        # that implementation while keeping CIPPInstance numerically valid.
        return 1, 1.0e12

    raise ValueError(
        "objective_variant must be 'paper_equation' "
        "or 'professor_code'."
    )


def load_professor_benchmark(
    excel_path: str | Path,
    *,
    party: str,
    number_of_states: int = 14,
    horizon: int = 30,
    objective_variant: ObjectiveVariant = "paper_equation",
) -> ProfessorBenchmark:
    """Load one deterministic CIPP benchmark from the professor's Excel."""

    normalized_party = (
        party.strip().upper()
    )

    if normalized_party not in {
        "D",
        "R",
    }:
        raise ValueError(
            "party must be either 'D' or 'R'."
        )

    if number_of_states not in {
        14,
        30,
        51,
    }:
        raise ValueError(
            "number_of_states must be one of 14, 30, or 51."
        )

    workbook = xlrd.open_workbook(
        str(excel_path)
    )

    reward_sheet = workbook.sheet_by_name(
        "Rewards"
    )

    name_sheet = workbook.sheet_by_name(
        "ID"
    )

    accommodation_sheet = (
        workbook.sheet_by_name(
            "Accomodation"
        )
    )

    activity_sheet = workbook.sheet_by_name(
        "Holding Activity"
    )

    row_indices = range(
        1,
        number_of_states + 1,
    )

    rewards = np.asarray(
        [
            reward_sheet.cell_value(
                row,
                0,
            )
            for row in row_indices
        ],
        dtype=np.float64,
    )

    costs = np.asarray(
        [
            accommodation_sheet.cell_value(
                row,
                0,
            )
            + activity_sheet.cell_value(
                row,
                0,
            )
            for row in row_indices
        ],
        dtype=np.float64,
    )

    location_names = tuple(
        str(
            name_sheet.cell_value(
                row,
                0,
            )
        )
        for row in row_indices
    )

    repeat_offset, budget = (
        _variant_parameters(
            objective_variant,
            horizon=horizon,
        )
    )

    instance_id = (
        f"{normalized_party}_"
        f"{number_of_states}S_"
        f"{horizon}P"
    )

    instance = CIPPInstance(
        n=number_of_states,
        H=horizon,
        rewards=rewards,
        costs=costs,
        budget=budget,
        alpha=7,
        idle_requirements=(
            paper_idle_requirements(
                horizon,
                alpha=7,
            )
        ),
        q=12,
        w=2,
        temporal_weights=(
            paper_temporal_weights(
                horizon
            )
        ),
        gamma=0.04,
        p=1,
        instance_id=instance_id,
        repeat_penalty_offset=(
            repeat_offset
        ),
    )

    return ProfessorBenchmark(
        instance=instance,
        location_names=location_names,
        party=normalized_party,
        objective_variant=objective_variant,
        table5_reference=(
            TABLE5_REFERENCES.get(
                instance_id
            )
        ),
    )


def generate_paper_like_instance(
    *,
    seed: int,
    number_of_states: int = 14,
    horizon: int = 30,
    objective_variant: ObjectiveVariant = "paper_equation",
    instance_id: str | None = None,
) -> CIPPInstance:
    """Generate a synthetic train/validation instance on the paper's scale.

    This function deliberately does not read either professor Excel file.
    Consequently, a policy trained with it can be evaluated on the Excel
    benchmark without leaking the test rewards or costs into training.
    """

    rng = np.random.default_rng(
        seed
    )

    rewards = rng.uniform(
        0.5,
        8.0,
        size=number_of_states,
    )

    costs = rng.uniform(
        20_000.0,
        100_000.0,
        size=number_of_states,
    )

    repeat_offset, budget = (
        _variant_parameters(
            objective_variant,
            horizon=horizon,
        )
    )

    return CIPPInstance(
        n=number_of_states,
        H=horizon,
        rewards=rewards,
        costs=costs,
        budget=budget,
        alpha=7,
        idle_requirements=(
            paper_idle_requirements(
                horizon,
                alpha=7,
            )
        ),
        q=12,
        w=2,
        temporal_weights=(
            paper_temporal_weights(
                horizon
            )
        ),
        gamma=0.04,
        p=1,
        instance_id=(
            instance_id
            or (
                f"paper-like-{objective_variant}-"
                f"n{number_of_states}-H{horizon}-"
                f"seed{seed}"
            )
        ),
        repeat_penalty_offset=(
            repeat_offset
        ),
    )
