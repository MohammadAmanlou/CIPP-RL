"""Professor benchmark loading and leakage-free paper-like generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re
from typing import Literal

import numpy as np
try:
    import xlrd  # type: ignore
except ModuleNotFoundError:  # CSV fallback supports budget-disabled experiments.
    xlrd = None

from src.core import CIPPInstance


ObjectiveVariant = Literal[
    "paper_equation",
    "professor_code",
]

BudgetMode = Literal[
    "auto",
    "disabled",
    "paper",
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
    """One Excel-backed benchmark plus publication metadata.

    Published identifiers use ``S`` for the number of visitable states.  Idle is
    represented internally as action 0 and is not counted in ``state_count``.
    Therefore ``D_14S_30P`` has 14 visit locations and 15 actions in total.
    """

    instance: CIPPInstance
    location_names: tuple[str, ...]
    party: str
    objective_variant: str
    table5_reference: Table5Reference | None
    state_count: int
    visit_location_count: int
    action_count: int
    idle_action: int = 0

    @property
    def total_state_count(self) -> int:
        """Backward-compatible alias for the published visit-state count."""

        return self.state_count


@dataclass(frozen=True, slots=True)
class ProfessorInstanceSpec:
    """Dimensions encoded by one published professor-instance identifier.

    In identifiers such as ``D_14S_30P``, ``14S`` means 14 visitable states.
    Idle/Rest is a separate internal action with index 0, so the action space has
    15 actions in total and the horizon contains 30 decision periods.
    """

    instance_id: str
    party: str
    state_count: int
    visit_location_count: int
    action_count: int
    horizon: int

    @property
    def total_state_count(self) -> int:
        """Backward-compatible alias for the published visit-state count."""

        return self.state_count


_PROFESSOR_INSTANCE_PATTERN = re.compile(
    r"^(?P<party>[DR])_(?P<locations>[1-9][0-9]*)S_(?P<horizon>[1-9][0-9]*)P$",
    re.IGNORECASE,
)


def parse_professor_instance_id(instance_id: str) -> ProfessorInstanceSpec:
    """Parse an instance ID so the instance, not CLI dimension flags, owns n/H."""

    if not isinstance(instance_id, str):
        raise TypeError("instance_id must be a string.")

    normalized = instance_id.strip().upper()
    match = _PROFESSOR_INSTANCE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            "Professor instance IDs must have the form D_14S_30P or "
            "R_51S_90P, where S is the number of visitable states, Idle is "
            "an additional internal action, and P is the number of periods."
        )

    state_count = int(match.group("locations"))
    horizon = int(match.group("horizon"))
    if state_count < 1:
        raise ValueError("A professor benchmark must contain at least one visitable state.")
    if horizon < 7:
        raise ValueError("Professor benchmark horizons must be at least 7 periods.")

    return ProfessorInstanceSpec(
        instance_id=normalized,
        party=match.group("party").upper(),
        state_count=state_count,
        visit_location_count=state_count,
        action_count=state_count + 1,
        horizon=horizon,
    )


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


def _processed_profile_candidates(excel_path: Path, party: str) -> tuple[Path, ...]:
    filename = f"CIPP-{party}.csv"
    return (
        excel_path.parent / "processed" / filename,
        excel_path.parent / "data" / "processed" / filename,
        Path("data") / "processed" / filename,
    )


def _read_processed_profile(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rewards: list[float] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"location_id", "reward"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain location_id and reward columns")
        for row in reader:
            names.append(str(row["location_id"]).strip())
            rewards.append(float(row["reward"]))
    return names, np.asarray(rewards, dtype=np.float64)


def load_professor_benchmark(
    excel_path: str | Path,
    *,
    instance_id: str | None = None,
    party: str | None = None,
    number_of_locations: int | None = None,
    horizon: int | None = None,
    number_of_states: int | None = None,
    objective_variant: ObjectiveVariant = "paper_equation",
    budget_mode: BudgetMode = "auto",
) -> ProfessorBenchmark:
    """Load one professor benchmark with one explicit internal Idle action.

    ``D_14S_30P`` means 14 visitable states and 30 periods.  Source row zero is
    the zero-reward, zero-cost Idle/Rest row.  It is validated and removed from
    the location arrays because ``CIPPInstance`` represents the same Idle choice
    internally as action 0.  Consequently, the action space has 15 actions.
    """

    if number_of_states is not None and int(number_of_states) < 1:
        raise ValueError("number_of_states is the visit-state count and must be positive")
    if number_of_locations is not None and int(number_of_locations) < 1:
        raise ValueError("number_of_locations must be positive")
    if number_of_states is not None and number_of_locations is not None:
        if int(number_of_states) != int(number_of_locations):
            raise ValueError(
                "number_of_states and number_of_locations both exclude Idle and must match"
            )

    if instance_id is not None:
        spec = parse_professor_instance_id(instance_id)
        if party is not None and party.strip().upper() != spec.party:
            raise ValueError("party disagrees with the supplied instance_id")
        if number_of_states is not None and int(number_of_states) != spec.state_count:
            raise ValueError("number_of_states disagrees with the supplied instance_id")
        if (
            number_of_locations is not None
            and int(number_of_locations) != spec.visit_location_count
        ):
            raise ValueError("number_of_locations disagrees with the supplied instance_id")
        if horizon is not None and int(horizon) != spec.horizon:
            raise ValueError("horizon disagrees with the supplied instance_id")
    else:
        if party is None or horizon is None:
            raise ValueError(
                "Pass instance_id, or pass party, horizon, and either number_of_states "
                "or number_of_locations; both counts exclude Idle"
            )
        if number_of_states is None and number_of_locations is None:
            raise ValueError("A total state count or visit-location count is required")
        state_count = (
            int(number_of_states)
            if number_of_states is not None
            else int(number_of_locations)
        )
        spec = parse_professor_instance_id(
            f"{party.strip().upper()}_{state_count}S_{int(horizon)}P"
        )

    normalized_party = spec.party
    state_count = spec.state_count
    visit_location_count = spec.visit_location_count
    action_count = spec.action_count
    horizon = spec.horizon
    if budget_mode not in {"auto", "disabled", "paper"}:
        raise ValueError("budget_mode must be auto, disabled, or paper")

    excel_path = Path(excel_path)
    budget_disabled = budget_mode == "disabled" or (
        budget_mode == "auto" and objective_variant == "professor_code"
    )

    rewards: np.ndarray
    location_names: tuple[str, ...]
    costs: np.ndarray
    repeat_offset, budget = _variant_parameters(objective_variant, horizon=horizon)

    if xlrd is not None and excel_path.exists():
        workbook = xlrd.open_workbook(str(excel_path))
        reward_sheet = workbook.sheet_by_name("Rewards")
        name_sheet = workbook.sheet_by_name("ID")
        accommodation_sheet = workbook.sheet_by_name("Accomodation")
        activity_sheet = workbook.sheet_by_name("Holding Activity")

        idle_name = str(name_sheet.cell_value(0, 0)).strip().lower()
        idle_reward = float(reward_sheet.cell_value(0, 0))
        idle_cost = float(accommodation_sheet.cell_value(0, 0)) + float(
            activity_sheet.cell_value(0, 0)
        )
        if idle_name not in {"rest", "idle"}:
            raise ValueError(f"Source row zero must be Idle/Rest, got {idle_name!r}")
        if abs(idle_reward) > 1e-12 or abs(idle_cost) > 1e-12:
            raise ValueError("Idle/Rest source row must have zero reward and zero cost")

        available_locations = min(reward_sheet.nrows, name_sheet.nrows) - 1
        if visit_location_count > available_locations:
            raise ValueError(
                f"Instance {spec.instance_id} requires {visit_location_count} visit "
                f"locations but {excel_path} contains {available_locations}"
            )
        row_indices = range(1, state_count + 1)
        rewards = np.asarray(
            [reward_sheet.cell_value(row, 0) for row in row_indices],
            dtype=np.float64,
        )
        location_names = tuple(
            str(name_sheet.cell_value(row, 0)).strip() for row in row_indices
        )
        if budget_disabled:
            costs = np.zeros(visit_location_count, dtype=np.float64)
            budget = 0.0
        else:
            available_cost_locations = min(
                accommodation_sheet.nrows, activity_sheet.nrows
            ) - 1
            if visit_location_count > available_cost_locations:
                raise ValueError("Workbook does not contain enough cost rows")
            costs = np.asarray(
                [
                    accommodation_sheet.cell_value(row, 0)
                    + activity_sheet.cell_value(row, 0)
                    for row in row_indices
                ],
                dtype=np.float64,
            )
            if budget_mode == "paper":
                _, budget = _variant_parameters("paper_equation", horizon=horizon)
    else:
        if not budget_disabled:
            raise ModuleNotFoundError(
                "xlrd is required for budget-enabled .xls loading; install requirements.txt"
            )
        csv_path = next(
            (candidate for candidate in _processed_profile_candidates(excel_path, normalized_party) if candidate.exists()),
            None,
        )
        if csv_path is None:
            raise FileNotFoundError(
                f"Could not read {excel_path}; xlrd is unavailable and no processed CSV was found"
            )
        all_names, all_rewards = _read_processed_profile(csv_path)
        if not all_names or all_names[0].strip().lower() not in {"rest", "idle"}:
            raise ValueError("Processed source row zero must be Idle/Rest")
        if abs(float(all_rewards[0])) > 1e-12:
            raise ValueError("Processed Idle/Rest row must have zero reward")
        if len(all_names) < action_count:
            raise ValueError(
                f"{csv_path} contains {len(all_names) - 1} visit states but "
                f"{state_count} are required"
            )
        rewards = all_rewards[1 : state_count + 1].astype(np.float64, copy=True)
        location_names = tuple(all_names[1 : state_count + 1])
        costs = np.zeros(visit_location_count, dtype=np.float64)
        budget = 0.0

    instance = CIPPInstance(
        n=visit_location_count,
        H=horizon,
        rewards=rewards,
        costs=costs,
        budget=budget,
        alpha=7,
        idle_requirements=paper_idle_requirements(horizon, alpha=7),
        q=12,
        w=2,
        temporal_weights=paper_temporal_weights(horizon),
        gamma=0.04,
        p=1,
        instance_id=spec.instance_id,
        repeat_penalty_offset=repeat_offset,
    )
    if instance.num_actions != action_count:
        raise RuntimeError("Internal action count must equal visit-state count plus Idle")

    return ProfessorBenchmark(
        instance=instance,
        location_names=location_names,
        party=normalized_party,
        objective_variant=objective_variant,
        table5_reference=TABLE5_REFERENCES.get(spec.instance_id),
        state_count=state_count,
        visit_location_count=visit_location_count,
        action_count=instance.num_actions,
    )


def benchmark_from_instance(
    instance: CIPPInstance,
    *,
    party: str = "CUSTOM",
    location_names: tuple[str, ...] | None = None,
) -> ProfessorBenchmark:
    """Wrap an arbitrary JSON-backed instance without overriding its n or H."""

    names = location_names or tuple(
        f"location_{index}" for index in range(1, instance.n + 1)
    )
    if len(names) != instance.n:
        raise ValueError("location_names must contain exactly instance.n entries.")

    return ProfessorBenchmark(
        instance=instance,
        location_names=names,
        party=party.strip().upper() or "CUSTOM",
        objective_variant="instance_defined",
        table5_reference=TABLE5_REFERENCES.get(instance.instance_id),
        state_count=instance.n,
        visit_location_count=instance.n,
        action_count=instance.num_actions,
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
