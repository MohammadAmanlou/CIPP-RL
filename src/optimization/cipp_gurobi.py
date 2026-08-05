"""Exact CIPP teacher using Gurobi, with a SciPy MILP verification fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal, Sequence

import numpy as np

from src.core import (
    CIPPInstance,
    evaluate_itinerary,
)


ExactSolver = Literal[
    "auto",
    "gurobi",
    "scipy",
]


@dataclass(frozen=True, slots=True)
class ExactCIPPSolution:
    """One exact or time-limited MILP solution."""

    instance_id: str
    solver: str
    status: str
    itinerary: tuple[int, ...]
    objective: float
    best_bound: float
    optimality_gap_percent: float
    runtime_seconds: float
    proven_optimal: bool
    feasible: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert the result to JSON-compatible values."""

        return {
            "instance_id": self.instance_id,
            "solver": self.solver,
            "status": self.status,
            "itinerary": list(
                self.itinerary
            ),
            "objective": self.objective,
            "best_bound": self.best_bound,
            "optimality_gap_percent": (
                self.optimality_gap_percent
            ),
            "runtime_seconds": (
                self.runtime_seconds
            ),
            "proven_optimal": (
                self.proven_optimal
            ),
            "feasible": self.feasible,
            "violations": list(
                self.violations
            ),
        }


def _solution_from_itinerary(
    instance: CIPPInstance,
    *,
    solver: str,
    status: str,
    itinerary: list[int],
    best_bound: float,
    optimality_gap_percent: float,
    runtime_seconds: float,
    proven_optimal: bool,
) -> ExactCIPPSolution:
    evaluation = evaluate_itinerary(
        instance,
        itinerary,
    )

    return ExactCIPPSolution(
        instance_id=instance.instance_id,
        solver=solver,
        status=status,
        itinerary=tuple(
            int(action)
            for action in itinerary
        ),
        objective=float(
            evaluation.objective
        ),
        best_bound=float(
            best_bound
        ),
        optimality_gap_percent=float(
            optimality_gap_percent
        ),
        runtime_seconds=float(
            runtime_seconds
        ),
        proven_optimal=bool(
            proven_optimal
        ),
        feasible=bool(
            evaluation.feasible
        ),
        violations=tuple(
            evaluation.violations
        ),
    )


def solve_cipp_gurobi(
    instance: CIPPInstance,
    *,
    time_limit_seconds: float = 3_600.0,
    mip_gap: float = 0.0,
    threads: int | None = None,
    output_directory: str | Path | None = None,
    verbose: bool = False,
    mip_start_itinerary: Sequence[int] | None = None,
) -> ExactCIPPSolution:
    """Solve CIPP with Gurobi and return a teacher itinerary.

    ``mip_start_itinerary`` is an optional hybrid experiment.  It never
    changes or trains the RL policy; it only supplies a feasible RL incumbent
    to Gurobi before optimization.
    """

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise RuntimeError(
            "gurobipy is not installed. Install Gurobi and "
            "activate a valid license, or use solver='scipy' "
            "for local verification."
        ) from error

    started = time.perf_counter()

    model = gp.Model(
        f"CIPP_{instance.instance_id}"
    )

    model.Params.OutputFlag = int(
        verbose
    )

    model.Params.TimeLimit = float(
        time_limit_seconds
    )

    model.Params.MIPGap = float(
        mip_gap
    )

    if threads is not None:
        model.Params.Threads = int(
            threads
        )

    output_path: Path | None = None

    if output_directory is not None:
        output_path = Path(
            output_directory
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.Params.LogFile = str(
            output_path
            / f"{instance.instance_id}.gurobi.log"
        )

    locations = range(
        instance.n
    )

    periods = range(
        instance.H
    )

    visit_counts = range(
        instance.q + 1
    )

    z = model.addVars(
        locations,
        periods,
        vtype=GRB.BINARY,
        name="Z",
    )

    s = model.addVars(
        locations,
        visit_counts,
        vtype=GRB.BINARY,
        name="S",
    )

    y = model.addVars(
        locations,
        periods,
        visit_counts,
        vtype=GRB.BINARY,
        name="Y",
    )

    idle = model.addVars(
        periods,
        vtype=GRB.BINARY,
        name="V",
    )

    model.setObjective(
        gp.quicksum(
            float(
                instance.temporal_weights[
                    period
                ]
            )
            * instance.repeat_factor(
                count
            )
            * float(
                instance.rewards[
                    location
                ]
            )
            * y[
                location,
                period,
                count,
            ]
            for location in locations
            for period in periods
            for count in visit_counts
            if count >= 1
        ),
        GRB.MAXIMIZE,
    )

    model.addConstrs(
        (
            gp.quicksum(
                z[
                    location,
                    period,
                ]
                for location in locations
            )
            <= 1 - idle[period]
            for period in periods
        ),
        name="DailyIdleCoupling",
    )

    model.addConstrs(
        (
            gp.quicksum(
                idle[period]
                for period in range(
                    window_start,
                    window_start
                    + instance.alpha,
                )
            )
            >= int(
                instance.idle_requirements[
                    window_start
                ]
            )
            for window_start in range(
                instance.num_rolling_windows
            )
        ),
        name="IdleWindow",
    )

    model.addConstrs(
        (
            gp.quicksum(
                s[
                    location,
                    count,
                ]
                for count in visit_counts
            )
            == 1
            for location in locations
        ),
        name="OneVisitCount",
    )

    model.addConstrs(
        (
            gp.quicksum(
                z[
                    location,
                    period,
                ]
                for period in periods
            )
            == gp.quicksum(
                count
                * s[
                    location,
                    count,
                ]
                for count in visit_counts
            )
            for location in locations
        ),
        name="VisitCountLink",
    )

    model.addConstrs(
        (
            y[
                location,
                period,
                count,
            ]
            <= s[
                location,
                count,
            ]
            for location in locations
            for period in periods
            for count in visit_counts
        ),
        name="YLeqS",
    )

    model.addConstrs(
        (
            y[
                location,
                period,
                count,
            ]
            <= z[
                location,
                period,
            ]
            for location in locations
            for period in periods
            for count in visit_counts
        ),
        name="YLeqZ",
    )

    model.addConstrs(
        (
            y[
                location,
                period,
                count,
            ]
            >= s[
                location,
                count,
            ]
            + z[
                location,
                period,
            ]
            - 1
            for location in locations
            for period in periods
            for count in visit_counts
        ),
        name="YGeqSZ",
    )

    model.addConstrs(
        (
            gp.quicksum(
                z[
                    location,
                    period,
                ]
                for period in periods
            )
            <= instance.q
            for location in locations
        ),
        name="TotalVisitCap",
    )

    model.addConstrs(
        (
            gp.quicksum(
                z[
                    location,
                    period,
                ]
                for period in range(
                    window_start,
                    window_start
                    + instance.alpha,
                )
            )
            <= instance.w
            for location in locations
            for window_start in range(
                instance.num_rolling_windows
            )
        ),
        name="RollingVisitCap",
    )

    model.addConstr(
        gp.quicksum(
            float(
                instance.costs[
                    location
                ]
            )
            * z[
                location,
                period,
            ]
            for location in locations
            for period in periods
        )
        <= instance.budget,
        name="Budget",
    )

    if mip_start_itinerary is not None:
        start_actions = tuple(int(action) for action in mip_start_itinerary)
        start_evaluation = evaluate_itinerary(instance, start_actions)
        if not start_evaluation.feasible:
            raise ValueError(
                "mip_start_itinerary must be a complete feasible solution: "
                + "; ".join(start_evaluation.violations)
            )
        for period in periods:
            action = start_actions[period]
            idle[period].Start = float(action == 0)
            for location in locations:
                z[location, period].Start = float(action == location + 1)
        for location in locations:
            selected_count = int(start_evaluation.visit_counts[location])
            for count in visit_counts:
                s[location, count].Start = float(count == selected_count)
                for period in periods:
                    y[location, period, count].Start = float(
                        count == selected_count
                        and start_actions[period] == location + 1
                    )

        if output_path is not None:
            model.write(str(output_path / f"{instance.instance_id}.rl_start.mst"))

    model.optimize()

    runtime_seconds = (
        time.perf_counter() - started
    )

    if model.SolCount < 1:
        raise RuntimeError(
            "Gurobi did not produce a feasible CIPP solution. "
            f"Status code: {model.Status}."
        )

    itinerary: list[int] = []

    for period in periods:
        selected = [
            location + 1
            for location in locations
            if z[
                location,
                period,
            ].X > 0.5
        ]

        itinerary.append(
            selected[0]
            if selected
            else 0
        )

    if output_path is not None:
        model.write(
            str(
                output_path
                / f"{instance.instance_id}.lp"
            )
        )

        model.write(
            str(
                output_path
                / f"{instance.instance_id}.sol"
            )
        )

    proven_optimal = (
        model.Status == GRB.OPTIMAL
    )

    status = (
        "optimal"
        if proven_optimal
        else f"gurobi_status_{model.Status}"
    )

    gap_percent = (
        100.0 * float(
            model.MIPGap
        )
    )

    return _solution_from_itinerary(
        instance,
        solver="gurobi",
        status=status,
        itinerary=itinerary,
        best_bound=float(
            model.ObjBound
        ),
        optimality_gap_percent=(
            gap_percent
        ),
        runtime_seconds=runtime_seconds,
        proven_optimal=proven_optimal,
    )


def solve_cipp_scipy(
    instance: CIPPInstance,
    *,
    time_limit_seconds: float = 3_600.0,
    mip_gap: float = 0.0,
    verbose: bool = False,
) -> ExactCIPPSolution:
    """Solve the same MILP with SciPy/HiGHS for CI and local verification."""

    from scipy.optimize import (
        Bounds,
        LinearConstraint,
        milp,
    )
    from scipy.sparse import coo_matrix

    n = instance.n
    horizon = instance.H
    counts = instance.q + 1

    z_offset = 0
    z_size = n * horizon

    s_offset = z_offset + z_size
    s_size = n * counts

    y_offset = s_offset + s_size
    y_size = n * horizon * counts

    v_offset = y_offset + y_size
    number_of_variables = (
        v_offset + horizon
    )

    def z_index(
        location: int,
        period: int,
    ) -> int:
        return (
            z_offset
            + location * horizon
            + period
        )

    def s_index(
        location: int,
        count: int,
    ) -> int:
        return (
            s_offset
            + location * counts
            + count
        )

    def y_index(
        location: int,
        period: int,
        count: int,
    ) -> int:
        return (
            y_offset
            + (
                location * horizon
                + period
            )
            * counts
            + count
        )

    def v_index(
        period: int,
    ) -> int:
        return (
            v_offset + period
        )

    objective = np.zeros(
        number_of_variables,
        dtype=np.float64,
    )

    for location in range(n):
        for period in range(
            horizon
        ):
            for count in range(
                1,
                counts,
            ):
                objective[
                    y_index(
                        location,
                        period,
                        count,
                    )
                ] = -(
                    float(
                        instance.rewards[
                            location
                        ]
                    )
                    * float(
                        instance.temporal_weights[
                            period
                        ]
                    )
                    * instance.repeat_factor(
                        count
                    )
                )

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(
        entries: list[
            tuple[int, float]
        ],
        lower: float,
        upper: float,
    ) -> None:
        row = len(
            lower_bounds
        )

        for column, coefficient in entries:
            row_indices.append(
                row
            )
            column_indices.append(
                column
            )
            coefficients.append(
                coefficient
            )

        lower_bounds.append(
            lower
        )
        upper_bounds.append(
            upper
        )

    negative_infinity = -np.inf
    positive_infinity = np.inf

    for period in range(
        horizon
    ):
        add_constraint(
            [
                *[
                    (
                        z_index(
                            location,
                            period,
                        ),
                        1.0,
                    )
                    for location in range(
                        n
                    )
                ],
                (
                    v_index(period),
                    1.0,
                ),
            ],
            negative_infinity,
            1.0,
        )

    for window_start in range(
        instance.num_rolling_windows
    ):
        add_constraint(
            [
                (
                    v_index(period),
                    1.0,
                )
                for period in range(
                    window_start,
                    window_start
                    + instance.alpha,
                )
            ],
            float(
                instance.idle_requirements[
                    window_start
                ]
            ),
            positive_infinity,
        )

    for location in range(n):
        add_constraint(
            [
                (
                    s_index(
                        location,
                        count,
                    ),
                    1.0,
                )
                for count in range(
                    counts
                )
            ],
            1.0,
            1.0,
        )

        add_constraint(
            [
                *[
                    (
                        z_index(
                            location,
                            period,
                        ),
                        1.0,
                    )
                    for period in range(
                        horizon
                    )
                ],
                *[
                    (
                        s_index(
                            location,
                            count,
                        ),
                        -float(count),
                    )
                    for count in range(
                        counts
                    )
                ],
            ],
            0.0,
            0.0,
        )

        add_constraint(
            [
                (
                    z_index(
                        location,
                        period,
                    ),
                    1.0,
                )
                for period in range(
                    horizon
                )
            ],
            negative_infinity,
            float(
                instance.q
            ),
        )

        for window_start in range(
            instance.num_rolling_windows
        ):
            add_constraint(
                [
                    (
                        z_index(
                            location,
                            period,
                        ),
                        1.0,
                    )
                    for period in range(
                        window_start,
                        window_start
                        + instance.alpha,
                    )
                ],
                negative_infinity,
                float(
                    instance.w
                ),
            )

        for period in range(
            horizon
        ):
            for count in range(
                counts
            ):
                y_column = y_index(
                    location,
                    period,
                    count,
                )

                s_column = s_index(
                    location,
                    count,
                )

                z_column = z_index(
                    location,
                    period,
                )

                add_constraint(
                    [
                        (
                            y_column,
                            1.0,
                        ),
                        (
                            s_column,
                            -1.0,
                        ),
                    ],
                    negative_infinity,
                    0.0,
                )

                add_constraint(
                    [
                        (
                            y_column,
                            1.0,
                        ),
                        (
                            z_column,
                            -1.0,
                        ),
                    ],
                    negative_infinity,
                    0.0,
                )

                add_constraint(
                    [
                        (
                            y_column,
                            -1.0,
                        ),
                        (
                            s_column,
                            1.0,
                        ),
                        (
                            z_column,
                            1.0,
                        ),
                    ],
                    negative_infinity,
                    1.0,
                )

    add_constraint(
        [
            (
                z_index(
                    location,
                    period,
                ),
                float(
                    instance.costs[
                        location
                    ]
                ),
            )
            for location in range(n)
            for period in range(
                horizon
            )
        ],
        negative_infinity,
        float(
            instance.budget
        ),
    )

    matrix = coo_matrix(
        (
            np.asarray(
                coefficients,
                dtype=np.float64,
            ),
            (
                np.asarray(
                    row_indices,
                    dtype=np.int64,
                ),
                np.asarray(
                    column_indices,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            len(lower_bounds),
            number_of_variables,
        ),
    ).tocsr()

    constraints = LinearConstraint(
        matrix,
        np.asarray(
            lower_bounds,
            dtype=np.float64,
        ),
        np.asarray(
            upper_bounds,
            dtype=np.float64,
        ),
    )

    started = time.perf_counter()

    result = milp(
        c=objective,
        integrality=np.ones(
            number_of_variables,
            dtype=np.int8,
        ),
        bounds=Bounds(
            np.zeros(
                number_of_variables,
                dtype=np.float64,
            ),
            np.ones(
                number_of_variables,
                dtype=np.float64,
            ),
        ),
        constraints=constraints,
        options={
            "disp": bool(
                verbose
            ),
            "time_limit": float(
                time_limit_seconds
            ),
            "mip_rel_gap": float(
                mip_gap
            ),
        },
    )

    runtime_seconds = (
        time.perf_counter() - started
    )

    if result.x is None:
        raise RuntimeError(
            "SciPy/HiGHS did not produce a feasible CIPP solution: "
            + str(
                result.message
            )
        )

    itinerary: list[int] = []

    for period in range(
        horizon
    ):
        selected = [
            location + 1
            for location in range(n)
            if result.x[
                z_index(
                    location,
                    period,
                )
            ] > 0.5
        ]

        itinerary.append(
            selected[0]
            if selected
            else 0
        )

    proven_optimal = (
        int(result.status) == 0
    )

    status = (
        "optimal"
        if proven_optimal
        else (
            "time_limit"
            if int(result.status) == 1
            else f"scipy_status_{result.status}"
        )
    )

    gap_fraction = float(
        getattr(
            result,
            "mip_gap",
            np.nan,
        )
    )

    dual_bound = float(
        getattr(
            result,
            "mip_dual_bound",
            -result.fun,
        )
    )

    # SciPy minimizes the negative CIPP objective.
    best_bound = -dual_bound

    return _solution_from_itinerary(
        instance,
        solver="scipy-highs",
        status=status,
        itinerary=itinerary,
        best_bound=best_bound,
        optimality_gap_percent=(
            100.0 * gap_fraction
        ),
        runtime_seconds=runtime_seconds,
        proven_optimal=proven_optimal,
    )


def solve_cipp_exact(
    instance: CIPPInstance,
    *,
    solver: ExactSolver = "auto",
    time_limit_seconds: float = 3_600.0,
    mip_gap: float = 0.0,
    threads: int | None = None,
    output_directory: str | Path | None = None,
    verbose: bool = False,
) -> ExactCIPPSolution:
    """Resolve the requested exact backend without mislabeling results."""

    if solver == "gurobi":
        return solve_cipp_gurobi(
            instance,
            time_limit_seconds=(
                time_limit_seconds
            ),
            mip_gap=mip_gap,
            threads=threads,
            output_directory=(
                output_directory
            ),
            verbose=verbose,
        )

    if solver == "scipy":
        return solve_cipp_scipy(
            instance,
            time_limit_seconds=(
                time_limit_seconds
            ),
            mip_gap=mip_gap,
            verbose=verbose,
        )

    if solver != "auto":
        raise ValueError(
            "solver must be 'auto', 'gurobi', or 'scipy'."
        )

    try:
        return solve_cipp_gurobi(
            instance,
            time_limit_seconds=(
                time_limit_seconds
            ),
            mip_gap=mip_gap,
            threads=threads,
            output_directory=(
                output_directory
            ),
            verbose=verbose,
        )
    except (
        ImportError,
        RuntimeError,
    ):
        return solve_cipp_scipy(
            instance,
            time_limit_seconds=(
                time_limit_seconds
            ),
            mip_gap=mip_gap,
            verbose=verbose,
        )
