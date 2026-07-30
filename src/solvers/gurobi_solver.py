"""Exact deterministic CIPP MILP used as the benchmark reference."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.core import CIPPInstance, evaluate_itinerary


@dataclass(frozen=True, slots=True)
class GurobiResult:
    method: str
    instance_id: str
    status: str
    has_solution: bool
    objective: float | None
    upper_bound: float | None
    mip_gap_percent: float | None
    runtime_seconds: float
    wall_runtime_seconds: float
    itinerary: tuple[int, ...] | None
    feasible: bool | None
    violations: tuple[str, ...]
    node_count: float | None
    solution_count: int
    certified_optimal: bool
    solver_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def solve_with_gurobi(
    instance: CIPPInstance,
    *,
    time_limit_seconds: float = 3600.0,
    threads: int | None = None,
    seed: int = 0,
    mip_gap: float = 0.0,
    log_path: str | Path | None = None,
    export_model_prefix: str | Path | None = None,
) -> GurobiResult:
    """Solve one canonical deterministic CIPP instance with Gurobi.

    Every heuristic and RL method is evaluated using the same
    :class:`CIPPInstance`; the extracted Gurobi itinerary is independently
    re-evaluated before it is accepted as a reference.
    """

    try:
        import gurobipy as gp  # type: ignore
        from gurobipy import GRB  # type: ignore
    except ImportError as error:  # pragma: no cover - local dependency
        raise RuntimeError(
            "Gurobi evaluation requires gurobipy and a working Gurobi license."
        ) from error

    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive.")
    if mip_gap < 0:
        raise ValueError("mip_gap must be nonnegative.")

    model = gp.Model("CIPP_canonical")
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)
    model.Params.Seed = int(seed)
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be positive.")
        model.Params.Threads = int(threads)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.Params.LogFile = str(path)

    locations = range(instance.n)
    days = range(instance.H)
    visit_counts = range(instance.q + 1)

    z = model.addVars(locations, days, vtype=GRB.BINARY, name="Z")
    x = model.addVars(
        locations, days, visit_counts, vtype=GRB.BINARY, name="X"
    )
    s_count = model.addVars(
        locations, visit_counts, vtype=GRB.BINARY, name="S"
    )
    idle = model.addVars(days, vtype=GRB.BINARY, name="V")

    model.setObjective(
        gp.quicksum(
            x[i, t, count]
            * float(instance.rewards[i])
            * float(instance.temporal_weights[t])
            * float(instance.repeat_factor(count))
            for i in locations
            for t in days
            for count in visit_counts
            if count >= 1
        ),
        GRB.MAXIMIZE,
    )

    # With p=1, every canonical day is exactly one visit or Idle.
    model.addConstrs(
        (
            gp.quicksum(z[i, t] for i in locations) + idle[t] == 1
            for t in days
        ),
        name="OneActionPerDay",
    )

    model.addConstrs(
        (
            gp.quicksum(
                idle[k] for k in range(start, start + instance.alpha)
            )
            >= int(instance.idle_requirements[start])
            for start in range(instance.num_rolling_windows)
        ),
        name="RestWindow",
    )

    model.addConstrs(
        (gp.quicksum(z[i, t] for t in days) <= instance.q for i in locations),
        name="TotalVisitCap",
    )
    model.addConstrs(
        (
            gp.quicksum(
                z[i, k] for k in range(start, start + instance.alpha)
            )
            <= instance.w
            for i in locations
            for start in range(instance.num_rolling_windows)
        ),
        name="RollingVisitCap",
    )

    model.addConstrs(
        (gp.quicksum(s_count[i, count] for count in visit_counts) == 1 for i in locations),
        name="OneCountPerLocation",
    )
    model.addConstrs(
        (
            gp.quicksum(z[i, t] for t in days)
            == gp.quicksum(
                count * s_count[i, count] for count in visit_counts
            )
            for i in locations
        ),
        name="VisitsEqualSelectedCount",
    )
    model.addConstrs(
        (
            x[i, t, count] <= s_count[i, count]
            for i in locations
            for t in days
            for count in visit_counts
        ),
        name="XLeqCount",
    )
    model.addConstrs(
        (
            x[i, t, count] <= z[i, t]
            for i in locations
            for t in days
            for count in visit_counts
        ),
        name="XLeqVisit",
    )
    model.addConstrs(
        (
            x[i, t, count] >= s_count[i, count] + z[i, t] - 1
            for i in locations
            for t in days
            for count in visit_counts
        ),
        name="XGeqCountPlusVisit",
    )

    if np.any(instance.costs > 0):
        model.addConstr(
            gp.quicksum(
                float(instance.costs[i]) * z[i, t]
                for i in locations
                for t in days
            )
            <= instance.budget,
            name="Budget",
        )

    started = time.perf_counter()
    model.optimize()
    wall_runtime = time.perf_counter() - started

    status_names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    status = status_names.get(model.Status, f"STATUS_{model.Status}")
    has_solution = model.SolCount > 0

    itinerary: tuple[int, ...] | None = None
    feasible: bool | None = None
    violations: tuple[str, ...] = ()
    objective: float | None = None
    upper_bound: float | None = None
    gap_percent: float | None = None

    if has_solution:
        actions: list[int] = []
        for t in days:
            selected = [i for i in locations if z[i, t].X > 0.5]
            if len(selected) > 1:
                raise RuntimeError("Gurobi returned more than one visit on a day.")
            actions.append(0 if not selected else selected[0] + 1)
        itinerary = tuple(actions)
        evaluation = evaluate_itinerary(instance, itinerary)
        feasible = evaluation.feasible
        violations = evaluation.violations
        objective = float(model.ObjVal)
        upper_bound = float(model.ObjBound)
        gap_percent = float(model.MIPGap * 100.0)

        if not evaluation.feasible:
            raise RuntimeError(
                "Gurobi itinerary fails the shared evaluator: "
                + "; ".join(evaluation.violations)
            )
        if not np.isclose(
            objective,
            evaluation.objective,
            rtol=1e-8,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Gurobi objective does not match the shared evaluator: "
                f"MILP={objective}, evaluator={evaluation.objective}."
            )

    if export_model_prefix is not None:
        prefix = Path(export_model_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(prefix.with_suffix(".lp")))
        if has_solution:
            model.write(str(prefix.with_suffix(".sol")))

    return GurobiResult(
        method="gurobi",
        instance_id=instance.instance_id,
        status=status,
        has_solution=has_solution,
        objective=objective,
        upper_bound=upper_bound,
        mip_gap_percent=gap_percent,
        runtime_seconds=float(model.Runtime),
        wall_runtime_seconds=float(wall_runtime),
        itinerary=itinerary,
        feasible=feasible,
        violations=violations,
        node_count=float(model.NodeCount),
        solution_count=int(model.SolCount),
        certified_optimal=status == "OPTIMAL",
        solver_seed=int(seed),
    )
