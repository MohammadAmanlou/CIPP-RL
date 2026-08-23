"""Adaptive Gurobi residual reference with a fixed realized prefix."""

from __future__ import annotations

import time
from typing import Sequence

from src.adaptive.context import ShockScenario
from src.core import CIPPInstance
from src.adaptive.context import evaluate_adaptive_itinerary


def solve_adaptive_gurobi(
    instance: CIPPInstance,
    scenario: ShockScenario,
    prefix: Sequence[int],
    *,
    time_limit_seconds: float = 3600.0,
    mip_gap: float = 0.0,
    threads: int | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Re-optimize the remaining campaign while fixing the realized prefix.

    The objective uses time-varying reward coefficients, while all original CIPP
    constraints remain unchanged. Prefix decisions are fixed exactly.
    """

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "gurobipy is required for the adaptive Gurobi baseline."
        ) from exc

    scenario.validate(instance)
    prefix = tuple(int(a) for a in prefix)
    if len(prefix) != scenario.switch_day:
        raise ValueError(
            "For the pilot, prefix length must equal scenario.switch_day; "
            f"got {len(prefix)} vs {scenario.switch_day}."
        )

    started = time.perf_counter()
    model = gp.Model(f"Adaptive_CIPP_{instance.instance_id}_{scenario.scenario_id}")
    model.Params.OutputFlag = int(verbose)
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    locations = range(instance.n)
    periods = range(instance.H)
    visit_counts = range(instance.q + 1)
    multipliers = scenario.multiplier_matrix(instance)

    z = model.addVars(locations, periods, vtype=GRB.BINARY, name="Z")
    s = model.addVars(locations, visit_counts, vtype=GRB.BINARY, name="S")
    y = model.addVars(locations, periods, visit_counts, vtype=GRB.BINARY, name="Y")
    idle = model.addVars(periods, vtype=GRB.BINARY, name="V")

    model.setObjective(
        gp.quicksum(
            float(instance.rewards[i])
            * float(multipliers[t, i])
            * float(instance.temporal_weights[t])
            * instance.repeat_factor(c)
            * y[i, t, c]
            for i in locations
            for t in periods
            for c in visit_counts
            if c >= 1
        ),
        GRB.MAXIMIZE,
    )

    model.addConstrs(
        (
            gp.quicksum(z[i, t] for i in locations) <= 1 - idle[t]
            for t in periods
        ),
        name="DailyIdleCoupling",
    )
    model.addConstrs(
        (
            gp.quicksum(
                idle[t]
                for t in range(start, start + instance.alpha)
            )
            >= int(instance.idle_requirements[start])
            for start in range(instance.num_rolling_windows)
        ),
        name="IdleWindow",
    )
    model.addConstrs(
        (
            gp.quicksum(s[i, c] for c in visit_counts) == 1
            for i in locations
        ),
        name="OneVisitCount",
    )
    model.addConstrs(
        (
            gp.quicksum(z[i, t] for t in periods)
            == gp.quicksum(c * s[i, c] for c in visit_counts)
            for i in locations
        ),
        name="VisitCountLink",
    )
    model.addConstrs(
        (
            y[i, t, c] <= s[i, c]
            for i in locations for t in periods for c in visit_counts
        ),
        name="YLeqS",
    )
    model.addConstrs(
        (
            y[i, t, c] <= z[i, t]
            for i in locations for t in periods for c in visit_counts
        ),
        name="YLeqZ",
    )
    model.addConstrs(
        (
            y[i, t, c] >= s[i, c] + z[i, t] - 1
            for i in locations for t in periods for c in visit_counts
        ),
        name="YGeqSZ",
    )
    model.addConstrs(
        (
            gp.quicksum(z[i, t] for t in periods) <= instance.q
            for i in locations
        ),
        name="TotalVisitCap",
    )
    model.addConstrs(
        (
            gp.quicksum(
                z[i, t]
                for t in range(start, start + instance.alpha)
            )
            <= instance.w
            for i in locations
            for start in range(instance.num_rolling_windows)
        ),
        name="RollingVisitCap",
    )
    model.addConstr(
        gp.quicksum(
            float(instance.costs[i]) * z[i, t]
            for i in locations for t in periods
        )
        <= instance.budget,
        name="Budget",
    )

    # Immutable realized prefix.
    for t, action in enumerate(prefix):
        model.addConstr(idle[t] == int(action == 0), name=f"FixIdle_{t}")
        for i in locations:
            model.addConstr(
                z[i, t] == int(action == i + 1),
                name=f"FixZ_{i}_{t}",
            )

    model.optimize()

    runtime = float(time.perf_counter() - started)
    if model.SolCount < 1:
        raise RuntimeError(
            f"Adaptive Gurobi returned no feasible solution; status={model.Status}"
        )

    itinerary = []
    for t in periods:
        selected = [i + 1 for i in locations if z[i, t].X > 0.5]
        itinerary.append(selected[0] if selected else 0)

    evaluation = evaluate_adaptive_itinerary(instance, scenario, itinerary)
    gap = 100.0 * float(model.MIPGap)
    return {
        "method": "adaptive_gurobi",
        "objective": float(evaluation["objective"]),
        "itinerary": itinerary,
        "runtime_seconds": runtime,
        "best_bound": float(model.ObjBound),
        "optimality_gap_percent": gap,
        "proven_optimal": bool(model.Status == GRB.OPTIMAL),
        "status": int(model.Status),
    }
