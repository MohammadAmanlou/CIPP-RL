"""Adaptive Gurobi references for fixed-prefix and two-stage replanning."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Sequence

from src.adaptive.context import (
    ShockScenario,
    evaluate_adaptive_itinerary,
)
from src.core import CIPPInstance


def solve_adaptive_gurobi(
    instance: CIPPInstance,
    scenario: ShockScenario,
    prefix: Sequence[int],
    *,
    time_limit_seconds: float = 3600.0,
    mip_gap: float = 0.0,
    threads: int | None = None,
    verbose: bool = False,
    output_directory: str | Path | None = None,
    model_label: str = "adaptive_reoptimization",
) -> dict[str, object]:
    """Re-optimize the remaining campaign while fixing the realized prefix.

    The model is still a full-horizon CIPP MILP. Decisions in ``prefix`` are
    fixed exactly, so budget, rolling visit caps, idle-window constraints,
    visit counts, and repeat effects all carry correctly from the realized
    past into the future.

    The shock is non-clairvoyant: pre-shock periods have multiplier 1 and only
    periods at/after ``scenario.switch_day`` use the new reward context.
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
            "prefix length must equal scenario.switch_day; "
            f"got {len(prefix)} vs {scenario.switch_day}."
        )

    output_path = None if output_directory is None else Path(output_directory)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = gp.Model(
        f"Adaptive_CIPP_{instance.instance_id}_{scenario.scenario_id}_{model_label}"
    )
    model.Params.OutputFlag = int(verbose)
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    if output_path is not None:
        model.Params.LogFile = str(output_path / f"{model_label}.gurobi.log")

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

    # Immutable realized history: past actions cannot be revised after the shock.
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

    itinerary: list[int] = []
    for t in periods:
        selected = [i + 1 for i in locations if z[i, t].X > 0.5]
        itinerary.append(selected[0] if selected else 0)

    evaluation = evaluate_adaptive_itinerary(instance, scenario, itinerary)
    gap = 100.0 * float(model.MIPGap)

    if output_path is not None:
        model.write(str(output_path / f"{model_label}.lp"))
        model.write(str(output_path / f"{model_label}.sol"))

    return {
        "method": "adaptive_gurobi",
        "objective": float(evaluation["objective"]),
        "itinerary": itinerary,
        "runtime_seconds": runtime,
        "best_bound": float(model.ObjBound),
        "optimality_gap_percent": gap,
        "proven_optimal": bool(model.Status == GRB.OPTIMAL),
        "status": int(model.Status),
        "prefix": list(prefix),
        "switch_day": int(scenario.switch_day),
    }


def solve_two_stage_adaptive_gurobi(
    instance: CIPPInstance,
    scenario: ShockScenario,
    *,
    initial_time_limit_seconds: float = 3600.0,
    reoptimization_time_limit_seconds: float = 3600.0,
    mip_gap: float = 0.0,
    threads: int | None = None,
    verbose: bool = False,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Plan statically, execute to the shock, then re-optimize the future.

    Stage 1 (time 0):
        Solve the original static CIPP over the full horizon, with no knowledge
        of the future shock.

    Execution:
        Execute exactly the first ``scenario.switch_day`` actions of that
        initial Gurobi plan.

    Stage 2 (shock time):
        Observe the new reward context, fix all executed actions, and solve the
        adaptive full-horizon MILP again. Only future actions are free.

    This is the end-to-end Gurobi replanning baseline described by the user.
    """

    from src.optimization import solve_cipp_gurobi

    scenario.validate(instance)
    output_path = None if output_directory is None else Path(output_directory)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    initial_output = None if output_path is None else output_path / "initial_static"
    reopt_output = None if output_path is None else output_path / "post_shock_reoptimization"

    initial = solve_cipp_gurobi(
        instance,
        time_limit_seconds=initial_time_limit_seconds,
        mip_gap=mip_gap,
        threads=threads,
        output_directory=initial_output,
        verbose=verbose,
    )

    initial_itinerary = tuple(int(a) for a in initial.itinerary)
    prefix = initial_itinerary[: scenario.switch_day]

    # What would happen if the original static plan were NOT revised after shock?
    no_replan = evaluate_adaptive_itinerary(
        instance,
        scenario,
        initial_itinerary,
    )

    reoptimized = solve_adaptive_gurobi(
        instance,
        scenario,
        prefix,
        time_limit_seconds=reoptimization_time_limit_seconds,
        mip_gap=mip_gap,
        threads=threads,
        verbose=verbose,
        output_directory=reopt_output,
        model_label="post_shock_reoptimization",
    )
    reoptimized["method"] = "gurobi_two_stage_replanned"

    initial_future = initial_itinerary[scenario.switch_day :]
    replanned_future = tuple(
        int(a) for a in reoptimized["itinerary"][scenario.switch_day :]
    )
    future_changes = sum(
        int(a != b)
        for a, b in zip(initial_future, replanned_future)
    )
    future_length = max(len(initial_future), 1)

    no_replan_objective = float(no_replan["objective"])
    final_objective = float(reoptimized["objective"])
    gain_absolute = final_objective - no_replan_objective
    gain_percent = (
        100.0 * gain_absolute / abs(no_replan_objective)
        if no_replan_objective != 0.0
        else 0.0
    )

    return {
        "method": "gurobi_two_stage_replanned",
        "switch_day": int(scenario.switch_day),
        "shock_occurs_between_periods": [
            int(scenario.switch_day),
            int(scenario.switch_day + 1),
        ],
        "initial_static_solution": initial.to_dict(),
        "executed_prefix": list(prefix),
        "static_plan_adaptive_objective_without_replanning": no_replan_objective,
        "reoptimized_solution": reoptimized,
        "final_objective": final_objective,
        "initial_planning_runtime_seconds": float(initial.runtime_seconds),
        "reoptimization_runtime_seconds": float(reoptimized["runtime_seconds"]),
        "total_planning_runtime_seconds": float(
            initial.runtime_seconds + float(reoptimized["runtime_seconds"])
        ),
        "reoptimization_gain_absolute": float(gain_absolute),
        "reoptimization_gain_percent": float(gain_percent),
        "future_actions_changed": int(future_changes),
        "future_periods": int(len(initial_future)),
        "future_change_fraction": float(future_changes / future_length),
        "initial_plan_proven_optimal": bool(initial.proven_optimal),
        "initial_plan_gap_percent": float(initial.optimality_gap_percent),
        "reoptimization_proven_optimal": bool(reoptimized["proven_optimal"]),
        "reoptimization_gap_percent": float(
            reoptimized["optimality_gap_percent"]
        ),
    }

def solve_clairvoyant_adaptive_gurobi(
    instance: CIPPInstance,
    scenario: ShockScenario,
    *,
    time_limit_seconds: float = 3600.0,
    mip_gap: float = 0.0,
    threads: int | None = None,
    verbose: bool = False,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Perfect-information Gurobi upper bound.

    This solver knows the complete future shock at day 0 and optimizes all
    periods jointly under the realized time-varying reward coefficients.

    It is NOT an operational competitor for Method C or two-stage Gurobi.
    It is a clairvoyant upper bound: a feasible non-clairvoyant online policy
    should not be expected to exceed it under the same objective/constraints.
    """

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "gurobipy is required for the clairvoyant Gurobi upper bound."
        ) from exc

    scenario.validate(instance)
    output_path = None if output_directory is None else Path(output_directory)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = gp.Model(
        f"Clairvoyant_Adaptive_CIPP_{instance.instance_id}_{scenario.scenario_id}"
    )
    model.Params.OutputFlag = int(verbose)
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    if output_path is not None:
        model.Params.LogFile = str(output_path / "clairvoyant.gurobi.log")

    locations = range(instance.n)
    periods = range(instance.H)
    visit_counts = range(instance.q + 1)
    multipliers = scenario.multiplier_matrix(instance)

    z = model.addVars(locations, periods, vtype=GRB.BINARY, name="Z")
    s = model.addVars(locations, visit_counts, vtype=GRB.BINARY, name="S")
    y = model.addVars(
        locations, periods, visit_counts, vtype=GRB.BINARY, name="Y"
    )
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
            for i in locations
            for t in periods
            for c in visit_counts
        ),
        name="YLeqS",
    )
    model.addConstrs(
        (
            y[i, t, c] <= z[i, t]
            for i in locations
            for t in periods
            for c in visit_counts
        ),
        name="YLeqZ",
    )
    model.addConstrs(
        (
            y[i, t, c] >= s[i, c] + z[i, t] - 1
            for i in locations
            for t in periods
            for c in visit_counts
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
            for i in locations
            for t in periods
        )
        <= instance.budget,
        name="Budget",
    )

    model.optimize()
    runtime = float(time.perf_counter() - started)

    if model.SolCount < 1:
        raise RuntimeError(
            "Clairvoyant adaptive Gurobi returned no feasible solution; "
            f"status={model.Status}"
        )

    itinerary: list[int] = []
    for t in periods:
        selected = [
            i + 1
            for i in locations
            if z[i, t].X > 0.5
        ]
        itinerary.append(selected[0] if selected else 0)

    evaluation = evaluate_adaptive_itinerary(
        instance,
        scenario,
        itinerary,
    )

    if output_path is not None:
        model.write(str(output_path / "clairvoyant.lp"))
        model.write(str(output_path / "clairvoyant.sol"))

    return {
        "method": "gurobi_clairvoyant_upper_bound",
        "objective": float(evaluation["objective"]),
        "itinerary": itinerary,
        "runtime_seconds": runtime,
        "best_bound": float(model.ObjBound),
        "optimality_gap_percent": 100.0 * float(model.MIPGap),
        "proven_optimal": bool(model.Status == GRB.OPTIMAL),
        "status": int(model.Status),
        "information_pattern": "perfect_future_shock_information_at_day_0",
    }

def solve_stochastic_prefix_gurobi(
    instance: CIPPInstance,
    scenarios: Sequence[ShockScenario],
    *,
    scenario_probabilities: Sequence[float] | None = None,
    time_limit_seconds: float = 900.0,
    mip_gap: float = 0.0,
    threads: int | None = None,
    verbose: bool = False,
    output_directory: str | Path | None = None,
    model_label: str = "stochastic_nonanticipative_prefix",
) -> dict[str, object]:
    """Scenario-based stochastic MILP with a shared non-clairvoyant prefix.

    Every supplied scenario must have the same first shock day. Decisions before
    that day are linked by non-anticipativity constraints, while each scenario
    has its own post-shock recourse. The objective is expected adaptive reward.

    This is the strong optimization benchmark missing from a purely myopic
    Gurobi-vs-RL comparison: unlike static day-0 Gurobi, it is given the shock
    distribution, just as the anticipatory RL policy is.
    """

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "gurobipy is required for the stochastic Gurobi benchmark."
        ) from exc

    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("at least one stochastic scenario is required")
    for scenario in scenarios:
        scenario.validate(instance)
    switch_day = int(scenarios[0].switch_day)
    if any(int(s.switch_day) != switch_day for s in scenarios):
        raise ValueError(
            "all stochastic scenarios must share the same switch_day"
        )

    if scenario_probabilities is None:
        probabilities = [1.0 / len(scenarios)] * len(scenarios)
    else:
        probabilities = [float(x) for x in scenario_probabilities]
        if len(probabilities) != len(scenarios):
            raise ValueError(
                "scenario_probabilities length must match scenarios"
            )
        if any(x < 0.0 for x in probabilities):
            raise ValueError("scenario probabilities must be non-negative")
        total = sum(probabilities)
        if total <= 0.0:
            raise ValueError("scenario probabilities must sum to > 0")
        probabilities = [x / total for x in probabilities]

    output_path = None if output_directory is None else Path(output_directory)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = gp.Model(
        f"Stochastic_CIPP_{instance.instance_id}_T{switch_day}"
    )
    model.Params.OutputFlag = int(verbose)
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    if output_path is not None:
        model.Params.LogFile = str(output_path / f"{model_label}.gurobi.log")

    S = range(len(scenarios))
    locations = range(instance.n)
    periods = range(instance.H)
    visit_counts = range(instance.q + 1)
    matrices = [scenario.multiplier_matrix(instance) for scenario in scenarios]

    z = model.addVars(S, locations, periods, vtype=GRB.BINARY, name="Z")
    s_var = model.addVars(
        S, locations, visit_counts, vtype=GRB.BINARY, name="S"
    )
    y = model.addVars(
        S, locations, periods, visit_counts, vtype=GRB.BINARY, name="Y"
    )
    idle = model.addVars(S, periods, vtype=GRB.BINARY, name="V")

    model.setObjective(
        gp.quicksum(
            probabilities[omega]
            * float(instance.rewards[i])
            * float(matrices[omega][t, i])
            * float(instance.temporal_weights[t])
            * instance.repeat_factor(c)
            * y[omega, i, t, c]
            for omega in S
            for i in locations
            for t in periods
            for c in visit_counts
            if c >= 1
        ),
        GRB.MAXIMIZE,
    )

    for omega in S:
        model.addConstrs(
            (
                gp.quicksum(z[omega, i, t] for i in locations)
                <= 1 - idle[omega, t]
                for t in periods
            ),
            name=f"DailyIdleCoupling_{omega}",
        )
        model.addConstrs(
            (
                gp.quicksum(
                    idle[omega, t]
                    for t in range(start, start + instance.alpha)
                )
                >= int(instance.idle_requirements[start])
                for start in range(instance.num_rolling_windows)
            ),
            name=f"IdleWindow_{omega}",
        )
        model.addConstrs(
            (
                gp.quicksum(
                    s_var[omega, i, c] for c in visit_counts
                )
                == 1
                for i in locations
            ),
            name=f"OneVisitCount_{omega}",
        )
        model.addConstrs(
            (
                gp.quicksum(z[omega, i, t] for t in periods)
                == gp.quicksum(
                    c * s_var[omega, i, c] for c in visit_counts
                )
                for i in locations
            ),
            name=f"VisitCountLink_{omega}",
        )
        model.addConstrs(
            (
                y[omega, i, t, c] <= s_var[omega, i, c]
                for i in locations
                for t in periods
                for c in visit_counts
            ),
            name=f"YLeqS_{omega}",
        )
        model.addConstrs(
            (
                y[omega, i, t, c] <= z[omega, i, t]
                for i in locations
                for t in periods
                for c in visit_counts
            ),
            name=f"YLeqZ_{omega}",
        )
        model.addConstrs(
            (
                y[omega, i, t, c]
                >= s_var[omega, i, c] + z[omega, i, t] - 1
                for i in locations
                for t in periods
                for c in visit_counts
            ),
            name=f"YGeqSZ_{omega}",
        )
        model.addConstrs(
            (
                gp.quicksum(z[omega, i, t] for t in periods)
                <= instance.q
                for i in locations
            ),
            name=f"TotalVisitCap_{omega}",
        )
        model.addConstrs(
            (
                gp.quicksum(
                    z[omega, i, t]
                    for t in range(start, start + instance.alpha)
                )
                <= instance.w
                for i in locations
                for start in range(instance.num_rolling_windows)
            ),
            name=f"RollingVisitCap_{omega}",
        )
        model.addConstr(
            gp.quicksum(
                float(instance.costs[i]) * z[omega, i, t]
                for i in locations
                for t in periods
            )
            <= instance.budget,
            name=f"Budget_{omega}",
        )

    # Shared pre-shock action history: the optimizer knows only the scenario
    # distribution, not which future realization will occur.
    for omega in range(1, len(scenarios)):
        for t in range(switch_day):
            model.addConstr(
                idle[omega, t] == idle[0, t],
                name=f"NonAntIdle_{omega}_{t}",
            )
            for i in locations:
                model.addConstr(
                    z[omega, i, t] == z[0, i, t],
                    name=f"NonAntZ_{omega}_{i}_{t}",
                )

    model.optimize()
    runtime = float(time.perf_counter() - started)
    if model.SolCount < 1:
        raise RuntimeError(
            f"Stochastic Gurobi returned no feasible solution; "
            f"status={model.Status}"
        )

    prefix = []
    for t in range(switch_day):
        selected = [
            i + 1 for i in locations if z[0, i, t].X > 0.5
        ]
        prefix.append(selected[0] if selected else 0)

    scenario_itineraries = []
    scenario_objectives = []
    for omega, scenario in enumerate(scenarios):
        itinerary = []
        for t in periods:
            selected = [
                i + 1
                for i in locations
                if z[omega, i, t].X > 0.5
            ]
            itinerary.append(selected[0] if selected else 0)
        scenario_itineraries.append(itinerary)
        scenario_objectives.append(
            float(
                evaluate_adaptive_itinerary(
                    instance,
                    scenario,
                    itinerary,
                )["objective"]
            )
        )

    expected_eval = float(
        sum(p * v for p, v in zip(probabilities, scenario_objectives))
    )

    if output_path is not None:
        model.write(str(output_path / f"{model_label}.lp"))
        model.write(str(output_path / f"{model_label}.sol"))

    return {
        "method": "stochastic_gurobi_nonanticipative_prefix",
        "switch_day": switch_day,
        "scenario_count": len(scenarios),
        "scenario_probabilities": probabilities,
        "prefix": prefix,
        "expected_objective": expected_eval,
        "model_objective": float(model.ObjVal),
        "best_bound": float(model.ObjBound),
        "optimality_gap_percent": 100.0 * float(model.MIPGap),
        "proven_optimal": bool(model.Status == GRB.OPTIMAL),
        "status": int(model.Status),
        "runtime_seconds": runtime,
        "scenario_objectives": scenario_objectives,
        "scenario_itineraries": scenario_itineraries,
    }

