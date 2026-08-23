"""Compare Method C with same-prefix and two-stage adaptive Gurobi.

This script produces two scientifically different Gurobi references:

1) same-prefix oracle:
   Uses the exact realized Attention prefix from the Method C experiment and
   finds the best feasible continuation after the shock. This isolates
   post-shock adaptation quality.

2) end-to-end two-stage Gurobi:
   At time 0, Gurobi plans the original static campaign with no shock knowledge.
   The first switch_day decisions are executed. After the shock, those decisions
   are frozen and Gurobi re-optimizes all remaining periods under the new
   reward context.

The script also evaluates the original static Gurobi plan under the shock
without replanning, so the value of Gurobi replanning itself is measurable.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json

import numpy as np

from src.adaptive import (
    ShockScenario,
    rank_flip_scenario,
    rank_flip_scenario_at_day,
    solve_adaptive_gurobi,
    solve_two_stage_adaptive_gurobi,
)
from src.utils import load_professor_benchmark, parse_professor_instance_id


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="D_14S_30P")
    p.add_argument("--data-directory", type=Path, default=Path("."))
    p.add_argument(
        "--method-c-output-directory",
        type=Path,
        default=Path("results/adaptive_method_c_full"),
        help="Root containing <instance>/seed_<seed>/ from the Method C run.",
    )
    p.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/gurobi_replanning_comparison"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--switch-fraction", type=float, default=0.50)
    p.add_argument(
        "--switch-day",
        type=int,
        default=None,
        help=(
            "Exact number of fully executed days before shock. "
            "If the saved Method C scenario exists, its switch day is used "
            "unless this value matches it exactly."
        ),
    )
    p.add_argument("--shock-boost", type=float, default=2.25)
    p.add_argument("--shock-suppress", type=float, default=0.55)
    p.add_argument("--initial-time-limit", type=float, default=3600.0)
    p.add_argument("--reopt-time-limit", type=float, default=3600.0)
    p.add_argument("--mip-gap", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--verbose-gurobi", action="store_true")
    return p.parse_args()


def _load_saved_scenario(
    path: Path,
    *,
    instance,
) -> ShockScenario | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenario = ShockScenario(
        switch_day=int(payload["switch_day_zero_based"]),
        post_multiplier=np.asarray(payload["post_multiplier"], dtype=np.float64),
        scenario_id="heldout_rank_flip_saved",
    )
    scenario.validate(instance)
    return scenario


def _load_existing_method_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    keep = []
    for row in rows:
        method = str(row.get("method", ""))
        if method.startswith("B_") or method.startswith("C_") or method.startswith("D_"):
            keep.append(row)
    return keep


def _gap_below(reference: float, value: float) -> float:
    if reference == 0.0:
        return 0.0
    return 100.0 * (reference - value) / abs(reference)


def main() -> None:
    a = _args()
    spec = parse_professor_instance_id(a.instance)
    benchmark = load_professor_benchmark(
        a.data_directory / f"CIPP-{spec.party}.xls",
        instance_id=spec.instance_id,
        objective_variant="professor_code",
        budget_mode="auto",
    )
    instance = benchmark.instance

    method_root = (
        a.method_c_output_directory
        / instance.instance_id
        / f"seed_{a.seed}"
    )
    output = (
        a.output_directory
        / instance.instance_id
        / f"seed_{a.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    saved_scenario_path = method_root / "scenario.json"
    scenario = _load_saved_scenario(saved_scenario_path, instance=instance)

    if scenario is not None:
        if a.switch_day is not None and a.switch_day != scenario.switch_day:
            raise ValueError(
                "Saved Method C scenario uses switch_day="
                f"{scenario.switch_day}, but --switch-day={a.switch_day}. "
                "For an exact Method C comparison, use the saved switch day. "
                "Run Method C eval-only separately for a new switch day."
            )
        scenario_source = "saved_method_c_scenario"
    elif a.switch_day is not None:
        scenario = rank_flip_scenario_at_day(
            instance,
            switch_day=a.switch_day,
            boost=a.shock_boost,
            suppress=a.shock_suppress,
            scenario_id="heldout_rank_flip_cli",
        )
        scenario_source = "cli_switch_day"
    else:
        scenario = rank_flip_scenario(
            instance,
            switch_fraction=a.switch_fraction,
            boost=a.shock_boost,
            suppress=a.shock_suppress,
            scenario_id="heldout_rank_flip_cli",
        )
        scenario_source = "cli_switch_fraction"

    scenario_payload = {
        "instance": instance.instance_id,
        "scenario_source": scenario_source,
        "switch_day_zero_based": int(scenario.switch_day),
        "switch_period_one_based": int(scenario.switch_day + 1),
        "meaning": (
            f"Days 1..{scenario.switch_day} are fully executed before the shock; "
            f"replanning starts for day {scenario.switch_day + 1}."
        ),
        "post_multiplier": scenario.post_multiplier.tolist(),
    }
    (output / "scenario_used.json").write_text(
        json.dumps(scenario_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # A) Same-prefix oracle: identical state/history as B/D/C.
    # --------------------------------------------------------------
    fixed_prefix_path = method_root / "fixed_prefix.json"
    same_prefix = None
    if fixed_prefix_path.exists():
        fixed_payload = json.loads(fixed_prefix_path.read_text(encoding="utf-8"))
        attention_prefix = tuple(int(x) for x in fixed_payload["prefix"])
        if len(attention_prefix) != scenario.switch_day:
            raise ValueError(
                "Method C fixed prefix length does not match the scenario: "
                f"{len(attention_prefix)} vs {scenario.switch_day}"
            )
        same_prefix = solve_adaptive_gurobi(
            instance,
            scenario,
            attention_prefix,
            time_limit_seconds=a.reopt_time_limit,
            mip_gap=a.mip_gap,
            threads=a.threads,
            verbose=a.verbose_gurobi,
            output_directory=output / "same_attention_prefix",
            model_label="same_attention_prefix",
        )
        same_prefix["method"] = "adaptive_gurobi_same_attention_prefix"
        (output / "same_attention_prefix_gurobi.json").write_text(
            json.dumps(same_prefix, indent=2) + "\n",
            encoding="utf-8",
        )

    # --------------------------------------------------------------
    # B) End-to-end Gurobi: Gurobi -> execute -> shock -> Gurobi.
    # --------------------------------------------------------------
    two_stage = solve_two_stage_adaptive_gurobi(
        instance,
        scenario,
        initial_time_limit_seconds=a.initial_time_limit,
        reoptimization_time_limit_seconds=a.reopt_time_limit,
        mip_gap=a.mip_gap,
        threads=a.threads,
        verbose=a.verbose_gurobi,
        output_directory=output / "two_stage_gurobi",
    )
    (output / "two_stage_gurobi.json").write_text(
        json.dumps(two_stage, indent=2) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Combined comparison table.
    # --------------------------------------------------------------
    existing = _load_existing_method_rows(method_root / "comparison.json")
    rows: list[dict[str, object]] = []

    same_obj = None if same_prefix is None else float(same_prefix["objective"])
    end_obj = float(two_stage["final_objective"])

    for row in existing:
        item = {
            "method": row["method"],
            "comparison_group": "same_attention_prefix",
            "prefix_source": "static_attention",
            "objective": float(row["objective"]),
            "runtime_seconds": float(row["runtime_seconds"]),
            "proven_optimal": "NA",
            "optimality_gap_percent": "NA",
            "gap_to_same_prefix_oracle_percent": (
                "NA"
                if same_obj is None
                else _gap_below(same_obj, float(row["objective"]))
            ),
            "gap_to_end_to_end_gurobi_percent": _gap_below(
                end_obj, float(row["objective"])
            ),
        }
        rows.append(item)

    if same_prefix is not None:
        rows.append({
            "method": "adaptive_gurobi_same_attention_prefix",
            "comparison_group": "same_attention_prefix",
            "prefix_source": "static_attention",
            "objective": float(same_prefix["objective"]),
            "runtime_seconds": float(same_prefix["runtime_seconds"]),
            "proven_optimal": bool(same_prefix["proven_optimal"]),
            "optimality_gap_percent": float(
                same_prefix["optimality_gap_percent"]
            ),
            "gap_to_same_prefix_oracle_percent": 0.0,
            "gap_to_end_to_end_gurobi_percent": _gap_below(
                end_obj, float(same_prefix["objective"])
            ),
        })

    initial_static = two_stage["initial_static_solution"]
    rows.append({
        "method": "gurobi_initial_static_plan_no_replan",
        "comparison_group": "gurobi_end_to_end",
        "prefix_source": "initial_static_gurobi",
        "objective": float(
            two_stage["static_plan_adaptive_objective_without_replanning"]
        ),
        "runtime_seconds": float(two_stage["initial_planning_runtime_seconds"]),
        "proven_optimal": bool(initial_static["proven_optimal"]),
        "optimality_gap_percent": float(
            initial_static["optimality_gap_percent"]
        ),
        "gap_to_same_prefix_oracle_percent": "NA",
        "gap_to_end_to_end_gurobi_percent": _gap_below(
            end_obj,
            float(two_stage["static_plan_adaptive_objective_without_replanning"]),
        ),
    })

    reopt = two_stage["reoptimized_solution"]
    rows.append({
        "method": "gurobi_two_stage_replanned",
        "comparison_group": "gurobi_end_to_end",
        "prefix_source": "initial_static_gurobi",
        "objective": float(two_stage["final_objective"]),
        "runtime_seconds": float(two_stage["reoptimization_runtime_seconds"]),
        "proven_optimal": bool(reopt["proven_optimal"]),
        "optimality_gap_percent": float(reopt["optimality_gap_percent"]),
        "gap_to_same_prefix_oracle_percent": "NA",
        "gap_to_end_to_end_gurobi_percent": 0.0,
    })

    columns = [
        "method",
        "comparison_group",
        "prefix_source",
        "objective",
        "runtime_seconds",
        "proven_optimal",
        "optimality_gap_percent",
        "gap_to_same_prefix_oracle_percent",
        "gap_to_end_to_end_gurobi_percent",
    ]
    with (output / "comparison_gurobi.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    (output / "comparison_gurobi.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== GUROBI REPLANNING COMPARISON ===")
    print(f"instance={instance.instance_id}")
    print(
        f"shock after {scenario.switch_day} completed days; "
        f"replanning begins on day {scenario.switch_day + 1}"
    )
    print(f"scenario_source={scenario_source}")

    if same_prefix is not None:
        print(
            "same-prefix Gurobi oracle: "
            f"objective={float(same_prefix['objective']):.3f} "
            f"time={float(same_prefix['runtime_seconds']):.3f}s "
            f"gap={float(same_prefix['optimality_gap_percent']):.6f}% "
            f"optimal={bool(same_prefix['proven_optimal'])}"
        )

    print(
        "initial static Gurobi (evaluated after shock, no replan): "
        f"objective={float(two_stage['static_plan_adaptive_objective_without_replanning']):.3f}"
    )
    print(
        "two-stage Gurobi after shock reoptimization: "
        f"objective={float(two_stage['final_objective']):.3f} "
        f"reopt_time={float(two_stage['reoptimization_runtime_seconds']):.3f}s "
        f"gain_from_replanning={float(two_stage['reoptimization_gain_percent']):+.3f}%"
    )
    print(
        "future plan changed in "
        f"{int(two_stage['future_actions_changed'])}/"
        f"{int(two_stage['future_periods'])} periods"
    )
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
