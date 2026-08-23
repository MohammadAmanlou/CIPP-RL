"""Run exact Gurobi references for the frozen V6 held-out suite.

Run this LOCALLY after the Kaggle V6 RL results have been copied to:
    results/AdaptiveV6/<instance>/seed_<seed>

Efficiency:
- The static day-0 Gurobi plan is solved ONCE because it is independent of the
  realized shock.
- For each held-out single shock, we solve:
    1) exact continuation from the static-Gurobi prefix (two-stage baseline),
    2) exact continuation from Method C's realized anticipatory prefix,
    3) clairvoyant perfect-information full-horizon upper bound.

This directly decomposes:
    prefix advantage  = oracle(Method-C prefix) - two-stage Gurobi
    adaptation gap    = oracle(Method-C prefix) - Method C
    final advantage   = Method C - two-stage Gurobi
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive import (
    ShockScenario,
    evaluate_adaptive_itinerary,
    scenario_from_dict,
    solve_adaptive_gurobi,
    solve_clairvoyant_adaptive_gurobi,
)
from src.optimization import solve_cipp_gurobi
from src.utils import load_professor_benchmark, parse_professor_instance_id


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="D_14S_30P")
    p.add_argument("--data-directory", type=Path, default=Path("."))
    p.add_argument(
        "--rl-output-directory",
        type=Path,
        default=Path("results/AdaptiveV6"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=float, default=3600.0)
    p.add_argument("--mip-gap", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--artifact-scenarios",
        type=int,
        default=1,
        help="Save LP/SOL/log artifacts for the first N suite scenarios only.",
    )
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _pct(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else 100.0 * numerator / abs(denominator)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


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

    root = a.rl_output_directory / instance.instance_id / f"seed_{a.seed}"
    suite_payload = json.loads(
        (root / "heldout_suite.json").read_text(encoding="utf-8")
    )
    suite = [scenario_from_dict(x) for x in suite_payload["scenarios"]]
    if not all(isinstance(x, ShockScenario) for x in suite):
        raise RuntimeError(
            "V6 exact Gurobi suite must contain only single-shock scenarios"
        )

    rl_rows = json.loads(
        (root / "suite_rl_results.json").read_text(encoding="utf-8")
    )
    c_rows = {
        int(row["scenario_index"]): row
        for row in rl_rows
        if str(row["method"]).startswith("C_rich_generalist_e2e_best_of_")
    }
    if len(c_rows) != len(suite):
        raise RuntimeError(
            f"Expected one Method C row per scenario; got {len(c_rows)} "
            f"for {len(suite)} scenarios"
        )

    retrain_rows = {}
    retrain_path = root / "retrain_suite_results.json"
    if retrain_path.exists():
        for row in json.loads(retrain_path.read_text(encoding="utf-8")):
            retrain_rows[int(row["scenario_index"])] = row

    gurobi_root = root / "gurobi_v6"
    gurobi_root.mkdir(parents=True, exist_ok=True)

    print("=== STATIC DAY-0 GUROBI: SOLVE ONCE ===")
    static_solution = solve_cipp_gurobi(
        instance,
        time_limit_seconds=a.time_limit,
        mip_gap=a.mip_gap,
        threads=a.threads,
        output_directory=gurobi_root / "initial_static",
        verbose=a.verbose,
    )
    static_itinerary = tuple(int(x) for x in static_solution.itinerary)
    if not static_solution.feasible:
        raise RuntimeError("Static Gurobi did not return a feasible solution")

    rows: list[dict[str, object]] = []
    started_all = time.perf_counter()

    for i, scenario in enumerate(suite):
        c_row = c_rows[i]
        method_c_prefix = tuple(int(x) for x in c_row["prefix"])
        if len(method_c_prefix) != scenario.switch_day:
            raise RuntimeError(
                f"scenario {i}: Method C prefix length mismatch "
                f"{len(method_c_prefix)} != {scenario.switch_day}"
            )
        static_gurobi_prefix = static_itinerary[: scenario.switch_day]

        artifact_root = (
            gurobi_root / f"scenario_{i:03d}"
            if i < a.artifact_scenarios
            else None
        )

        two_stage = solve_adaptive_gurobi(
            instance,
            scenario,
            static_gurobi_prefix,
            time_limit_seconds=a.time_limit,
            mip_gap=a.mip_gap,
            threads=a.threads,
            verbose=a.verbose if i < a.artifact_scenarios else False,
            output_directory=(
                None if artifact_root is None else artifact_root / "two_stage_reopt"
            ),
            model_label="two_stage_post_shock",
        )

        c_prefix_oracle = solve_adaptive_gurobi(
            instance,
            scenario,
            method_c_prefix,
            time_limit_seconds=a.time_limit,
            mip_gap=a.mip_gap,
            threads=a.threads,
            verbose=a.verbose if i < a.artifact_scenarios else False,
            output_directory=(
                None if artifact_root is None else artifact_root / "method_c_prefix_oracle"
            ),
            model_label="method_c_prefix_oracle",
        )

        clairvoyant = solve_clairvoyant_adaptive_gurobi(
            instance,
            scenario,
            time_limit_seconds=a.time_limit,
            mip_gap=a.mip_gap,
            threads=a.threads,
            verbose=a.verbose if i < a.artifact_scenarios else False,
            output_directory=(
                None if artifact_root is None else artifact_root / "clairvoyant"
            ),
        )

        no_replan = evaluate_adaptive_itinerary(
            instance,
            scenario,
            static_itinerary,
        )

        c_obj = float(c_row["objective"])
        two_obj = float(two_stage["objective"])
        prefix_oracle_obj = float(c_prefix_oracle["objective"])
        clair_obj = float(clairvoyant["objective"])
        no_replan_obj = float(no_replan["objective"])

        row = {
            "scenario_index": i,
            "scenario_id": scenario.scenario_id,
            "family": scenario.family,
            "switch_day": int(scenario.switch_day),
            "method_c_objective": c_obj,
            "two_stage_gurobi_objective": two_obj,
            "method_c_prefix_oracle_objective": prefix_oracle_obj,
            "clairvoyant_gurobi_objective": clair_obj,
            "static_no_replan_objective": no_replan_obj,
            "prefix_advantage_absolute": prefix_oracle_obj - two_obj,
            "prefix_advantage_percent": _pct(
                prefix_oracle_obj - two_obj,
                two_obj,
            ),
            "adaptation_gap_absolute": prefix_oracle_obj - c_obj,
            "adaptation_gap_percent": _pct(
                prefix_oracle_obj - c_obj,
                prefix_oracle_obj,
            ),
            "final_advantage_absolute": c_obj - two_obj,
            "final_advantage_percent": _pct(
                c_obj - two_obj,
                two_obj,
            ),
            "clairvoyant_gap_percent": _pct(
                clair_obj - c_obj,
                clair_obj,
            ),
            "two_stage_replanning_gain_percent": _pct(
                two_obj - no_replan_obj,
                no_replan_obj,
            ),
            "method_c_beats_two_stage": bool(c_obj > two_obj + 1e-9),
            "method_c_prefix_beats_static_gurobi_prefix": bool(
                prefix_oracle_obj > two_obj + 1e-9
            ),
            "two_stage_proven_optimal": bool(two_stage["proven_optimal"]),
            "method_c_prefix_oracle_proven_optimal": bool(
                c_prefix_oracle["proven_optimal"]
            ),
            "clairvoyant_proven_optimal": bool(clairvoyant["proven_optimal"]),
            "two_stage_runtime_seconds": float(two_stage["runtime_seconds"]),
            "method_c_prefix_oracle_runtime_seconds": float(
                c_prefix_oracle["runtime_seconds"]
            ),
            "clairvoyant_runtime_seconds": float(
                clairvoyant["runtime_seconds"]
            ),
            "method_c_prefix": list(method_c_prefix),
            "static_gurobi_prefix": list(static_gurobi_prefix),
        }

        if i in retrain_rows:
            r = retrain_rows[i]
            r_obj = float(r["objective"])
            row.update({
                "R_retrain_objective": r_obj,
                "R_retrain_final_advantage_absolute": r_obj - two_obj,
                "R_retrain_final_advantage_percent": _pct(
                    r_obj - two_obj,
                    two_obj,
                ),
                "R_retrain_beats_two_stage": bool(r_obj > two_obj + 1e-9),
                "R_retrain_online_training_seconds": float(
                    r["online_training_seconds"]
                ),
                "R_retrain_inference_runtime_seconds": float(
                    r["inference_runtime_seconds"]
                ),
            })

        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[gurobi-suite] {i + 1}/{len(suite)} "
                f"C={c_obj:.3f} two-stage={two_obj:.3f} "
                f"final={row['final_advantage_percent']:+.3f}% "
                f"prefix={row['prefix_advantage_percent']:+.3f}%",
                flush=True,
            )

    elapsed = float(time.perf_counter() - started_all)

    (root / "gurobi_suite_results.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        root / "gurobi_suite_results.csv",
        [
            {
                k: v
                for k, v in row.items()
                if k not in {"method_c_prefix", "static_gurobi_prefix"}
            }
            for row in rows
        ],
    )

    final = np.asarray(
        [float(x["final_advantage_percent"]) for x in rows],
        dtype=np.float64,
    )
    prefix = np.asarray(
        [float(x["prefix_advantage_percent"]) for x in rows],
        dtype=np.float64,
    )
    adaptation = np.asarray(
        [float(x["adaptation_gap_percent"]) for x in rows],
        dtype=np.float64,
    )
    clair = np.asarray(
        [float(x["clairvoyant_gap_percent"]) for x in rows],
        dtype=np.float64,
    )

    summary = {
        "instance": instance.instance_id,
        "scenario_count": len(rows),
        "static_initial_gurobi": static_solution.to_dict(),
        "total_suite_gurobi_seconds_excluding_static_initial": elapsed,
        "all_two_stage_proven_optimal": all(
            bool(x["two_stage_proven_optimal"]) for x in rows
        ),
        "all_method_c_prefix_oracles_proven_optimal": all(
            bool(x["method_c_prefix_oracle_proven_optimal"]) for x in rows
        ),
        "all_clairvoyant_proven_optimal": all(
            bool(x["clairvoyant_proven_optimal"]) for x in rows
        ),
        "method_c_vs_two_stage": {
            "win_count": int(sum(bool(x["method_c_beats_two_stage"]) for x in rows)),
            "win_rate": float(
                np.mean([bool(x["method_c_beats_two_stage"]) for x in rows])
            ),
            "final_advantage_percent": _summary(final),
        },
        "anticipatory_prefix_value": {
            "prefix_win_count": int(
                sum(
                    bool(x["method_c_prefix_beats_static_gurobi_prefix"])
                    for x in rows
                )
            ),
            "prefix_win_rate": float(
                np.mean(
                    [
                        bool(x["method_c_prefix_beats_static_gurobi_prefix"])
                        for x in rows
                    ]
                )
            ),
            "prefix_advantage_percent": _summary(prefix),
        },
        "post_shock_adaptation": {
            "adaptation_gap_percent": _summary(adaptation),
        },
        "clairvoyant_gap_percent": _summary(clair),
        "by_family": {},
    }

    families = sorted({str(x["family"]) for x in rows})
    for family in families:
        subset = [x for x in rows if x["family"] == family]
        vals = np.asarray(
            [float(x["final_advantage_percent"]) for x in subset],
            dtype=np.float64,
        )
        prefix_vals = np.asarray(
            [float(x["prefix_advantage_percent"]) for x in subset],
            dtype=np.float64,
        )
        summary["by_family"][family] = {
            "count": len(subset),
            "method_c_win_rate": float(
                np.mean([bool(x["method_c_beats_two_stage"]) for x in subset])
            ),
            "final_advantage_percent": _summary(vals),
            "prefix_advantage_percent": _summary(prefix_vals),
        }

    r_subset = [x for x in rows if "R_retrain_objective" in x]
    if r_subset:
        r_adv = np.asarray(
            [float(x["R_retrain_final_advantage_percent"]) for x in r_subset],
            dtype=np.float64,
        )
        summary["R_retraining_subset"] = {
            "count": len(r_subset),
            "win_rate_vs_two_stage": float(
                np.mean([bool(x["R_retrain_beats_two_stage"]) for x in r_subset])
            ),
            "final_advantage_percent": _summary(r_adv),
            "mean_online_training_seconds": float(
                np.mean(
                    [float(x["R_retrain_online_training_seconds"]) for x in r_subset]
                )
            ),
        }

    (root / "gurobi_suite_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    # Main pilot = scenario 0.
    (root / "main_complete_comparison.json").write_text(
        json.dumps(rows[0], indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== V6 GUROBI SUITE SUMMARY ===")
    print(
        "Method C win rate vs two-stage Gurobi: "
        f"{100.0 * summary['method_c_vs_two_stage']['win_rate']:.1f}%"
    )
    print(
        "Method C prefix win rate vs static-Gurobi prefix: "
        f"{100.0 * summary['anticipatory_prefix_value']['prefix_win_rate']:.1f}%"
    )
    print(
        "Mean final advantage: "
        f"{summary['method_c_vs_two_stage']['final_advantage_percent']['mean']:+.3f}%"
    )
    print(
        "Mean prefix advantage: "
        f"{summary['anticipatory_prefix_value']['prefix_advantage_percent']['mean']:+.3f}%"
    )
    print(
        "Mean adaptation gap: "
        f"{summary['post_shock_adaptation']['adaptation_gap_percent']['mean']:.3f}%"
    )
    if "R_retraining_subset" in summary:
        print(
            "R shock-retraining subset win rate vs two-stage: "
            f"{100.0 * summary['R_retraining_subset']['win_rate_vs_two_stage']:.1f}%"
        )
    print(f"Saved to: {root}")


if __name__ == "__main__":
    main()
