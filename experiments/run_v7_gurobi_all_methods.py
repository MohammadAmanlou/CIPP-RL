"""FINAL V7 local Gurobi evaluation for every prefix/recourse idea.

Reads Kaggle outputs from:
    results/AdaptiveV7/<instance>/seed_<seed>

Exact/hybrid methods
--------------------
G-G   static exact Gurobi prefix -> exact adaptive Gurobi suffix
S-G   static RL prefix -> exact adaptive Gurobi suffix
C-G   broad generalist RL prefix -> exact adaptive Gurobi suffix
P-G   recourse-aware RL prefix -> exact adaptive Gurobi suffix
I-G   instance-specific recourse-aware RL prefix -> exact adaptive Gurobi suffix
SG-G  stochastic nonanticipative Gurobi prefix -> exact adaptive suffix
CLV   clairvoyant Gurobi perfect-information upper bound

The script also joins all Kaggle RL results so one final CSV/JSON contains every
old and new method side-by-side.
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
    sample_training_scenario,
    scenario_from_dict,
    solve_adaptive_gurobi,
    solve_clairvoyant_adaptive_gurobi,
    solve_stochastic_prefix_gurobi,
)
from src.optimization import solve_cipp_gurobi
from src.utils import load_professor_benchmark, parse_professor_instance_id


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="D_14S_30P")
    p.add_argument("--data-directory", type=Path, default=Path("."))
    p.add_argument(
        "--rl-output-directory",
        type=Path,
        default=Path("results/AdaptiveV7"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=float, default=3600.0)
    p.add_argument("--mip-gap", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--artifact-scenarios", type=int, default=1)

    # Fair anticipatory optimization benchmark.
    p.add_argument("--skip-stochastic", action="store_true")
    p.add_argument("--stochastic-scenarios", type=int, default=8)
    p.add_argument("--stochastic-time-limit", type=float, default=900.0)
    p.add_argument("--stochastic-mip-gap", type=float, default=0.01)
    return p.parse_args()


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pct(delta: float, reference: float) -> float:
    return 0.0 if reference == 0.0 else 100.0 * delta / abs(reference)


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _build_stochastic_bank(instance, *, switch_day: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    bank = []
    for j in range(count):
        sampled = sample_training_scenario(
            instance,
            rng,
            progress=(j + 0.5) / max(count, 1),
            allow_multi_shock=False,
        )
        bank.append(
            ShockScenario(
                switch_day=int(switch_day),
                post_multiplier=np.asarray(
                    sampled.post_multiplier,
                    dtype=np.float64,
                ),
                scenario_id=f"stochastic_train_T{switch_day}_{j:03d}",
                family=f"stochastic_{sampled.family}",
            )
        )
    return bank


def _index_rl_rows(rows):
    result = {}
    for row in rows:
        result.setdefault(int(row["scenario_index"]), {})[
            str(row["method"])
        ] = row
    return result


def _find_method(methods: dict[str, dict], prefix: str):
    matches = [v for k, v in methods.items() if k.startswith(prefix)]
    if not matches:
        return None
    # If multiple rows match, retain the best achieved objective.
    return max(matches, key=lambda x: float(x["objective"]))


def main():
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
    suite = [
        scenario_from_dict(x) for x in suite_payload["scenarios"]
    ]
    if not all(isinstance(x, ShockScenario) for x in suite):
        raise RuntimeError("V7 exact suite must contain only single shocks")

    prefixes = {
        int(x["scenario_index"]): x
        for x in json.loads(
            (root / "suite_prefixes.json").read_text(encoding="utf-8")
        )
    }
    rl_rows = json.loads(
        (root / "suite_rl_all_methods.json").read_text(encoding="utf-8")
    )
    rl_by_scenario = _index_rl_rows(rl_rows)

    expensive_by_scenario = {}
    expensive_path = root / "expensive_residual_suite.json"
    if expensive_path.exists():
        expensive_rows = json.loads(
            expensive_path.read_text(encoding="utf-8")
        )
        expensive_by_scenario = _index_rl_rows(expensive_rows)

    gurobi_root = root / "gurobi_v7"
    gurobi_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Static exact Gurobi is independent of the realized shock: solve once.
    # --------------------------------------------------------------
    print("=== V7 STATIC EXACT GUROBI: SOLVE ONCE ===")
    static_solution = solve_cipp_gurobi(
        instance,
        time_limit_seconds=a.time_limit,
        mip_gap=a.mip_gap,
        threads=a.threads,
        output_directory=gurobi_root / "initial_static",
        verbose=a.verbose,
    )
    if not static_solution.feasible:
        raise RuntimeError("static Gurobi returned no feasible solution")
    static_itinerary = tuple(int(x) for x in static_solution.itinerary)

    # --------------------------------------------------------------
    # Strong stochastic-Gurobi prefixes, cached by switch day.
    # --------------------------------------------------------------
    stochastic_cache = {}
    unique_days = sorted({int(s.switch_day) for s in suite})
    if not a.skip_stochastic:
        print(
            f"=== STOCHASTIC NONANTICIPATIVE GUROBI: "
            f"{len(unique_days)} UNIQUE SHOCK DAYS ==="
        )
        for k, day in enumerate(unique_days):
            bank = _build_stochastic_bank(
                instance,
                switch_day=day,
                count=a.stochastic_scenarios,
                seed=a.seed + 700_000_000 + day * 1009,
            )
            result = solve_stochastic_prefix_gurobi(
                instance,
                bank,
                time_limit_seconds=a.stochastic_time_limit,
                mip_gap=a.stochastic_mip_gap,
                threads=a.threads,
                verbose=a.verbose and k == 0,
                output_directory=(
                    gurobi_root / f"stochastic_T{day}"
                    if k < a.artifact_scenarios
                    else None
                ),
                model_label=f"stochastic_T{day}",
            )
            stochastic_cache[day] = result
            print(
                f"[stochastic] T={day} prefix={result['prefix']} "
                f"gap={result['optimality_gap_percent']:.3f}% "
                f"time={result['runtime_seconds']:.1f}s",
                flush=True,
            )

    rows = []
    started_all = time.perf_counter()

    for idx, scenario in enumerate(suite):
        px = prefixes[idx]
        prefix_map = {
            "S": tuple(int(x) for x in px["S_static_rl"]),
            "C": tuple(int(x) for x in px["C_generalist"]),
            "P": tuple(int(x) for x in px["P_recourse_aware"]),
            "I": tuple(int(x) for x in px["I_instance_specific"]),
            "G": static_itinerary[: scenario.switch_day],
        }
        if not a.skip_stochastic:
            prefix_map["SG"] = tuple(
                int(x)
                for x in stochastic_cache[int(scenario.switch_day)]["prefix"]
            )

        artifact_root = (
            gurobi_root / f"scenario_{idx:03d}"
            if idx < a.artifact_scenarios
            else None
        )

        exact = {}
        for name, prefix in prefix_map.items():
            exact[name] = solve_adaptive_gurobi(
                instance,
                scenario,
                prefix,
                time_limit_seconds=a.time_limit,
                mip_gap=a.mip_gap,
                threads=a.threads,
                verbose=a.verbose and idx < a.artifact_scenarios,
                output_directory=(
                    None
                    if artifact_root is None
                    else artifact_root / f"{name}_G"
                ),
                model_label=f"{name}_prefix_exact_recourse",
            )

        clairvoyant = solve_clairvoyant_adaptive_gurobi(
            instance,
            scenario,
            time_limit_seconds=a.time_limit,
            mip_gap=a.mip_gap,
            threads=a.threads,
            verbose=a.verbose and idx < a.artifact_scenarios,
            output_directory=(
                None
                if artifact_root is None
                else artifact_root / "clairvoyant"
            ),
        )
        no_replan = evaluate_adaptive_itinerary(
            instance,
            scenario,
            static_itinerary,
        )

        methods = rl_by_scenario[idx]
        expensive = expensive_by_scenario.get(idx, {})

        c_c = _find_method(methods, "C_C_e2e_best_of_")
        p_best_candidates = [
            x
            for prefix in ("P_P_best_of_", "P_C_best_of_")
            if (x := _find_method(methods, prefix)) is not None
        ]
        p_best = (
            max(p_best_candidates, key=lambda x: float(x["objective"]))
            if p_best_candidates
            else None
        )
        i_best_candidates = [
            x
            for prefix in ("I_I_best_of_", "I_C_best_of_")
            if (x := _find_method(methods, prefix)) is not None
        ]
        i_best = (
            max(i_best_candidates, key=lambda x: float(x["objective"]))
            if i_best_candidates
            else None
        )
        d = _find_method(methods, "D_frozen_static_best_of_")
        c_s = _find_method(methods, "C_S_static_prefix_best_of_")

        g_g = float(exact["G"]["objective"])
        row = {
            "scenario_index": idx,
            "scenario_id": scenario.scenario_id,
            "family": scenario.family,
            "switch_day": int(scenario.switch_day),

            "B_objective": float(
                _find_method(methods, "B_frozen_static_greedy")["objective"]
            ),
            "D_objective": None if d is None else float(d["objective"]),
            "C_S_objective": None if c_s is None else float(c_s["objective"]),
            "C_C_objective": None if c_c is None else float(c_c["objective"]),
            "P_best_RL_objective": (
                None if p_best is None else float(p_best["objective"])
            ),
            "I_best_RL_objective": (
                None if i_best is None else float(i_best["objective"])
            ),

            "static_gurobi_no_replan_objective": float(no_replan["objective"]),
            "G_G_two_stage_objective": g_g,
            "S_G_objective": float(exact["S"]["objective"]),
            "C_G_objective": float(exact["C"]["objective"]),
            "P_G_objective": float(exact["P"]["objective"]),
            "I_G_objective": float(exact["I"]["objective"]),
            "clairvoyant_objective": float(clairvoyant["objective"]),

            "S_prefix_advantage_vs_G_G_percent": _pct(
                float(exact["S"]["objective"]) - g_g, g_g
            ),
            "C_prefix_advantage_vs_G_G_percent": _pct(
                float(exact["C"]["objective"]) - g_g, g_g
            ),
            "P_prefix_advantage_vs_G_G_percent": _pct(
                float(exact["P"]["objective"]) - g_g, g_g
            ),
            "I_prefix_advantage_vs_G_G_percent": _pct(
                float(exact["I"]["objective"]) - g_g, g_g
            ),
            "C_adaptation_gap_to_C_G_percent": (
                None
                if c_c is None
                else _pct(
                    float(exact["C"]["objective"]) - float(c_c["objective"]),
                    float(exact["C"]["objective"]),
                )
            ),
            "P_adaptation_gap_to_P_G_percent": (
                None
                if p_best is None
                else _pct(
                    float(exact["P"]["objective"]) - float(p_best["objective"]),
                    float(exact["P"]["objective"]),
                )
            ),
            "I_adaptation_gap_to_I_G_percent": (
                None
                if i_best is None
                else _pct(
                    float(exact["I"]["objective"]) - float(i_best["objective"]),
                    float(exact["I"]["objective"]),
                )
            ),
            "C_C_vs_G_G_percent": (
                None
                if c_c is None
                else _pct(float(c_c["objective"]) - g_g, g_g)
            ),
            "P_RL_vs_G_G_percent": (
                None
                if p_best is None
                else _pct(float(p_best["objective"]) - g_g, g_g)
            ),
            "I_RL_vs_G_G_percent": (
                None
                if i_best is None
                else _pct(float(i_best["objective"]) - g_g, g_g)
            ),

            "G_G_proven_optimal": bool(exact["G"]["proven_optimal"]),
            "S_G_proven_optimal": bool(exact["S"]["proven_optimal"]),
            "C_G_proven_optimal": bool(exact["C"]["proven_optimal"]),
            "P_G_proven_optimal": bool(exact["P"]["proven_optimal"]),
            "I_G_proven_optimal": bool(exact["I"]["proven_optimal"]),
            "clairvoyant_proven_optimal": bool(
                clairvoyant["proven_optimal"]
            ),
        }

        if "SG" in exact:
            sg_g = float(exact["SG"]["objective"])
            stochastic_model = stochastic_cache[int(scenario.switch_day)]
            row.update(
                {
                    "SG_G_objective": sg_g,
                    "SG_G_vs_G_G_percent": _pct(sg_g - g_g, g_g),
                    "SG_prefix_model_gap_percent": float(
                        stochastic_model["optimality_gap_percent"]
                    ),
                    "SG_prefix_model_proven_optimal": bool(
                        stochastic_model["proven_optimal"]
                    ),
                }
            )

        for method_name in ("S_R", "C_R", "P_R", "I_R"):
            r = _find_method(expensive, method_name)
            if r is not None:
                obj = float(r["objective"])
                row[f"{method_name}_objective"] = obj
                row[f"{method_name}_vs_G_G_percent"] = _pct(
                    obj - g_g, g_g
                )
                row[f"{method_name}_online_training_seconds"] = float(
                    r["online_training_seconds"]
                )

        rows.append(row)
        if (idx + 1) % 10 == 0 or idx == 0:
            best_prefix = max(
                (
                    ("S-G", float(exact["S"]["objective"])),
                    ("C-G", float(exact["C"]["objective"])),
                    ("P-G", float(exact["P"]["objective"])),
                    ("I-G", float(exact["I"]["objective"])),
                ),
                key=lambda x: x[1],
            )
            print(
                f"[V7 Gurobi] {idx + 1}/{len(suite)} "
                f"G-G={g_g:.3f} best_hybrid={best_prefix[0]}:"
                f"{best_prefix[1]:.3f}",
                flush=True,
            )

    elapsed = float(time.perf_counter() - started_all)
    _json(root / "v7_all_methods_complete_results.json", rows)
    _write_csv(root / "v7_all_methods_complete_results.csv", rows)

    # --------------------------------------------------------------
    # Final aggregate across every idea.
    # --------------------------------------------------------------
    objective_columns = [
        "B_objective",
        "D_objective",
        "C_S_objective",
        "C_C_objective",
        "P_best_RL_objective",
        "I_best_RL_objective",
        "static_gurobi_no_replan_objective",
        "G_G_two_stage_objective",
        "S_G_objective",
        "C_G_objective",
        "P_G_objective",
        "I_G_objective",
        "SG_G_objective",
        "S_R_objective",
        "C_R_objective",
        "P_R_objective",
        "I_R_objective",
        "clairvoyant_objective",
    ]
    summary = {
        "instance": instance.instance_id,
        "scenario_count": len(rows),
        "static_initial_gurobi": static_solution.to_dict(),
        "gurobi_suite_runtime_seconds_excluding_stochastic_prefix_training": elapsed,
        "methods": {},
        "win_rates_vs_G_G": {},
        "prefix_advantages_vs_G_G": {},
        "by_family": {},
    }

    for col in objective_columns:
        vals = [
            float(r[col])
            for r in rows
            if col in r and r[col] is not None
        ]
        if vals:
            summary["methods"][col] = _stats(vals)

    for col in (
        "C_C_objective",
        "P_best_RL_objective",
        "I_best_RL_objective",
        "S_G_objective",
        "C_G_objective",
        "P_G_objective",
        "I_G_objective",
        "SG_G_objective",
        "S_R_objective",
        "C_R_objective",
        "P_R_objective",
        "I_R_objective",
    ):
        subset = [
            r for r in rows if col in r and r[col] is not None
        ]
        if subset:
            diffs = [
                _pct(
                    float(r[col]) - float(r["G_G_two_stage_objective"]),
                    float(r["G_G_two_stage_objective"]),
                )
                for r in subset
            ]
            summary["win_rates_vs_G_G"][col] = {
                "count": len(subset),
                "win_count": int(sum(x > 1e-12 for x in diffs)),
                "win_rate": float(np.mean([x > 1e-12 for x in diffs])),
                "advantage_percent": _stats(diffs),
            }

    for prefix in ("S", "C", "P", "I"):
        vals = [
            float(r[f"{prefix}_prefix_advantage_vs_G_G_percent"])
            for r in rows
        ]
        summary["prefix_advantages_vs_G_G"][prefix] = {
            "win_count": int(sum(x > 1e-12 for x in vals)),
            "win_rate": float(np.mean([x > 1e-12 for x in vals])),
            "advantage_percent": _stats(vals),
        }

    families = sorted({str(r["family"]) for r in rows})
    for family in families:
        subset = [r for r in rows if r["family"] == family]
        family_payload = {}
        for col in (
            "C_C_objective",
            "P_best_RL_objective",
            "I_best_RL_objective",
            "G_G_two_stage_objective",
            "C_G_objective",
            "P_G_objective",
            "I_G_objective",
            "SG_G_objective",
        ):
            vals = [
                float(r[col])
                for r in subset
                if col in r and r[col] is not None
            ]
            if vals:
                family_payload[col] = _stats(vals)
        summary["by_family"][family] = family_payload

    _json(root / "v7_all_methods_complete_summary.json", summary)
    _json(root / "v7_main_pilot_complete.json", rows[0])

    print("\n=== V7 FINAL SUMMARY ===")
    for method, data in summary["win_rates_vs_G_G"].items():
        print(
            f"{method}: win_rate_vs_G-G={100*data['win_rate']:.1f}% "
            f"mean_adv={data['advantage_percent']['mean']:+.3f}%"
        )
    print("\nPrefix quality:")
    for prefix, data in summary["prefix_advantages_vs_G_G"].items():
        print(
            f"{prefix}: prefix_win_rate={100*data['win_rate']:.1f}% "
            f"mean_adv={data['advantage_percent']['mean']:+.3f}%"
        )
    print(f"\nSaved to: {root}")


if __name__ == "__main__":
    main()
