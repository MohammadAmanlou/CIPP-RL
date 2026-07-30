"""Benchmark all implemented methods on one smallest real-data instance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines import (
    run_dqn_backtracking_rollouts,
    run_dqn_greedy_policy,
    run_greedy_policy,
    run_random_feasible_policy,
)
from src.envs import CIPPEnv
from src.models import DQNAgent
from src.solvers import solve_with_gurobi
from src.utils import ObservationNormalizer, load_professor_instance


def _default_data_path(party: str) -> Path:
    processed = Path(f"data/processed/CIPP-{party}.csv")
    return processed if processed.exists() else Path(f"data/CIPP-{party}.xls")


def _gap_percent(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None:
        return None
    return float(100.0 * (reference - value) / max(abs(reference), 1e-12))


def _aggregate(method: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError(f"No runs supplied for {method}.")
    objectives = np.asarray([run["objective"] for run in runs], dtype=np.float64)
    runtimes = np.asarray([run["runtime_seconds"] for run in runs], dtype=np.float64)
    feasible = np.asarray([run["feasible"] for run in runs], dtype=np.float64)
    rollouts = np.asarray([run.get("rollouts", 1) for run in runs], dtype=np.float64)
    best_index = int(np.argmax(objectives))
    best = runs[best_index]
    return {
        "method": method,
        "best_objective": float(objectives[best_index]),
        "mean_objective": float(objectives.mean()),
        "std_objective": float(objectives.std(ddof=0)),
        "min_objective": float(objectives.min()),
        "max_objective": float(objectives.max()),
        "mean_runtime_seconds": float(runtimes.mean()),
        "total_runtime_seconds": float(runtimes.sum()),
        "feasible_rate": float(feasible.mean()),
        "runs": len(runs),
        "mean_search_rollouts": float(rollouts.mean()),
        "total_search_rollouts": int(rollouts.sum()),
        "best_run_index": best_index,
        "best_itinerary": best["itinerary"],
        "best_idle_days": int(best["idle_days"]),
        "best_unique_locations": int(best["unique_locations"]),
        "details": runs,
    }


def _best_of_random(runs: list[dict[str, Any]]) -> dict[str, Any]:
    base = _aggregate(f"random_best_of_{len(runs)}", runs)
    best_index = base["best_run_index"]
    best = runs[best_index]
    total_runtime = float(sum(run["runtime_seconds"] for run in runs))
    return {
        **base,
        "mean_objective": base["best_objective"],
        "std_objective": 0.0,
        "mean_runtime_seconds": total_runtime,
        "total_runtime_seconds": total_runtime,
        "runs": 1,
        "mean_search_rollouts": float(len(runs)),
        "total_search_rollouts": len(runs),
        "details": runs,
        "best_idle_days": int(best["idle_days"]),
        "best_unique_locations": int(best["unique_locations"]),
    }


def _load_checkpoint(path: Path, device: str):
    agent, payload = DQNAgent.from_checkpoint(path, device=device)
    normalizer = ObservationNormalizer.from_dict(payload["normalizer"])
    return agent, normalizer, payload


def _write_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "Instance",
        "Method",
        "BFS",
        "Mean Objective",
        "Std Objective",
        "Best Gap to Reference (%)",
        "Mean Gap to Reference (%)",
        "CPU Mean (s)",
        "CPU Total (s)",
        "Runs/Seeds",
        "Mean Search Rollouts",
        "Total Search Rollouts",
        "Feasible Rate",
        "Idle Days (Best)",
        "Unique Locations (Best)",
        "Reference Type",
        "Gurobi MIP Gap (%)",
        "Status",
    ]
    with (output_dir / "comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def display(value: Any, decimals: int = 3) -> str:
        if value is None or value == "":
            return "--"
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return str(value)

    markdown = [
        "| Instance | Method | BFS | Mean | Std | Best gap (%) | Mean gap (%) | CPU total (s) | Runs | Rollouts | Feasible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {Instance} | {Method} | {BFS} | {Mean} | {Std} | {BestGap} | {MeanGap} | {CPU} | {Runs} | {Rollouts} | {Rate} |".format(
                Instance=row["Instance"],
                Method=row["Method"],
                BFS=display(row["BFS"], 2),
                Mean=display(row["Mean Objective"], 2),
                Std=display(row["Std Objective"], 2),
                BestGap=display(row["Best Gap to Reference (%)"], 2),
                MeanGap=display(row["Mean Gap to Reference (%)"], 2),
                CPU=display(row["CPU Total (s)"], 4),
                Runs=row["Runs/Seeds"],
                Rollouts=display(row["Mean Search Rollouts"], 1),
                Rate=display(row["Feasible Rate"], 3),
            )
        )
    (output_dir / "comparison_table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Performance on the smallest real-data CIPP instance.}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Instance & Method & BFS & Mean & Std & Best gap (\%) & CPU total (s) & Feasible \\",
        r"\midrule",
    ]
    for row in rows:
        instance = str(row["Instance"]).replace("_", r"\_")
        method = str(row["Method"]).replace("_", r"\_")
        latex.append(
            f"{instance} & {method} & {display(row['BFS'], 1)} & "
            f"{display(row['Mean Objective'], 1)} & {display(row['Std Objective'], 1)} & "
            f"{display(row['Best Gap to Reference (%)'], 2)} & "
            f"{display(row['CPU Total (s)'], 3)} & "
            f"{display(row['Feasible Rate'], 2)} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (output_dir / "comparison_table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--party", choices=["D", "R"], default="D")
    parser.add_argument("--d-data", type=Path, default=_default_data_path("D"))
    parser.add_argument("--r-data", type=Path, default=_default_data_path("R"))
    parser.add_argument("--cities-parameter", type=int, default=16)
    parser.add_argument(
        "--instance-mode",
        choices=["supplied-code", "paper-14"],
        default="supplied-code",
    )
    parser.add_argument("--real-location-count", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--rollouts", type=int, default=30)
    parser.add_argument("--alternatives-per-state", type=int, default=3)
    parser.add_argument("--random-runs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gurobi-time-limit", type=float, default=3600.0)
    parser.add_argument("--gurobi-threads", type=int, default=None)
    parser.add_argument("--gurobi-seed", type=int, default=0)
    parser.add_argument("--skip-gurobi", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/smallest_real")
    )
    args = parser.parse_args()

    if args.rollouts < 1 or args.random_runs < 1:
        raise ValueError("rollouts and random-runs must be positive.")

    data_path = args.d_data if args.party == "D" else args.r_data
    instance, location_ids = load_professor_instance(
        data_path,
        party=args.party,
        cities_parameter=args.cities_parameter,
        horizon=args.horizon,
        instance_mode=args.instance_mode,
        real_location_count=args.real_location_count,
    )

    detailed: dict[str, Any] = {
        "instance": {
            "instance_id": instance.instance_id,
            "party": args.party,
            "source_data": str(data_path),
            "cities_parameter": args.cities_parameter,
            "instance_mode": args.instance_mode,
            "n_real_actions": instance.n,
            "H": instance.H,
            "location_ids": list(location_ids),
            "rewards": instance.rewards.tolist(),
            "temporal_weights": instance.temporal_weights.tolist(),
            "idle_requirements": instance.idle_requirements.tolist(),
            "repeat_count_offset": instance.repeat_count_offset,
            "budget_constraint_active": bool(np.any(instance.costs > 0)),
        }
    }

    random_runs = [
        run_random_feasible_policy(instance, seed=100_000 + index).to_dict()
        for index in range(args.random_runs)
    ]
    random_distribution = _aggregate(
        f"random_feasible_{args.random_runs}_runs", random_runs
    )
    random_best = _best_of_random(random_runs)
    greedy = _aggregate(
        "greedy_exact_increment", [run_greedy_policy(instance).to_dict()]
    )

    dqn_greedy_runs: list[dict[str, Any]] = []
    dqn_rollout_runs: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []

    for checkpoint in args.checkpoints:
        agent, normalizer, payload = _load_checkpoint(checkpoint, args.device)
        expected_observation_dim = CIPPEnv(instance).reset()[0].size
        if agent.action_dim != instance.num_actions:
            raise ValueError(
                f"Checkpoint {checkpoint} action_dim={agent.action_dim} does not "
                f"match real instance actions={instance.num_actions}."
            )
        if agent.observation_dim != expected_observation_dim:
            raise ValueError(
                f"Checkpoint {checkpoint} observation_dim={agent.observation_dim} "
                f"does not match benchmark observation_dim={expected_observation_dim}."
            )
        if normalizer.observation_dim != expected_observation_dim:
            raise ValueError("Checkpoint normalizer has the wrong dimension.")

        dqn_greedy_runs.append(
            run_dqn_greedy_policy(
                instance, agent=agent, normalizer=normalizer
            ).to_dict()
        )
        training_metadata = payload.get("training_metadata", {})
        reward_scale = float(training_metadata.get("reward_scale", 1.0))
        dqn_rollout_runs.append(
            run_dqn_backtracking_rollouts(
                instance,
                agent=agent,
                normalizer=normalizer,
                rollout_budget=args.rollouts,
                alternatives_per_state=args.alternatives_per_state,
                reward_scale=reward_scale,
            ).to_dict()
        )
        checkpoint_metadata.append(
            {
                "path": str(checkpoint),
                "training_metadata": training_metadata,
                "rollout_reward_scale": reward_scale,
            }
        )

    dqn_greedy = _aggregate("dqn_greedy", dqn_greedy_runs)
    dqn_rollout = _aggregate(
        f"dqn_backtracking_{args.rollouts}", dqn_rollout_runs
    )

    gurobi_result = None
    reference_value: float | None = None
    reference_name = "not available"
    if not args.skip_gurobi:
        gurobi_result = solve_with_gurobi(
            instance,
            time_limit_seconds=args.gurobi_time_limit,
            threads=args.gurobi_threads,
            seed=args.gurobi_seed,
            log_path=args.output_dir / "gurobi.log",
            export_model_prefix=args.output_dir / "gurobi_model",
        )
        if gurobi_result.objective is not None:
            reference_value = gurobi_result.objective
            reference_name = (
                "proven optimum"
                if gurobi_result.certified_optimal
                else "Gurobi incumbent (not certified optimal)"
            )

    summaries = [
        random_distribution,
        random_best,
        greedy,
        dqn_greedy,
        dqn_rollout,
    ]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.append(
            {
                "Instance": instance.instance_id,
                "Method": summary["method"],
                "BFS": summary["best_objective"],
                "Mean Objective": summary["mean_objective"],
                "Std Objective": summary["std_objective"],
                "Best Gap to Reference (%)": _gap_percent(
                    reference_value, summary["best_objective"]
                ),
                "Mean Gap to Reference (%)": _gap_percent(
                    reference_value, summary["mean_objective"]
                ),
                "CPU Mean (s)": summary["mean_runtime_seconds"],
                "CPU Total (s)": summary["total_runtime_seconds"],
                "Runs/Seeds": summary["runs"],
                "Mean Search Rollouts": summary["mean_search_rollouts"],
                "Total Search Rollouts": summary["total_search_rollouts"],
                "Feasible Rate": summary["feasible_rate"],
                "Idle Days (Best)": summary["best_idle_days"],
                "Unique Locations (Best)": summary["best_unique_locations"],
                "Reference Type": reference_name,
                "Gurobi MIP Gap (%)": None,
                "Status": "FEASIBLE",
            }
        )

    if gurobi_result is not None:
        gurobi_idle = None
        gurobi_unique = None
        if gurobi_result.itinerary is not None:
            from src.core import evaluate_itinerary

            evaluation = evaluate_itinerary(instance, gurobi_result.itinerary)
            gurobi_idle = evaluation.idle_days
            gurobi_unique = int(np.count_nonzero(evaluation.visit_counts))
        rows.append(
            {
                "Instance": instance.instance_id,
                "Method": "gurobi",
                "BFS": gurobi_result.objective,
                "Mean Objective": gurobi_result.objective,
                "Std Objective": 0.0 if gurobi_result.objective is not None else None,
                "Best Gap to Reference (%)": (
                    0.0 if gurobi_result.objective is not None else None
                ),
                "Mean Gap to Reference (%)": (
                    0.0 if gurobi_result.objective is not None else None
                ),
                "CPU Mean (s)": gurobi_result.runtime_seconds,
                "CPU Total (s)": gurobi_result.runtime_seconds,
                "Runs/Seeds": 1,
                "Mean Search Rollouts": 1.0,
                "Total Search Rollouts": 1,
                "Feasible Rate": 1.0 if gurobi_result.feasible else 0.0,
                "Idle Days (Best)": gurobi_idle,
                "Unique Locations (Best)": gurobi_unique,
                "Reference Type": reference_name,
                "Gurobi MIP Gap (%)": gurobi_result.mip_gap_percent,
                "Status": gurobi_result.status,
            }
        )

    detailed.update(
        {
            "checkpoint_metadata": checkpoint_metadata,
            "methods": {summary["method"]: summary for summary in summaries},
            "gurobi": None if gurobi_result is None else gurobi_result.to_dict(),
            "reference_value": reference_value,
            "reference_name": reference_name,
            "table_rows": rows,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark_details.json").write_text(
        json.dumps(detailed, indent=2) + "\n", encoding="utf-8"
    )
    _write_tables(args.output_dir, rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
