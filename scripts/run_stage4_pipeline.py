"""Run DQN training and smallest-real-instance benchmarking end to end."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _combine_tables(paths: list[Path], output_path: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            current = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current
            elif current != fieldnames:
                raise ValueError("Comparison tables use different schemas.")
            rows.extend(reader)
    if not fieldnames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--party", choices=["D", "R", "both"], default="both")
    parser.add_argument(
        "--instance-mode",
        choices=["supplied-code", "paper-14"],
        default="supplied-code",
    )
    parser.add_argument("--cities-parameter", type=int, default=16)
    parser.add_argument("--real-location-count", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=30)
    parser.add_argument("--alternatives-per-state", type=int, default=3)
    parser.add_argument("--random-runs", type=int, default=30)
    parser.add_argument("--gurobi-time-limit", type=float, default=3600.0)
    parser.add_argument("--gurobi-threads", type=int, default=None)
    parser.add_argument("--gurobi-seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/dqn_real_matched"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/stage4_smallest_real"),
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-gurobi", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    common_shape = [
        "--instance-mode",
        args.instance_mode,
        "--cities-parameter",
        str(args.cities_parameter),
        "--horizon",
        str(args.horizon),
    ]
    if args.real_location_count is not None:
        common_shape.extend(["--real-location-count", str(args.real_location_count)])

    if not args.skip_training:
        train_command = [
            python,
            "-m",
            "experiments.train_dqn_real_matched",
            "--episodes",
            str(args.episodes),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--output-dir",
            str(args.checkpoint_dir),
            *common_shape,
        ]
        if args.device:
            train_command.extend(["--device", args.device])
        _run(train_command)

    checkpoints = [
        args.checkpoint_dir / f"seed_{seed}" / "best.pt" for seed in args.seeds
    ]
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing best checkpoints:\n" + "\n".join(missing))

    parties = ["D", "R"] if args.party == "both" else [args.party]
    table_paths: list[Path] = []
    for party in parties:
        output = args.results_dir / party
        command = [
            python,
            "-m",
            "experiments.benchmark_smallest_real_instance",
            "--party",
            party,
            "--checkpoints",
            *[str(path) for path in checkpoints],
            "--rollouts",
            str(args.rollouts),
            "--alternatives-per-state",
            str(args.alternatives_per_state),
            "--random-runs",
            str(args.random_runs),
            "--gurobi-time-limit",
            str(args.gurobi_time_limit),
            "--gurobi-seed",
            str(args.gurobi_seed),
            "--output-dir",
            str(output),
            *common_shape,
        ]
        if args.gurobi_threads is not None:
            command.extend(["--gurobi-threads", str(args.gurobi_threads)])
        if args.device:
            command.extend(["--device", args.device])
        if args.skip_gurobi:
            command.append("--skip-gurobi")
        _run(command)
        table_paths.append(output / "comparison_table.csv")

    if len(table_paths) > 1:
        combined = args.results_dir / "comparison_table_all.csv"
        _combine_tables(table_paths, combined)
        print(f"\nCombined table: {combined}", flush=True)


if __name__ == "__main__":
    main()
