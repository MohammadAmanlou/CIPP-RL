"""Compare random feasible and greedy CIPP baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines import (
    run_greedy_policy,
    run_random_feasible_policy,
)

from src.utils import (
    generate_cipp_instance,
)


def _method_summary(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate policy results."""

    objectives = np.asarray(
        [
            result["objective"]
            for result in results
        ],
        dtype=np.float64,
    )

    runtimes = np.asarray(
        [
            result["runtime_seconds"]
            for result in results
        ],
        dtype=np.float64,
    )

    feasible = np.asarray(
        [
            result["feasible"]
            for result in results
        ],
        dtype=np.float64,
    )

    return {
        "mean_objective": float(
            np.mean(objectives)
        ),
        "median_objective": float(
            np.median(objectives)
        ),
        "std_objective": float(
            np.std(
                objectives,
                ddof=0,
            )
        ),
        "min_objective": float(
            np.min(objectives)
        ),
        "max_objective": float(
            np.max(objectives)
        ),
        "mean_runtime_seconds": float(
            np.mean(runtimes)
        ),
        "feasible_rate": float(
            np.mean(feasible)
        ),
    }


def run_comparison(
    *,
    number_of_instances: int,
    n: int,
    horizon: int,
    base_seed: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run both baselines on the same fixed instances."""

    if number_of_instances < 1:
        raise ValueError(
            "number_of_instances must be positive."
        )

    random_results: list[
        dict[str, Any]
    ] = []

    greedy_results: list[
        dict[str, Any]
    ] = []

    paired_results: list[
        dict[str, Any]
    ] = []

    for index in range(
        number_of_instances
    ):
        instance_seed = (
            base_seed + index
        )

        random_policy_seed = (
            base_seed
            + 100_000
            + index
        )

        instance = generate_cipp_instance(
            n=n,
            H=horizon,
            alpha=5,
            q=6,
            w=2,
            seed=instance_seed,
            instance_id=(
                f"week3-test-{index:04d}"
            ),
        )

        random_result = (
            run_random_feasible_policy(
                instance,
                seed=random_policy_seed,
            )
        )

        greedy_result = (
            run_greedy_policy(
                instance
            )
        )

        random_results.append(
            random_result.to_dict()
        )

        greedy_results.append(
            greedy_result.to_dict()
        )

        difference = (
            greedy_result.objective
            - random_result.objective
        )

        paired_results.append(
            {
                "instance_id": (
                    instance.instance_id
                ),
                "instance_seed": (
                    instance_seed
                ),
                "random_policy_seed": (
                    random_policy_seed
                ),
                "random_objective": (
                    random_result.objective
                ),
                "greedy_objective": (
                    greedy_result.objective
                ),
                "greedy_minus_random": (
                    difference
                ),
            }
        )

    differences = np.asarray(
        [
            item["greedy_minus_random"]
            for item in paired_results
        ],
        dtype=np.float64,
    )

    tolerance = 1e-9

    wins = int(
        np.count_nonzero(
            differences > tolerance
        )
    )

    ties = int(
        np.count_nonzero(
            np.abs(differences)
            <= tolerance
        )
    )

    losses = int(
        np.count_nonzero(
            differences < -tolerance
        )
    )

    payload: dict[str, Any] = {
        "configuration": {
            "number_of_instances": (
                number_of_instances
            ),
            "n": n,
            "H": horizon,
            "base_seed": base_seed,
        },
        "random_feasible": (
            _method_summary(
                random_results
            )
        ),
        "greedy_exact_increment": (
            _method_summary(
                greedy_results
            )
        ),
        "paired_comparison": {
            "mean_greedy_minus_random": float(
                np.mean(differences)
            ),
            "median_greedy_minus_random": float(
                np.median(differences)
            ),
            "greedy_wins": wins,
            "ties": ties,
            "greedy_losses": losses,
            "greedy_win_rate": float(
                wins
                / number_of_instances
            ),
        },
        "instances": paired_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic CIPP "
            "construction baselines."
        )
    )

    parser.add_argument(
        "--instances",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/week3_baselines.json"
        ),
    )

    arguments = parser.parse_args()

    payload = run_comparison(
        number_of_instances=(
            arguments.instances
        ),
        n=arguments.n,
        horizon=arguments.horizon,
        base_seed=arguments.seed,
        output_path=arguments.output,
    )

    random_summary = payload[
        "random_feasible"
    ]

    greedy_summary = payload[
        "greedy_exact_increment"
    ]

    paired = payload[
        "paired_comparison"
    ]

    print(
        "Instances:",
        arguments.instances,
    )

    print(
        "Random mean objective:",
        random_summary[
            "mean_objective"
        ],
    )

    print(
        "Greedy mean objective:",
        greedy_summary[
            "mean_objective"
        ],
    )

    print(
        "Mean greedy - random:",
        paired[
            "mean_greedy_minus_random"
        ],
    )

    print(
        "Greedy wins / ties / losses:",
        paired["greedy_wins"],
        "/",
        paired["ties"],
        "/",
        paired["greedy_losses"],
    )

    print(
        "Random feasible rate:",
        random_summary[
            "feasible_rate"
        ],
    )

    print(
        "Greedy feasible rate:",
        greedy_summary[
            "feasible_rate"
        ],
    )

    print(
        "Saved to:",
        arguments.output,
    )


if __name__ == "__main__":
    main()