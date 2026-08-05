"""Paper-style Week 4 benchmark on professor-provided real instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines import (
    PolicyResult,
    run_greedy_policy,
)
from src.core import (
    CIPPInstance,
    visit_counts,
)
from src.models import (
    DQNAgent,
    PPOAgent,
)
from src.optimization import (
    ExactCIPPSolution,
    solve_cipp_exact,
)
from src.training import (
    evaluate_dqn_best_of_rollouts,
    evaluate_dqn_policy,
    evaluate_ppo_best_of_rollouts,
    evaluate_ppo_policy,
)
from src.utils import (
    ObservationNormalizer,
    ProfessorBenchmark,
    load_professor_benchmark,
)


def _validate_checkpoint_variant(
    payload: dict[str, Any],
    expected_variant: str,
    checkpoint_label: str,
) -> None:
    training_metadata = payload.get(
        "training_metadata",
        {},
    )

    checkpoint_variant = (
        training_metadata.get(
            "objective_variant"
        )
    )

    if (
        checkpoint_variant is not None
        and checkpoint_variant
        != expected_variant
    ):
        raise ValueError(
            f"{checkpoint_label} was trained with "
            f"objective_variant={checkpoint_variant}, but "
            f"the benchmark requested {expected_variant}."
        )


def _validate_checkpoint_instance(
    payload: dict[str, Any],
    expected_instance_id: str,
    checkpoint_label: str,
) -> None:
    """Prevent accidental use of a D-specific checkpoint on R or vice versa."""

    metadata = payload.get(
        "training_metadata",
        {},
    )

    source_ids = set(
        metadata.get(
            "training_instance_ids",
            metadata.get(
                "teacher_instance_ids",
                [],
            ),
        )
    )

    if (
        metadata.get(
            "real_benchmark_used_for_training",
            False,
        )
        and source_ids
        and expected_instance_id
        not in source_ids
    ):
        raise ValueError(
            f"{checkpoint_label} is instance-specific for "
            f"{sorted(source_ids)}, not {expected_instance_id}."
        )


def _hhi(
    instance: CIPPInstance,
    result: PolicyResult,
) -> float:
    counts = visit_counts(
        instance,
        result.itinerary,
    ).astype(
        np.float64
    )

    total = float(
        counts.sum()
    )

    if total == 0.0:
        return 0.0

    shares = counts / total

    return float(
        np.square(
            shares
        ).sum()
    )


def _policy_row(
    *,
    benchmark: ProfessorBenchmark,
    result: PolicyResult,
    rollouts: int,
    reference_objective: float | None,
    reference_kind: str | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    if (
        reference_objective is not None
        and reference_objective != 0
    ):
        gap = (
            100.0
            * (
                reference_objective
                - result.objective
            )
            / abs(
                reference_objective
            )
        )
    else:
        gap = None

    if (
        benchmark.instance.budget
        >= 1.0e11
    ):
        budget_utilization = None
    else:
        budget_utilization = (
            100.0
            * result.total_cost
            / benchmark.instance.budget
        )

    row = {
        "instance": (
            benchmark.instance
            .instance_id
        ),
        "objective_variant": (
            benchmark.objective_variant
        ),
        "method": result.method,
        "objective": result.objective,
        "gap_to_reference_percent": (
            gap
        ),
        "reference_kind": (
            reference_kind
        ),
        "runtime_seconds": (
            result.runtime_seconds
        ),
        "feasible": result.feasible,
        "constraint_violations": len(
            result.violations
        ),
        "rollouts": rollouts,
        "budget_utilization_percent": (
            budget_utilization
        ),
        "idle_days": result.idle_days,
        "unique_states": (
            result.unique_locations
        ),
        "hhi": _hhi(
            benchmark.instance,
            result,
        ),
    }

    detail = {
        **row,
        "itinerary": list(
            result.itinerary
        ),
        "itinerary_names": [
            (
                "Idle"
                if action == 0
                else benchmark
                .location_names[
                    action - 1
                ]
            )
            for action in result.itinerary
        ],
        "total_cost": (
            result.total_cost
        ),
        "violations": list(
            result.violations
        ),
    }

    return row, detail


def _reference_row(
    benchmark: ProfessorBenchmark,
    *,
    exact_solution: ExactCIPPSolution
    | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    float | None,
    str | None,
]:
    if exact_solution is not None:
        row = {
            "instance": (
                benchmark.instance
                .instance_id
            ),
            "objective_variant": (
                benchmark
                .objective_variant
            ),
            "method": (
                f"Exact-{exact_solution.solver}"
            ),
            "objective": (
                exact_solution.objective
            ),
            "gap_to_reference_percent": (
                exact_solution
                .optimality_gap_percent
            ),
            "reference_kind": (
                "current_exact_run"
            ),
            "runtime_seconds": (
                exact_solution
                .runtime_seconds
            ),
            "feasible": (
                exact_solution.feasible
            ),
            "constraint_violations": len(
                exact_solution.violations
            ),
            "rollouts": 1,
            "budget_utilization_percent": (
                None
            ),
            "idle_days": None,
            "unique_states": None,
            "hhi": None,
        }

        detail = {
            **row,
            **exact_solution.to_dict(),
        }

        return (
            row,
            detail,
            exact_solution.objective,
            "current_exact_run",
        )

    published = (
        benchmark.table5_reference
    )

    if published is None:
        return (
            {},
            {},
            None,
            None,
        )

    row = {
        "instance": (
            benchmark.instance
            .instance_id
        ),
        "objective_variant": (
            benchmark.objective_variant
        ),
        "method": (
            "Gurobi-Table5"
        ),
        "objective": published.bfs,
        "gap_to_reference_percent": (
            published
            .optimality_gap_percent
        ),
        "reference_kind": (
            "published_table5"
        ),
        "runtime_seconds": (
            published.runtime_seconds
        ),
        "feasible": True,
        "constraint_violations": 0,
        "rollouts": 1,
        "budget_utilization_percent": (
            None
        ),
        "idle_days": None,
        "unique_states": None,
        "hhi": None,
    }

    detail = {
        **row,
        "note": (
            "Table 5 reports aggregate solution values, "
            "not the itinerary."
        ),
    }

    if (
        benchmark.objective_variant
        != "professor_code"
    ):
        # The supplied July 13 code and the paper equation differ.
        # Never compute a misleading gap across objective definitions.
        return (
            row,
            detail,
            None,
            "published_table5_incomparable_variant",
        )

    return (
        row,
        detail,
        published.bfs,
        "published_table5",
    )


def _format_value(
    value: Any,
    *,
    digits: int = 3,
) -> str:
    if value is None:
        return "NA"

    if isinstance(
        value,
        bool,
    ):
        return (
            "Yes"
            if value
            else "No"
        )

    if isinstance(
        value,
        float,
    ):
        return f"{value:.{digits}f}"

    return str(
        value
    )


def _write_markdown(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    columns = [
        "instance",
        "method",
        "objective",
        "gap_to_reference_percent",
        "runtime_seconds",
        "feasible",
        "constraint_violations",
        "rollouts",
    ]

    labels = [
        "Instance",
        "Method",
        "Objective",
        "Gap (%)",
        "CPU (s)",
        "Feasible",
        "Violations",
        "Rollouts",
    ]

    lines = [
        "| "
        + " | ".join(labels)
        + " |",
        "| "
        + " | ".join(
            ["---"] * len(labels)
        )
        + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _format_value(
                    row[column]
                )
                for column in columns
            )
            + " |"
        )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def _write_latex(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"Instance & Method & Objective & Gap (\%) "
            r"& CPU (s) & Feasible & Viol. & Rollouts \\"
        ),
        r"\midrule",
    ]

    for row in rows:
        method = str(
            row["method"]
        ).replace(
            "_",
            r"\_",
        )

        instance = str(
            row["instance"]
        ).replace(
            "_",
            r"\_",
        )

        lines.append(
            " & ".join(
                [
                    instance,
                    method,
                    _format_value(
                        row["objective"]
                    ),
                    _format_value(
                        row[
                            "gap_to_reference_percent"
                        ]
                    ),
                    _format_value(
                        row[
                            "runtime_seconds"
                        ]
                    ),
                    _format_value(
                        row["feasible"]
                    ),
                    _format_value(
                        row[
                            "constraint_violations"
                        ]
                    ),
                    _format_value(
                        row["rollouts"]
                    ),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen DQN/PPO checkpoints on the "
            "professor-provided real CIPP benchmarks."
        )
    )

    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--imitation-ppo-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dqn-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--democrat-excel",
        type=Path,
        default=Path(
            "CIPP-D.xls"
        ),
    )

    parser.add_argument(
        "--republican-excel",
        type=Path,
        default=Path(
            "CIPP-R.xls"
        ),
    )

    parser.add_argument(
        "--parties",
        choices=[
            "D",
            "R",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--number-of-states",
        type=int,
        choices=[
            14,
            30,
            51,
        ],
        default=14,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        choices=[
            30,
            45,
            60,
            75,
            90,
        ],
        default=30,
    )

    parser.add_argument(
        "--objective-variant",
        choices=[
            "paper_equation",
            "professor_code",
        ],
        default="professor_code",
    )

    parser.add_argument(
        "--rollouts",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--dqn-rollout-epsilon",
        type=float,
        default=0.10,
        help=(
            "Epsilon used only while constructing DQN Best-of-N candidates."
        ),
    )

    parser.add_argument(
        "--run-exact",
        action="store_true",
    )

    parser.add_argument(
        "--exact-solver",
        choices=[
            "auto",
            "gurobi",
            "scipy",
        ],
        default="auto",
    )

    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=3_600.0,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/week4/benchmark"
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default="cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    arguments = parser.parse_args()

    if (
        arguments.ppo_checkpoint is None
        and arguments
        .imitation_ppo_checkpoint
        is None
        and arguments.dqn_checkpoint
        is None
    ):
        raise ValueError(
            "provide at least one PPO, Imitation+PPO, or DQN checkpoint."
        )

    checkpoint_payloads: list[
        dict[str, Any]
    ] = []

    ppo_agent = None
    ppo_normalizer = None
    ppo_payload = None

    if arguments.ppo_checkpoint:
        ppo_agent, ppo_payload = (
            PPOAgent.from_checkpoint(
                arguments.ppo_checkpoint,
                device=arguments.device,
            )
        )

        _validate_checkpoint_variant(
            ppo_payload,
            arguments.objective_variant,
            "PPO checkpoint",
        )

        ppo_normalizer = (
            ObservationNormalizer.from_dict(
                ppo_payload[
                    "normalizer"
                ]
            )
        )

        checkpoint_payloads.append(
            ppo_payload
        )

    imitation_agent = None
    imitation_normalizer = None
    imitation_payload = None

    if (
        arguments
        .imitation_ppo_checkpoint
    ):
        imitation_agent, imitation_payload = (
            PPOAgent.from_checkpoint(
                arguments
                .imitation_ppo_checkpoint,
                device=arguments.device,
            )
        )

        _validate_checkpoint_variant(
            imitation_payload,
            arguments.objective_variant,
            "Imitation+PPO checkpoint",
        )

        imitation_normalizer = (
            ObservationNormalizer.from_dict(
                imitation_payload[
                    "normalizer"
                ]
            )
        )

        checkpoint_payloads.append(
            imitation_payload
        )

    dqn_agent = None
    dqn_normalizer = None
    dqn_payload = None

    if arguments.dqn_checkpoint:
        dqn_agent, dqn_payload = (
            DQNAgent.from_checkpoint(
                arguments.dqn_checkpoint,
                device=arguments.device,
            )
        )

        _validate_checkpoint_variant(
            dqn_payload,
            arguments.objective_variant,
            "DQN checkpoint",
        )

        dqn_normalizer = (
            ObservationNormalizer.from_dict(
                dqn_payload[
                    "normalizer"
                ]
            )
        )

        checkpoint_payloads.append(
            dqn_payload
        )

    parties = (
        ["D", "R"]
        if arguments.parties == "both"
        else [arguments.parties]
    )

    all_rows: list[
        dict[str, Any]
    ] = []

    all_details: list[
        dict[str, Any]
    ] = []

    for party_index, party in enumerate(
        parties
    ):
        excel_path = (
            arguments.democrat_excel
            if party == "D"
            else arguments.republican_excel
        )

        benchmark = (
            load_professor_benchmark(
                excel_path,
                party=party,
                number_of_states=(
                    arguments
                    .number_of_states
                ),
                horizon=(
                    arguments.horizon
                ),
                objective_variant=(
                    arguments
                    .objective_variant
                ),
            )
        )

        if ppo_payload is not None:
            _validate_checkpoint_instance(
                ppo_payload,
                benchmark
                .instance.instance_id,
                "PPO checkpoint",
            )

        if imitation_payload is not None:
            _validate_checkpoint_instance(
                imitation_payload,
                benchmark
                .instance.instance_id,
                "Imitation+PPO checkpoint",
            )

        if (
            dqn_agent is not None
            and dqn_normalizer
            is not None
        ):
            _validate_checkpoint_instance(
                dqn_payload,
                benchmark
                .instance.instance_id,
                "DQN checkpoint",
            )

        exact_solution = None

        if arguments.run_exact:
            exact_solution = (
                solve_cipp_exact(
                    benchmark.instance,
                    solver=(
                        arguments
                        .exact_solver
                    ),
                    time_limit_seconds=(
                        arguments
                        .exact_time_limit
                    ),
                    output_directory=(
                        arguments
                        .output_directory
                        / "exact_solver"
                    ),
                )
            )

        (
            reference_row,
            reference_detail,
            reference_objective,
            reference_kind,
        ) = _reference_row(
            benchmark,
            exact_solution=(
                exact_solution
            ),
        )

        policy_results: list[
            tuple[
                PolicyResult,
                int,
            ]
        ] = [
            (
                run_greedy_policy(
                    benchmark.instance
                ),
                1,
            ),
        ]

        if (
            dqn_agent is not None
            and dqn_normalizer is not None
        ):
            policy_results.insert(
                len(
                    policy_results
                ),
                (
                    evaluate_dqn_policy(
                        benchmark.instance,
                        dqn_agent,
                        dqn_normalizer,
                        method="DQN-Greedy-1",
                    ),
                    1,
                ),
            )

            policy_results.append(
                (
                    evaluate_dqn_best_of_rollouts(
                        benchmark.instance,
                        dqn_agent,
                        dqn_normalizer,
                        number_of_rollouts=(
                            arguments.rollouts
                        ),
                        exploration_epsilon=(
                            arguments
                            .dqn_rollout_epsilon
                        ),
                        seed=(
                            arguments.seed
                            + 250_000
                            + 10_000
                            * party_index
                        ),
                        method=(
                            "DQN-Best-of-"
                            f"{arguments.rollouts}"
                        ),
                    ),
                    arguments.rollouts,
                )
            )

        if (
            ppo_agent is not None
            and ppo_normalizer
            is not None
        ):
            policy_results.extend(
                [
                    (
                        evaluate_ppo_policy(
                            benchmark.instance,
                            ppo_agent,
                            ppo_normalizer,
                            deterministic=True,
                            method=(
                                "PPO-Greedy-1"
                            ),
                        ),
                        1,
                    ),
                    (
                        evaluate_ppo_best_of_rollouts(
                            benchmark.instance,
                            ppo_agent,
                            ppo_normalizer,
                            number_of_rollouts=(
                                arguments.rollouts
                            ),
                            seed=(
                                arguments.seed
                                + 10_000
                                * party_index
                            ),
                            method=(
                                "PPO-Best-of-"
                                f"{arguments.rollouts}"
                            ),
                        ),
                        arguments.rollouts,
                    ),
                ]
            )

        if (
            imitation_agent is not None
            and imitation_normalizer
            is not None
        ):
            policy_results.extend(
                [
                    (
                        evaluate_ppo_policy(
                            benchmark.instance,
                            imitation_agent,
                            imitation_normalizer,
                            deterministic=True,
                            method=(
                                "Imitation+PPO-Greedy-1"
                            ),
                        ),
                        1,
                    ),
                    (
                        evaluate_ppo_best_of_rollouts(
                            benchmark.instance,
                            imitation_agent,
                            imitation_normalizer,
                            number_of_rollouts=(
                                arguments.rollouts
                            ),
                            seed=(
                                arguments.seed
                                + 500_000
                                + 10_000
                                * party_index
                            ),
                            method=(
                                "Imitation+PPO-"
                                f"Best-of-{arguments.rollouts}"
                            ),
                        ),
                        arguments.rollouts,
                    ),
                ]
            )

        for result, rollouts in (
            policy_results
        ):
            row, detail = (
                _policy_row(
                    benchmark=benchmark,
                    result=result,
                    rollouts=rollouts,
                    reference_objective=(
                        reference_objective
                    ),
                    reference_kind=(
                        reference_kind
                    ),
                )
            )

            all_rows.append(
                row
            )

            all_details.append(
                detail
            )

        if reference_row:
            all_rows.append(
                reference_row
            )

            all_details.append(
                reference_detail
            )

    output_directory = (
        arguments.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / "comparison_table.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                all_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    markdown_path = (
        output_directory
        / "comparison_table.md"
    )

    _write_markdown(
        all_rows,
        markdown_path,
    )

    latex_path = (
        output_directory
        / "comparison_table.tex"
    )

    _write_latex(
        all_rows,
        latex_path,
    )

    details_path = (
        output_directory
        / "benchmark_details.json"
    )

    details_path.write_text(
        json.dumps(
            {
                "configuration": {
                    key: (
                        str(value)
                        if isinstance(
                            value,
                            Path,
                        )
                        else value
                    )
                    for key, value
                    in vars(
                        arguments
                    ).items()
                },
                "test_only_frozen_checkpoints": (
                    True
                ),
                "benchmark_instances_used_for_training": (
                    any(
                        payload.get(
                            "training_metadata",
                            {},
                        ).get(
                            "real_benchmark_used_for_training",
                            False,
                        )
                        for payload
                        in checkpoint_payloads
                    )
                ),
                "results": all_details,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Saved {csv_path}"
    )
    print(
        f"Saved {markdown_path}"
    )
    print(
        f"Saved {latex_path}"
    )
    print(
        f"Saved {details_path}"
    )


if __name__ == "__main__":
    main()
