"""Generate an exact synthetic or professor-instance teacher for PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.envs import CIPPEnv
from src.models import (
    PPOAgent,
    PPOConfig,
    build_imitation_dataset,
    pretrain_policy_by_imitation,
)
from src.optimization import (
    solve_cipp_exact,
)
from src.training import (
    evaluate_ppo_policy,
    fit_training_normalizer,
)
from src.utils import (
    generate_paper_like_instance,
    load_professor_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve CIPP instances exactly, convert solutions to "
            "state-action pairs, and pre-train PPO."
        )
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/week4/imitation"
        ),
    )

    parser.add_argument(
        "--professor-excel",
        type=Path,
        default=None,
        help=(
            "Use one real professor instance as the exact teacher."
        ),
    )

    parser.add_argument(
        "--party",
        choices=[
            "D",
            "R",
        ],
        default=None,
    )

    parser.add_argument(
        "--teacher-instances",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--normalizer-instances",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--number-of-states",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--horizon",
        type=int,
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
        "--solver",
        choices=[
            "auto",
            "gurobi",
            "scipy",
        ],
        default="auto",
    )

    parser.add_argument(
        "--teacher-time-limit",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--allow-nonoptimal-teachers",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--hidden-dimension",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default="cpu",
    )

    arguments = parser.parse_args()

    output_directory = (
        arguments.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if arguments.professor_excel is not None:
        if arguments.party is None:
            raise ValueError(
                "--party is required when --professor-excel is used."
            )

        benchmark = load_professor_benchmark(
            arguments.professor_excel,
            party=arguments.party,
            number_of_states=(
                arguments.number_of_states
            ),
            horizon=arguments.horizon,
            objective_variant=(
                arguments.objective_variant
            ),
        )

        normalizer_instances = [
            benchmark.instance
        ]

        teacher_instances = [
            benchmark.instance
        ]

        normalizer_episodes = (
            arguments.normalizer_instances
        )
        experiment_mode = (
            "professor_instance_specific"
        )
    else:
        normalizer_instances = [
            generate_paper_like_instance(
                seed=(
                    arguments.seed + index
                ),
                number_of_states=(
                    arguments.number_of_states
                ),
                horizon=arguments.horizon,
                objective_variant=(
                    arguments.objective_variant
                ),
                instance_id=(
                    f"imitation-normalizer-"
                    f"{arguments.seed + index}"
                ),
            )
            for index in range(
                arguments.normalizer_instances
            )
        ]

        teacher_instances = [
            generate_paper_like_instance(
                seed=(
                    arguments.seed
                    + 10_000
                    + index
                ),
                number_of_states=(
                    arguments.number_of_states
                ),
                horizon=arguments.horizon,
                objective_variant=(
                    arguments.objective_variant
                ),
                instance_id=(
                    f"imitation-teacher-"
                    f"{arguments.seed + 10_000 + index}"
                ),
            )
            for index in range(
                arguments.teacher_instances
            )
        ]

        normalizer_episodes = 1
        experiment_mode = (
            "synthetic_generalization"
        )

    normalizer = (
        fit_training_normalizer(
            normalizer_instances,
            episodes_per_instance=(
                normalizer_episodes
            ),
            seed=(
                arguments.seed
                + 200_000
            ),
        )
    )

    solutions = []
    demonstrations = []

    exact_output_directory = (
        output_directory
        / "exact_solver"
    )

    for index, instance in enumerate(
        teacher_instances,
        start=1,
    ):
        solution = solve_cipp_exact(
            instance,
            solver=arguments.solver,
            time_limit_seconds=(
                arguments
                .teacher_time_limit
            ),
            output_directory=(
                exact_output_directory
            ),
        )

        if not solution.feasible:
            raise RuntimeError(
                "exact teacher returned an infeasible itinerary."
            )

        if (
            not solution.proven_optimal
            and not arguments
            .allow_nonoptimal_teachers
        ):
            raise RuntimeError(
                "teacher solution was not proven optimal. "
                "Increase --teacher-time-limit or explicitly "
                "use --allow-nonoptimal-teachers."
            )

        solutions.append(
            solution
        )

        demonstrations.append(
            (
                instance,
                solution.itinerary,
            )
        )

        print(
            f"teacher={index}/"
            f"{len(teacher_instances)} "
            f"solver={solution.solver} "
            f"objective={solution.objective:.3f} "
            f"gap={solution.optimality_gap_percent:.4f}%"
        )

    dataset = build_imitation_dataset(
        demonstrations,
        normalizer=normalizer,
    )

    observation, _ = CIPPEnv(
        teacher_instances[0]
    ).reset()

    agent = PPOAgent(
        observation_dim=int(
            observation.size
        ),
        action_dim=(
            teacher_instances[
                0
            ].num_actions
        ),
        config=PPOConfig(
            hidden_dim=(
                arguments.hidden_dimension
            ),
            learning_rate=(
                arguments.learning_rate
            ),
        ),
        seed=arguments.seed,
        device=arguments.device,
    )

    imitation_metrics = (
        pretrain_policy_by_imitation(
            agent,
            dataset,
            epochs=arguments.epochs,
            batch_size=(
                arguments.batch_size
            ),
            learning_rate=(
                arguments.learning_rate
            ),
            seed=arguments.seed,
        )
    )

    teacher_policy_results = [
        evaluate_ppo_policy(
            instance,
            agent,
            normalizer,
            deterministic=True,
            method=(
                "imitation_greedy"
            ),
        ).to_dict()
        for instance in teacher_instances
    ]

    for teacher_result in (
        teacher_policy_results
    ):
        print(
            "imitation_evaluation "
            f"instance="
            f"{teacher_result['instance_id']} "
            f"objective="
            f"{teacher_result['objective']:.3f} "
            f"feasible="
            f"{int(teacher_result['feasible'])}"
        )

    metadata = {
        "algorithm": (
            "masked_cross_entropy_imitation"
        ),
        "objective_variant": (
            arguments.objective_variant
        ),
        "experiment_mode": (
            experiment_mode
        ),
        "party": arguments.party,
        "professor_excel": (
            str(
                arguments.professor_excel
            )
            if arguments
            .professor_excel
            is not None
            else None
        ),
        "teacher_solver_requested": (
            arguments.solver
        ),
        "teacher_instance_ids": [
            instance.instance_id
            for instance
            in teacher_instances
        ],
        "normalizer_instance_ids": [
            instance.instance_id
            for instance
            in normalizer_instances
        ],
        "real_benchmark_used_for_training": (
            arguments
            .professor_excel
            is not None
        ),
        "imitation_metrics": (
            imitation_metrics
        ),
    }

    checkpoint_path = (
        output_directory
        / "imitation_initialization.pt"
    )

    agent.save_checkpoint(
        checkpoint_path,
        normalizer=(
            normalizer.to_dict()
        ),
        training_metadata=metadata,
    )

    result_payload = {
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
            in vars(arguments).items()
        },
        "imitation_metrics": (
            imitation_metrics
        ),
        "teacher_solutions": [
            solution.to_dict()
            for solution in solutions
        ],
        "teacher_policy_results": (
            teacher_policy_results
        ),
        "checkpoint": str(
            checkpoint_path
        ),
    }

    (
        output_directory
        / "imitation_results.json"
    ).write_text(
        json.dumps(
            result_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "checkpoint": str(
                    checkpoint_path
                ),
                "metrics": (
                    imitation_metrics
                ),
                "teacher_policy_results": (
                    teacher_policy_results
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
