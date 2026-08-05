"""Train masked PPO on synthetic or professor-provided CIPP instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.baselines import (
    run_greedy_policy,
    run_random_feasible_policy,
)
from src.envs import CIPPEnv
from src.models import (
    PPOAgent,
    PPOConfig,
)
from src.training import (
    collect_ppo_rollouts,
    evaluate_ppo_policy,
    fit_training_normalizer,
)
from src.utils import (
    ObservationNormalizer,
    generate_paper_like_instance,
    load_professor_benchmark,
)


def _resolve_device(
    requested: str,
) -> str:
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return requested


def _validation_summary(
    *,
    instances,
    agent: PPOAgent,
    normalizer: ObservationNormalizer,
    viability_caches: dict[
        int,
        dict[
            tuple[int, ...],
            np.ndarray,
        ],
    ]
    | None = None,
) -> dict[str, float]:
    results = [
        evaluate_ppo_policy(
            instance,
            agent,
            normalizer,
            deterministic=True,
            method="ppo_validation_greedy",
            viability_cache=(
                viability_caches.get(
                    id(instance)
                )
                if viability_caches
                is not None
                else None
            ),
        )
        for instance in instances
    ]

    objectives = np.asarray(
        [
            result.objective
            for result in results
        ],
        dtype=np.float64,
    )

    feasible = np.asarray(
        [
            result.feasible
            for result in results
        ],
        dtype=np.float64,
    )

    return {
        "mean_objective": float(
            objectives.mean()
        ),
        "std_objective": float(
            objectives.std()
        ),
        "minimum_objective": float(
            objectives.min()
        ),
        "maximum_objective": float(
            objectives.max()
        ),
        "feasible_rate": float(
            feasible.mean()
        ),
    }


def _validation_baseline_summary(
    *,
    instances,
    seed: int,
) -> dict[str, float]:
    """Evaluate fixed non-learning references on validation once."""

    greedy_objectives = np.asarray(
        [
            run_greedy_policy(
                instance
            ).objective
            for instance in instances
        ],
        dtype=np.float64,
    )

    random_objectives = np.asarray(
        [
            run_random_feasible_policy(
                instance,
                seed=(
                    seed
                    + 300_000
                    + index
                ),
            ).objective
            for index, instance
            in enumerate(instances)
        ],
        dtype=np.float64,
    )

    return {
        "greedy_mean_objective": float(
            greedy_objectives.mean()
        ),
        "random_one_mean_objective": float(
            random_objectives.mean()
        ),
    }


def _ppo_config(
    arguments: argparse.Namespace,
    *,
    hidden_dimension: int | None = None,
) -> PPOConfig:
    """Build the effective PPO optimizer configuration from CLI arguments."""

    return PPOConfig(
        hidden_dim=(
            arguments.hidden_dimension
            if hidden_dimension is None
            else hidden_dimension
        ),
        learning_rate=(
            arguments.learning_rate
        ),
        reward_scale=(
            arguments.reward_scale
        ),
        discount_factor=(
            arguments.discount_factor
        ),
        gae_lambda=(
            arguments.gae_lambda
        ),
        clip_epsilon=(
            arguments.clip_epsilon
        ),
        value_loss_coefficient=(
            arguments
            .value_loss_coefficient
        ),
        entropy_coefficient=(
            arguments.entropy_coefficient
        ),
        gradient_clip_norm=(
            arguments.gradient_clip_norm
        ),
        update_epochs=(
            arguments.update_epochs
        ),
        minibatch_size=(
            arguments.minibatch_size
        ),
    )


def train(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Run one reproducible plain or imitation-initialized PPO experiment."""

    output_directory = Path(
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

        training_instances = [
            benchmark.instance
        ]

        validation_instances = [
            benchmark.instance
        ]

        training_seeds: list[int] = []
        validation_seeds: list[int] = []
        experiment_mode = (
            "professor_instance_specific"
        )
    else:
        training_seeds = [
            arguments.seed + index
            for index in range(
                arguments.training_instances
            )
        ]

        validation_seeds = [
            arguments.seed
            + 100_000
            + index
            for index in range(
                arguments.validation_instances
            )
        ]

        training_instances = [
            generate_paper_like_instance(
                seed=seed,
                number_of_states=(
                    arguments.number_of_states
                ),
                horizon=arguments.horizon,
                objective_variant=(
                    arguments.objective_variant
                ),
                instance_id=(
                    f"ppo-train-{seed}"
                ),
            )
            for seed in training_seeds
        ]

        validation_instances = [
            generate_paper_like_instance(
                seed=seed,
                number_of_states=(
                    arguments.number_of_states
                ),
                horizon=arguments.horizon,
                objective_variant=(
                    arguments.objective_variant
                ),
                instance_id=(
                    f"ppo-validation-{seed}"
                ),
            )
            for seed in validation_seeds
        ]

        experiment_mode = (
            "synthetic_generalization"
        )

    validation_baselines = (
        _validation_baseline_summary(
            instances=validation_instances,
            seed=arguments.seed,
        )
    )

    print(
        "validation_references "
        f"random_one_mean="
        f"{validation_baselines['random_one_mean_objective']:.3f} "
        f"greedy_mean="
        f"{validation_baselines['greedy_mean_objective']:.3f}"
    )

    device = _resolve_device(
        arguments.device
    )

    if arguments.initial_checkpoint:
        loaded_agent, checkpoint_payload = (
            PPOAgent.from_checkpoint(
                arguments.initial_checkpoint,
                device=device,
            )
        )

        normalizer = (
            ObservationNormalizer.from_dict(
                checkpoint_payload[
                    "normalizer"
                ]
            )
        )

        initialization = (
            "imitation_checkpoint"
        )

        checkpoint_metadata = (
            checkpoint_payload.get(
                "training_metadata",
                {},
            )
        )

        if arguments.professor_excel is not None:
            source_ids = set(
                checkpoint_metadata.get(
                    "teacher_instance_ids",
                    checkpoint_metadata.get(
                        "training_instance_ids",
                        [],
                    ),
                )
            )

            expected_id = (
                training_instances[
                    0
                ].instance_id
            )

            if (
                source_ids
                and expected_id
                not in source_ids
            ):
                raise ValueError(
                    "initial checkpoint was not trained on "
                    f"the requested instance {expected_id}."
                )

        checkpoint_hidden_dimension = int(
            checkpoint_payload[
                "config"
            ][
                "hidden_dim"
            ]
        )

        agent = PPOAgent(
            observation_dim=(
                loaded_agent
                .observation_dim
            ),
            action_dim=(
                loaded_agent.action_dim
            ),
            config=_ppo_config(
                arguments,
                hidden_dimension=(
                    checkpoint_hidden_dimension
                ),
            ),
            seed=arguments.seed,
            device=device,
        )

        agent.network.load_state_dict(
            loaded_agent
            .network.state_dict()
        )
    else:
        if arguments.professor_excel is not None:
            normalizer = (
                fit_training_normalizer(
                    training_instances,
                    episodes_per_instance=(
                        arguments
                        .normalizer_instances
                    ),
                    seed=(
                        arguments.seed
                        + 200_000
                    ),
                )
            )
        else:
            normalizer_instance_count = min(
                arguments.normalizer_instances,
                len(training_instances),
            )

            normalizer = (
                fit_training_normalizer(
                    training_instances[
                        :normalizer_instance_count
                    ],
                    episodes_per_instance=1,
                    seed=(
                        arguments.seed
                        + 200_000
                    ),
                )
            )

        first_observation, _ = CIPPEnv(
            training_instances[0]
        ).reset()

        agent = PPOAgent(
            observation_dim=int(
                first_observation.size
            ),
            action_dim=(
                training_instances[
                    0
                ].num_actions
            ),
            config=_ppo_config(
                arguments
            ),
            seed=arguments.seed,
            device=device,
        )

        initialization = "random"

    if (
        agent.action_dim
        != training_instances[
            0
        ].num_actions
    ):
        raise ValueError(
            "checkpoint action dimension does not match the training class."
        )

    history: list[dict[str, Any]] = []
    best_validation = -np.inf
    best_update = 0
    validations_without_improvement = 0
    completed_updates = 0
    stopped_early = False
    best_checkpoint = (
        output_directory
        / "checkpoint_best.pt"
    )

    base_metadata = {
        "algorithm": "PPO",
        "initialization": initialization,
        "initial_checkpoint": (
            str(
                arguments.initial_checkpoint
            )
            if arguments.initial_checkpoint
            else None
        ),
        "number_of_states": (
            arguments.number_of_states
        ),
        "horizon": arguments.horizon,
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
        "training_instance_ids": [
            instance.instance_id
            for instance
            in training_instances
        ],
        "validation_instance_ids": [
            instance.instance_id
            for instance
            in validation_instances
        ],
        "training_seeds": (
            training_seeds
        ),
        "validation_seeds": (
            validation_seeds
        ),
        "real_benchmark_used_for_training": (
            arguments
            .professor_excel
            is not None
        ),
        "device": device,
        "reward_scale": float(
            agent.config.reward_scale
        ),
        "validation_baselines": (
            validation_baselines
        ),
    }

    configuration = {
        **vars(arguments),
        "output_directory": str(
            output_directory
        ),
        "resolved_device": device,
    }

    configuration_path = (
        output_directory
        / "experiment_config.json"
    )

    configuration_path.write_text(
        json.dumps(
            configuration,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    partial_history_path = (
        output_directory
        / "training_history.partial.json"
    )

    viability_caches: dict[
        int,
        dict[
            tuple[int, ...],
            np.ndarray,
        ],
    ] = {
        id(instance): {}
        for instance in (
            training_instances
            + validation_instances
        )
    }

    initial_validation = (
        _validation_summary(
            instances=(
                validation_instances
            ),
            agent=agent,
            normalizer=normalizer,
            viability_caches=(
                viability_caches
            ),
        )
    )

    initial_validation[
        "gap_to_greedy_percent"
    ] = float(
        100.0
        * (
            validation_baselines[
                "greedy_mean_objective"
            ]
            - initial_validation[
                "mean_objective"
            ]
        )
        / abs(
            validation_baselines[
                "greedy_mean_objective"
            ]
        )
    )

    best_validation = (
        initial_validation[
            "mean_objective"
        ]
    )

    agent.save_checkpoint(
        best_checkpoint,
        normalizer=(
            normalizer.to_dict()
        ),
        training_metadata={
            **base_metadata,
            "selected_update": 0,
            "best_validation": (
                initial_validation
            ),
        },
    )

    print(
        "update=0 "
        "initial_validation_mean="
        f"{best_validation:.3f} "
        "gap_to_greedy="
        f"{initial_validation['gap_to_greedy_percent']:.2f}%"
    )

    for update_index in range(
        1,
        arguments.updates + 1,
    ):
        batch, rollout_summary = (
            collect_ppo_rollouts(
                agent,
                normalizer=normalizer,
                instances=(
                    training_instances
                ),
                number_of_episodes=(
                    arguments
                    .episodes_per_update
                ),
                seed=(
                    arguments.seed
                    + 1_000_000
                    + update_index
                    * arguments
                    .episodes_per_update
                ),
                viability_caches=(
                    viability_caches
                ),
            )
        )

        update_metrics = (
            agent.update(
                batch
            )
        )

        record: dict[str, Any] = {
            "update": update_index,
            "rollout": rollout_summary,
            "optimization": update_metrics,
        }

        should_validate = (
            update_index
            % arguments.validation_interval
            == 0
            or update_index
            == arguments.updates
        )

        if should_validate:
            validation = (
                _validation_summary(
                    instances=(
                        validation_instances
                    ),
                    agent=agent,
                    normalizer=normalizer,
                    viability_caches=(
                        viability_caches
                    ),
                )
            )

            record[
                "validation"
            ] = validation

            validation[
                "gap_to_greedy_percent"
            ] = float(
                100.0
                * (
                    validation_baselines[
                        "greedy_mean_objective"
                    ]
                    - validation[
                        "mean_objective"
                    ]
                )
                / abs(
                    validation_baselines[
                        "greedy_mean_objective"
                    ]
                )
            )

            if (
                validation[
                    "mean_objective"
                ]
                > (
                    best_validation
                    + arguments
                    .early_stopping_min_delta
                )
            ):
                best_validation = (
                    validation[
                        "mean_objective"
                    ]
                )
                best_update = update_index
                validations_without_improvement = 0

                agent.save_checkpoint(
                    best_checkpoint,
                    normalizer=(
                        normalizer.to_dict()
                    ),
                    training_metadata={
                        **base_metadata,
                        "selected_update": (
                            update_index
                        ),
                        "best_validation": (
                            validation
                        ),
                    },
                )
            else:
                validations_without_improvement += 1

        history.append(
            record
        )
        completed_updates = update_index

        print(
            f"update={update_index} "
            f"train_mean="
            f"{rollout_summary['mean_episode_objective']:.3f} "
            f"feasible="
            f"{rollout_summary['feasible_rate']:.3f}"
            + (
                " validation_mean="
                f"{record['validation']['mean_objective']:.3f}"
                " gap_to_greedy="
                f"{record['validation']['gap_to_greedy_percent']:.2f}%"
                if "validation" in record
                else ""
            )
            + " kl="
            f"{update_metrics['approximate_kl']:.5f}"
            + " clip_fraction="
            f"{update_metrics['clip_fraction']:.3f}"
            + " entropy="
            f"{update_metrics['entropy']:.3f}"
        )

        if should_validate:
            partial_history_path.write_text(
                json.dumps(
                    history,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            if (
                arguments
                .early_stopping_patience
                > 0
                and update_index
                >= arguments.minimum_updates
                and validations_without_improvement
                >= arguments
                .early_stopping_patience
            ):
                stopped_early = True
                print(
                    "early_stopping=1 "
                    f"best_update={best_update} "
                    f"best_validation="
                    f"{best_validation:.3f}"
                )
                break

    last_checkpoint = (
        output_directory
        / "checkpoint_last.pt"
    )

    agent.save_checkpoint(
        last_checkpoint,
        normalizer=(
            normalizer.to_dict()
        ),
        training_metadata={
            **base_metadata,
            "selected_update": (
                completed_updates
            ),
            "best_validation_mean": float(
                best_validation
            ),
            "best_update": int(
                best_update
            ),
            "stopped_early": bool(
                stopped_early
            ),
        },
    )

    history_path = (
        output_directory
        / "training_history.json"
    )

    history_path.write_text(
        json.dumps(
            history,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    validation_points = [
        (
            record["update"],
            record[
                "validation"
            ]["mean_objective"],
        )
        for record in history
        if "validation" in record
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        [
            record["update"]
            for record in history
        ],
        [
            record[
                "rollout"
            ][
                "mean_episode_objective"
            ]
            for record in history
        ],
        label="Training rollout mean",
        alpha=0.75,
    )

    axis.plot(
        [
            point[0]
            for point in validation_points
        ],
        [
            point[1]
            for point in validation_points
        ],
        label="Fixed validation greedy mean",
        marker="o",
    )

    axis.set_xlabel(
        "PPO update"
    )

    axis.set_ylabel(
        "Exact CIPP objective"
    )

    axis.set_title(
        "Masked PPO learning curve"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "learning_curve.png",
        dpi=180,
    )

    plt.close(
        figure
    )

    configuration_path.write_text(
        json.dumps(
            {
                **configuration,
                "completed_updates": int(
                    completed_updates
                ),
                "best_update": int(
                    best_update
                ),
                "stopped_early": bool(
                    stopped_early
                ),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "best_checkpoint": str(
            best_checkpoint
        ),
        "last_checkpoint": str(
            last_checkpoint
        ),
        "best_validation_mean": float(
            best_validation
        ),
        "best_update": int(
            best_update
        ),
        "completed_updates": int(
            completed_updates
        ),
        "stopped_early": bool(
            stopped_early
        ),
        "validation_baselines": (
            validation_baselines
        ),
        "history_path": str(
            history_path
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train feasibility-masked PPO on synthetic instances "
            "or directly optimize one professor-provided instance."
        )
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/week4/ppo"
        ),
    )

    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--professor-excel",
        type=Path,
        default=None,
        help=(
            "When provided, train and select checkpoints directly on "
            "this professor instance instead of synthetic instances."
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
        "--updates",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--training-instances",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--validation-instances",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--normalizer-instances",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--validation-interval",
        type=int,
        default=10,
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
        "--hidden-dimension",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1e-3,
        help=(
            "Scale environment rewards used by the critic and GAE. "
            "Exact reported CIPP objectives remain unscaled."
        ),
    )

    parser.add_argument(
        "--discount-factor",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--value-loss-coefficient",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many validation checks without improvement; "
            "zero disables early stopping."
        ),
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--minimum-updates",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser


def main() -> None:
    arguments = (
        _parser().parse_args()
    )

    result = train(
        arguments
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
