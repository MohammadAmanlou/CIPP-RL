"""Train a feasibility-masked DQN directly on one professor CIPP instance."""

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
    DQNAgent,
    DQNConfig,
    ReplayBuffer,
)
from src.training import (
    evaluate_dqn_policy,
    fit_training_normalizer,
)
from src.utils import (
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


def _epsilon(
    episode: int,
    *,
    start: float,
    end: float,
    decay_episodes: int,
) -> float:
    if decay_episodes <= 1:
        return float(
            end
        )

    progress = min(
        1.0,
        max(
            0.0,
            (episode - 1)
            / (decay_episodes - 1),
        ),
    )

    return float(
        start
        + progress
        * (end - start)
    )


def train(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Run one reproducible instance-specific DQN experiment."""

    output_directory = Path(
        arguments.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
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

    instance = benchmark.instance
    device = _resolve_device(
        arguments.device
    )

    normalizer = fit_training_normalizer(
        [
            instance
        ],
        episodes_per_instance=(
            arguments.normalizer_episodes
        ),
        seed=(
            arguments.seed
            + 200_000
        ),
    )

    initial_observation, _ = CIPPEnv(
        instance
    ).reset()

    agent = DQNAgent(
        observation_dim=int(
            initial_observation.size
        ),
        action_dim=(
            instance.num_actions
        ),
        config=DQNConfig(
            hidden_dim=(
                arguments.hidden_dimension
            ),
            learning_rate=(
                arguments.learning_rate
            ),
            reward_scale=(
                arguments.reward_scale
            ),
            gamma=arguments.discount_factor,
            gradient_clip_norm=(
                arguments.gradient_clip_norm
            ),
            double_dqn=False,
        ),
        seed=arguments.seed,
        device=device,
    )

    replay = ReplayBuffer(
        capacity=(
            arguments.replay_capacity
        ),
        observation_dim=(
            agent.observation_dim
        ),
        action_dim=agent.action_dim,
        seed=arguments.seed,
    )

    viability_cache: dict[
        tuple[int, ...],
        np.ndarray,
    ] = {}

    greedy = run_greedy_policy(
        instance
    )

    random_reference = np.mean(
        [
            run_random_feasible_policy(
                instance,
                seed=(
                    arguments.seed
                    + 300_000
                    + index
                ),
            ).objective
            for index in range(
                arguments.reference_rollouts
            )
        ]
    )

    print(
        "validation_references "
        f"random_mean={random_reference:.3f} "
        f"greedy={greedy.objective:.3f}"
    )

    metadata = {
        "algorithm": "masked_dqn",
        "experiment_mode": (
            "professor_instance_specific"
        ),
        "objective_variant": (
            arguments.objective_variant
        ),
        "party": arguments.party,
        "professor_excel": str(
            arguments.professor_excel
        ),
        "training_instance_ids": [
            instance.instance_id
        ],
        "validation_instance_ids": [
            instance.instance_id
        ],
        "real_benchmark_used_for_training": (
            True
        ),
        "device": device,
        "reward_scale": float(
            arguments.reward_scale
        ),
        "validation_baselines": {
            "random_mean_objective": (
                float(
                    random_reference
                )
            ),
            "greedy_objective": (
                greedy.objective
            ),
        },
    }

    configuration_path = (
        output_directory
        / "experiment_config.json"
    )

    configuration = {
        **vars(arguments),
        "output_directory": str(
            output_directory
        ),
        "professor_excel": str(
            arguments.professor_excel
        ),
        "resolved_device": device,
    }

    configuration_path.write_text(
        json.dumps(
            configuration,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    best_checkpoint = (
        output_directory
        / "checkpoint_best.pt"
    )

    initial_result = (
        evaluate_dqn_policy(
            instance,
            agent,
            normalizer,
            method="dqn_validation",
            viability_cache=(
                viability_cache
            ),
        )
    )

    best_objective = (
        initial_result.objective
    )
    best_episode = 0

    agent.save_checkpoint(
        best_checkpoint,
        normalizer=(
            normalizer.to_dict()
        ),
        training_metadata={
            **metadata,
            "selected_episode": 0,
            "best_validation_objective": (
                best_objective
            ),
        },
    )

    print(
        "episode=0 "
        "initial_validation="
        f"{best_objective:.3f}"
    )

    history: list[
        dict[str, Any]
    ] = []
    total_steps = 0
    validations_without_improvement = 0
    stopped_early = False
    completed_episodes = 0
    decay_episodes = max(
        1,
        int(
            arguments.episodes
            * arguments.epsilon_decay_fraction
        ),
    )

    partial_history_path = (
        output_directory
        / "training_history.partial.json"
    )

    for episode in range(
        1,
        arguments.episodes + 1,
    ):
        epsilon = _epsilon(
            episode,
            start=(
                arguments.epsilon_start
            ),
            end=arguments.epsilon_end,
            decay_episodes=(
                decay_episodes
            ),
        )

        environment = CIPPEnv(
            instance,
            seed=(
                arguments.seed
                + episode
            ),
            viability_cache=(
                viability_cache
            ),
        )

        observation, info = (
            environment.reset(
                seed=(
                    arguments.seed
                    + episode
                )
            )
        )

        losses: list[float] = []

        while not environment.done:
            normalized = (
                normalizer.transform(
                    observation
                ).astype(
                    np.float32
                )
            )

            action_mask = np.asarray(
                info["action_mask"],
                dtype=np.bool_,
            )

            action = agent.select_action(
                normalized,
                action_mask,
                epsilon=epsilon,
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                next_info,
            ) = environment.step(
                action
            )

            done = bool(
                terminated
                or truncated
            )

            next_normalized = (
                normalizer.transform(
                    next_observation
                ).astype(
                    np.float32
                )
            )

            replay.add(
                state=normalized,
                action=action,
                reward=(
                    float(reward)
                    * agent.config
                    .reward_scale
                ),
                next_state=(
                    next_normalized
                ),
                done=done,
                action_mask=(
                    action_mask
                ),
                next_action_mask=(
                    next_info[
                        "action_mask"
                    ]
                ),
            )

            total_steps += 1

            if (
                len(replay)
                >= max(
                    arguments
                    .warmup_transitions,
                    arguments.batch_size,
                )
            ):
                for _ in range(
                    arguments
                    .updates_per_step
                ):
                    losses.append(
                        agent.optimize(
                            replay.sample(
                                arguments
                                .batch_size
                            )
                        )
                    )

            if (
                total_steps
                % arguments
                .target_sync_steps
                == 0
            ):
                agent.sync_target_network()

            observation = (
                next_observation
            )
            info = next_info

        final_evaluation = info[
            "final_evaluation"
        ]

        record: dict[str, Any] = {
            "episode": episode,
            "epsilon": epsilon,
            "training_objective": float(
                final_evaluation[
                    "objective"
                ]
            ),
            "feasible": bool(
                final_evaluation[
                    "feasible"
                ]
            ),
            "mean_loss": (
                float(
                    np.mean(
                        losses
                    )
                )
                if losses
                else None
            ),
            "replay_size": len(
                replay
            ),
            "total_steps": (
                total_steps
            ),
        }

        should_validate = (
            episode
            % arguments
            .validation_interval
            == 0
            or episode
            == arguments.episodes
        )

        if should_validate:
            validation = (
                evaluate_dqn_policy(
                    instance,
                    agent,
                    normalizer,
                    method=(
                        "dqn_validation"
                    ),
                    viability_cache=(
                        viability_cache
                    ),
                )
            )

            gap_to_greedy = (
                100.0
                * (
                    greedy.objective
                    - validation.objective
                )
                / abs(
                    greedy.objective
                )
            )

            record[
                "validation_objective"
            ] = validation.objective
            record[
                "gap_to_greedy_percent"
            ] = float(
                gap_to_greedy
            )

            if (
                validation.objective
                > (
                    best_objective
                    + arguments
                    .early_stopping_min_delta
                )
            ):
                best_objective = (
                    validation.objective
                )
                best_episode = episode
                validations_without_improvement = 0

                agent.save_checkpoint(
                    best_checkpoint,
                    normalizer=(
                        normalizer.to_dict()
                    ),
                    training_metadata={
                        **metadata,
                        "selected_episode": (
                            episode
                        ),
                        "best_validation_objective": (
                            best_objective
                        ),
                    },
                )
            else:
                validations_without_improvement += 1

        history.append(
            record
        )
        completed_episodes = episode

        print(
            f"episode={episode} "
            "train_objective="
            f"{record['training_objective']:.3f} "
            f"epsilon={epsilon:.3f} "
            "feasible="
            f"{int(record['feasible'])}"
            + (
                " validation="
                f"{record['validation_objective']:.3f}"
                " gap_to_greedy="
                f"{record['gap_to_greedy_percent']:.2f}%"
                if should_validate
                else ""
            )
            + (
                " loss="
                f"{record['mean_loss']:.5f}"
                if record[
                    "mean_loss"
                ]
                is not None
                else ""
            )
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
                and episode
                >= arguments.minimum_episodes
                and validations_without_improvement
                >= arguments
                .early_stopping_patience
            ):
                stopped_early = True
                print(
                    "early_stopping=1 "
                    f"best_episode={best_episode} "
                    f"best_validation="
                    f"{best_objective:.3f}"
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
            **metadata,
            "selected_episode": (
                completed_episodes
            ),
            "best_episode": (
                best_episode
            ),
            "best_validation_objective": (
                best_objective
            ),
            "stopped_early": (
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

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        [
            record["episode"]
            for record in history
        ],
        [
            record[
                "training_objective"
            ]
            for record in history
        ],
        label=(
            "Training epsilon-greedy"
        ),
        alpha=0.65,
    )

    validation_records = [
        record
        for record in history
        if "validation_objective"
        in record
    ]

    axis.plot(
        [
            record["episode"]
            for record
            in validation_records
        ],
        [
            record[
                "validation_objective"
            ]
            for record
            in validation_records
        ],
        label="DQN Greedy-1",
        marker="o",
    )

    axis.axhline(
        greedy.objective,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Greedy baseline",
    )

    axis.set_xlabel(
        "Training episode"
    )
    axis.set_ylabel(
        "Exact CIPP objective"
    )
    axis.set_title(
        f"Instance-specific DQN: "
        f"{instance.instance_id}"
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
                "completed_episodes": (
                    completed_episodes
                ),
                "best_episode": (
                    best_episode
                ),
                "stopped_early": (
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
        "best_validation_objective": (
            best_objective
        ),
        "best_episode": (
            best_episode
        ),
        "completed_episodes": (
            completed_episodes
        ),
        "stopped_early": (
            stopped_early
        ),
        "history_path": str(
            history_path
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train masked DQN directly on one professor-provided instance."
        )
    )

    parser.add_argument(
        "--professor-excel",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--party",
        choices=[
            "D",
            "R",
        ],
        required=True,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/week4_instance/dqn"
        ),
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
        "--episodes",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--normalizer-episodes",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--hidden-dimension",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--warmup-transitions",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--updates-per-step",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--target-sync-steps",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--epsilon-decay-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--reference-rollouts",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--minimum-episodes",
        type=int,
        default=100,
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
    result = train(
        _parser().parse_args()
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
