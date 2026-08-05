"""Tests for masked PPO, imitation learning, and exact teachers."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from src.baselines import (
    run_greedy_policy,
)
from src.core import (
    CIPPInstance,
    evaluate_itinerary,
)
from src.envs import CIPPEnv
from src.models import (
    DQNAgent,
    DQNConfig,
    PPOAgent,
    PPOConfig,
    build_imitation_dataset,
    compute_gae,
    masked_categorical,
    pretrain_policy_by_imitation,
)
from src.optimization import (
    solve_cipp_scipy,
)
from src.training import (
    collect_ppo_rollouts,
    evaluate_dqn_best_of_rollouts,
    fit_training_normalizer,
)
from src.utils import (
    ObservationNormalizer,
    generate_paper_like_instance,
    load_professor_benchmark,
)


def _small_instance(
    seed: int = 1,
) -> CIPPInstance:
    return generate_paper_like_instance(
        seed=seed,
        number_of_states=4,
        horizon=8,
        objective_variant=(
            "paper_equation"
        ),
    )


def test_masked_categorical_assigns_zero_probability() -> None:
    logits = torch.tensor(
        [
            [
                1.0,
                100.0,
                2.0,
            ]
        ]
    )

    mask = torch.tensor(
        [
            [
                True,
                False,
                True,
            ]
        ]
    )

    distribution = (
        masked_categorical(
            logits,
            mask,
        )
    )

    assert distribution.probs[
        0,
        1,
    ].item() == pytest.approx(
        0.0
    )

    assert distribution.probs[
        0
    ].sum().item() == pytest.approx(
        1.0
    )


def test_ppo_never_selects_a_masked_action() -> None:
    agent = PPOAgent(
        observation_dim=5,
        action_dim=4,
        config=PPOConfig(
            hidden_dim=16,
        ),
        seed=3,
    )

    observation = np.zeros(
        5,
        dtype=np.float32,
    )

    mask = np.asarray(
        [
            False,
            True,
            False,
            True,
        ]
    )

    for deterministic in (
        False,
        True,
    ):
        for _ in range(50):
            action, _, _ = (
                agent.select_action(
                    observation,
                    mask,
                    deterministic=(
                        deterministic
                    ),
                )
            )

            assert action in {
                1,
                3,
            }


def test_dqn_best_of_rollouts_is_feasible() -> None:
    instance = _small_instance(
        17
    )

    normalizer = (
        fit_training_normalizer(
            [
                instance
            ],
            episodes_per_instance=2,
            seed=18,
        )
    )

    observation, _ = CIPPEnv(
        instance
    ).reset()

    agent = DQNAgent(
        observation_dim=int(
            observation.size
        ),
        action_dim=(
            instance.num_actions
        ),
        config=DQNConfig(
            hidden_dim=16,
        ),
        seed=19,
    )

    result = (
        evaluate_dqn_best_of_rollouts(
            instance,
            agent,
            normalizer,
            number_of_rollouts=3,
            exploration_epsilon=0.2,
            seed=20,
        )
    )

    assert result.feasible is True
    assert len(
        result.itinerary
    ) == instance.H


def test_gae_resets_at_episode_boundaries() -> None:
    advantages, returns = (
        compute_gae(
            np.asarray(
                [
                    1.0,
                    2.0,
                    10.0,
                ]
            ),
            np.zeros(3),
            np.asarray(
                [
                    False,
                    True,
                    True,
                ]
            ),
            discount_factor=1.0,
            gae_lambda=1.0,
        )
    )

    np.testing.assert_allclose(
        advantages,
        np.asarray(
            [
                3.0,
                2.0,
                10.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        returns,
        advantages,
    )


def test_rollout_update_is_finite_and_feasible() -> None:
    instances = [
        _small_instance(1),
        _small_instance(2),
    ]

    normalizer = (
        fit_training_normalizer(
            instances,
            seed=10,
        )
    )

    observation, _ = CIPPEnv(
        instances[0]
    ).reset()

    agent = PPOAgent(
        observation_dim=int(
            observation.size
        ),
        action_dim=(
            instances[0].num_actions
        ),
        config=PPOConfig(
            hidden_dim=32,
            update_epochs=2,
            minibatch_size=8,
        ),
        seed=11,
    )

    batch, summary = (
        collect_ppo_rollouts(
            agent,
            normalizer=normalizer,
            instances=instances,
            number_of_episodes=3,
            seed=12,
        )
    )

    metrics = agent.update(
        batch
    )

    assert summary[
        "feasible_rate"
    ] == pytest.approx(
        1.0
    )

    assert batch.size == 24

    assert all(
        np.isfinite(value)
        for value in metrics.values()
    )


def test_rollout_reward_scaling_preserves_reported_objective() -> None:
    instance = _small_instance(
        4
    )

    normalizer = (
        fit_training_normalizer(
            [instance],
            seed=9,
        )
    )

    observation, _ = CIPPEnv(
        instance
    ).reset()

    reward_scale = 1e-3

    agent = PPOAgent(
        observation_dim=int(
            observation.size
        ),
        action_dim=(
            instance.num_actions
        ),
        config=PPOConfig(
            hidden_dim=16,
            reward_scale=reward_scale,
            discount_factor=1.0,
            gae_lambda=1.0,
            update_epochs=1,
            minibatch_size=8,
        ),
        seed=10,
    )

    with torch.no_grad():
        for parameter in (
            agent.network.parameters()
        ):
            parameter.zero_()

    batch, summary = (
        collect_ppo_rollouts(
            agent,
            normalizer=normalizer,
            instances=[instance],
            number_of_episodes=1,
            seed=11,
        )
    )

    assert summary[
        "reward_scale"
    ] == pytest.approx(
        reward_scale
    )

    assert batch.returns[
        0
    ] == pytest.approx(
        summary[
            "mean_episode_objective"
        ]
        * reward_scale,
        rel=1e-5,
    )


def test_imitation_pretraining_and_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    instance = _small_instance(
        5
    )

    normalizer = (
        fit_training_normalizer(
            [instance],
            seed=6,
        )
    )

    teacher = run_greedy_policy(
        instance
    )

    dataset = build_imitation_dataset(
        [
            (
                instance,
                teacher.itinerary,
            )
        ],
        normalizer=normalizer,
    )

    observation, _ = CIPPEnv(
        instance
    ).reset()

    agent = PPOAgent(
        observation_dim=int(
            observation.size
        ),
        action_dim=(
            instance.num_actions
        ),
        config=PPOConfig(
            hidden_dim=32,
        ),
        seed=7,
    )

    metrics = (
        pretrain_policy_by_imitation(
            agent,
            dataset,
            epochs=20,
            batch_size=8,
            learning_rate=1e-3,
            seed=8,
        )
    )

    assert metrics[
        "number_of_examples"
    ] == pytest.approx(
        instance.H
    )

    assert 0.0 <= metrics[
        "masked_action_accuracy"
    ] <= 1.0

    path = agent.save_checkpoint(
        tmp_path / "ppo.pt",
        normalizer=(
            normalizer.to_dict()
        ),
        training_metadata={
            "objective_variant": (
                "paper_equation"
            )
        },
    )

    restored, payload = (
        PPOAgent.from_checkpoint(
            path
        )
    )

    assert (
        restored.observation_dim
        == agent.observation_dim
    )

    assert (
        payload[
            "training_metadata"
        ][
            "objective_variant"
        ]
        == "paper_equation"
    )


def test_scipy_teacher_matches_brute_force() -> None:
    instance = CIPPInstance(
        n=2,
        H=4,
        rewards=[
            5.0,
            3.0,
        ],
        costs=[
            1.0,
            1.0,
        ],
        budget=4.0,
        alpha=2,
        idle_requirements=[
            0,
            0,
            0,
        ],
        q=2,
        w=1,
        temporal_weights=[
            2.0,
            1.0,
            1.0,
            2.0,
        ],
        gamma=0.1,
        instance_id=(
            "tiny-exact"
        ),
    )

    feasible_results = [
        evaluate_itinerary(
            instance,
            itinerary,
        )
        for itinerary in product(
            range(
                instance.num_actions
            ),
            repeat=instance.H,
        )
    ]

    expected = max(
        result.objective
        for result in feasible_results
        if result.feasible
    )

    solution = solve_cipp_scipy(
        instance,
        time_limit_seconds=10.0,
    )

    assert solution.proven_optimal
    assert solution.feasible
    assert solution.objective == pytest.approx(
        expected
    )


def test_professor_loader_keeps_variants_explicit() -> None:
    workbook = Path(
        "CIPP-D.xls"
    )

    if not workbook.exists():
        pytest.skip(
            "Professor workbook not included."
        )

    paper = load_professor_benchmark(
        workbook,
        party="D",
        number_of_states=14,
        horizon=30,
        objective_variant=(
            "paper_equation"
        ),
        budget_mode="disabled",
    )

    legacy = load_professor_benchmark(
        workbook,
        party="D",
        number_of_states=14,
        horizon=30,
        objective_variant=(
            "professor_code"
        ),
    )

    assert paper.instance.n == 14
    assert paper.instance.num_actions == 15
    assert paper.location_names[0] == (
        "Alabama"
    )

    assert paper.instance.repeat_factor(
        1
    ) == pytest.approx(
        0.96
    )

    assert legacy.instance.repeat_factor(
        1
    ) == pytest.approx(
        1.0
    )

    assert legacy.table5_reference is not None
    assert (
        legacy.table5_reference.bfs
        == pytest.approx(
            15_611.6
        )
    )
