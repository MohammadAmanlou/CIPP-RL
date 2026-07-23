"""Tests for normalization, replay buffer, and masked DQN."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs import (
    CIPPEnv,
)

from src.models import (
    DQNAgent,
    DQNConfig,
    ReplayBuffer,
)

from src.utils import (
    ObservationNormalizer,
    generate_cipp_instance,
)


def test_observation_normalizer_round_trip_payload() -> None:
    values = np.array(
        [
            [1.0, 2.0, 5.0],
            [3.0, 2.0, 9.0],
        ]
    )

    normalizer = (
        ObservationNormalizer.fit(
            values
        )
    )

    restored = (
        ObservationNormalizer.from_dict(
            normalizer.to_dict()
        )
    )

    transformed = restored.transform(
        values
    )

    assert transformed.shape == (
        values.shape
    )

    assert transformed[
        :, 0
    ].mean() == pytest.approx(
        0.0
    )

    assert np.all(
        np.isfinite(transformed)
    )


def test_replay_buffer_stores_masks() -> None:
    buffer = ReplayBuffer(
        capacity=10,
        observation_dim=3,
        action_dim=4,
        seed=1,
    )

    current_mask = np.array(
        [
            True,
            False,
            True,
            False,
        ]
    )

    next_mask = np.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    buffer.add(
        state=np.array(
            [1.0, 2.0, 3.0]
        ),
        action=2,
        reward=4.0,
        next_state=np.array(
            [2.0, 3.0, 4.0]
        ),
        done=False,
        action_mask=current_mask,
        next_action_mask=next_mask,
    )

    batch = buffer.sample(
        1
    )

    assert batch.action_masks.shape == (
        1,
        4,
    )

    assert (
        batch.next_action_masks.shape
        == (1, 4)
    )

    assert bool(
        batch.action_masks[
            0,
            batch.actions[0],
        ]
    )


def test_dqn_action_selection_never_uses_masked_action() -> None:
    agent = DQNAgent(
        observation_dim=5,
        action_dim=4,
        config=DQNConfig(
            hidden_dim=16
        ),
        seed=5,
    )

    observation = np.zeros(
        5,
        dtype=np.float32,
    )

    mask = np.array(
        [
            False,
            True,
            False,
            True,
        ]
    )

    # epsilon=0 tests exploitation.
    # epsilon=1 tests exploration.
    for epsilon in (
        0.0,
        1.0,
    ):
        for _ in range(50):
            action = (
                agent.select_action(
                    observation,
                    mask,
                    epsilon=epsilon,
                )
            )

            assert action in {
                1,
                3,
            }


def test_dqn_optimization_handles_terminal_empty_next_mask() -> None:
    agent = DQNAgent(
        observation_dim=3,
        action_dim=2,
        config=DQNConfig(
            hidden_dim=16
        ),
        seed=2,
    )

    buffer = ReplayBuffer(
        capacity=4,
        observation_dim=3,
        action_dim=2,
        seed=2,
    )

    buffer.add(
        state=np.zeros(3),
        action=0,
        reward=1.0,
        next_state=np.ones(3),
        done=True,
        action_mask=np.array(
            [True, True]
        ),
        next_action_mask=np.array(
            [False, False]
        ),
    )

    loss = agent.optimize(
        buffer.sample(1)
    )

    assert np.isfinite(
        loss
    )


def test_untrained_masked_dqn_rollout_remains_feasible() -> None:
    instance = generate_cipp_instance(
        n=6,
        H=12,
        alpha=5,
        q=4,
        w=2,
        seed=91,
    )

    environment = CIPPEnv(
        instance
    )

    observation, info = (
        environment.reset()
    )

    normalizer = ObservationNormalizer.fit(
        np.stack(
            [
                observation,
                observation + 1.0,
            ]
        )
    )

    agent = DQNAgent(
        observation_dim=(
            observation.size
        ),
        action_dim=(
            instance.num_actions
        ),
        config=DQNConfig(
            hidden_dim=32
        ),
        seed=9,
    )

    while not environment.done:
        normalized = (
            normalizer.transform(
                observation
            ).astype(
                np.float32
            )
        )

        action = agent.select_action(
            normalized,
            info["action_mask"],
            epsilon=0.5,
        )

        (
            observation,
            _,
            _,
            _,
            info,
        ) = environment.step(
            action
        )

    assert info[
        "final_evaluation"
    ]["feasible"] is True