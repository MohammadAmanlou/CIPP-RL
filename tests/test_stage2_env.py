"""Week 2 tests for the deterministic CIPP environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.core import (
    CIPPInstance,
    evaluate_itinerary,
)

from src.envs import (
    CIPPEnv,
    all_idle_completion,
    get_viability_mask,
)

from src.utils import (
    generate_cipp_instance,
)


def make_instance(
    **overrides,
) -> CIPPInstance:
    """Create a small deterministic test instance."""

    values = {
        "n": 3,
        "H": 5,
        "rewards": [
            100.0,
            80.0,
            60.0,
        ],
        "costs": [
            10.0,
            20.0,
            5.0,
        ],
        "budget": 60.0,
        "alpha": 3,
        "idle_requirements": [
            1,
            1,
            1,
        ],
        "q": 3,
        "w": 2,
        "temporal_weights": [
            1.0,
            0.8,
            1.2,
            1.0,
            1.1,
        ],
        "gamma": 0.1,
        "instance_id": "week2-test",
    }

    values.update(
        overrides
    )

    return CIPPInstance(
        **values
    )


def test_reset_returns_state_and_mask() -> None:
    env = CIPPEnv(
        make_instance(),
        seed=1,
    )

    observation, info = env.reset()

    assert observation.ndim == 1
    assert info["day"] == 0

    assert info[
        "action_mask"
    ].shape == (4,)

    assert bool(
        info["action_mask"].any()
    )


def test_first_and_repeat_rewards_are_exact() -> None:
    instance = make_instance(
        idle_requirements=[
            0,
            0,
            0,
        ]
    )

    env = CIPPEnv(
        instance
    )

    env.reset()

    first_reward = env.compute_reward(
        1
    )

    expected_first_reward = (
        100.0
        * 0.9
        * 1.0
    )

    assert first_reward == pytest.approx(
        expected_first_reward
    )

    env.step(1)

    second_reward = env.compute_reward(
        1
    )

    old_value = (
        100.0
        * 0.9
        * 1.0
    )

    new_value = (
        100.0
        * 0.8
        * (1.0 + 0.8)
    )

    assert second_reward == pytest.approx(
        new_value - old_value
    )


def test_mask_blocks_total_visit_cap() -> None:
    instance = make_instance(
        q=1,
        idle_requirements=[
            0,
            0,
            0,
        ],
    )

    mask = get_viability_mask(
        instance,
        [1],
    )

    assert not bool(mask[1])
    assert bool(mask[0])


def test_mask_blocks_budget_violation() -> None:
    instance = make_instance(
        budget=15.0,
        idle_requirements=[
            0,
            0,
            0,
        ],
    )

    mask = get_viability_mask(
        instance,
        [],
    )

    assert bool(mask[1])
    assert not bool(mask[2])
    assert bool(mask[3])


def test_mask_blocks_rolling_visit_violation() -> None:
    instance = make_instance(
        w=1,
        idle_requirements=[
            0,
            0,
            0,
        ],
    )

    mask = get_viability_mask(
        instance,
        [1],
    )

    assert not bool(mask[1])


def test_mask_prevents_future_idle_dead_end() -> None:
    instance = CIPPInstance(
        n=2,
        H=5,
        rewards=[
            10.0,
            9.0,
        ],
        costs=[
            1.0,
            1.0,
        ],
        budget=5.0,
        alpha=5,
        idle_requirements=[
            3,
        ],
        q=5,
        w=5,
        temporal_weights=[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        gamma=0.0,
        instance_id=(
            "idle-dead-end"
        ),
    )

    mask = get_viability_mask(
        instance,
        [1, 2],
    )

    assert bool(mask[0])
    assert not bool(mask[1])
    assert not bool(mask[2])


def test_every_masked_action_has_a_feasible_completion() -> None:
    instance = generate_cipp_instance(
        n=6,
        H=12,
        alpha=5,
        q=4,
        w=2,
        seed=12,
    )

    env = CIPPEnv(
        instance,
        seed=5,
    )

    env.reset()

    for _ in range(6):
        prefix = env.itinerary
        mask = env.get_action_mask()

        for action in np.flatnonzero(
            mask
        ):
            candidate_prefix = np.append(
                prefix,
                int(action),
            )

            completion = all_idle_completion(
                instance,
                candidate_prefix,
            )

            result = evaluate_itinerary(
                instance,
                completion,
            )

            assert result.feasible

        env.step(
            env.sample_viable_action()
        )


def test_invalid_masked_action_raises() -> None:
    instance = make_instance(
        budget=5.0,
        idle_requirements=[
            0,
            0,
            0,
        ],
    )

    env = CIPPEnv(
        instance
    )

    env.reset()

    with pytest.raises(
        ValueError,
        match="not viable",
    ):
        env.step(1)


def test_rewards_telescope_to_final_objective() -> None:
    instance = make_instance()

    env = CIPPEnv(
        instance,
        seed=7,
    )

    env.reset()

    rewards: list[float] = []

    while not env.done:
        action = (
            env.sample_viable_action()
        )

        _, reward, _, _, _ = (
            env.step(action)
        )

        rewards.append(
            reward
        )

    result = evaluate_itinerary(
        instance,
        env.itinerary,
    )

    assert result.feasible

    assert sum(
        rewards
    ) == pytest.approx(
        result.objective,
        abs=1e-9,
    )

    assert env.cumulative_reward == pytest.approx(
        result.objective,
        abs=1e-9,
    )


def test_terminal_information_contains_final_evaluation() -> None:
    env = CIPPEnv(
        make_instance()
    )

    env.reset()

    info = {}

    while not env.done:
        (
            _,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(0)

        assert not truncated

    assert terminated

    assert info[
        "final_evaluation"
    ]["feasible"] is True

    with pytest.raises(
        RuntimeError,
        match="after episode termination",
    ):
        env.step(0)


def test_one_thousand_masked_episodes_have_zero_violations() -> None:
    for episode in range(
        1000
    ):
        instance = generate_cipp_instance(
            n=4 + episode % 5,
            H=10 + episode % 7,
            alpha=5,
            q=4,
            w=2,
            seed=10_000 + episode,
        )

        env = CIPPEnv(
            instance,
            seed=20_000 + episode,
        )

        env.reset()

        while not env.done:
            mask = (
                env.get_action_mask()
            )

            assert bool(
                mask.any()
            )

            action = (
                env.sample_viable_action()
            )

            env.step(
                action
            )

        result = evaluate_itinerary(
            instance,
            env.itinerary,
        )

        assert result.feasible
        assert result.violations == ()

        assert env.cumulative_reward == pytest.approx(
            result.objective,
            abs=1e-8,
        )