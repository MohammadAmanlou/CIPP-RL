"""Tests for Week 3 construction baselines."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import (
    run_greedy_policy,
    run_random_feasible_policy,
    select_greedy_action,
    viable_action_rewards,
)

from src.core import (
    CIPPInstance,
    evaluate_itinerary,
)

from src.envs import (
    CIPPEnv,
)

from src.utils import (
    generate_cipp_instance,
)


def make_simple_instance() -> CIPPInstance:
    """Return one small manually verifiable instance."""

    return CIPPInstance(
        n=3,
        H=5,
        rewards=[
            100.0,
            80.0,
            60.0,
        ],
        costs=[
            10.0,
            20.0,
            5.0,
        ],
        budget=60.0,
        alpha=3,
        idle_requirements=[
            1,
            1,
            1,
        ],
        q=3,
        w=2,
        temporal_weights=[
            1.0,
            0.8,
            1.2,
            1.0,
            1.1,
        ],
        gamma=0.1,
        instance_id="stage3-simple",
    )


def test_random_policy_is_reproducible_and_feasible() -> None:
    instance = generate_cipp_instance(
        seed=31
    )

    first = run_random_feasible_policy(
        instance,
        seed=99,
    )

    second = run_random_feasible_policy(
        instance,
        seed=99,
    )

    assert (
        first.itinerary
        == second.itinerary
    )

    assert first.objective == pytest.approx(
        second.objective
    )

    assert first.feasible is True

    evaluation = evaluate_itinerary(
        instance,
        first.itinerary,
    )

    assert evaluation.feasible is True

    assert first.objective == pytest.approx(
        evaluation.objective
    )


def test_greedy_selects_largest_exact_increment() -> None:
    environment = CIPPEnv(
        make_simple_instance()
    )

    environment.reset()

    values = viable_action_rewards(
        environment
    )

    action = select_greedy_action(
        environment
    )

    assert action == int(
        np.argmax(values)
    )

    assert action == 1


def test_greedy_policy_is_deterministic_and_feasible() -> None:
    instance = generate_cipp_instance(
        seed=44
    )

    first = run_greedy_policy(
        instance
    )

    second = run_greedy_policy(
        instance
    )

    assert (
        first.itinerary
        == second.itinerary
    )

    assert first.objective == pytest.approx(
        second.objective
    )

    assert first.feasible is True

    evaluation = evaluate_itinerary(
        instance,
        first.itinerary,
    )

    assert evaluation.feasible is True

    assert first.objective == pytest.approx(
        evaluation.objective
    )


def test_baselines_remain_feasible_on_many_instances() -> None:
    for index in range(20):
        instance = generate_cipp_instance(
            n=6,
            H=12,
            alpha=5,
            q=4,
            w=2,
            seed=500 + index,
        )

        random_result = (
            run_random_feasible_policy(
                instance,
                seed=1_500 + index,
            )
        )

        greedy_result = (
            run_greedy_policy(
                instance
            )
        )

        assert (
            random_result.feasible
            is True
        )

        assert (
            greedy_result.feasible
            is True
        )

        assert (
            random_result.violations
            == ()
        )

        assert (
            greedy_result.violations
            == ()
        )


def test_greedy_does_not_choose_negative_visit_over_idle() -> None:
    instance = CIPPInstance(
        n=1,
        H=2,
        rewards=[10.0],
        costs=[1.0],
        budget=2.0,
        alpha=2,
        idle_requirements=[0],
        q=2,
        w=2,
        temporal_weights=[
            1.0,
            0.01,
        ],
        gamma=0.5,
        instance_id=(
            "negative-repeat-increment"
        ),
    )

    environment = CIPPEnv(
        instance
    )

    environment.reset()

    environment.step(1)

    values = viable_action_rewards(
        environment
    )

    assert values[1] < 0.0

    assert select_greedy_action(
        environment
    ) == 0