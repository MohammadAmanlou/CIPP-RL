"""Stage 4 tests for real-data alignment and DQN rollout search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.baselines import run_dqn_backtracking_rollouts, run_dqn_greedy_policy
from src.core import CIPPInstance, deterministic_objective
from src.envs import CIPPEnv, get_viability_mask, prefix_has_feasible_completion
from src.models import DQNAgent, DQNConfig
from src.utils import (
    ObservationNormalizer,
    generate_professor_matched_instance,
    load_professor_instance,
    professor_idle_requirements,
    professor_temporal_weights,
    reward_profiles_from_data,
)


def test_professor_temporal_weights_match_supplied_formula() -> None:
    weights = professor_temporal_weights(30)
    assert weights.shape == (30,)
    assert weights[0] == pytest.approx((1.0 - 15.5) ** 2 + 7.25**2)
    assert weights[-1] == pytest.approx((30.0 - 15.5) ** 2 + 7.25**2)
    assert weights[0] == pytest.approx(weights[-1])


def test_professor_idle_schedule_uses_complete_seven_day_windows() -> None:
    requirements = professor_idle_requirements(30)
    assert requirements.shape == (24,)
    np.testing.assert_array_equal(requirements, np.array([2] * 22 + [0] * 2))


def test_professor_repeat_factor_and_evaluator_use_same_offset() -> None:
    instance = CIPPInstance(
        n=1,
        H=2,
        rewards=[100.0],
        costs=[0.0],
        budget=0.0,
        alpha=2,
        idle_requirements=[0],
        q=12,
        w=2,
        temporal_weights=[1.0, 1.0],
        gamma=0.04,
        repeat_count_offset=1,
    )
    assert instance.repeat_factor(1) == pytest.approx(1.0)
    assert instance.repeat_factor(2) == pytest.approx(0.96)
    assert deterministic_objective(instance, [1, 1]) == pytest.approx(192.0)

    env = CIPPEnv(instance)
    env.reset()
    _, first_reward, _, _, _ = env.step(1)
    _, second_reward, _, _, _ = env.step(1)
    assert first_reward == pytest.approx(100.0)
    assert second_reward == pytest.approx(92.0)
    assert first_reward + second_reward == pytest.approx(192.0)


def test_supplied_code_and_paper_modes_are_not_mislabeled() -> None:
    data = Path("data/processed/CIPP-D.csv")
    supplied, supplied_ids = load_professor_instance(
        data,
        party="D",
        cities_parameter=16,
        horizon=30,
        instance_mode="supplied-code",
    )
    paper, paper_ids = load_professor_instance(
        data,
        party="D",
        cities_parameter=16,
        horizon=30,
        instance_mode="paper-14",
    )
    assert supplied.n == 15
    assert len(supplied_ids) == 15
    assert supplied.instance_id == "D_15S_30P_CODE"
    assert paper.n == 14
    assert len(paper_ids) == 14
    assert paper.instance_id == "D_14S_30P_PAPER_SHAPE"
    assert supplied_ids[0] == "Alabama"


def test_empirical_training_generator_is_reproducible_and_permuted() -> None:
    profiles = reward_profiles_from_data(
        [Path("data/processed/CIPP-D.csv"), Path("data/processed/CIPP-R.csv")],
        cities_parameter=16,
        instance_mode="paper-14",
    )
    first = generate_professor_matched_instance(
        n=14,
        horizon=30,
        seed=123,
        reward_profiles=profiles,
    )
    second = generate_professor_matched_instance(
        n=14,
        horizon=30,
        seed=123,
        reward_profiles=profiles,
    )
    third = generate_professor_matched_instance(
        n=14,
        horizon=30,
        seed=124,
        reward_profiles=profiles,
    )
    np.testing.assert_allclose(first.rewards, second.rewards)
    assert not np.array_equal(first.rewards, third.rewards)


def test_fast_mask_matches_explicit_all_idle_viability() -> None:
    instance = generate_professor_matched_instance(
        n=6,
        horizon=12,
        seed=55,
        reward_range=(1.0, 10.0),
    )
    env = CIPPEnv(instance, seed=77)
    env.reset()
    while not env.done:
        prefix = env.itinerary
        fast = get_viability_mask(instance, prefix)
        explicit = np.zeros(instance.num_actions, dtype=np.bool_)
        for action in range(instance.num_actions):
            explicit[action] = prefix_has_feasible_completion(
                instance,
                np.append(prefix, action),
            )
        np.testing.assert_array_equal(fast, explicit)
        env.step(env.sample_viable_action())


def test_backtracking_rollouts_include_greedy_and_keep_weights_frozen() -> None:
    instance = generate_professor_matched_instance(
        n=5,
        horizon=10,
        seed=123,
        reward_range=(50.0, 150.0),
    )
    env = CIPPEnv(instance)
    observation, _ = env.reset()
    normalizer = ObservationNormalizer.fit(
        np.stack([observation, observation + 1.0], axis=0)
    )
    agent = DQNAgent(
        observation_dim=observation.size,
        action_dim=instance.num_actions,
        config=DQNConfig(hidden_dim=32),
        seed=7,
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in agent.online_network.state_dict().items()
    }

    greedy = run_dqn_greedy_policy(instance, agent=agent, normalizer=normalizer)
    searched = run_dqn_backtracking_rollouts(
        instance,
        agent=agent,
        normalizer=normalizer,
        rollout_budget=5,
        alternatives_per_state=2,
        reward_scale=100.0,
    )

    assert greedy.feasible
    assert searched.feasible
    assert searched.objective >= greedy.objective - 1e-9
    assert 1 <= searched.rollouts <= 5
    assert searched.metadata["weights_updated_during_test"] is False
    assert searched.metadata["reward_scale"] == pytest.approx(100.0)
    assert searched.metadata["branch_priority"] == (
        "exact_scaled_prefix_return_plus_Q"
    )
    for name, tensor in agent.online_network.state_dict().items():
        assert torch.equal(before[name], tensor)


def test_gurobi_matches_bruteforce_when_available() -> None:
    import importlib.util
    import itertools

    if importlib.util.find_spec("gurobipy") is None:
        pytest.skip("gurobipy is not installed in this test environment.")

    from src.core import evaluate_itinerary
    from src.solvers import solve_with_gurobi

    instance = CIPPInstance(
        n=2,
        H=4,
        rewards=[10.0, 8.0],
        costs=[0.0, 0.0],
        budget=0.0,
        alpha=3,
        idle_requirements=[1, 1],
        q=3,
        w=2,
        temporal_weights=[2.0, 1.0, 1.0, 2.0],
        gamma=0.1,
        repeat_count_offset=1,
        instance_id="tiny-gurobi-crosscheck",
    )
    feasible_values = []
    for itinerary in itertools.product(range(instance.num_actions), repeat=instance.H):
        result = evaluate_itinerary(instance, itinerary)
        if result.feasible:
            feasible_values.append(result.objective)
    brute_force = max(feasible_values)

    result = solve_with_gurobi(instance, time_limit_seconds=30, threads=1, seed=0)
    assert result.status == "OPTIMAL"
    assert result.objective == pytest.approx(brute_force)
    assert result.feasible is True
