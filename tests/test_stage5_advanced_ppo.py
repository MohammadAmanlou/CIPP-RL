"""Correctness tests for structured PPO and RL-only improvement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.advanced import (
    AdvancedPPOAgent,
    AdvancedPPOConfig,
    ConstructionTrainingConfig,
    ImprovementPPOAgent,
    ImprovementPPOConfig,
    NetworkConfig,
    StructuredFeatureBuilder,
    evaluate_construction_policy,
    evaluate_rl_improvement,
    policy_beam_search,
    train_construction_agent,
    train_improvement_agent,
)
from src.advanced.improvement import build_candidate_set
from src.advanced.training import collect_construction_batch
from src.core import evaluate_itinerary
from src.envs import CIPPEnv
from src.utils import (
    benchmark_from_instance,
    generate_paper_like_instance,
    load_professor_benchmark,
    parse_professor_instance_id,
)


def _instance():
    return generate_paper_like_instance(
        seed=91,
        number_of_states=4,
        horizon=8,
        objective_variant="professor_code",
        instance_id="advanced-test",
    )


def _agent(architecture: str, *, residual: bool = False, count: bool = False):
    instance = _instance()
    builder = StructuredFeatureBuilder(instance)
    return AdvancedPPOAgent(
        feature_builder=builder,
        network_config=NetworkConfig(
            architecture=architecture,
            hidden_dim=32,
            attention_heads=4,
            attention_layers=1,
            residual_marginal=residual,
            count_planner=count,
            max_visit_count=instance.q,
        ),
        ppo_config=AdvancedPPOConfig(
            update_epochs=2,
            minibatch_size=8,
            advantage_mode="hybrid" if count else "gae",
        ),
        seed=92,
    )


def test_14s_means_14_visit_locations_plus_idle() -> None:
    benchmark = load_professor_benchmark(
        Path("CIPP-D.xls"),
        instance_id="D_14S_30P",
        objective_variant="professor_code",
        budget_mode="auto",
    )
    assert benchmark.total_state_count == 14
    assert benchmark.visit_location_count == 14
    assert benchmark.instance.n == 14
    assert benchmark.action_count == 15
    assert benchmark.instance.num_actions == 15
    assert benchmark.idle_action == 0
    assert len(benchmark.location_names) == 14
    np.testing.assert_array_equal(benchmark.instance.costs, np.zeros(14))
    assert benchmark.instance.budget == pytest.approx(0.0)


def test_instance_id_defines_non_hardcoded_location_and_period_counts() -> None:
    spec = parse_professor_instance_id("d_17s_23p")
    assert spec.instance_id == "D_17S_23P"
    assert spec.total_state_count == 17
    assert spec.visit_location_count == 17
    assert spec.horizon == 23

    benchmark = load_professor_benchmark(
        Path("CIPP-D.xls"),
        instance_id=spec.instance_id,
        objective_variant="professor_code",
    )
    assert benchmark.instance.n == 17
    assert benchmark.instance.H == 23
    assert benchmark.instance.num_actions == 18
    assert benchmark.table5_reference is None


def test_json_backed_instance_owns_its_dimensions() -> None:
    instance = generate_paper_like_instance(
        seed=101,
        number_of_states=6,
        horizon=9,
        objective_variant="professor_code",
        instance_id="custom-6-by-9",
    )
    benchmark = benchmark_from_instance(instance)
    assert benchmark.visit_location_count == 6
    assert benchmark.action_count == 7
    assert benchmark.instance.H == 9


def test_structured_features_have_entity_aligned_shapes_and_finite_values() -> None:
    instance = _instance()
    environment = CIPPEnv(instance)
    environment.reset()
    state = StructuredFeatureBuilder(instance).build(environment)
    assert state.locations.shape == (instance.n, 14)
    assert state.global_features.shape == (12,)
    assert state.action_mask.shape == (instance.num_actions,)
    assert np.all(np.isfinite(state.locations))
    assert np.all(np.isfinite(state.global_features))


@pytest.mark.parametrize(
    "architecture,residual,count",
    [
        ("stable_mlp", False, False),
        ("attention", False, False),
        ("hierarchical_attention", False, False),
        ("hierarchical_attention", True, True),
    ],
)
def test_every_advanced_architecture_updates_and_remains_feasible(
    architecture: str, residual: bool, count: bool
) -> None:
    agent = _agent(architecture, residual=residual, count=count)
    batch, summary, _ = collect_construction_batch(
        agent,
        number_of_episodes=4,
        pomo_group_size=2 if count else 1,
        seed=93,
    )
    metrics = agent.update(batch, progress=0.5)
    result = evaluate_construction_policy(
        agent, deterministic=True, seed=94, method="test"
    )
    assert summary["feasible_rate"] == pytest.approx(1.0)
    assert result.feasible is True
    assert batch.locations.shape[0] == batch.size
    assert all(np.isfinite(value) for value in metrics.values())


def test_policy_beam_search_returns_a_feasible_complete_itinerary() -> None:
    agent = _agent("hierarchical_attention", residual=True, count=True)
    result = policy_beam_search(agent, beam_width=4, expansion_width=2)
    assert result.feasible is True
    assert len(result.itinerary) == agent.instance.H


def test_vectorized_candidate_objectives_match_canonical_evaluator() -> None:
    agent = _agent("hierarchical_attention", residual=True, count=True)
    start = evaluate_construction_policy(
        agent, deterministic=True, seed=95, method="start"
    )
    config = ImprovementPPOConfig(max_candidates=24, max_moves=3, hidden_dim=32)
    candidates, _ = build_candidate_set(
        agent.instance,
        start.itinerary,
        current_best=start.objective,
        move_index=0,
        config=config,
        seed=96,
    )
    for index in np.flatnonzero(candidates.mask):
        canonical = evaluate_itinerary(agent.instance, candidates.itineraries[index])
        assert canonical.feasible is True
        assert candidates.objectives[index] == pytest.approx(canonical.objective)


def test_rl_improvement_keeps_best_incumbent_and_never_degrades_start() -> None:
    construction = _agent("hierarchical_attention", residual=True, count=True)
    start = evaluate_construction_policy(
        construction, deterministic=True, seed=97, method="start"
    )
    improvement = ImprovementPPOAgent(
        construction.instance,
        ImprovementPPOConfig(max_candidates=24, max_moves=4, hidden_dim=32),
        seed=98,
    )
    improved = evaluate_rl_improvement(improvement, start, seed=99)
    assert improved.feasible is True
    assert improved.objective >= start.objective - 1e-8


def test_construction_training_prints_episode_based_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _agent("stable_mlp")
    train_construction_agent(
        agent,
        config=ConstructionTrainingConfig(
            updates=2,
            episodes_per_update=2,
            validation_interval=2,
            validation_rollouts=1,
            self_imitation_interval=0,
        ),
        output_directory=tmp_path / "construction",
        seed=100,
        method_name="stable_mlp",
        log_every_episodes=3,
    )
    output = capsys.readouterr().out
    assert "phase=construction" in output
    assert "episodes=2/4" in output
    assert "episodes=4/4" in output
    assert "mean_objective=" in output
    assert "loss=" in output
    assert "eta_seconds=" in output


def test_improvement_training_prints_episode_based_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    construction = _agent("hierarchical_attention", residual=True, count=True)
    improvement = ImprovementPPOAgent(
        construction.instance,
        ImprovementPPOConfig(
            max_candidates=16,
            max_moves=2,
            hidden_dim=32,
            minibatch_size=8,
        ),
        seed=101,
    )
    train_improvement_agent(
        improvement,
        construction,
        updates=2,
        episodes_per_update=1,
        construction_rollouts=1,
        output_directory=tmp_path / "improvement",
        seed=102,
        log_every_episodes=1,
    )
    output = capsys.readouterr().out
    assert "phase=improvement" in output
    assert "episodes=1/2" in output
    assert "episodes=2/2" in output
    assert "best_seen=" in output
    assert "policy_loss=" in output


def test_construction_full_early_stopping_uses_fixed_validation(tmp_path: Path) -> None:
    agent = _agent("stable_mlp")
    output = tmp_path / "early_construction"
    train_construction_agent(
        agent,
        config=ConstructionTrainingConfig(
            updates=8,
            episodes_per_update=2,
            validation_interval=1,
            validation_rollouts=2,
            self_imitation_interval=0,
            early_stopping_patience=1,
            early_stopping_min_delta=1.0e9,
            early_stopping_warmup_updates=1,
            history_flush_interval=1,
        ),
        output_directory=output,
        seed=103,
        method_name="stable_mlp",
        log_every_episodes=0,
    )
    import json
    summary = json.loads((output / "training_summary.json").read_text())
    assert summary["completed_updates"] < 8
    assert summary["early_stopping_enabled"] is True
    assert str(summary["stop_reason"]).startswith("early_stopping")
    assert (output / "checkpoint_best.pt").exists()


def test_improvement_full_early_stopping_uses_fixed_starts(tmp_path: Path) -> None:
    construction = _agent("hierarchical_attention", residual=True, count=True)
    improvement = ImprovementPPOAgent(
        construction.instance,
        ImprovementPPOConfig(max_candidates=16, max_moves=2, hidden_dim=32, minibatch_size=8),
        seed=104,
    )
    output = tmp_path / "early_improvement"
    from src.advanced import ImprovementTrainingConfig
    train_improvement_agent(
        improvement,
        construction,
        config=ImprovementTrainingConfig(
            updates=8,
            episodes_per_update=1,
            construction_rollouts=1,
            validation_interval=1,
            validation_starts=1,
            validation_construction_rollouts=1,
            early_stopping_patience=1,
            early_stopping_min_delta=1.0e9,
            early_stopping_warmup_updates=1,
            history_flush_interval=1,
        ),
        output_directory=output,
        seed=105,
        log_every_episodes=0,
    )
    import json
    summary = json.loads((output / "training_summary.json").read_text())
    assert summary["completed_updates"] < 8
    assert str(summary["stop_reason"]).startswith("early_stopping")
    assert (output / "fixed_validation_starts.json").exists()
