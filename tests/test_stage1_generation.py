"""Tests that complete the Week 1 deterministic CIPP foundation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core import (
    deterministic_objective,
    evaluate_itinerary,
)

from src.utils import (
    generate_cipp_instance,
    load_instance,
    permute_instance,
    relabel_itinerary,
    save_instance,
)


def assert_instances_equal(
    first,
    second,
) -> None:
    """Assert equality of every CIPP instance field."""

    assert first.instance_id == second.instance_id
    assert first.n == second.n
    assert first.H == second.H

    assert first.budget == pytest.approx(
        second.budget
    )

    assert first.alpha == second.alpha
    assert first.q == second.q
    assert first.w == second.w

    assert first.gamma == pytest.approx(
        second.gamma
    )

    assert first.p == second.p

    np.testing.assert_allclose(
        first.rewards,
        second.rewards,
    )

    np.testing.assert_allclose(
        first.costs,
        second.costs,
    )

    np.testing.assert_allclose(
        first.temporal_weights,
        second.temporal_weights,
    )

    np.testing.assert_array_equal(
        first.idle_requirements,
        second.idle_requirements,
    )


def test_same_seed_produces_same_instance() -> None:
    first = generate_cipp_instance(
        seed=123
    )

    second = generate_cipp_instance(
        seed=123
    )

    assert_instances_equal(
        first,
        second,
    )


def test_different_seeds_produce_different_instances() -> None:
    first = generate_cipp_instance(
        seed=1
    )

    second = generate_cipp_instance(
        seed=2
    )

    assert not np.array_equal(
        first.rewards,
        second.rewards,
    )

    assert not np.array_equal(
        first.costs,
        second.costs,
    )


def test_generated_instance_has_expected_shapes() -> None:
    instance = generate_cipp_instance(
        n=14,
        H=30,
        alpha=5,
        seed=42,
    )

    assert instance.rewards.shape == (14,)
    assert instance.costs.shape == (14,)
    assert instance.temporal_weights.shape == (30,)
    assert instance.idle_requirements.shape == (26,)

    assert instance.temporal_weights.mean() == pytest.approx(
        1.0
    )

    assert np.all(
        instance.idle_requirements
        <= instance.alpha
    )

    assert np.all(
        np.diff(
            instance.idle_requirements
        )
        <= 0
    )


def test_all_idle_is_feasible_for_many_instances() -> None:
    for seed in range(50):
        instance = generate_cipp_instance(
            seed=seed
        )

        result = evaluate_itinerary(
            instance,
            [0] * instance.H,
        )

        assert result.feasible is True

        assert result.objective == pytest.approx(
            0.0
        )

        assert result.total_cost == pytest.approx(
            0.0
        )

        assert result.violations == ()


def test_save_load_round_trip(
    tmp_path: Path,
) -> None:
    original = generate_cipp_instance(
        seed=77
    )

    path = save_instance(
        original,
        tmp_path / "instance.json",
    )

    loaded = load_instance(path)

    assert_instances_equal(
        original,
        loaded,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    assert payload["schema_version"] == 1


def test_location_permutation_preserves_objective() -> None:
    instance = generate_cipp_instance(
        n=4,
        H=8,
        alpha=4,
        q=4,
        w=3,
        max_idle_requirement=1,
        seed=9,
    )

    itinerary = np.array(
        [1, 2, 0, 4, 1, 3, 0, 2]
    )

    permutation = [2, 0, 3, 1]

    permuted_instance = permute_instance(
        instance,
        permutation,
    )

    relabelled_itinerary = relabel_itinerary(
        itinerary,
        permutation,
    )

    original_value = deterministic_objective(
        instance,
        itinerary,
    )

    permuted_value = deterministic_objective(
        permuted_instance,
        relabelled_itinerary,
    )

    assert permuted_value == pytest.approx(
        original_value
    )


def test_invalid_generator_arguments_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="alpha",
    ):
        generate_cipp_instance(
            H=5,
            alpha=6,
        )

    with pytest.raises(
        ValueError,
        match="budget_tightness",
    ):
        generate_cipp_instance(
            budget_tightness=1.2
        )

    with pytest.raises(
        ValueError,
        match="gamma",
    ):
        generate_cipp_instance(
            q=4,
            gamma=0.3,
        )