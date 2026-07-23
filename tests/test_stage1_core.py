"""Tests for Stage 1 deterministic CIPP implementation."""

from typing import Any

import numpy as np
import pytest

from src.core import (
    CIPPInstance,
    deterministic_objective,
    evaluate_itinerary,
    temporal_exposures,
    total_cost,
    visit_counts,
)


def make_instance(
    **overrides: Any,
) -> CIPPInstance:
    """Create the fixed small instance used in Stage 1 tests."""

    parameters: dict[str, Any] = {
        "n": 3,
        "H": 5,
        "rewards": [100.0, 80.0, 60.0],
        "costs": [10.0, 20.0, 15.0],
        "budget": 60.0,
        "alpha": 3,
        "idle_requirements": [1, 1, 1],
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
        "p": 1,
        "instance_id": "golden-3S-5P",
    }

    parameters.update(overrides)

    return CIPPInstance(**parameters)


def test_valid_instance() -> None:
    instance = make_instance()

    assert instance.n == 3
    assert instance.H == 5
    assert instance.tmax == 5
    assert instance.num_actions == 4
    assert instance.num_rolling_windows == 3


def test_instance_arrays_are_read_only() -> None:
    instance = make_instance()

    with pytest.raises(ValueError):
        instance.rewards[0] = 999.0


def test_invalid_gamma_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="gamma must satisfy",
    ):
        make_instance(gamma=0.34)


def test_invalid_vector_length_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="rewards must have shape",
    ):
        make_instance(
            rewards=[100.0, 80.0]
        )


def test_golden_visit_counts() -> None:
    instance = make_instance()
    itinerary = [1, 2, 0, 1, 3]

    np.testing.assert_array_equal(
        visit_counts(instance, itinerary),
        np.array([2, 1, 1]),
    )


def test_golden_temporal_exposures() -> None:
    instance = make_instance()
    itinerary = [1, 2, 0, 1, 3]

    np.testing.assert_allclose(
        temporal_exposures(
            instance,
            itinerary,
        ),
        np.array([2.0, 0.8, 1.1]),
    )


def test_golden_objective_is_277() -> None:
    instance = make_instance()
    itinerary = [1, 2, 0, 1, 3]

    assert deterministic_objective(
        instance,
        itinerary,
    ) == pytest.approx(277.0)


def test_golden_cost_is_55() -> None:
    instance = make_instance()
    itinerary = [1, 2, 0, 1, 3]

    assert total_cost(
        instance,
        itinerary,
    ) == pytest.approx(55.0)


def test_golden_itinerary_is_feasible() -> None:
    instance = make_instance()
    itinerary = [1, 2, 0, 1, 3]

    result = evaluate_itinerary(
        instance,
        itinerary,
    )

    assert result.feasible is True
    assert result.violations == ()
    assert result.idle_days == 1
    assert result.objective == pytest.approx(277.0)


def test_all_idle_itinerary_is_feasible() -> None:
    instance = make_instance()

    result = evaluate_itinerary(
        instance,
        [0, 0, 0, 0, 0],
    )

    assert result.feasible is True
    assert result.objective == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(0.0)
    assert result.idle_days == 5


def test_budget_violation_is_detected() -> None:
    instance = make_instance()

    result = evaluate_itinerary(
        instance,
        [2, 2, 0, 2, 1],
    )

    assert result.feasible is False
    assert any(
        violation.startswith("budget_exceeded")
        for violation in result.violations
    )


def test_total_visit_cap_violation_is_detected() -> None:
    instance = make_instance()

    result = evaluate_itinerary(
        instance,
        [1, 1, 0, 1, 1],
    )

    assert result.feasible is False
    assert any(
        violation.startswith("total_visit_cap")
        for violation in result.violations
    )


def test_rolling_visit_violation_is_detected() -> None:
    instance = make_instance()

    result = evaluate_itinerary(
        instance,
        [1, 1, 1, 0, 0],
    )

    assert result.feasible is False
    assert any(
        violation.startswith("rolling_visit_cap")
        for violation in result.violations
    )


def test_idle_violation_is_detected() -> None:
    instance = make_instance(
        budget=100.0
    )

    result = evaluate_itinerary(
        instance,
        [1, 2, 3, 1, 2],
    )

    assert result.feasible is False
    assert any(
        violation.startswith("idle_requirement")
        for violation in result.violations
    )


def test_invalid_action_is_reported() -> None:
    instance = make_instance()

    result = evaluate_itinerary(
        instance,
        [1, 2, 4, 0, 0],
    )

    assert result.feasible is False
    assert result.violations[0].startswith(
        "invalid_itinerary"
    )


def test_wrong_horizon_is_reported() -> None:
    instance = make_instance()

    result = evaluate_itinerary(
        instance,
        [1, 2, 0],
    )

    assert result.feasible is False
    assert result.violations[0].startswith(
        "invalid_itinerary"
    )