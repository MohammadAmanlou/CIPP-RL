"""Structured, optimization-informed observations for advanced CIPP PPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core import CIPPInstance
from src.envs import CIPPEnv


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


LOCATION_FEATURE_NAMES = (
    "reward",
    "cost",
    "visit_count",
    "temporal_exposure",
    "repeat_factor_now",
    "repeat_factor_next",
    "exact_marginal",
    "recent_window_count",
    "remaining_visit_capacity",
    "feasible_now",
    "current_temporal_weight",
    "reward_rank",
    "exposure_share",
    "visit_share",
)

GLOBAL_FEATURE_NAMES = (
    "day",
    "remaining_periods",
    "remaining_budget",
    "current_temporal_weight",
    "current_objective",
    "recent_idle_fraction",
    "minimum_idle_slack",
    "valid_action_fraction",
    "mean_visit_utilization",
    "maximum_visit_utilization",
    "future_temporal_mean",
    "future_temporal_max",
)

MARGINAL_FEATURE_INDEX = LOCATION_FEATURE_NAMES.index("exact_marginal")
COUNT_FEATURE_INDEX = LOCATION_FEATURE_NAMES.index("visit_count")


@dataclass(frozen=True, slots=True)
class StructuredState:
    """One tensor-ready structured CIPP state."""

    locations: FloatArray
    global_features: FloatArray
    action_mask: BoolArray


@dataclass(frozen=True, slots=True)
class FeatureScale:
    """Stable per-instance scales used by the feature builder."""

    reward: float
    cost: float
    temporal: float
    exposure: float
    objective: float
    budget: float

    @classmethod
    def from_instance(cls, instance: CIPPInstance) -> "FeatureScale":
        reward = max(float(np.max(instance.rewards)), 1.0)
        cost = max(float(np.max(instance.costs)), 1.0)
        temporal = max(float(np.max(instance.temporal_weights)), 1.0)
        exposure = max(float(np.sum(instance.temporal_weights)), 1.0)
        # This is deliberately a scale, not an optimization bound.  It keeps
        # rewards and critic targets in a stable order of magnitude.
        objective = max(reward * exposure * max(instance.n, 1), 1.0)
        budget = max(float(instance.budget), cost, 1.0)
        return cls(reward, cost, temporal, exposure, objective, budget)


class StructuredFeatureBuilder:
    """Convert a CIPP environment state into location tokens and context."""

    location_dim = len(LOCATION_FEATURE_NAMES)
    global_dim = len(GLOBAL_FEATURE_NAMES)

    def __init__(self, instance: CIPPInstance) -> None:
        self.instance = instance
        self.scale = FeatureScale.from_instance(instance)
        order = np.argsort(np.argsort(instance.rewards))
        self._reward_rank = order.astype(np.float64) / max(instance.n - 1, 1)

    def _exact_marginals(self, environment: CIPPEnv) -> np.ndarray:
        if environment.done:
            return np.zeros(self.instance.n, dtype=np.float64)
        old_count = environment.visit_counts.astype(np.int64)
        old_exposure = environment.temporal_exposures.astype(np.float64)
        new_count = old_count + 1
        new_exposure = old_exposure + float(
            self.instance.temporal_weights[environment.day]
        )
        old_factor = np.asarray(
            [self.instance.repeat_factor(int(value)) for value in old_count],
            dtype=np.float64,
        )
        new_factor = np.asarray(
            [self.instance.repeat_factor(int(value)) for value in new_count],
            dtype=np.float64,
        )
        return self.instance.rewards * (
            new_factor * new_exposure - old_factor * old_exposure
        )

    def _minimum_idle_slack(self, environment: CIPPEnv) -> float:
        day = environment.day
        if day == 0 or self.instance.num_rolling_windows == 0:
            return 1.0
        itinerary = environment.itinerary
        first = max(0, day - self.instance.alpha)
        last = min(day - 1, self.instance.num_rolling_windows - 1)
        slacks: list[float] = []
        for start in range(first, last + 1):
            end = start + self.instance.alpha
            observed_end = min(day, end)
            idle = int(np.count_nonzero(itinerary[start:observed_end] == 0))
            remaining_slots = max(0, end - day)
            required = int(self.instance.idle_requirements[start])
            slacks.append((idle + remaining_slots - required) / self.instance.alpha)
        return float(min(slacks, default=1.0))

    def build(self, environment: CIPPEnv) -> StructuredState:
        if environment.instance is not self.instance:
            raise ValueError("feature builder and environment use different instances")

        mask = environment.get_action_mask()
        counts = environment.visit_counts.astype(np.float64)
        exposures = environment.temporal_exposures.astype(np.float64)
        itinerary = environment.itinerary
        history_length = max(self.instance.alpha - 1, 0)
        recent = itinerary[-history_length:] if history_length else itinerary[:0]
        recent_counts = np.asarray(
            [np.count_nonzero(recent == action) for action in range(1, self.instance.n + 1)],
            dtype=np.float64,
        )
        marginals = self._exact_marginals(environment)
        current_lambda = (
            float(self.instance.temporal_weights[environment.day])
            if not environment.done
            else 0.0
        )
        current_factors = np.asarray(
            [self.instance.repeat_factor(int(value)) for value in counts],
            dtype=np.float64,
        )
        next_factors = np.asarray(
            [self.instance.repeat_factor(int(value + 1)) for value in counts],
            dtype=np.float64,
        )
        visit_total = max(float(np.sum(counts)), 1.0)
        exposure_total = max(float(np.sum(exposures)), 1.0)
        locations = np.stack(
            [
                self.instance.rewards / self.scale.reward,
                self.instance.costs / self.scale.cost,
                counts / max(self.instance.q, 1),
                exposures / self.scale.exposure,
                current_factors,
                next_factors,
                marginals / self.scale.objective,
                recent_counts / max(self.instance.w, 1),
                np.maximum(self.instance.q - counts, 0.0) / max(self.instance.q, 1),
                mask[1:].astype(np.float64),
                np.full(self.instance.n, current_lambda / self.scale.temporal),
                self._reward_rank,
                exposures / exposure_total,
                counts / visit_total,
            ],
            axis=1,
        ).astype(np.float32)

        future = self.instance.temporal_weights[environment.day :]
        recent_idle = int(np.count_nonzero(recent == 0))
        remaining_budget = environment.remaining_budget / self.scale.budget
        global_features = np.asarray(
            [
                environment.day / max(self.instance.H, 1),
                (self.instance.H - environment.day) / max(self.instance.H, 1),
                np.clip(remaining_budget, -1.0, 1.0),
                current_lambda / self.scale.temporal,
                environment.cumulative_reward / self.scale.objective,
                recent_idle / max(self.instance.alpha, 1),
                self._minimum_idle_slack(environment),
                float(np.mean(mask)),
                float(np.mean(counts / max(self.instance.q, 1))),
                float(np.max(counts / max(self.instance.q, 1), initial=0.0)),
                float(np.mean(future) / self.scale.temporal) if future.size else 0.0,
                float(np.max(future) / self.scale.temporal) if future.size else 0.0,
            ],
            dtype=np.float32,
        )
        return StructuredState(
            locations=locations,
            global_features=global_features,
            action_mask=mask.astype(np.bool_),
        )

    def environment_from_prefix(self, prefix: NDArray[np.integer] | list[int]) -> CIPPEnv:
        environment = CIPPEnv(self.instance)
        environment.reset()
        for action in prefix:
            environment.step(int(action))
        return environment

