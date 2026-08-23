"""Adaptive reward-context environment and feature builder for CIPP.

Method C keeps the Attention architecture unchanged, but makes the reward context
observable by replacing the reward-dependent location features at every period.
This preserves compatibility with the existing static Attention checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from src.advanced.features import (
    LOCATION_FEATURE_NAMES,
    StructuredFeatureBuilder,
    StructuredState,
)
from src.core import CIPPInstance, evaluate_itinerary
from src.envs import CIPPEnv


FloatArray = NDArray[np.float64]


REWARD_FEATURE_INDEX = LOCATION_FEATURE_NAMES.index("reward")
EXPOSURE_FEATURE_INDEX = LOCATION_FEATURE_NAMES.index("temporal_exposure")
MARGINAL_FEATURE_INDEX = LOCATION_FEATURE_NAMES.index("exact_marginal")
REWARD_RANK_FEATURE_INDEX = LOCATION_FEATURE_NAMES.index("reward_rank")


@dataclass(frozen=True, slots=True)
class ShockScenario:
    """One piecewise-stationary reward shock.

    Before ``switch_day`` all reward multipliers are 1.
    Starting at ``switch_day`` (zero-based), location i has multiplier
    ``post_multiplier[i]``.

    The adaptive objective is the original CIPP objective generalized to
    time-varying reward coefficients:

        sum_i base_reward_i * repeat_factor(total_visits_i)
              * sum_t multiplier[t, i] * temporal_weight[t] * z[i, t]

    so the static CIPP objective is recovered exactly when all multipliers are 1.
    """

    switch_day: int
    post_multiplier: FloatArray
    scenario_id: str = "shock"

    def __post_init__(self) -> None:
        multiplier = np.asarray(self.post_multiplier, dtype=np.float64).copy()
        if multiplier.ndim != 1:
            raise ValueError("post_multiplier must be a 1-D vector")
        if not np.all(np.isfinite(multiplier)):
            raise ValueError("post_multiplier must be finite")
        if np.any(multiplier <= 0.0):
            raise ValueError("all reward multipliers must be strictly positive")
        multiplier.setflags(write=False)
        object.__setattr__(self, "post_multiplier", multiplier)

    def validate(self, instance: CIPPInstance) -> None:
        if self.post_multiplier.shape != (instance.n,):
            raise ValueError(
                f"scenario multiplier shape {self.post_multiplier.shape} "
                f"does not match n={instance.n}"
            )
        if not 1 <= self.switch_day < instance.H:
            raise ValueError(
                f"switch_day must satisfy 1 <= switch_day < H; "
                f"got {self.switch_day}, H={instance.H}"
            )

    def multiplier_at(self, day: int) -> FloatArray:
        if day < self.switch_day:
            return np.ones_like(self.post_multiplier)
        return self.post_multiplier

    def multiplier_matrix(self, instance: CIPPInstance) -> FloatArray:
        self.validate(instance)
        matrix = np.ones((instance.H, instance.n), dtype=np.float64)
        matrix[self.switch_day :, :] = self.post_multiplier[None, :]
        return matrix


def rank_flip_scenario(
    instance: CIPPInstance,
    *,
    switch_fraction: float = 0.50,
    boost: float = 2.25,
    suppress: float = 0.55,
    scenario_id: str = "rank_flip",
) -> ShockScenario:
    """Held-out deterministic shock for the first pilot.

    Locations in the bottom third of the original reward ranking become high
    priority; locations in the top third are suppressed. This intentionally
    creates a real route-replanning signal rather than a tiny perturbation.
    """

    if not 0.1 <= switch_fraction <= 0.9:
        raise ValueError("switch_fraction must be between 0.1 and 0.9")
    order = np.argsort(instance.rewards)
    group = max(instance.n // 3, 1)
    multiplier = np.ones(instance.n, dtype=np.float64)
    multiplier[order[:group]] = float(boost)
    multiplier[order[-group:]] = float(suppress)
    switch_day = int(round(instance.H * switch_fraction))
    switch_day = min(max(switch_day, 1), instance.H - 1)
    return ShockScenario(
        switch_day=switch_day,
        post_multiplier=multiplier,
        scenario_id=scenario_id,
    )


def sample_training_scenario(
    instance: CIPPInstance,
    rng: np.random.Generator,
    *,
    switch_fractions: tuple[float, ...] = (0.25, 0.40, 0.50, 0.60, 0.75),
    min_boost: float = 1.35,
    max_boost: float = 2.50,
    min_suppress: float = 0.40,
    max_suppress: float = 0.85,
) -> ShockScenario:
    """Sample diverse shocks without exposing the held-out rank-flip exactly."""

    switch_fraction = float(rng.choice(np.asarray(switch_fractions)))
    switch_day = int(round(instance.H * switch_fraction))
    switch_day = min(max(switch_day, 1), instance.H - 1)

    # Randomly choose disjoint boosted/suppressed subsets.
    indices = rng.permutation(instance.n)
    group = max(instance.n // 3, 1)
    boosted = indices[:group]
    suppressed = indices[group : 2 * group]

    multiplier = np.ones(instance.n, dtype=np.float64)
    multiplier[boosted] = rng.uniform(min_boost, max_boost, size=boosted.size)
    multiplier[suppressed] = rng.uniform(min_suppress, max_suppress, size=suppressed.size)

    return ShockScenario(
        switch_day=switch_day,
        post_multiplier=multiplier,
        scenario_id=f"train_{switch_day}_{int(rng.integers(1_000_000_000))}",
    )


class AdaptiveCIPPEnv(CIPPEnv):
    """CIPP environment with a time-varying location reward context.

    Feasibility is unchanged and therefore all original action masking remains
    valid. Only the objective is generalized to a time-varying reward multiplier.
    """

    def __init__(
        self,
        instance: CIPPInstance,
        scenario: ShockScenario,
        seed: int | None = None,
    ) -> None:
        scenario.validate(instance)
        self.scenario = scenario
        self._adaptive_exposures = np.zeros(instance.n, dtype=np.float64)
        super().__init__(instance, seed=seed)

    @property
    def current_multiplier(self) -> FloatArray:
        return self.scenario.multiplier_at(self.day).copy()

    @property
    def adaptive_exposures(self) -> FloatArray:
        return self._adaptive_exposures.copy()

    @property
    def shock_active(self) -> bool:
        return self.day >= self.scenario.switch_day

    def reset(self, seed: int | None = None):
        self._adaptive_exposures = np.zeros(self.instance.n, dtype=np.float64)
        return super().reset(seed=seed)

    def compute_reward(self, action: int) -> float:
        if self.done:
            raise RuntimeError("Cannot compute reward after episode termination.")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action = int(action)
        if action < 0 or action > self.instance.n:
            raise ValueError(f"action must be between 0 and {self.instance.n}")
        mask = self.get_action_mask()
        if not bool(mask[action]):
            raise ValueError(f"Action {action} is not viable on day {self.day + 1}.")
        if action == 0:
            return 0.0

        location = action - 1
        old_count = int(self._visit_counts[location])
        new_count = old_count + 1

        old_exposure = float(self._adaptive_exposures[location])
        multiplier = float(self.scenario.multiplier_at(self.day)[location])
        new_exposure = (
            old_exposure
            + multiplier * float(self.instance.temporal_weights[self.day])
        )

        old_contribution = (
            float(self.instance.rewards[location])
            * self.instance.repeat_factor(old_count)
            * old_exposure
        )
        new_contribution = (
            float(self.instance.rewards[location])
            * self.instance.repeat_factor(new_count)
            * new_exposure
        )
        return float(new_contribution - old_contribution)

    def step(self, action: int):
        day_before = self.day
        action = int(action)
        multiplier = self.scenario.multiplier_at(day_before)
        result = super().step(action)
        if action > 0:
            location = action - 1
            self._adaptive_exposures[location] += (
                float(multiplier[location])
                * float(self.instance.temporal_weights[day_before])
            )
        return result


class AdaptiveFeatureBuilder(StructuredFeatureBuilder):
    """Reward-context-conditioned features with static checkpoint compatibility.

    The tensor dimensions are intentionally identical to StructuredFeatureBuilder.
    Therefore an existing static Attention checkpoint can initialize Method C.
    """

    def build(self, environment: CIPPEnv) -> StructuredState:
        state = super().build(environment)
        if not isinstance(environment, AdaptiveCIPPEnv):
            return state

        locations = state.locations.copy()
        global_features = state.global_features.copy()

        multiplier = environment.current_multiplier
        effective_rewards = self.instance.rewards * multiplier

        # Current reward context is directly observable per location.
        locations[:, REWARD_FEATURE_INDEX] = (
            effective_rewards / self.scale.reward
        ).astype(np.float32)

        # Context-correct exact marginal.
        counts = environment.visit_counts.astype(np.int64)
        old_exposure = environment.adaptive_exposures.astype(np.float64)
        if environment.done:
            marginals = np.zeros(self.instance.n, dtype=np.float64)
        else:
            new_count = counts + 1
            current_lambda = float(self.instance.temporal_weights[environment.day])
            new_exposure = old_exposure + multiplier * current_lambda
            old_factor = np.asarray(
                [self.instance.repeat_factor(int(v)) for v in counts],
                dtype=np.float64,
            )
            new_factor = np.asarray(
                [self.instance.repeat_factor(int(v)) for v in new_count],
                dtype=np.float64,
            )
            marginals = self.instance.rewards * (
                new_factor * new_exposure - old_factor * old_exposure
            )
        locations[:, MARGINAL_FEATURE_INDEX] = (
            marginals / self.scale.objective
        ).astype(np.float32)

        # Current reward rank is also context-dependent.
        order = np.argsort(np.argsort(effective_rewards))
        reward_rank = order.astype(np.float64) / max(self.instance.n - 1, 1)
        locations[:, REWARD_RANK_FEATURE_INDEX] = reward_rank.astype(np.float32)

        # Exposure in the objective should reflect the context-weighted exposure.
        locations[:, EXPOSURE_FEATURE_INDEX] = (
            environment.adaptive_exposures / self.scale.exposure
        ).astype(np.float32)

        return StructuredState(
            locations=locations,
            global_features=global_features,
            action_mask=state.action_mask.copy(),
        )


def replay_prefix(
    instance: CIPPInstance,
    scenario: ShockScenario,
    prefix: Iterable[int],
    *,
    seed: int = 0,
) -> AdaptiveCIPPEnv:
    environment = AdaptiveCIPPEnv(instance, scenario, seed=seed)
    environment.reset(seed=seed)
    for action in prefix:
        environment.step(int(action))
    return environment


def evaluate_adaptive_itinerary(
    instance: CIPPInstance,
    scenario: ShockScenario,
    itinerary: Iterable[int],
) -> dict[str, object]:
    actions = tuple(int(a) for a in itinerary)
    static_eval = evaluate_itinerary(instance, actions)
    if not static_eval.feasible:
        return {
            "objective": float("-inf"),
            "feasible": False,
            "violations": list(static_eval.violations),
        }
    env = AdaptiveCIPPEnv(instance, scenario)
    env.reset()
    for action in actions:
        env.step(action)
    return {
        "objective": float(env.cumulative_reward),
        "feasible": True,
        "violations": [],
        "visit_counts": env.visit_counts.tolist(),
        "itinerary": list(actions),
    }
