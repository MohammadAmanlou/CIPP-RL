"""Adaptive reward-context environments and shock curricula for CIPP V6.

V6 keeps the original static CIPP feasibility model unchanged while allowing
reward coefficients to change over time.

Key design rules:
- Before a shock is realized, its future identity is NOT observable.
- Method C is trained over a rich distribution of shocks, including no-shock,
  boost-only, suppress-only, mixed, concentrated, distributed, and multi-shock
  episodes.
- The final held-out benchmark suite contains only single-shock scenarios so
  it can be compared exactly and efficiently with two-stage Gurobi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias

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
    """One permanent single shock.

    Before ``switch_day`` every multiplier is 1. Starting at ``switch_day``
    (zero-based), location i uses ``post_multiplier[i]``.
    """

    switch_day: int
    post_multiplier: FloatArray
    scenario_id: str = "shock"
    family: str = "single"

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


@dataclass(frozen=True, slots=True)
class MultiShockScenario:
    """Piecewise-stationary reward process used for training robustness.

    ``change_days[j]`` is the first period using
    ``post_multipliers[j]``. The final held-out Gurobi suite intentionally
    excludes this class; it is training-only in V6.
    """

    change_days: tuple[int, ...]
    post_multipliers: tuple[FloatArray, ...]
    scenario_id: str = "multi_shock"
    family: str = "multi_shock"

    def __post_init__(self) -> None:
        days = tuple(int(x) for x in self.change_days)
        if not days:
            raise ValueError("MultiShockScenario requires at least one change day")
        if tuple(sorted(days)) != days or len(set(days)) != len(days):
            raise ValueError("change_days must be strictly increasing")
        if len(days) != len(self.post_multipliers):
            raise ValueError("change_days and post_multipliers must have equal length")
        multipliers = []
        for value in self.post_multipliers:
            arr = np.asarray(value, dtype=np.float64).copy()
            if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ValueError("each post multiplier must be finite, positive, and 1-D")
            arr.setflags(write=False)
            multipliers.append(arr)
        object.__setattr__(self, "change_days", days)
        object.__setattr__(self, "post_multipliers", tuple(multipliers))

    @property
    def switch_day(self) -> int:
        return int(self.change_days[0])

    @property
    def post_multiplier(self) -> FloatArray:
        """Compatibility view: multiplier immediately after first shock."""
        return self.post_multipliers[0]

    def validate(self, instance: CIPPInstance) -> None:
        for day in self.change_days:
            if not 1 <= day < instance.H:
                raise ValueError(
                    f"every change day must satisfy 1 <= day < H; got {day}"
                )
        for multiplier in self.post_multipliers:
            if multiplier.shape != (instance.n,):
                raise ValueError(
                    f"scenario multiplier shape {multiplier.shape} "
                    f"does not match n={instance.n}"
                )

    def multiplier_at(self, day: int) -> FloatArray:
        current = np.ones_like(self.post_multipliers[0])
        for change_day, multiplier in zip(self.change_days, self.post_multipliers):
            if day < change_day:
                break
            current = multiplier
        return current

    def multiplier_matrix(self, instance: CIPPInstance) -> FloatArray:
        self.validate(instance)
        matrix = np.ones((instance.H, instance.n), dtype=np.float64)
        for day in range(instance.H):
            matrix[day] = self.multiplier_at(day)
        return matrix


AdaptiveScenario: TypeAlias = ShockScenario | MultiShockScenario


RICH_CURRICULUM_DESCRIPTION = {
    "version": "rich_v6",
    "timing": "random integer switch day across approximately days 4..26 for H=30",
    "families": [
        "no_shock",
        "mixed",
        "boost_only",
        "suppress_only",
        "concentrated",
        "distributed",
        "multi_shock_training_only",
    ],
    "magnitude_ranges": {
        "ordinary_boost": [1.10, 3.00],
        "ordinary_suppress": [0.25, 0.95],
        "concentrated_boost": [2.20, 4.00],
        "concentrated_suppress": [0.15, 0.55],
        "distributed_boost": [1.10, 1.80],
        "distributed_suppress": [0.55, 0.95],
    },
    "curriculum": {
        "early": "moderate single shocks + no-shock",
        "middle": "full single-shock mixture + some multi-shock",
        "late": "more concentrated/extreme and multi-shock episodes",
    },
}


def scenario_to_dict(scenario: AdaptiveScenario) -> dict[str, object]:
    if isinstance(scenario, MultiShockScenario):
        return {
            "kind": "multi",
            "scenario_id": scenario.scenario_id,
            "family": scenario.family,
            "change_days": list(scenario.change_days),
            "post_multipliers": [x.tolist() for x in scenario.post_multipliers],
        }
    return {
        "kind": "single",
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "switch_day": int(scenario.switch_day),
        "post_multiplier": scenario.post_multiplier.tolist(),
    }


def scenario_from_dict(payload: dict[str, object]) -> AdaptiveScenario:
    kind = str(payload.get("kind", "single"))
    if kind == "multi":
        return MultiShockScenario(
            change_days=tuple(int(x) for x in payload["change_days"]),
            post_multipliers=tuple(
                np.asarray(x, dtype=np.float64)
                for x in payload["post_multipliers"]
            ),
            scenario_id=str(payload.get("scenario_id", "multi_shock")),
            family=str(payload.get("family", "multi_shock")),
        )
    return ShockScenario(
        switch_day=int(payload["switch_day"]),
        post_multiplier=np.asarray(payload["post_multiplier"], dtype=np.float64),
        scenario_id=str(payload.get("scenario_id", "shock")),
        family=str(payload.get("family", "single")),
    )


def rank_flip_scenario(
    instance: CIPPInstance,
    *,
    switch_fraction: float = 0.50,
    boost: float = 2.25,
    suppress: float = 0.55,
    scenario_id: str = "rank_flip",
) -> ShockScenario:
    if not 0.1 <= switch_fraction <= 0.9:
        raise ValueError("switch_fraction must be between 0.1 and 0.9")
    switch_day = int(round(instance.H * switch_fraction))
    return rank_flip_scenario_at_day(
        instance,
        switch_day=min(max(switch_day, 1), instance.H - 1),
        boost=boost,
        suppress=suppress,
        scenario_id=scenario_id,
    )


def rank_flip_scenario_at_day(
    instance: CIPPInstance,
    *,
    switch_day: int,
    boost: float = 2.25,
    suppress: float = 0.55,
    scenario_id: str = "rank_flip",
) -> ShockScenario:
    switch_day = int(switch_day)
    if not 1 <= switch_day < instance.H:
        raise ValueError(
            f"switch_day must satisfy 1 <= switch_day < H; "
            f"got {switch_day}, H={instance.H}"
        )
    order = np.argsort(instance.rewards)
    group = max(instance.n // 3, 1)
    multiplier = np.ones(instance.n, dtype=np.float64)
    multiplier[order[:group]] = float(boost)
    multiplier[order[-group:]] = float(suppress)
    return ShockScenario(
        switch_day=switch_day,
        post_multiplier=multiplier,
        scenario_id=scenario_id,
        family="heldout_rank_flip",
    )


def _random_switch_day(instance: CIPPInstance, rng: np.random.Generator) -> int:
    # Keep some meaningful pre- and post-shock horizon. For H=30 => 4..26.
    low = max(2, int(round(0.13 * instance.H)))
    high = min(instance.H - 2, int(round(0.87 * instance.H)))
    return int(rng.integers(low, high + 1))


def _draw_subset(
    indices: np.ndarray,
    rng: np.random.Generator,
    min_count: int,
    max_count: int,
) -> np.ndarray:
    max_count = min(max_count, indices.size)
    min_count = min(min_count, max_count)
    count = int(rng.integers(min_count, max_count + 1))
    return rng.choice(indices, size=count, replace=False)


def _single_family_scenario(
    instance: CIPPInstance,
    rng: np.random.Generator,
    *,
    family: str,
    scenario_id: str,
    switch_day: int | None = None,
) -> ShockScenario:
    switch_day = _random_switch_day(instance, rng) if switch_day is None else int(switch_day)
    n = instance.n
    all_idx = np.arange(n)
    multiplier = np.ones(n, dtype=np.float64)
    max_affected = max(1, min(6, n - 1))

    if family == "no_shock":
        pass
    elif family == "boost_only":
        boosted = _draw_subset(all_idx, rng, 1, max_affected)
        multiplier[boosted] = rng.uniform(1.10, 3.00, size=boosted.size)
    elif family == "suppress_only":
        suppressed = _draw_subset(all_idx, rng, 1, max_affected)
        multiplier[suppressed] = rng.uniform(0.25, 0.95, size=suppressed.size)
    elif family == "concentrated":
        count = int(rng.integers(1, min(2, n) + 1))
        affected = rng.choice(all_idx, size=count, replace=False)
        for idx in affected:
            if rng.random() < 0.65:
                multiplier[idx] = float(rng.uniform(2.20, 4.00))
            else:
                multiplier[idx] = float(rng.uniform(0.15, 0.55))
    elif family == "distributed":
        count = int(rng.integers(max(2, n // 2), n + 1))
        affected = rng.choice(all_idx, size=count, replace=False)
        signs = rng.random(count) < 0.55
        for idx, positive in zip(affected, signs):
            multiplier[idx] = (
                float(rng.uniform(1.10, 1.80))
                if positive
                else float(rng.uniform(0.55, 0.95))
            )
    elif family == "mixed":
        perm = rng.permutation(n)
        max_each = max(1, min(6, n // 2))
        k_boost = int(rng.integers(1, max_each + 1))
        remaining = n - k_boost
        k_suppress = int(rng.integers(1, min(max_each, remaining) + 1))
        boosted = perm[:k_boost]
        suppressed = perm[k_boost : k_boost + k_suppress]
        multiplier[boosted] = rng.uniform(1.10, 3.00, size=boosted.size)
        multiplier[suppressed] = rng.uniform(0.25, 0.95, size=suppressed.size)
    else:
        raise ValueError(f"unknown single-shock family: {family}")

    return ShockScenario(
        switch_day=switch_day,
        post_multiplier=multiplier,
        scenario_id=scenario_id,
        family=family,
    )


def sample_training_scenario(
    instance: CIPPInstance,
    rng: np.random.Generator,
    *,
    progress: float = 0.5,
    allow_multi_shock: bool = True,
) -> AdaptiveScenario:
    """Sample one V6 curriculum episode.

    The family mixture changes gradually with training progress. Importantly,
    the held-out deterministic rank-flip is not generated by this function.
    """

    progress = float(np.clip(progress, 0.0, 1.0))
    if progress < 0.25:
        weights = {
            "no_shock": 0.15,
            "mixed": 0.45,
            "boost_only": 0.20,
            "suppress_only": 0.10,
            "concentrated": 0.10,
            "distributed": 0.00,
            "multi_shock": 0.00,
        }
    elif progress < 0.70:
        weights = {
            "no_shock": 0.10,
            "mixed": 0.30,
            "boost_only": 0.15,
            "suppress_only": 0.10,
            "concentrated": 0.15,
            "distributed": 0.10,
            "multi_shock": 0.10 if allow_multi_shock else 0.0,
        }
    else:
        weights = {
            "no_shock": 0.10,
            "mixed": 0.25,
            "boost_only": 0.10,
            "suppress_only": 0.10,
            "concentrated": 0.20,
            "distributed": 0.10,
            "multi_shock": 0.15 if allow_multi_shock else 0.0,
        }

    names = list(weights)
    probs = np.asarray([weights[x] for x in names], dtype=np.float64)
    probs /= probs.sum()
    family = str(rng.choice(np.asarray(names), p=probs))
    token = int(rng.integers(1_000_000_000))

    if family != "multi_shock":
        return _single_family_scenario(
            instance,
            rng,
            family=family,
            scenario_id=f"train_{family}_{token}",
        )

    # Two or three permanent regime changes. Training-only.
    number_of_changes = int(rng.integers(2, 4))
    low = max(2, int(round(0.13 * instance.H)))
    high = min(instance.H - 2, int(round(0.87 * instance.H)))
    candidate_days = np.arange(low, high + 1)
    if candidate_days.size < number_of_changes:
        number_of_changes = candidate_days.size
    change_days = tuple(
        sorted(int(x) for x in rng.choice(candidate_days, size=number_of_changes, replace=False))
    )
    regimes = []
    for j in range(number_of_changes):
        family_j = str(
            rng.choice(
                np.asarray(["mixed", "boost_only", "suppress_only", "concentrated", "distributed"]),
                p=np.asarray([0.40, 0.15, 0.10, 0.20, 0.15]),
            )
        )
        single = _single_family_scenario(
            instance,
            rng,
            family=family_j,
            scenario_id=f"_tmp_{j}",
            switch_day=change_days[j],
        )
        regimes.append(single.post_multiplier)
    return MultiShockScenario(
        change_days=change_days,
        post_multipliers=tuple(regimes),
        scenario_id=f"train_multi_{token}",
        family="multi_shock",
    )


def generate_validation_scenarios(
    instance: CIPPInstance,
    *,
    count: int,
    seed: int,
) -> list[ShockScenario]:
    """Fixed validation scenarios, single-shock only, never used for updates."""
    rng = np.random.default_rng(seed)
    scenarios = []
    families = ["no_shock", "mixed", "boost_only", "suppress_only", "concentrated", "distributed"]
    for i in range(count):
        family = families[i % len(families)]
        scenarios.append(
            _single_family_scenario(
                instance,
                rng,
                family=family,
                scenario_id=f"validation_{i:03d}_{family}",
            )
        )
    return scenarios


def generate_heldout_single_shock_suite(
    instance: CIPPInstance,
    *,
    count: int = 100,
    seed: int = 260824,
) -> list[ShockScenario]:
    """Deterministic single-shock benchmark suite for RL/Gurobi comparison.

    Scenario 0 is the original pilot rank-flip. The rest are held-out random and
    structured scenarios. They are saved to JSON before evaluation so local
    Gurobi uses exactly the same realizations.
    """
    if count < 10:
        raise ValueError("held-out suite should contain at least 10 scenarios")
    rng = np.random.default_rng(seed)
    suite: list[ShockScenario] = [
        rank_flip_scenario_at_day(
            instance,
            switch_day=instance.H // 2,
            boost=2.25,
            suppress=0.55,
            scenario_id="test_000_pilot_rank_flip",
        )
    ]

    # Roughly 70% unseen samples from the broad single-shock distribution.
    id_count = max(1, int(round(0.70 * count)) - 1)
    families = ["no_shock", "mixed", "boost_only", "suppress_only", "concentrated", "distributed"]
    for i in range(id_count):
        family = families[i % len(families)]
        suite.append(
            _single_family_scenario(
                instance,
                rng,
                family=family,
                scenario_id=f"test_{len(suite):03d}_id_{family}",
            )
        )

    order_reward = np.argsort(instance.rewards)
    order_cost = np.argsort(instance.costs)
    group = max(1, instance.n // 3)

    # Remaining scenarios are structured OOD patterns never sampled exactly in training.
    while len(suite) < count:
        idx = len(suite)
        mode = (idx - (1 + id_count)) % 5
        day = int(rng.integers(max(3, instance.H // 6), min(instance.H - 2, 5 * instance.H // 6) + 1))
        multiplier = np.ones(instance.n, dtype=np.float64)

        if mode == 0:
            # Rank flip with varied severity/timing.
            multiplier[order_reward[:group]] = float(rng.uniform(2.4, 3.5))
            multiplier[order_reward[-group:]] = float(rng.uniform(0.25, 0.60))
            family = "ood_rank_flip"
        elif mode == 1:
            # One low-base-reward city becomes exceptionally valuable.
            target = int(rng.choice(order_reward[: max(2, group)]))
            multiplier[target] = float(rng.uniform(3.2, 4.5))
            family = "ood_extreme_single_boost"
        elif mode == 2:
            # High-reward locations collapse together.
            affected = order_reward[-max(2, group):]
            multiplier[affected] = rng.uniform(0.18, 0.45, size=affected.size)
            family = "ood_top_reward_collapse"
        elif mode == 3:
            # High-cost locations become attractive, stressing budget optionality.
            affected = order_cost[-max(2, group):]
            multiplier[affected] = rng.uniform(1.8, 3.2, size=affected.size)
            family = "ood_high_cost_boost"
        else:
            # Low-cost locations become less attractive while a few expensive ones rise.
            low = order_cost[:max(2, group)]
            high = order_cost[-max(1, group // 2):]
            multiplier[low] = rng.uniform(0.35, 0.70, size=low.size)
            multiplier[high] = rng.uniform(2.0, 3.3, size=high.size)
            family = "ood_budget_reallocation"

        suite.append(
            ShockScenario(
                switch_day=day,
                post_multiplier=multiplier,
                scenario_id=f"test_{idx:03d}_{family}",
                family=family,
            )
        )

    return suite[:count]


class AdaptiveCIPPEnv(CIPPEnv):
    """CIPP environment with time-varying location reward context."""

    def __init__(
        self,
        instance: CIPPInstance,
        scenario: AdaptiveScenario,
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
    """Context-conditioned features with static checkpoint compatibility."""

    def build(self, environment: CIPPEnv) -> StructuredState:
        state = super().build(environment)
        if not isinstance(environment, AdaptiveCIPPEnv):
            return state

        locations = state.locations.copy()
        global_features = state.global_features.copy()
        multiplier = environment.current_multiplier
        effective_rewards = self.instance.rewards * multiplier

        locations[:, REWARD_FEATURE_INDEX] = (
            effective_rewards / self.scale.reward
        ).astype(np.float32)

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

        order = np.argsort(np.argsort(effective_rewards))
        reward_rank = order.astype(np.float64) / max(self.instance.n - 1, 1)
        locations[:, REWARD_RANK_FEATURE_INDEX] = reward_rank.astype(np.float32)

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
    scenario: AdaptiveScenario,
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
    scenario: AdaptiveScenario,
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
