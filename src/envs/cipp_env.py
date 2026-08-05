"""Gym-style deterministic CIPP environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core import (
    CIPPInstance,
    evaluate_itinerary,
)

from src.envs.masking import (
    get_viability_mask,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(
    frozen=True,
    slots=True,
)
class CIPPEnvSnapshot:
    """Read-only snapshot of the environment."""

    day: int
    remaining_budget: float
    itinerary: tuple[int, ...]
    visit_counts: tuple[int, ...]
    temporal_exposures: tuple[float, ...]
    cumulative_reward: float
    done: bool


class CIPPEnv:
    """Sequential environment for one deterministic CIPP instance.

    Action encoding:

    - 0 means Idle;
    - actions 1 through n visit a location.
    """

    def __init__(
        self,
        instance: CIPPInstance,
        seed: int | None = None,
        viability_cache: dict[
            tuple[int, ...],
            BoolArray,
        ]
        | None = None,
    ) -> None:
        self.instance = instance
        self._rng = np.random.default_rng(
            seed
        )

        self._itinerary: list[int] = []

        self._visit_counts = np.zeros(
            instance.n,
            dtype=np.int64,
        )

        self._temporal_exposures = np.zeros(
            instance.n,
            dtype=np.float64,
        )

        self._spent_budget = 0.0
        self._cumulative_reward = 0.0
        self._done = False
        self._cached_mask_day = -1
        self._cached_action_mask: BoolArray | None = None
        self._viability_cache = viability_cache

    @property
    def day(self) -> int:
        """Current zero-based day index."""

        return len(self._itinerary)

    @property
    def done(self) -> bool:
        """Whether the episode has ended."""

        return self._done

    @property
    def itinerary(self) -> IntArray:
        """Return a copy of the current itinerary."""

        return np.asarray(
            self._itinerary,
            dtype=np.int64,
        ).copy()

    @property
    def visit_counts(self) -> IntArray:
        """Return a copy of location visit counts."""

        return self._visit_counts.copy()

    @property
    def temporal_exposures(self) -> FloatArray:
        """Return accumulated temporal exposures."""

        return self._temporal_exposures.copy()

    @property
    def remaining_budget(self) -> float:
        """Return the unspent campaign budget."""

        return float(
            self.instance.budget
            - self._spent_budget
        )

    @property
    def cumulative_reward(self) -> float:
        """Return the sum of received rewards."""

        return float(
            self._cumulative_reward
        )

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[
        FloatArray,
        dict[str, Any],
    ]:
        """Reset the environment to day zero."""

        if seed is not None:
            self._rng = np.random.default_rng(
                seed
            )

        self._itinerary = []

        self._visit_counts = np.zeros(
            self.instance.n,
            dtype=np.int64,
        )

        self._temporal_exposures = np.zeros(
            self.instance.n,
            dtype=np.float64,
        )

        self._spent_budget = 0.0
        self._cumulative_reward = 0.0
        self._done = False
        self._cached_mask_day = -1
        self._cached_action_mask = None

        return (
            self.get_observation(),
            self._info(),
        )

    def get_action_mask(self) -> BoolArray:
        """Return the current viability mask.

        The mask is cached for the current day because action selection and
        ``step`` request the same mask repeatedly during DQN training.
        """

        prefix = tuple(self._itinerary)

        if self._viability_cache is not None:
            shared_mask = self._viability_cache.get(prefix)

            if shared_mask is None:
                shared_mask = get_viability_mask(
                    self.instance,
                    prefix,
                )
                self._viability_cache[prefix] = shared_mask.copy()

            return shared_mask.copy()

        if self._cached_action_mask is None or self._cached_mask_day != self.day:
            self._cached_action_mask = get_viability_mask(
                self.instance,
                self._itinerary,
            )
            self._cached_mask_day = self.day
        return self._cached_action_mask.copy()

    def compute_reward(
        self,
        action: int,
    ) -> float:
        """Compute the exact objective increment.

        This method does not change the environment.
        """

        if self._done:
            raise RuntimeError(
                "Cannot compute reward after episode termination."
            )

        if isinstance(action, bool) or not isinstance(
            action,
            (int, np.integer),
        ):
            raise TypeError(
                "action must be an integer."
            )

        normalized_action = int(action)

        if (
            normalized_action < 0
            or normalized_action > self.instance.n
        ):
            raise ValueError(
                "action must be between "
                f"0 and {self.instance.n}."
            )

        action_mask = self.get_action_mask()

        if not bool(
            action_mask[normalized_action]
        ):
            raise ValueError(
                f"Action {normalized_action} "
                f"is not viable on day {self.day + 1}."
            )

        if normalized_action == 0:
            return 0.0

        location_index = (
            normalized_action - 1
        )

        old_count = int(
            self._visit_counts[
                location_index
            ]
        )

        new_count = old_count + 1

        old_exposure = float(
            self._temporal_exposures[
                location_index
            ]
        )

        new_exposure = (
            old_exposure
            + float(
                self.instance.temporal_weights[
                    self.day
                ]
            )
        )

        old_contribution = (
            float(
                self.instance.rewards[
                    location_index
                ]
            )
            * self.instance.repeat_factor(
                old_count
            )
            * old_exposure
        )

        new_contribution = (
            float(
                self.instance.rewards[
                    location_index
                ]
            )
            * self.instance.repeat_factor(
                new_count
            )
            * new_exposure
        )

        return float(
            new_contribution
            - old_contribution
        )

    def step(
        self,
        action: int,
    ) -> tuple[
        FloatArray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Apply one action and advance by one day."""

        if self._done:
            raise RuntimeError(
                "Cannot call step after episode termination."
            )

        reward = self.compute_reward(
            action
        )

        normalized_action = int(action)

        if normalized_action > 0:
            location_index = (
                normalized_action - 1
            )

            self._visit_counts[
                location_index
            ] += 1

            self._temporal_exposures[
                location_index
            ] += self.instance.temporal_weights[
                self.day
            ]

            self._spent_budget += float(
                self.instance.costs[
                    location_index
                ]
            )

        self._itinerary.append(
            normalized_action
        )
        self._cached_action_mask = None
        self._cached_mask_day = -1

        self._cumulative_reward += (
            reward
        )

        self._done = (
            self.day == self.instance.H
        )

        info = self._info()

        if self._done:
            final_result = evaluate_itinerary(
                self.instance,
                self._itinerary,
            )

            info["final_evaluation"] = (
                final_result.to_dict()
            )

            if not final_result.feasible:
                raise RuntimeError(
                    "Environment produced an infeasible "
                    "final itinerary: "
                    + "; ".join(
                        final_result.violations
                    )
                )

        observation = (
            self.get_observation()
        )

        terminated = self._done
        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def sample_viable_action(
        self,
    ) -> int:
        """Sample one random action from the viability mask."""

        valid_actions = np.flatnonzero(
            self.get_action_mask()
        )

        if valid_actions.size == 0:
            raise RuntimeError(
                "No viable action is available."
            )

        return int(
            self._rng.choice(
                valid_actions
            )
        )

    def get_observation(
        self,
    ) -> FloatArray:
        """Construct a complete Markov observation.

        The observation includes:

        - current day;
        - remaining budget;
        - current temporal coefficient;
        - instance parameters;
        - visit counts;
        - temporal exposures;
        - future temporal coefficients;
        - idle requirements;
        - last alpha-1 actions.

        History padding uses -1, which is different from Idle=0.
        """

        if not self._done:
            current_lambda = float(
                self.instance.temporal_weights[
                    self.day
                ]
            )
        else:
            current_lambda = 0.0

        history_length = max(
            0,
            self.instance.alpha - 1,
        )

        history = np.full(
            history_length,
            -1.0,
            dtype=np.float64,
        )

        if history_length:
            recent_actions = (
                self._itinerary[
                    -history_length:
                ]
            )

            if recent_actions:
                history[
                    -len(recent_actions):
                ] = recent_actions

        global_features = np.array(
            [
                float(self.day),
                float(self.instance.H),
                self.remaining_budget,
                current_lambda,
                float(self.instance.alpha),
                float(self.instance.q),
                float(self.instance.w),
                float(self.instance.gamma),
                float(self.instance.repeat_count_offset),
                float(self.instance.p),
            ],
            dtype=np.float64,
        )

        observation = np.concatenate(
            [
                global_features,
                self.instance.rewards,
                self.instance.costs,
                self._visit_counts.astype(
                    np.float64
                ),
                self._temporal_exposures,
                self.instance.temporal_weights,
                self.instance.idle_requirements.astype(
                    np.float64
                ),
                history,
            ]
        )

        return observation

    def get_snapshot(
        self,
    ) -> CIPPEnvSnapshot:
        """Return a read-only state snapshot."""

        return CIPPEnvSnapshot(
            day=self.day,
            remaining_budget=(
                self.remaining_budget
            ),
            itinerary=tuple(
                self._itinerary
            ),
            visit_counts=tuple(
                int(value)
                for value
                in self._visit_counts
            ),
            temporal_exposures=tuple(
                float(value)
                for value
                in self._temporal_exposures
            ),
            cumulative_reward=(
                self.cumulative_reward
            ),
            done=self.done,
        )

    def _info(
        self,
    ) -> dict[str, Any]:
        """Construct the environment information dictionary."""

        return {
            "day": self.day,
            "remaining_budget": (
                self.remaining_budget
            ),
            "cumulative_reward": (
                self.cumulative_reward
            ),
            "action_mask": (
                self.get_action_mask()
            ),
        }
