"""Fixed-capacity replay buffer for masked DQN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """One sampled DQN minibatch."""

    states: FloatArray
    actions: IntArray
    rewards: FloatArray
    next_states: FloatArray
    dones: BoolArray
    action_masks: BoolArray
    next_action_masks: BoolArray


class ReplayBuffer:
    """Preallocated replay storage including action masks."""

    def __init__(
        self,
        *,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError(
                "capacity must be positive."
            )

        if observation_dim < 1:
            raise ValueError(
                "observation_dim must be positive."
            )

        if action_dim < 1:
            raise ValueError(
                "action_dim must be positive."
            )

        self.capacity = int(
            capacity
        )

        self.observation_dim = int(
            observation_dim
        )

        self.action_dim = int(
            action_dim
        )

        self._rng = np.random.default_rng(
            seed
        )

        self._position = 0
        self._size = 0

        self._states = np.empty(
            (
                self.capacity,
                self.observation_dim,
            ),
            dtype=np.float32,
        )

        self._actions = np.empty(
            self.capacity,
            dtype=np.int64,
        )

        self._rewards = np.empty(
            self.capacity,
            dtype=np.float32,
        )

        self._next_states = np.empty(
            (
                self.capacity,
                self.observation_dim,
            ),
            dtype=np.float32,
        )

        self._dones = np.empty(
            self.capacity,
            dtype=np.bool_,
        )

        self._action_masks = np.empty(
            (
                self.capacity,
                self.action_dim,
            ),
            dtype=np.bool_,
        )

        self._next_action_masks = np.empty(
            (
                self.capacity,
                self.action_dim,
            ),
            dtype=np.bool_,
        )

    def __len__(
        self,
    ) -> int:
        """Return number of currently stored transitions."""

        return self._size

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        """Store one transition."""

        state_values = np.asarray(
            state,
            dtype=np.float32,
        )

        next_state_values = np.asarray(
            next_state,
            dtype=np.float32,
        )

        current_mask = np.asarray(
            action_mask,
            dtype=np.bool_,
        )

        following_mask = np.asarray(
            next_action_mask,
            dtype=np.bool_,
        )

        if state_values.shape != (
            self.observation_dim,
        ):
            raise ValueError(
                "state has the wrong shape."
            )

        if next_state_values.shape != (
            self.observation_dim,
        ):
            raise ValueError(
                "next_state has the wrong shape."
            )

        if current_mask.shape != (
            self.action_dim,
        ):
            raise ValueError(
                "action_mask has the wrong shape."
            )

        if following_mask.shape != (
            self.action_dim,
        ):
            raise ValueError(
                "next_action_mask has the wrong shape."
            )

        if (
            action < 0
            or action >= self.action_dim
        ):
            raise ValueError(
                "action is outside the replay buffer action range."
            )

        if not bool(
            current_mask[action]
        ):
            raise ValueError(
                "stored action must be allowed by action_mask."
            )

        if (
            not done
            and not bool(
                following_mask.any()
            )
        ):
            raise ValueError(
                "a nonterminal transition must have "
                "a viable next action."
            )

        index = self._position

        self._states[index] = (
            state_values
        )

        self._actions[index] = int(
            action
        )

        self._rewards[index] = float(
            reward
        )

        self._next_states[index] = (
            next_state_values
        )

        self._dones[index] = bool(
            done
        )

        self._action_masks[index] = (
            current_mask
        )

        self._next_action_masks[index] = (
            following_mask
        )

        self._position = (
            self._position + 1
        ) % self.capacity

        self._size = min(
            self._size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size: int,
    ) -> ReplayBatch:
        """Sample transitions uniformly without replacement."""

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        if self._size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} transitions "
                f"from buffer size {self._size}."
            )

        indices = self._rng.choice(
            self._size,
            size=batch_size,
            replace=False,
        )

        return ReplayBatch(
            states=self._states[
                indices
            ].copy(),
            actions=self._actions[
                indices
            ].copy(),
            rewards=self._rewards[
                indices
            ].copy(),
            next_states=self._next_states[
                indices
            ].copy(),
            dones=self._dones[
                indices
            ].copy(),
            action_masks=self._action_masks[
                indices
            ].copy(),
            next_action_masks=(
                self._next_action_masks[
                    indices
                ].copy()
            ),
        )