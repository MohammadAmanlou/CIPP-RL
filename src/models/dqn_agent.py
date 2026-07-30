"""Masked Deep Q-Network agent for Week 3 CIPP instances."""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)

from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from src.models.replay_buffer import (
    ReplayBatch,
)


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """DQN model and optimizer configuration."""

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gradient_clip_norm: float = 1.0
    double_dqn: bool = True


class QNetwork(nn.Module):
    """Fixed-size MLP Q-network for the Week 3 baseline."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return one Q-value per action."""

        return self.network(
            observations
        )


class DQNAgent:
    """DQN with masking during behavior and Bellman updates."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if (
            observation_dim < 1
            or action_dim < 1
        ):
            raise ValueError(
                "observation_dim and action_dim must be positive."
            )

        self.observation_dim = int(
            observation_dim
        )

        self.action_dim = int(
            action_dim
        )

        self.config = (
            config
            or DQNConfig()
        )

        self.device = torch.device(
            device
        )

        self._rng = np.random.default_rng(
            seed
        )

        torch.manual_seed(
            seed
        )

        self.online_network = QNetwork(
            self.observation_dim,
            self.action_dim,
            self.config.hidden_dim,
        ).to(
            self.device
        )

        self.target_network = QNetwork(
            self.observation_dim,
            self.action_dim,
            self.config.hidden_dim,
        ).to(
            self.device
        )

        self.sync_target_network()

        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )

    def predict_q_values(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return Q-values, optionally replacing masked entries by -inf."""

        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (self.observation_dim,):
            raise ValueError("observation has the wrong shape.")

        was_training = self.online_network.training
        self.online_network.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            values = self.online_network(state_tensor).squeeze(0)

            if action_mask is not None:
                mask = np.asarray(action_mask, dtype=np.bool_)
                if mask.shape != (self.action_dim,):
                    raise ValueError("action_mask has the wrong shape.")
                mask_tensor = torch.as_tensor(
                    mask, dtype=torch.bool, device=self.device
                )
                values = values.masked_fill(~mask_tensor, -torch.inf)

            output = values.detach().cpu().numpy().astype(np.float64, copy=True)

        if was_training:
            self.online_network.train()
        return output

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float,
    ) -> int:
        """Select an action using masked epsilon-greedy."""

        state = np.asarray(
            observation,
            dtype=np.float32,
        )

        mask = np.asarray(
            action_mask,
            dtype=np.bool_,
        )

        if state.shape != (
            self.observation_dim,
        ):
            raise ValueError(
                "observation has the wrong shape."
            )

        if mask.shape != (
            self.action_dim,
        ):
            raise ValueError(
                "action_mask has the wrong shape."
            )

        if not bool(
            mask.any()
        ):
            raise RuntimeError(
                "no viable action is available."
            )

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between zero and one."
            )

        viable_actions = np.flatnonzero(
            mask
        )

        # Exploration is also restricted to viable actions.
        if self._rng.random() < epsilon:
            return int(
                self._rng.choice(
                    viable_actions
                )
            )

        masked_q_values = self.predict_q_values(
            state,
            mask,
        )

        return int(np.argmax(masked_q_values))

    def optimize(
        self,
        batch: ReplayBatch,
    ) -> float:
        """Perform one masked DQN optimization step."""

        states = torch.as_tensor(
            batch.states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            batch.actions,
            dtype=torch.int64,
            device=self.device,
        )

        rewards = torch.as_tensor(
            batch.rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            batch.next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            batch.dones,
            dtype=torch.bool,
            device=self.device,
        )

        next_masks = torch.as_tensor(
            batch.next_action_masks,
            dtype=torch.bool,
            device=self.device,
        )

        invalid_nonterminal = (
            ~dones
            & ~next_masks.any(
                dim=1
            )
        )

        if bool(
            invalid_nonterminal.any().item()
        ):
            raise ValueError(
                "every nonterminal transition must "
                "have a viable next action."
            )

        all_q_values = self.online_network(
            states
        )

        chosen_q_values = all_q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            target_next_q = self.target_network(next_states)

            if self.config.double_dqn:
                online_next_q = self.online_network(next_states)
                masked_online_next_q = online_next_q.masked_fill(
                    ~next_masks, -torch.inf
                )
                next_actions = masked_online_next_q.argmax(dim=1)
                next_max = target_next_q.gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                masked_target_next_q = target_next_q.masked_fill(
                    ~next_masks, -torch.inf
                )
                next_max = masked_target_next_q.max(dim=1).values

            # Terminal states can have an empty next mask and no future value.
            next_max = torch.where(
                dones, torch.zeros_like(next_max), next_max
            )
            targets = rewards + self.config.gamma * next_max

        loss = F.smooth_l1_loss(
            chosen_q_values,
            targets,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip_norm,
        )

        self.optimizer.step()

        return float(
            loss.detach()
            .cpu()
            .item()
        )

    def sync_target_network(
        self,
    ) -> None:
        """Copy online network parameters to target network."""

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def checkpoint_payload(
        self,
        *,
        normalizer: dict[str, Any],
        training_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct a complete reproducible checkpoint."""

        return {
            "observation_dim": (
                self.observation_dim
            ),
            "action_dim": (
                self.action_dim
            ),
            "config": asdict(
                self.config
            ),
            "online_state_dict": (
                self.online_network.state_dict()
            ),
            "target_state_dict": (
                self.target_network.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "normalizer": normalizer,
            "training_metadata": (
                training_metadata
            ),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        normalizer: dict[str, Any],
        training_metadata: dict[str, Any],
    ) -> Path:
        """Save model, optimizer, normalizer, and metadata."""

        output_path = Path(
            path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.checkpoint_payload(
                normalizer=normalizer,
                training_metadata=(
                    training_metadata
                ),
            ),
            output_path,
        )

        return output_path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[
        "DQNAgent",
        dict[str, Any],
    ]:
        """Restore an agent and its checkpoint metadata."""

        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        agent = cls(
            observation_dim=int(
                payload[
                    "observation_dim"
                ]
            ),
            action_dim=int(
                payload[
                    "action_dim"
                ]
            ),
            config=DQNConfig(
                **payload["config"]
            ),
            seed=0,
            device=device,
        )

        agent.online_network.load_state_dict(
            payload[
                "online_state_dict"
            ]
        )

        agent.target_network.load_state_dict(
            payload[
                "target_state_dict"
            ]
        )

        agent.optimizer.load_state_dict(
            payload[
                "optimizer_state_dict"
            ]
        )

        return agent, payload