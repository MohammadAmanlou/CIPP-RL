"""Feasibility-masked categorical PPO for deterministic CIPP."""

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
from torch.distributions import Categorical


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO architecture and optimization hyperparameters."""

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    reward_scale: float = 1e-3
    discount_factor: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gradient_clip_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 256
    normalize_advantages: bool = True


@dataclass(frozen=True, slots=True)
class PPOBatch:
    """One frozen on-policy rollout batch."""

    states: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    action_masks: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of transitions."""

        return int(
            self.actions.size
        )


class ActorCriticNetwork(nn.Module):
    """Shared MLP with categorical-policy and scalar-value heads."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return policy logits and state values."""

        features = self.backbone(
            observations
        )

        logits = self.policy_head(
            features
        )

        values = self.value_head(
            features
        ).squeeze(-1)

        return logits, values


def masked_categorical(
    logits: torch.Tensor,
    action_masks: torch.Tensor,
) -> Categorical:
    """Construct a categorical distribution after feasibility masking."""

    masks = action_masks.to(
        dtype=torch.bool,
        device=logits.device,
    )

    if masks.shape != logits.shape:
        raise ValueError(
            "action_masks must have the same shape as policy logits."
        )

    if not bool(
        masks.any(
            dim=-1
        ).all().item()
    ):
        raise ValueError(
            "every policy row must contain at least one viable action."
        )

    masked_logits = logits.masked_fill(
        ~masks,
        torch.finfo(
            logits.dtype
        ).min,
    )

    return Categorical(
        logits=masked_logits
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    discount_factor: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Compute generalized advantages across one or more complete episodes."""

    reward_values = np.asarray(
        rewards,
        dtype=np.float64,
    )

    state_values = np.asarray(
        values,
        dtype=np.float64,
    )

    terminal_flags = np.asarray(
        dones,
        dtype=np.bool_,
    )

    if (
        reward_values.ndim != 1
        or state_values.shape
        != reward_values.shape
        or terminal_flags.shape
        != reward_values.shape
    ):
        raise ValueError(
            "rewards, values, and dones must be equal-length vectors."
        )

    advantages = np.zeros_like(
        reward_values
    )

    running_advantage = 0.0

    for step in range(
        reward_values.size - 1,
        -1,
        -1,
    ):
        if step == reward_values.size - 1:
            next_value = float(
                last_value
            )
        else:
            next_value = float(
                state_values[
                    step + 1
                ]
            )

        nonterminal = (
            0.0
            if terminal_flags[step]
            else 1.0
        )

        temporal_difference = (
            reward_values[step]
            + discount_factor
            * next_value
            * nonterminal
            - state_values[step]
        )

        running_advantage = (
            temporal_difference
            + discount_factor
            * gae_lambda
            * nonterminal
            * running_advantage
        )

        advantages[step] = (
            running_advantage
        )

    returns = (
        advantages + state_values
    )

    return (
        advantages.astype(
            np.float32
        ),
        returns.astype(
            np.float32
        ),
    )


class PPOAgent:
    """Categorical PPO with masking in sampling, evaluation, and updates."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
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
            or PPOConfig()
        )

        if (
            not np.isfinite(
                self.config.reward_scale
            )
            or self.config.reward_scale
            <= 0.0
        ):
            raise ValueError(
                "reward_scale must be finite and positive."
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

        self.network = ActorCriticNetwork(
            observation_dim=(
                self.observation_dim
            ),
            action_dim=self.action_dim,
            hidden_dim=(
                self.config.hidden_dim
            ),
        ).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
        )

    def _validated_inputs(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
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

        return state, mask

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[
        int,
        float,
        float,
    ]:
        """Select one feasible action and return action, log-probability, value."""

        state, mask = (
            self._validated_inputs(
                observation,
                action_mask,
            )
        )

        self.network.eval()

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            mask_tensor = torch.as_tensor(
                mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)

            logits, value = self.network(
                state_tensor
            )

            distribution = (
                masked_categorical(
                    logits,
                    mask_tensor,
                )
            )

            if deterministic:
                masked_logits = (
                    distribution.logits
                )

                action_tensor = torch.argmax(
                    masked_logits,
                    dim=-1,
                )
            else:
                action_tensor = (
                    distribution.sample()
                )

            log_probability = (
                distribution.log_prob(
                    action_tensor
                )
            )

        self.network.train()

        return (
            int(
                action_tensor.item()
            ),
            float(
                log_probability.item()
            ),
            float(
                value.item()
            ),
        )

    def action_probabilities(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Return a normalized probability vector with zeros on masked actions."""

        state, mask = (
            self._validated_inputs(
                observation,
                action_mask,
            )
        )

        self.network.eval()

        with torch.no_grad():
            logits, _ = self.network(
                torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
            )

            distribution = (
                masked_categorical(
                    logits,
                    torch.as_tensor(
                        mask,
                        dtype=torch.bool,
                        device=self.device,
                    ).unsqueeze(0),
                )
            )

            probabilities = (
                distribution.probs
                .squeeze(0)
                .cpu()
                .numpy()
            )

        self.network.train()

        return probabilities.astype(
            np.float64
        )

    def update(
        self,
        batch: PPOBatch,
    ) -> dict[str, float]:
        """Perform clipped PPO updates on one on-policy batch."""

        if batch.size < 1:
            raise ValueError(
                "PPO batch must not be empty."
            )

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

        old_log_probabilities = (
            torch.as_tensor(
                batch.old_log_probabilities,
                dtype=torch.float32,
                device=self.device,
            )
        )

        returns = torch.as_tensor(
            batch.returns,
            dtype=torch.float32,
            device=self.device,
        )

        advantages = torch.as_tensor(
            batch.advantages,
            dtype=torch.float32,
            device=self.device,
        )

        action_masks = torch.as_tensor(
            batch.action_masks,
            dtype=torch.bool,
            device=self.device,
        )

        if bool(
            (
                ~action_masks[
                    torch.arange(
                        batch.size,
                        device=self.device,
                    ),
                    actions,
                ]
            ).any().item()
        ):
            raise ValueError(
                "PPO batch contains an action that was masked."
            )

        if (
            self.config.normalize_advantages
            and batch.size > 1
        ):
            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std(
                    unbiased=False
                )
                + 1e-8
            )

        metric_totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
        }

        number_of_minibatches = 0

        for _ in range(
            self.config.update_epochs
        ):
            permutation = (
                self._rng.permutation(
                    batch.size
                )
            )

            for start in range(
                0,
                batch.size,
                self.config.minibatch_size,
            ):
                indices = permutation[
                    start:
                    start
                    + self.config.minibatch_size
                ]

                index_tensor = torch.as_tensor(
                    indices,
                    dtype=torch.int64,
                    device=self.device,
                )

                logits, predicted_values = (
                    self.network(
                        states[
                            index_tensor
                        ]
                    )
                )

                distribution = (
                    masked_categorical(
                        logits,
                        action_masks[
                            index_tensor
                        ],
                    )
                )

                new_log_probabilities = (
                    distribution.log_prob(
                        actions[
                            index_tensor
                        ]
                    )
                )

                entropy = (
                    distribution.entropy()
                    .mean()
                )

                log_ratio = (
                    new_log_probabilities
                    - old_log_probabilities[
                        index_tensor
                    ]
                )

                ratio = torch.exp(
                    log_ratio
                )

                minibatch_advantages = (
                    advantages[
                        index_tensor
                    ]
                )

                unclipped_objective = (
                    ratio
                    * minibatch_advantages
                )

                clipped_objective = (
                    torch.clamp(
                        ratio,
                        1.0
                        - self.config.clip_epsilon,
                        1.0
                        + self.config.clip_epsilon,
                    )
                    * minibatch_advantages
                )

                policy_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                value_loss = (
                    0.5
                    * (
                        predicted_values
                        - returns[
                            index_tensor
                        ]
                    )
                    .pow(2)
                    .mean()
                )

                loss = (
                    policy_loss
                    + self.config
                    .value_loss_coefficient
                    * value_loss
                    - self.config
                    .entropy_coefficient
                    * entropy
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.gradient_clip_norm,
                )

                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = (
                        (
                            torch.exp(
                                log_ratio
                            )
                            - 1
                            - log_ratio
                        )
                        .mean()
                    )

                    clip_fraction = (
                        (
                            torch.abs(
                                ratio - 1.0
                            )
                            > self.config
                            .clip_epsilon
                        )
                        .float()
                        .mean()
                    )

                values_to_add = {
                    "loss": loss,
                    "policy_loss": (
                        policy_loss
                    ),
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approximate_kl": (
                        approximate_kl
                    ),
                    "clip_fraction": (
                        clip_fraction
                    ),
                }

                for key, value in (
                    values_to_add.items()
                ):
                    metric_totals[key] += float(
                        value.detach()
                        .cpu()
                        .item()
                    )

                number_of_minibatches += 1

        return {
            key: value
            / number_of_minibatches
            for key, value
            in metric_totals.items()
        }

    def checkpoint_payload(
        self,
        *,
        normalizer: dict[str, Any],
        training_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct a complete reproducible PPO checkpoint."""

        return {
            "model_type": "masked_categorical_ppo",
            "observation_dim": (
                self.observation_dim
            ),
            "action_dim": (
                self.action_dim
            ),
            "config": asdict(
                self.config
            ),
            "network_state_dict": (
                self.network.state_dict()
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
        """Save network, optimizer, normalizer, and metadata."""

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
        "PPOAgent",
        dict[str, Any],
    ]:
        """Restore a PPO agent and its checkpoint payload."""

        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        if payload.get(
            "model_type"
        ) != "masked_categorical_ppo":
            raise ValueError(
                "checkpoint is not a masked categorical PPO model."
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
            config=PPOConfig(
                **payload["config"]
            ),
            seed=0,
            device=device,
        )

        agent.network.load_state_dict(
            payload[
                "network_state_dict"
            ]
        )

        agent.optimizer.load_state_dict(
            payload[
                "optimizer_state_dict"
            ]
        )

        return agent, payload
