"""Stable PPO implementation for structured CIPP policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from src.advanced.features import StructuredFeatureBuilder, StructuredState
from src.advanced.networks import NetworkConfig, build_network, set_active_search_mode


AdvantageMode = Literal["gae", "group_relative", "hybrid"]


@dataclass(frozen=True, slots=True)
class AdvancedPPOConfig:
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    discount_factor: float = 1.0
    gae_lambda: float = 1.0
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_loss_coefficient: float = 0.5
    q_loss_coefficient: float = 0.10
    count_loss_coefficient: float = 0.05
    entropy_start: float = 0.02
    entropy_end: float = 0.001
    target_kl: float = 0.02
    gradient_clip_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 512
    reward_target_scale: float = 10.0
    normalize_advantages: bool = True
    advantage_mode: AdvantageMode = "gae"
    group_advantage_weight: float = 0.5
    active_search_mode: Literal["full", "eas"] = "full"


@dataclass(frozen=True, slots=True)
class AdvancedPPOBatch:
    locations: np.ndarray
    global_features: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    final_visit_counts: np.ndarray

    @property
    def size(self) -> int:
        return int(self.actions.size)


def masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    masks = masks.to(dtype=torch.bool, device=logits.device)
    if logits.shape != masks.shape:
        raise ValueError("policy logits and action masks must have equal shapes")
    if not bool(masks.any(dim=-1).all().item()):
        raise ValueError("each policy row must have at least one feasible action")
    masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked_logits)


def compute_episode_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    discount_factor: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rewards.ndim != 1 or values.shape != rewards.shape:
        raise ValueError("rewards and values must be equal-length vectors")
    advantages = np.zeros_like(rewards)
    running = 0.0
    for index in range(rewards.size - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < rewards.size else 0.0
        delta = rewards[index] + discount_factor * next_value - values[index]
        running = delta + discount_factor * gae_lambda * running
        advantages[index] = running
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


class AdvancedPPOAgent:
    """PPO agent shared by Stable-MLP, Attention, Hierarchical, and HACIPP."""

    model_type = "advanced_cipp_ppo_v1"

    def __init__(
        self,
        *,
        feature_builder: StructuredFeatureBuilder,
        network_config: NetworkConfig,
        ppo_config: AdvancedPPOConfig,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.feature_builder = feature_builder
        self.instance = feature_builder.instance
        self.network_config = network_config
        self.config = ppo_config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.network = build_network(
            number_of_locations=self.instance.n,
            location_dim=feature_builder.location_dim,
            global_dim=feature_builder.global_dim,
            config=network_config,
        ).to(self.device)
        set_active_search_mode(self.network, ppo_config.active_search_mode)
        actor_parameters: list[nn.Parameter] = []
        critic_parameters: list[nn.Parameter] = []
        for name, parameter in self.network.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(fragment in name for fragment in ("critic", "value", "q_")):
                critic_parameters.append(parameter)
            else:
                actor_parameters.append(parameter)
        groups: list[dict[str, Any]] = []
        if actor_parameters:
            groups.append({"params": actor_parameters, "lr": ppo_config.actor_learning_rate})
        if critic_parameters:
            groups.append({"params": critic_parameters, "lr": ppo_config.critic_learning_rate})
        if not groups:
            raise ValueError("active-search configuration left no trainable parameters")
        self.optimizer = torch.optim.Adam(groups)
        self._base_group_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

    @property
    def reward_scale(self) -> float:
        return self.config.reward_target_scale / self.feature_builder.scale.objective

    def _tensors(self, state: StructuredState) -> tuple[torch.Tensor, ...]:
        return self._batched_tensors([state])

    def _batched_tensors(
        self, states: list[StructuredState]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not states:
            raise ValueError("states must not be empty")
        locations = torch.as_tensor(
            np.stack([state.locations for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        global_features = torch.as_tensor(
            np.stack([state.global_features for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.as_tensor(
            np.stack([state.action_mask for state in states]),
            dtype=torch.bool,
            device=self.device,
        )
        return locations, global_features, masks

    def batch_actions(
        self,
        states: list[StructuredState],
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for many synchronized environments in one network call.

        This preserves the number of trajectories and PPO samples while avoiding
        one batch-size-one GPU forward pass per trajectory and day.
        """

        was_training = self.network.training
        self.network.eval()
        with torch.inference_mode():
            locations, global_features, masks = self._batched_tensors(states)
            output = self.network(locations, global_features, masks)
            distribution = masked_distribution(output.logits, masks)
            actions = (
                torch.argmax(distribution.logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_probabilities = distribution.log_prob(actions)
        self.network.train(was_training)
        return (
            actions.cpu().numpy().astype(np.int64, copy=False),
            log_probabilities.cpu().numpy().astype(np.float64, copy=False),
            output.values.cpu().numpy().astype(np.float64, copy=False),
        )

    def batch_probabilities_and_values(
        self, states: list[StructuredState]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return masked action probabilities and values in one batched call."""

        was_training = self.network.training
        self.network.eval()
        with torch.inference_mode():
            locations, global_features, masks = self._batched_tensors(states)
            output = self.network(locations, global_features, masks)
            distribution = masked_distribution(output.logits, masks)
            probabilities = distribution.probs.cpu().numpy().astype(np.float64, copy=False)
            values = output.values.cpu().numpy().astype(np.float64, copy=False)
        self.network.train(was_training)
        return probabilities, values

    def select_action(
        self, state: StructuredState, *, deterministic: bool = False
    ) -> tuple[int, float, float]:
        actions, log_probabilities, values = self.batch_actions(
            [state], deterministic=deterministic
        )
        return int(actions[0]), float(log_probabilities[0]), float(values[0])

    def probabilities_and_value(self, state: StructuredState) -> tuple[np.ndarray, float]:
        probabilities, values = self.batch_probabilities_and_values([state])
        return probabilities[0], float(values[0])

    def entropy_coefficient(self, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        return self.config.entropy_start + progress * (
            self.config.entropy_end - self.config.entropy_start
        )

    def set_learning_rate_fraction(self, fraction: float) -> None:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        for group, base_lr in zip(self.optimizer.param_groups, self._base_group_lrs):
            group["lr"] = base_lr * fraction

    def update(self, batch: AdvancedPPOBatch, *, progress: float) -> dict[str, float]:
        if batch.size < 1:
            raise ValueError("PPO batch must not be empty")
        tensors = {
            "locations": torch.as_tensor(batch.locations, dtype=torch.float32, device=self.device),
            "global": torch.as_tensor(batch.global_features, dtype=torch.float32, device=self.device),
            "masks": torch.as_tensor(batch.action_masks, dtype=torch.bool, device=self.device),
            "actions": torch.as_tensor(batch.actions, dtype=torch.long, device=self.device),
            "old_logp": torch.as_tensor(batch.old_log_probabilities, dtype=torch.float32, device=self.device),
            "old_values": torch.as_tensor(batch.old_values, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device),
            "counts": torch.as_tensor(batch.final_visit_counts, dtype=torch.long, device=self.device),
        }
        if self.config.normalize_advantages and batch.size > 1:
            advantages = tensors["advantages"]
            tensors["advantages"] = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "q_loss": 0.0,
            "count_loss": 0.0,
            "normalized_entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
        }
        minibatches = 0
        epochs_completed = 0
        early_kl_stop = False
        entropy_coefficient = self.entropy_coefficient(progress)

        for _ in range(self.config.update_epochs):
            epoch_kls: list[float] = []
            permutation = self.rng.permutation(batch.size)
            for start in range(0, batch.size, self.config.minibatch_size):
                indices_np = permutation[
                    start : start + self.config.minibatch_size
                ]
                indices = torch.as_tensor(indices_np, dtype=torch.long, device=self.device)
                output = self.network(
                    tensors["locations"][indices],
                    tensors["global"][indices],
                    tensors["masks"][indices],
                )
                distribution = masked_distribution(output.logits, tensors["masks"][indices])
                new_logp = distribution.log_prob(tensors["actions"][indices])
                log_ratio = new_logp - tensors["old_logp"][indices]
                ratio = torch.exp(log_ratio)
                advantages = tensors["advantages"][indices]
                policy_loss = -torch.min(
                    ratio * advantages,
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * advantages,
                ).mean()

                old_values = tensors["old_values"][indices]
                clipped_values = old_values + torch.clamp(
                    output.values - old_values,
                    -self.config.value_clip_epsilon,
                    self.config.value_clip_epsilon,
                )
                target_returns = tensors["returns"][indices]
                value_loss = torch.maximum(
                    F.smooth_l1_loss(output.values, target_returns, reduction="none"),
                    F.smooth_l1_loss(clipped_values, target_returns, reduction="none"),
                ).mean()
                selected_q = output.q_values.gather(
                    1, tensors["actions"][indices].unsqueeze(1)
                ).squeeze(1)
                q_loss = F.smooth_l1_loss(selected_q, target_returns)

                count_loss = torch.zeros((), device=self.device)
                if output.count_logits is not None:
                    count_targets = tensors["counts"][indices].clamp(
                        0, self.network_config.max_visit_count
                    )
                    count_loss = F.cross_entropy(
                        output.count_logits.reshape(-1, output.count_logits.shape[-1]),
                        count_targets.reshape(-1),
                    )

                valid_count = tensors["masks"][indices].sum(dim=-1).float()
                entropy_denominator = torch.log(torch.clamp(valid_count, min=2.0))
                normalized_entropy = (distribution.entropy() / entropy_denominator).mean()
                loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    + self.config.q_loss_coefficient * q_loss
                    + self.config.count_loss_coefficient * count_loss
                    - entropy_coefficient * normalized_entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.network.parameters() if parameter.requires_grad],
                    self.config.gradient_clip_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float().mean()
                    )
                values = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "q_loss": q_loss,
                    "count_loss": count_loss,
                    "normalized_entropy": normalized_entropy,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                }
                for key, value in values.items():
                    metrics[key] += float(value.detach().cpu().item())
                epoch_kls.append(float(approximate_kl.detach().cpu().item()))
                minibatches += 1
            epochs_completed += 1
            if epoch_kls and float(np.mean(epoch_kls)) > self.config.target_kl:
                early_kl_stop = True
                break

        result = {key: value / max(minibatches, 1) for key, value in metrics.items()}
        result.update(
            {
                "epochs_completed": float(epochs_completed),
                "early_kl_stop": float(early_kl_stop),
                "entropy_coefficient": float(entropy_coefficient),
                "learning_rate_fraction": float(1.0 - progress),
            }
        )
        return result

    def self_imitation_update(
        self,
        states: list[StructuredState],
        actions: list[int],
        *,
        epochs: int = 1,
    ) -> float:
        """Masked behavior cloning on trajectories generated by this RL agent."""

        if not states:
            return 0.0
        locations = torch.as_tensor(
            np.stack([state.locations for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        global_features = torch.as_tensor(
            np.stack([state.global_features for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.as_tensor(
            np.stack([state.action_mask for state in states]),
            dtype=torch.bool,
            device=self.device,
        )
        targets = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        losses: list[float] = []
        for _ in range(max(epochs, 1)):
            output = self.network(locations, global_features, masks)
            distribution = masked_distribution(output.logits, masks)
            loss = -distribution.log_prob(targets).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in self.network.parameters() if parameter.requires_grad],
                self.config.gradient_clip_norm,
            )
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses))

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "network_config": asdict(self.network_config),
                "ppo_config": asdict(self.config),
                "instance": {
                    "instance_id": self.instance.instance_id,
                    "n": self.instance.n,
                    "H": self.instance.H,
                    "q": self.instance.q,
                },
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metadata": metadata,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        feature_builder: StructuredFeatureBuilder,
        device: str = "cpu",
        load_optimizer: bool = True,
    ) -> tuple["AdvancedPPOAgent", dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("model_type") != cls.model_type:
            raise ValueError("checkpoint is not an advanced CIPP PPO model")
        expected = payload["instance"]
        actual = feature_builder.instance
        if int(expected["n"]) != actual.n or int(expected["H"]) != actual.H:
            raise ValueError("checkpoint and requested instance dimensions differ")
        agent = cls(
            feature_builder=feature_builder,
            network_config=NetworkConfig(**payload["network_config"]),
            ppo_config=AdvancedPPOConfig(**payload["ppo_config"]),
            device=device,
        )
        agent.network.load_state_dict(payload["network_state_dict"])
        if load_optimizer:
            agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return agent, payload
