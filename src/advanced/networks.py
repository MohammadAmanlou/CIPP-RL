"""Neural architectures used by the comparable advanced PPO variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from src.advanced.features import COUNT_FEATURE_INDEX, MARGINAL_FEATURE_INDEX


Architecture = Literal["stable_mlp", "attention", "hierarchical_attention"]


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    architecture: Architecture = "hierarchical_attention"
    hidden_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 2
    dropout: float = 0.0
    residual_marginal: bool = False
    count_planner: bool = False
    max_visit_count: int = 12

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class NetworkOutput:
    logits: torch.Tensor
    values: torch.Tensor
    q_values: torch.Tensor
    count_logits: torch.Tensor | None


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def _joint_hierarchical_logits(
    visit_gate: torch.Tensor,
    location_scores: torch.Tensor,
    action_masks: torch.Tensor,
) -> torch.Tensor:
    """Return normalized Idle/Visit/location logits before final masking."""

    location_masks = action_masks[:, 1:].bool()
    very_negative = torch.finfo(location_scores.dtype).min
    masked_scores = location_scores.masked_fill(~location_masks, very_negative)
    has_location = location_masks.any(dim=-1, keepdim=True)
    safe_scores = torch.where(has_location, masked_scores, torch.zeros_like(masked_scores))
    location_log_probabilities = F.log_softmax(safe_scores, dim=-1)
    idle_log_probability = F.logsigmoid(-visit_gate)
    visit_log_probability = F.logsigmoid(visit_gate)
    return torch.cat(
        [idle_log_probability, visit_log_probability + location_log_probabilities],
        dim=-1,
    )


class StableMLPNetwork(nn.Module):
    """Separate actor and critic MLPs over the same structured state."""

    def __init__(
        self,
        *,
        number_of_locations: int,
        location_dim: int,
        global_dim: int,
        config: NetworkConfig,
    ) -> None:
        super().__init__()
        self.number_of_locations = number_of_locations
        self.config = config
        input_dim = number_of_locations * location_dim + global_dim
        action_dim = number_of_locations + 1
        self.actor = _mlp(input_dim, config.hidden_dim, action_dim)
        self.critic = _mlp(input_dim, config.hidden_dim, 1)
        self.q_head = _mlp(input_dim, config.hidden_dim, action_dim)
        self.instance_adapter = nn.Parameter(torch.zeros(input_dim))

    def forward(
        self,
        locations: torch.Tensor,
        global_features: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> NetworkOutput:
        flat = torch.cat([locations.flatten(start_dim=1), global_features], dim=-1)
        flat = flat + self.instance_adapter.unsqueeze(0)
        raw_logits = self.actor(flat)
        if self.config.architecture == "hierarchical_attention":
            logits = _joint_hierarchical_logits(
                raw_logits[:, :1], raw_logits[:, 1:], action_masks
            )
        else:
            logits = raw_logits
        return NetworkOutput(
            logits=logits,
            values=self.critic(flat).squeeze(-1),
            q_values=self.q_head(flat),
            count_logits=None,
        )


class _TokenEncoder(nn.Module):
    def __init__(self, input_dim: int, config: NetworkConfig) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.attention_layers,
            enable_nested_tensor=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.input(features))


class AttentionActorCritic(nn.Module):
    """Permutation-equivariant actor with a fully separate attention critic."""

    def __init__(
        self,
        *,
        number_of_locations: int,
        location_dim: int,
        global_dim: int,
        config: NetworkConfig,
    ) -> None:
        super().__init__()
        self.number_of_locations = number_of_locations
        self.location_dim = location_dim
        self.config = config

        self.actor_encoder = _TokenEncoder(location_dim, config)
        self.actor_global = nn.Sequential(
            nn.Linear(global_dim, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim)
        )
        self.actor_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.actor_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.location_bias = nn.Linear(config.hidden_dim, 1)
        self.idle_head = _mlp(2 * config.hidden_dim, config.hidden_dim, 1)
        self.visit_gate = _mlp(2 * config.hidden_dim, config.hidden_dim, 1)

        self.critic_encoder = _TokenEncoder(location_dim, config)
        self.critic_global = nn.Sequential(
            nn.Linear(global_dim, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim)
        )
        self.value_head = _mlp(2 * config.hidden_dim, config.hidden_dim, 1)
        self.q_location = nn.Linear(config.hidden_dim, 1)
        self.q_idle = _mlp(2 * config.hidden_dim, config.hidden_dim, 1)

        self.instance_adapter = nn.Parameter(torch.zeros(config.hidden_dim))
        self.marginal_scale = nn.Parameter(torch.tensor(1.0))
        self.count_scale = nn.Parameter(torch.tensor(1.0))
        self.count_head = (
            nn.Linear(config.hidden_dim, config.max_visit_count + 1)
            if config.count_planner
            else None
        )

    def forward(
        self,
        locations: torch.Tensor,
        global_features: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> NetworkOutput:
        actor_tokens = self.actor_encoder(locations)
        global_actor = self.actor_global(global_features) + self.instance_adapter
        pooled_actor = actor_tokens.mean(dim=1)
        context = global_actor + pooled_actor
        query = self.actor_query(context).unsqueeze(1)
        keys = self.actor_key(actor_tokens)
        location_scores = (query * keys).sum(dim=-1) / (keys.shape[-1] ** 0.5)
        location_scores = location_scores + self.location_bias(actor_tokens).squeeze(-1)

        count_logits = self.count_head(actor_tokens) if self.count_head is not None else None
        if count_logits is not None:
            probabilities = F.softmax(count_logits, dim=-1)
            count_values = torch.arange(
                count_logits.shape[-1], dtype=count_logits.dtype, device=count_logits.device
            )
            planned = (probabilities * count_values).sum(dim=-1) / max(
                self.config.max_visit_count, 1
            )
            deficit = planned - locations[:, :, COUNT_FEATURE_INDEX]
            location_scores = location_scores + F.softplus(self.count_scale) * deficit

        if self.config.residual_marginal:
            location_scores = location_scores + F.softplus(self.marginal_scale) * (
                locations[:, :, MARGINAL_FEATURE_INDEX] * self.number_of_locations
            )

        joined_actor = torch.cat([global_actor, pooled_actor], dim=-1)
        if self.config.architecture == "hierarchical_attention":
            logits = _joint_hierarchical_logits(
                self.visit_gate(joined_actor), location_scores, action_masks
            )
        else:
            idle_score = self.idle_head(joined_actor)
            logits = torch.cat([idle_score, location_scores], dim=-1)

        critic_tokens = self.critic_encoder(locations)
        global_critic = self.critic_global(global_features)
        pooled_critic = critic_tokens.mean(dim=1)
        joined_critic = torch.cat([global_critic, pooled_critic], dim=-1)
        values = self.value_head(joined_critic).squeeze(-1)
        q_values = torch.cat(
            [self.q_idle(joined_critic), self.q_location(critic_tokens).squeeze(-1)], dim=-1
        )
        return NetworkOutput(logits, values, q_values, count_logits)


def build_network(
    *,
    number_of_locations: int,
    location_dim: int,
    global_dim: int,
    config: NetworkConfig,
) -> nn.Module:
    if config.architecture == "stable_mlp":
        return StableMLPNetwork(
            number_of_locations=number_of_locations,
            location_dim=location_dim,
            global_dim=global_dim,
            config=config,
        )
    if config.architecture in {"attention", "hierarchical_attention"}:
        return AttentionActorCritic(
            number_of_locations=number_of_locations,
            location_dim=location_dim,
            global_dim=global_dim,
            config=config,
        )
    raise ValueError(f"unsupported architecture: {config.architecture}")


def set_active_search_mode(network: nn.Module, mode: Literal["full", "eas"]) -> None:
    """Configure full active search or efficient adapter/decoder search."""

    if mode == "full":
        for parameter in network.parameters():
            parameter.requires_grad = True
        return
    if mode != "eas":
        raise ValueError("active search mode must be 'full' or 'eas'")

    trainable_fragments = (
        "instance_adapter",
        "actor_key",
        "actor_query",
        "location_bias",
        "idle_head",
        "visit_gate",
        "marginal_scale",
        "count_scale",
    )
    for name, parameter in network.named_parameters():
        parameter.requires_grad = any(fragment in name for fragment in trainable_fragments)

