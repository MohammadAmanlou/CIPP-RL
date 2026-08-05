"""Cross-entropy imitation warm start from exact CIPP itineraries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.core import CIPPInstance
from src.envs import CIPPEnv
from src.models.ppo_agent import (
    PPOAgent,
    masked_categorical,
)
from src.utils.normalization import (
    ObservationNormalizer,
)


@dataclass(frozen=True, slots=True)
class ImitationDataset:
    """State-action-mask triples extracted from exact solutions."""

    states: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    source_instance_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        """Return the number of teacher transitions."""

        return int(
            self.actions.size
        )


def extract_teacher_trajectory(
    instance: CIPPInstance,
    itinerary: tuple[int, ...] | list[int],
    normalizer: ObservationNormalizer,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Replay one exact itinerary into normalized state-action pairs."""

    if len(itinerary) != instance.H:
        raise ValueError(
            "teacher itinerary length must equal instance.H."
        )

    environment = CIPPEnv(
        instance
    )

    observation, info = (
        environment.reset()
    )

    states: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []

    for action in itinerary:
        mask = np.asarray(
            info["action_mask"],
            dtype=np.bool_,
        )

        normalized_action = int(
            action
        )

        if not bool(
            mask[normalized_action]
        ):
            raise ValueError(
                "teacher itinerary contains an action "
                "that is masked by the CIPP environment."
            )

        states.append(
            normalizer.transform(
                observation
            ).astype(
                np.float32
            )
        )

        actions.append(
            normalized_action
        )

        masks.append(
            mask.copy()
        )

        (
            observation,
            _,
            _,
            _,
            info,
        ) = environment.step(
            normalized_action
        )

    return (
        np.stack(states),
        np.asarray(
            actions,
            dtype=np.int64,
        ),
        np.stack(masks),
    )


def build_imitation_dataset(
    demonstrations: list[
        tuple[
            CIPPInstance,
            tuple[int, ...] | list[int],
        ]
    ],
    *,
    normalizer: ObservationNormalizer,
) -> ImitationDataset:
    """Combine exact teacher trajectories into one supervised dataset."""

    if not demonstrations:
        raise ValueError(
            "at least one demonstration is required."
        )

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    source_ids: list[str] = []

    for instance, itinerary in demonstrations:
        (
            states,
            actions,
            masks,
        ) = extract_teacher_trajectory(
            instance,
            itinerary,
            normalizer,
        )

        all_states.append(
            states
        )

        all_actions.append(
            actions
        )

        all_masks.append(
            masks
        )

        source_ids.extend(
            [instance.instance_id]
            * actions.size
        )

    return ImitationDataset(
        states=np.concatenate(
            all_states,
            axis=0,
        ),
        actions=np.concatenate(
            all_actions,
            axis=0,
        ),
        action_masks=np.concatenate(
            all_masks,
            axis=0,
        ),
        source_instance_ids=tuple(
            source_ids
        ),
    )


def pretrain_policy_by_imitation(
    agent: PPOAgent,
    dataset: ImitationDataset,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> dict[str, float]:
    """Minimize masked cross-entropy before PPO fine-tuning."""

    if dataset.size < 1:
        raise ValueError(
            "imitation dataset must not be empty."
        )

    if epochs < 1 or batch_size < 1:
        raise ValueError(
            "epochs and batch_size must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    optimizer = torch.optim.Adam(
        agent.network.parameters(),
        lr=float(
            learning_rate
        ),
    )

    states = torch.as_tensor(
        dataset.states,
        dtype=torch.float32,
        device=agent.device,
    )

    actions = torch.as_tensor(
        dataset.actions,
        dtype=torch.int64,
        device=agent.device,
    )

    masks = torch.as_tensor(
        dataset.action_masks,
        dtype=torch.bool,
        device=agent.device,
    )

    loss_total = 0.0
    accuracy_total = 0.0
    number_of_batches = 0

    agent.network.train()

    for _ in range(
        epochs
    ):
        permutation = rng.permutation(
            dataset.size
        )

        for start in range(
            0,
            dataset.size,
            batch_size,
        ):
            indices = permutation[
                start:
                start + batch_size
            ]

            index_tensor = torch.as_tensor(
                indices,
                dtype=torch.int64,
                device=agent.device,
            )

            logits, _ = agent.network(
                states[
                    index_tensor
                ]
            )

            batch_masks = masks[
                index_tensor
            ]

            distribution = (
                masked_categorical(
                    logits,
                    batch_masks,
                )
            )

            masked_logits = (
                distribution.logits
            )

            loss = F.cross_entropy(
                masked_logits,
                actions[
                    index_tensor
                ],
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                agent.network.parameters(),
                agent.config
                .gradient_clip_norm,
            )

            optimizer.step()

            with torch.no_grad():
                predictions = torch.argmax(
                    masked_logits,
                    dim=-1,
                )

                accuracy = (
                    predictions
                    == actions[
                        index_tensor
                    ]
                ).float().mean()

            loss_total += float(
                loss.detach()
                .cpu()
                .item()
            )

            accuracy_total += float(
                accuracy.detach()
                .cpu()
                .item()
            )

            number_of_batches += 1

    return {
        "cross_entropy_loss": (
            loss_total
            / number_of_batches
        ),
        "masked_action_accuracy": (
            accuracy_total
            / number_of_batches
        ),
        "number_of_examples": float(
            dataset.size
        ),
    }
