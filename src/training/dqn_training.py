"""Reproducible multi-instance masked Double-DQN training."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.baselines import run_dqn_greedy_policy
from src.core import CIPPInstance
from src.envs import CIPPEnv
from src.models import DQNAgent, DQNConfig, ReplayBuffer
from src.utils import ObservationNormalizer


InstanceFactory = Callable[[int, str], CIPPInstance]


@dataclass(frozen=True, slots=True)
class DQNTrainingConfig:
    episodes: int = 20_000
    replay_capacity: int = 250_000
    batch_size: int = 256
    warmup_steps: int = 5_000
    train_every_steps: int = 1
    target_sync_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.80
    validation_interval_episodes: int = 250
    validation_instances: int = 100
    normalizer_instances: int = 250
    early_stopping_patience: int = 30
    min_improvement: float = 1e-6
    training_data_seed: int = 10_000_000
    validation_data_seed: int = 20_000_000
    normalizer_data_seed: int = 30_000_000

    def validate(self) -> None:
        integer_positive = {
            "episodes": self.episodes,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "train_every_steps": self.train_every_steps,
            "target_sync_steps": self.target_sync_steps,
            "validation_interval_episodes": self.validation_interval_episodes,
            "validation_instances": self.validation_instances,
            "normalizer_instances": self.normalizer_instances,
        }
        for name, value in integer_positive.items():
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity.")
        if not (0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0):
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1.")
        if not (0.0 < self.epsilon_decay_fraction <= 1.0):
            raise ValueError("epsilon_decay_fraction must be in (0, 1].")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be nonnegative.")
        if self.min_improvement < 0 or not np.isfinite(self.min_improvement):
            raise ValueError("min_improvement must be finite and nonnegative.")


def _epsilon(step: int, decay_steps: int, config: DQNTrainingConfig) -> float:
    if decay_steps <= 0:
        return config.epsilon_end
    fraction = min(max(step / decay_steps, 0.0), 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def fit_training_normalizer(
    *,
    instance_factory: InstanceFactory,
    base_seed: int,
    number_of_instances: int,
) -> ObservationNormalizer:
    """Fit feature statistics on training-only random masked trajectories."""

    observations: list[np.ndarray] = []
    for index in range(number_of_instances):
        instance = instance_factory(base_seed + index, f"normalizer-{index:05d}")
        env = CIPPEnv(instance, seed=base_seed + 1_000_000 + index)
        observation, _ = env.reset()
        observations.append(observation)
        while not env.done:
            action = env.sample_viable_action()
            observation, _, _, _, _ = env.step(action)
            observations.append(observation)
    return ObservationNormalizer.fit(np.stack(observations, axis=0))


def _validate(
    *,
    agent: DQNAgent,
    normalizer: ObservationNormalizer,
    validation_instances: list[CIPPInstance],
) -> dict[str, float]:
    objectives: list[float] = []
    runtimes: list[float] = []
    for instance in validation_instances:
        result = run_dqn_greedy_policy(
            instance,
            agent=agent,
            normalizer=normalizer,
        )
        objectives.append(result.objective)
        runtimes.append(result.runtime_seconds)

    values = np.asarray(objectives, dtype=np.float64)
    return {
        "mean_objective": float(values.mean()),
        "median_objective": float(np.median(values)),
        "std_objective": float(values.std(ddof=0)),
        "min_objective": float(values.min()),
        "max_objective": float(values.max()),
        "mean_runtime_seconds": float(np.mean(runtimes)),
        "feasible_rate": 1.0,
    }


def _save_learning_curves(
    metrics: list[dict[str, float | int]],
    output_dir: Path,
) -> dict[str, str]:
    """Write separate training/validation/loss plots."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.asarray([int(row["episode"]) for row in metrics])
    train_values = np.asarray([float(row["train_objective"]) for row in metrics])
    losses = np.asarray([float(row["mean_loss"]) for row in metrics])
    validation = np.asarray(
        [float(row["validation_mean_objective"]) for row in metrics]
    )

    paths: dict[str, str] = {}

    window = max(1, min(250, len(train_values)))
    if len(train_values) >= window:
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = np.convolve(train_values, kernel, mode="valid")
        smooth_episodes = episodes[window - 1 :]
    else:
        smoothed = train_values
        smooth_episodes = episodes
    plt.figure(figsize=(8, 5))
    plt.plot(smooth_episodes, smoothed)
    plt.xlabel("Training episode")
    plt.ylabel("Rolling mean training objective")
    plt.title("Masked DQN training objective")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "training_objective_curve.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths["training_objective_curve"] = str(path)

    finite_validation = np.isfinite(validation)
    plt.figure(figsize=(8, 5))
    plt.plot(episodes[finite_validation], validation[finite_validation], marker="o")
    plt.xlabel("Training episode")
    plt.ylabel("Mean validation objective")
    plt.title("Masked DQN validation learning curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "validation_learning_curve.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths["validation_learning_curve"] = str(path)

    finite_loss = np.isfinite(losses)
    plt.figure(figsize=(8, 5))
    plt.plot(episodes[finite_loss], losses[finite_loss])
    plt.xlabel("Training episode")
    plt.ylabel("Mean Huber loss")
    plt.title("Masked DQN optimization loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "training_loss_curve.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths["training_loss_curve"] = str(path)
    return paths


def train_dqn(
    *,
    instance_factory: InstanceFactory,
    output_dir: str | Path,
    seed: int,
    training_config: DQNTrainingConfig | None = None,
    model_config: DQNConfig | None = None,
    device: str | torch.device = "cpu",
    reward_scale: float,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Train one DQN seed on a fixed stream of independent instances.

    Different model seeds see the same training and validation instance seeds.
    This makes seed comparisons reflect network initialization and exploration,
    rather than a different validation set for every run.
    """

    cfg = training_config or DQNTrainingConfig()
    cfg.validate()
    model_cfg = model_config or DQNConfig()
    if reward_scale <= 0 or not np.isfinite(reward_scale):
        raise ValueError("reward_scale must be positive and finite.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.csv"
    best_checkpoint = output / "best.pt"
    last_checkpoint = output / "last.pt"

    probe = instance_factory(cfg.training_data_seed, "shape-probe")
    probe_env = CIPPEnv(probe)
    probe_observation, _ = probe_env.reset()

    normalizer = fit_training_normalizer(
        instance_factory=instance_factory,
        base_seed=cfg.normalizer_data_seed,
        number_of_instances=cfg.normalizer_instances,
    )
    if normalizer.observation_dim != probe_observation.size:
        raise RuntimeError("Normalizer and environment observation dimensions differ.")

    agent = DQNAgent(
        observation_dim=probe_observation.size,
        action_dim=probe.num_actions,
        config=model_cfg,
        seed=seed,
        device=device,
    )
    replay = ReplayBuffer(
        capacity=cfg.replay_capacity,
        observation_dim=probe_observation.size,
        action_dim=probe.num_actions,
        seed=seed + 1,
    )

    validation_instances = [
        instance_factory(
            cfg.validation_data_seed + index,
            f"validation-{index:05d}",
        )
        for index in range(cfg.validation_instances)
    ]

    estimated_steps = cfg.episodes * probe.H
    decay_steps = max(1, int(cfg.epsilon_decay_fraction * estimated_steps))
    total_steps = 0
    optimization_steps = 0
    best_validation = -math.inf
    no_improvement_checks = 0
    started = time.perf_counter()
    metrics: list[dict[str, float | int]] = []

    fieldnames = [
        "episode",
        "environment_steps",
        "optimization_steps",
        "epsilon",
        "train_objective",
        "mean_loss",
        "validation_mean_objective",
        "validation_std_objective",
        "elapsed_seconds",
    ]

    with metrics_path.open("w", newline="", encoding="utf-8") as metric_file:
        writer = csv.DictWriter(metric_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(1, cfg.episodes + 1):
            instance = instance_factory(
                cfg.training_data_seed + episode,
                f"train-{episode:07d}",
            )
            env_seed = seed * 10_000_000 + episode
            env = CIPPEnv(instance, seed=env_seed)
            observation, info = env.reset(seed=env_seed)
            episode_losses: list[float] = []

            while not env.done:
                epsilon = _epsilon(total_steps, decay_steps, cfg)
                normalized_state = normalizer.transform(observation).astype(np.float32)
                action_mask = info["action_mask"]
                action = agent.select_action(
                    normalized_state,
                    action_mask,
                    epsilon=epsilon,
                )
                next_observation, reward, terminated, _, next_info = env.step(action)
                normalized_next_state = normalizer.transform(next_observation).astype(
                    np.float32
                )
                replay.add(
                    state=normalized_state,
                    action=action,
                    reward=float(reward / reward_scale),
                    next_state=normalized_next_state,
                    done=terminated,
                    action_mask=action_mask,
                    next_action_mask=next_info["action_mask"],
                )

                total_steps += 1
                if (
                    total_steps >= cfg.warmup_steps
                    and len(replay) >= cfg.batch_size
                    and total_steps % cfg.train_every_steps == 0
                ):
                    episode_losses.append(agent.optimize(replay.sample(cfg.batch_size)))
                    optimization_steps += 1

                if total_steps % cfg.target_sync_steps == 0:
                    agent.sync_target_network()

                observation = next_observation
                info = next_info

            validation_mean = float("nan")
            validation_std = float("nan")
            should_validate = (
                episode == 1
                or episode % cfg.validation_interval_episodes == 0
                or episode == cfg.episodes
            )

            if should_validate:
                validation = _validate(
                    agent=agent,
                    normalizer=normalizer,
                    validation_instances=validation_instances,
                )
                validation_mean = validation["mean_objective"]
                validation_std = validation["std_objective"]

                metadata = {
                    "seed": seed,
                    "episode": episode,
                    "environment_steps": total_steps,
                    "optimization_steps": optimization_steps,
                    "reward_scale": reward_scale,
                    "training_config": asdict(cfg),
                    "model_config": asdict(model_cfg),
                    "validation": validation,
                    "run_metadata": run_metadata or {},
                    "elapsed_seconds": time.perf_counter() - started,
                }

                if validation_mean > best_validation + cfg.min_improvement:
                    best_validation = validation_mean
                    no_improvement_checks = 0
                    agent.save_checkpoint(
                        best_checkpoint,
                        normalizer=normalizer.to_dict(),
                        training_metadata=metadata,
                    )
                else:
                    no_improvement_checks += 1

            row: dict[str, float | int] = {
                "episode": episode,
                "environment_steps": total_steps,
                "optimization_steps": optimization_steps,
                "epsilon": _epsilon(total_steps, decay_steps, cfg),
                "train_objective": float(env.cumulative_reward),
                "mean_loss": (
                    float(np.mean(episode_losses))
                    if episode_losses
                    else float("nan")
                ),
                "validation_mean_objective": validation_mean,
                "validation_std_objective": validation_std,
                "elapsed_seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            metric_file.flush()
            metrics.append(row)

            if (
                should_validate
                and cfg.early_stopping_patience > 0
                and no_improvement_checks >= cfg.early_stopping_patience
            ):
                break

    final_metadata = {
        "seed": seed,
        "episode": int(metrics[-1]["episode"]),
        "environment_steps": total_steps,
        "optimization_steps": optimization_steps,
        "reward_scale": reward_scale,
        "training_config": asdict(cfg),
        "model_config": asdict(model_cfg),
        "run_metadata": run_metadata or {},
        "best_validation_mean_objective": best_validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    agent.save_checkpoint(
        last_checkpoint,
        normalizer=normalizer.to_dict(),
        training_metadata=final_metadata,
    )
    if not best_checkpoint.exists():
        agent.save_checkpoint(
            best_checkpoint,
            normalizer=normalizer.to_dict(),
            training_metadata=final_metadata,
        )

    figure_paths = _save_learning_curves(metrics, output / "figures")
    summary: dict[str, object] = {
        **final_metadata,
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "metrics_path": str(metrics_path),
        "figures": figure_paths,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
