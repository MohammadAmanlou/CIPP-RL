"""Train masked Double-DQN checkpoints for the smallest real CIPP shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models import DQNConfig
from src.training import DQNTrainingConfig, train_dqn
from src.utils import (
    generate_professor_matched_instance,
    professor_temporal_weights,
    resolve_real_location_count,
    reward_profiles_from_data,
)


def _default_data_path(party: str) -> Path:
    processed = Path(f"data/processed/CIPP-{party}.csv")
    if processed.exists():
        return processed
    return Path(f"data/CIPP-{party}.xls")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train fixed-size masked Double-DQN models on a stream of "
            "domain-matched synthetic CIPP instances."
        )
    )
    parser.add_argument("--d-data", type=Path, default=_default_data_path("D"))
    parser.add_argument("--r-data", type=Path, default=_default_data_path("R"))
    parser.add_argument("--cities-parameter", type=int, default=16)
    parser.add_argument(
        "--instance-mode",
        choices=["supplied-code", "paper-14"],
        default="supplied-code",
        help=(
            "supplied-code uses Cities=16 => 15 real locations; paper-14 uses "
            "14 real locations as described in the revised paper."
        ),
    )
    parser.add_argument("--real-location-count", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--validation-instances", type=int, default=100)
    parser.add_argument("--normalizer-instances", type=int, default=250)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--target-sync-steps", type=int, default=1_000)
    parser.add_argument("--replay-capacity", type=int, default=250_000)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--jitter-fraction", type=float, default=0.10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/dqn_real_matched"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    real_count = resolve_real_location_count(
        cities_parameter=args.cities_parameter,
        instance_mode=args.instance_mode,
        real_location_count=args.real_location_count,
    )
    profiles = reward_profiles_from_data(
        [args.d_data, args.r_data],
        cities_parameter=args.cities_parameter,
        instance_mode=args.instance_mode,
        real_location_count=args.real_location_count,
    )
    if any(profile.size != real_count for profile in profiles):
        raise RuntimeError("Calibration profile size does not match the model shape.")

    temporal_weights = professor_temporal_weights(args.horizon)
    maximum_reward = max(float(profile.max()) for profile in profiles) * 1.5
    # At most one visit is made per day and repeat factors are <= 1.
    reward_scale = max(maximum_reward * float(temporal_weights.sum()), 1.0)

    def factory(seed: int, instance_id: str):
        return generate_professor_matched_instance(
            n=real_count,
            horizon=args.horizon,
            seed=seed,
            reward_profiles=profiles,
            jitter_fraction=args.jitter_fraction,
            instance_id=instance_id,
        )

    run_metadata = {
        "training_distribution": "empirical-profile blend + scale + jitter + permutation",
        "d_data": str(args.d_data),
        "r_data": str(args.r_data),
        "cities_parameter": args.cities_parameter,
        "instance_mode": args.instance_mode,
        "real_location_count": real_count,
        "horizon": args.horizon,
        "jitter_fraction": args.jitter_fraction,
        "calibration_profiles": [profile.tolist() for profile in profiles],
        "test_instance_used_for_gradient_updates": False,
    }

    summaries = []
    for seed in args.seeds:
        seed_output = args.output_dir / f"seed_{seed}"
        summary = train_dqn(
            instance_factory=factory,
            output_dir=seed_output,
            seed=seed,
            training_config=DQNTrainingConfig(
                episodes=args.episodes,
                replay_capacity=args.replay_capacity,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                target_sync_steps=args.target_sync_steps,
                validation_interval_episodes=args.validation_interval,
                validation_instances=args.validation_instances,
                normalizer_instances=args.normalizer_instances,
                early_stopping_patience=args.early_stopping_patience,
            ),
            model_config=DQNConfig(
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                gamma=1.0,
                gradient_clip_norm=1.0,
                double_dqn=True,
            ),
            device=args.device,
            reward_scale=reward_scale,
            run_metadata=run_metadata,
        )
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reward_scale": reward_scale,
        "n": real_count,
        "H": args.horizon,
        "instance_mode": args.instance_mode,
        "device": args.device,
        "run_metadata": run_metadata,
        "runs": summaries,
    }
    (args.output_dir / "all_seeds_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
