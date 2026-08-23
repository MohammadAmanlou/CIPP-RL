"""Train Method C and compare three RL adaptive baselines against adaptive Gurobi.

Pilot recommendation:
    D_14S_30P, shock at 50%, held-out rank-flip reward change.

RL methods:
    B: Frozen Attention greedy (no reward adaptation)
    D: Frozen Attention + residual Best-of-K search
    C: Context-conditioned Attention + residual Best-of-K

Gurobi:
    Adaptive re-optimization with the exact same fixed prefix and reward shock.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import time

import numpy as np
import torch

from src.advanced import AdvancedPPOAgent, StructuredFeatureBuilder
from src.adaptive import (
    AdaptiveFeatureBuilder,
    AdaptiveTrainingConfig,
    rank_flip_scenario,
    residual_best_of_k,
    residual_greedy,
    solve_adaptive_gurobi,
    train_method_c,
)
from src.utils import load_professor_benchmark, parse_professor_instance_id


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="D_14S_30P")
    p.add_argument("--data-directory", type=Path, default=Path("."))
    p.add_argument("--output-directory", type=Path, default=Path("results/adaptive_method_c"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--switch-fraction", type=float, default=0.50)
    p.add_argument("--shock-boost", type=float, default=2.25)
    p.add_argument("--shock-suppress", type=float, default=0.55)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--episodes-per-update", type=int, default=64)
    p.add_argument("--validation-interval", type=int, default=10)
    p.add_argument("--validation-scenarios", type=int, default=8)
    p.add_argument("--validation-rollouts", type=int, default=16)
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--early-stopping-warmup", type=int, default=80)
    p.add_argument("--early-stopping-min-delta", type=float, default=2.0)
    p.add_argument("--adaptive-lr-scale", type=float, default=0.10)
    p.add_argument("--adaptive-update-epochs", type=int, default=2)
    p.add_argument("--final-rollouts", type=int, default=512)
    p.add_argument("--gurobi-time-limit", type=float, default=3600.0)
    p.add_argument("--gurobi-mip-gap", type=float, default=0.0)
    p.add_argument("--skip-gurobi", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--static-checkpoint", type=Path, default=None)
    return p.parse_args()


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def _default_static_checkpoint(instance_id: str, seed: int) -> Path:
    return (
        Path("results")
        / instance_id
        / instance_id
        / "attention"
        / f"seed_{seed}"
        / "checkpoint_best.pt"
    )


def _make_prefix(static_agent, scenario, seed: int) -> tuple[int, ...]:
    """Generate the common realized prefix before the shock with static Attention."""
    from src.envs import CIPPEnv

    env = CIPPEnv(static_agent.instance, seed=seed)
    env.reset(seed=seed)
    prefix = []
    while env.day < scenario.switch_day:
        state = static_agent.feature_builder.build(env)
        action, _, _ = static_agent.select_action(state, deterministic=True)
        env.step(action)
        prefix.append(int(action))
    return tuple(prefix)


def main() -> None:
    a = _args()
    device = _device(a.device)
    spec = parse_professor_instance_id(a.instance)
    benchmark = load_professor_benchmark(
        a.data_directory / f"CIPP-{spec.party}.xls",
        instance_id=spec.instance_id,
        objective_variant="professor_code",
        budget_mode="auto",
    )
    instance = benchmark.instance

    output = a.output_directory / instance.instance_id / f"seed_{a.seed}"
    output.mkdir(parents=True, exist_ok=True)

    scenario = rank_flip_scenario(
        instance,
        switch_fraction=a.switch_fraction,
        boost=a.shock_boost,
        suppress=a.shock_suppress,
        scenario_id="heldout_rank_flip",
    )
    (output / "scenario.json").write_text(
        json.dumps(
            {
                "instance": instance.instance_id,
                "switch_day_zero_based": scenario.switch_day,
                "switch_period_one_based": scenario.switch_day + 1,
                "post_multiplier": scenario.post_multiplier.tolist(),
                "boost": a.shock_boost,
                "suppress": a.shock_suppress,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    static_checkpoint = (
        a.static_checkpoint
        if a.static_checkpoint is not None
        else _default_static_checkpoint(instance.instance_id, a.seed)
    )
    if not static_checkpoint.exists():
        raise FileNotFoundError(
            f"Static Attention checkpoint not found: {static_checkpoint}\n"
            "Run the existing static Attention experiment first or pass "
            "--static-checkpoint."
        )

    # Same static checkpoint, two observation semantics:
    # - static_builder hides reward change (Methods B/D)
    # - adaptive_builder exposes current reward context (Method C)
    static_builder = StructuredFeatureBuilder(instance)
    static_agent, _ = AdvancedPPOAgent.load(
        static_checkpoint,
        feature_builder=static_builder,
        device=device,
        load_optimizer=False,
    )

    adaptive_builder = AdaptiveFeatureBuilder(instance)
    adaptive_init, _ = AdvancedPPOAgent.load(
        static_checkpoint,
        feature_builder=adaptive_builder,
        device=device,
        load_optimizer=False,
    )

    method_c_dir = output / "method_c"
    method_c_checkpoint = method_c_dir / "checkpoint_best.pt"

    if not a.eval_only:
        train_config = AdaptiveTrainingConfig(
            updates=a.updates,
            episodes_per_update=a.episodes_per_update,
            validation_interval=a.validation_interval,
            validation_scenarios=a.validation_scenarios,
            validation_rollouts_per_scenario=a.validation_rollouts,
            early_stopping_patience=a.early_stopping_patience,
            early_stopping_warmup_updates=a.early_stopping_warmup,
            early_stopping_min_delta=a.early_stopping_min_delta,
            learning_rate_scale=a.adaptive_lr_scale,
            update_epochs=a.adaptive_update_epochs,
        )
        train_method_c(
            adaptive_init,
            config=train_config,
            output_directory=method_c_dir,
            seed=a.seed,
        )

    if not method_c_checkpoint.exists():
        raise FileNotFoundError(
            f"Method C checkpoint not found: {method_c_checkpoint}"
        )

    method_c_agent, _ = AdvancedPPOAgent.load(
        method_c_checkpoint,
        feature_builder=adaptive_builder,
        device=device,
        load_optimizer=False,
    )

    # All adaptive methods receive exactly the same realized prefix.
    prefix = _make_prefix(static_agent, scenario, a.seed)
    (output / "fixed_prefix.json").write_text(
        json.dumps({"prefix": list(prefix)}, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []

    b = residual_greedy(
        static_agent,
        scenario=scenario,
        prefix=prefix,
        seed=a.seed + 10_000,
        context_aware=False,
    )
    rows.append({
        "method": "B_frozen_attention_greedy",
        **b,
    })

    d = residual_best_of_k(
        static_agent,
        scenario=scenario,
        prefix=prefix,
        rollouts=a.final_rollouts,
        seed=a.seed + 20_000,
        context_aware=False,
    )
    rows.append({
        "method": f"D_frozen_attention_residual_best_of_{a.final_rollouts}",
        **d,
    })

    c = residual_best_of_k(
        method_c_agent,
        scenario=scenario,
        prefix=prefix,
        rollouts=a.final_rollouts,
        seed=a.seed + 30_000,
        context_aware=True,
    )
    rows.append({
        "method": f"C_context_attention_residual_best_of_{a.final_rollouts}",
        **c,
    })

    if not a.skip_gurobi:
        g = solve_adaptive_gurobi(
            instance,
            scenario,
            prefix,
            time_limit_seconds=a.gurobi_time_limit,
            mip_gap=a.gurobi_mip_gap,
            verbose=False,
        )
        rows.append(g)

    # Add comparison fields.
    gurobi_row = next((r for r in rows if r["method"] == "adaptive_gurobi"), None)
    if gurobi_row is not None:
        g_obj = float(gurobi_row["objective"])
        g_time = float(gurobi_row["runtime_seconds"])
        for row in rows:
            obj = float(row["objective"])
            row["delta_vs_adaptive_gurobi_percent"] = (
                100.0 * (obj - g_obj) / abs(g_obj) if g_obj != 0 else 0.0
            )
            row["gurobi_over_method_runtime"] = (
                g_time / float(row["runtime_seconds"])
                if float(row["runtime_seconds"]) > 0
                else None
            )

    # Strip huge itineraries from the compact CSV but keep JSON complete.
    (output / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    compact = []
    for row in rows:
        compact.append({
            key: value
            for key, value in row.items()
            if key != "itinerary"
        })

    # Keep every evaluated method in the compact CSV, including the greedy
    # baseline. Use a stable schema so missing optional Gurobi fields do not
    # silently remove rows.
    columns = [
        "method",
        "objective",
        "runtime_seconds",
        "rollouts",
        "delta_vs_adaptive_gurobi_percent",
        "gurobi_over_method_runtime",
        "best_bound",
        "optimality_gap_percent",
        "proven_optimal",
        "status",
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
            restval="NA",
        )
        writer.writeheader()
        writer.writerows(compact)

    expected_rows = 3 if a.skip_gurobi else 4
    if len(compact) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} comparison rows, found {len(compact)}"
        )

    print("\n=== ADAPTIVE PILOT ===")
    print(f"instance={instance.instance_id}")
    print(f"switch_day={scenario.switch_day} (zero-based)")
    print(f"prefix={list(prefix)}")
    for row in sorted(rows, key=lambda r: float(r["objective"]), reverse=True):
        delta = row.get("delta_vs_adaptive_gurobi_percent")
        delta_text = "" if delta is None else f" delta_vs_gurobi={delta:+.3f}%"
        print(
            f"{row['method']}: objective={float(row['objective']):.3f} "
            f"time={float(row['runtime_seconds']):.3f}s{delta_text}"
        )
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
