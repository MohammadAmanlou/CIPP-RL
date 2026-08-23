"""FINAL Method C experiment: leakage-free, anticipatory, and Gurobi-complete.

This is the canonical V5 runner.

It supports two execution environments:

Kaggle / GPU:
    Train Method C and produce all RL-side results with --skip-gurobi.

Local machine with Gurobi:
    Re-run with --eval-only (same output directory) and without --skip-gurobi
    to add exact Gurobi references.

The experiment reports two scientifically distinct groups:

A) SAME STATIC-ATTENTION PREFIX (post-shock adaptation ablation)
   B: frozen Attention greedy
   D: frozen Attention + best-of-K
   C: context-aware Method C + best-of-K
   exact Gurobi oracle from the SAME static prefix

B) END-TO-END ONLINE PLANNING (main result)
   Method C chooses its own pre-shock prefix with no future-shock information,
   then adapts after the shock.
   It is compared with:
     - static Gurobi plan with no replanning
     - two-stage Gurobi: static solve -> execute -> shock -> exact re-solve
     - exact Gurobi continuation from Method C's own realized prefix
     - clairvoyant Gurobi upper bound (future shock known at day 0)

No best-of-K/hindsight selection is allowed before the shock for Method C.
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
import platform
import time

import numpy as np
import torch

from src.advanced.features import StructuredFeatureBuilder
from src.advanced.ppo import AdvancedPPOAgent
from src.adaptive import (
    AdaptiveFeatureBuilder,
    AdaptiveTrainingConfig,
    anticipatory_then_greedy,
    anticipatory_then_residual_best_of_k,
    rank_flip_scenario,
    rank_flip_scenario_at_day,
    residual_best_of_k,
    residual_greedy,
    solve_adaptive_gurobi,
    solve_clairvoyant_adaptive_gurobi,
    solve_two_stage_adaptive_gurobi,
    train_method_c,
)
from src.optimization import solve_cipp_gurobi
from src.utils import load_professor_benchmark, parse_professor_instance_id


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="D_14S_30P")
    p.add_argument("--data-directory", type=Path, default=Path("."))
    p.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/AdaptiveFinal"),
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)

    # Held-out evaluation shock.
    p.add_argument("--switch-fraction", type=float, default=0.50)
    p.add_argument(
        "--switch-day",
        type=int,
        default=None,
        help=(
            "Exact number of fully executed days before shock. "
            "Example: 15 means days 1..15 execute before the shock and "
            "replanning begins on day 16. Overrides --switch-fraction."
        ),
    )
    p.add_argument("--shock-boost", type=float, default=2.25)
    p.add_argument("--shock-suppress", type=float, default=0.55)

    # Method C training.
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--episodes-per-update", type=int, default=64)
    p.add_argument("--validation-interval", type=int, default=10)
    p.add_argument("--validation-scenarios", type=int, default=8)
    p.add_argument("--validation-rollouts", type=int, default=16)
    p.add_argument("--early-stopping-warmup", type=int, default=80)
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--early-stopping-min-delta", type=float, default=2.0)
    p.add_argument("--adaptive-lr-scale", type=float, default=0.10)
    p.add_argument("--adaptive-update-epochs", type=int, default=2)

    # Online post-shock search budget.
    p.add_argument("--final-rollouts", type=int, default=512)

    # Checkpoints / execution mode.
    p.add_argument("--static-checkpoint", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")

    # Exact references.
    p.add_argument("--skip-gurobi", action="store_true")
    p.add_argument("--gurobi-initial-time-limit", type=float, default=3600.0)
    p.add_argument("--gurobi-reopt-time-limit", type=float, default=3600.0)
    p.add_argument("--gurobi-clairvoyant-time-limit", type=float, default=3600.0)
    p.add_argument("--gurobi-mip-gap", type=float, default=0.0)
    p.add_argument("--gurobi-threads", type=int, default=None)
    p.add_argument("--verbose-gurobi", action="store_true")
    return p.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_static_checkpoint(instance_id: str, seed: int) -> Path:
    """Resolve static Attention checkpoint in new and legacy layouts."""
    candidates = [
        Path("results")
        / "Static"
        / instance_id
        / instance_id
        / "attention"
        / f"seed_{seed}"
        / "checkpoint_best.pt",
        Path("results")
        / instance_id
        / instance_id
        / "attention"
        / f"seed_{seed}"
        / "checkpoint_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _scenario_from_args(a: argparse.Namespace, instance):
    if a.switch_day is not None:
        return rank_flip_scenario_at_day(
            instance,
            switch_day=a.switch_day,
            boost=a.shock_boost,
            suppress=a.shock_suppress,
            scenario_id="heldout_rank_flip",
        )
    return rank_flip_scenario(
        instance,
        switch_fraction=a.switch_fraction,
        boost=a.shock_boost,
        suppress=a.shock_suppress,
        scenario_id="heldout_rank_flip",
    )


def _make_static_attention_prefix(
    static_agent,
    scenario,
    seed: int,
) -> tuple[int, ...]:
    """Generate common pre-shock prefix using truly static Attention."""
    from src.envs import CIPPEnv

    env = CIPPEnv(static_agent.instance, seed=seed)
    env.reset(seed=seed)
    prefix: list[int] = []
    while env.day < scenario.switch_day:
        state = static_agent.feature_builder.build(env)
        action, _, _ = static_agent.select_action(
            state,
            deterministic=True,
        )
        env.step(int(action))
        prefix.append(int(action))
    return tuple(prefix)


def _gap_below(reference: float, value: float) -> float:
    if reference == 0.0:
        return 0.0
    return 100.0 * (reference - value) / abs(reference)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "method",
        "comparison_group",
        "prefix_source",
        "objective",
        "runtime_seconds",
        "rollouts",
        "proven_optimal",
        "optimality_gap_percent",
        "gap_to_group_oracle_percent",
        "gap_to_two_stage_gurobi_percent",
        "gap_to_clairvoyant_upper_bound_percent",
        "information_pattern",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
            restval="NA",
        )
        writer.writeheader()
        writer.writerows(rows)


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
    scenario = _scenario_from_args(a, instance)

    output = (
        a.output_directory
        / instance.instance_id
        / f"seed_{a.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    static_checkpoint = (
        a.static_checkpoint
        if a.static_checkpoint is not None
        else _default_static_checkpoint(instance.instance_id, a.seed)
    )
    if not static_checkpoint.exists():
        raise FileNotFoundError(
            f"Static Attention checkpoint not found: {static_checkpoint}\n"
            "Pass --static-checkpoint explicitly if your layout differs."
        )

    scenario_payload = {
        "instance": instance.instance_id,
        "switch_day_zero_based": int(scenario.switch_day),
        "switch_period_one_based": int(scenario.switch_day + 1),
        "meaning": (
            f"Days 1..{scenario.switch_day} are completed before the shock; "
            f"day {scenario.switch_day + 1} is the first post-shock decision."
        ),
        "post_multiplier": scenario.post_multiplier.tolist(),
        "boost": float(a.shock_boost),
        "suppress": float(a.shock_suppress),
    }
    (output / "scenario.json").write_text(
        json.dumps(scenario_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    run_metadata = {
        "instance": instance.instance_id,
        "seed": int(a.seed),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "python": platform.python_version(),
        "static_checkpoint": str(static_checkpoint),
        "final_rollouts": int(a.final_rollouts),
        "information_safety": {
            "frozen_B_D_parallel_static_observation_env": True,
            "method_c_pre_shock_future_identity_visible": False,
            "method_c_pre_shock_best_of_k_selection": False,
            "best_of_k_allowed_only_after_shock": True,
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    # Same pretrained weights, different observation semantics.
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

    # ==============================================================
    # GROUP A: SAME STATIC-ATTENTION PREFIX
    # ==============================================================
    static_prefix = _make_static_attention_prefix(
        static_agent,
        scenario,
        a.seed,
    )
    (output / "static_attention_prefix.json").write_text(
        json.dumps({"prefix": list(static_prefix)}, indent=2) + "\n",
        encoding="utf-8",
    )

    b = residual_greedy(
        static_agent,
        scenario=scenario,
        prefix=static_prefix,
        seed=a.seed + 10_000,
        context_aware=False,
    )
    b["method"] = "B_frozen_attention_greedy"
    b["comparison_group"] = "same_static_attention_prefix"
    b["prefix_source"] = "static_attention"
    b["information_pattern"] = "shock_unaware_frozen_policy"

    d = residual_best_of_k(
        static_agent,
        scenario=scenario,
        prefix=static_prefix,
        rollouts=a.final_rollouts,
        seed=a.seed + 20_000,
        context_aware=False,
    )
    d["method"] = (
        f"D_frozen_attention_residual_best_of_{a.final_rollouts}"
    )
    d["comparison_group"] = "same_static_attention_prefix"
    d["prefix_source"] = "static_attention"
    d["information_pattern"] = "shock_unaware_frozen_policy_postshock_search"

    c_static = residual_best_of_k(
        method_c_agent,
        scenario=scenario,
        prefix=static_prefix,
        rollouts=a.final_rollouts,
        seed=a.seed + 30_000,
        context_aware=True,
    )
    c_static["method"] = (
        f"C_context_static_prefix_best_of_{a.final_rollouts}"
    )
    c_static["comparison_group"] = "same_static_attention_prefix"
    c_static["prefix_source"] = "static_attention"
    c_static["information_pattern"] = "shock_context_visible_only_when_realized"

    # ==============================================================
    # GROUP B: END-TO-END ONLINE METHOD C
    # ==============================================================
    c_e2e_greedy = anticipatory_then_greedy(
        method_c_agent,
        scenario=scenario,
        seed=a.seed + 40_000,
        deterministic_prefix=True,
    )
    c_e2e_greedy["method"] = "C_e2e_anticipatory_greedy"
    c_e2e_greedy["comparison_group"] = "end_to_end_online"
    c_e2e_greedy["information_pattern"] = (
        "anticipatory_learned_policy_no_future_shock_identity"
    )

    c_e2e = anticipatory_then_residual_best_of_k(
        method_c_agent,
        scenario=scenario,
        rollouts=a.final_rollouts,
        seed=a.seed + 50_000,
        deterministic_prefix=True,
    )
    c_e2e["method"] = (
        f"C_e2e_anticipatory_postshock_best_of_{a.final_rollouts}"
    )
    c_e2e["comparison_group"] = "end_to_end_online"
    c_e2e["information_pattern"] = (
        "anticipatory_prefix_no_future_info_then_postshock_adaptive_search"
    )

    method_c_prefix = tuple(int(x) for x in c_e2e["prefix"])
    (output / "method_c_anticipatory_prefix.json").write_text(
        json.dumps({"prefix": list(method_c_prefix)}, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = [
        b,
        d,
        c_static,
        c_e2e_greedy,
        c_e2e,
    ]

    # ==============================================================
    # GUROBI REFERENCES
    # ==============================================================
    if not a.skip_gurobi:
        same_static_oracle = solve_adaptive_gurobi(
            instance,
            scenario,
            static_prefix,
            time_limit_seconds=a.gurobi_reopt_time_limit,
            mip_gap=a.gurobi_mip_gap,
            threads=a.gurobi_threads,
            verbose=a.verbose_gurobi,
            output_directory=output / "gurobi_same_static_prefix",
            model_label="same_static_attention_prefix",
        )
        same_static_oracle["method"] = (
            "adaptive_gurobi_same_static_attention_prefix"
        )
        same_static_oracle["comparison_group"] = (
            "same_static_attention_prefix"
        )
        same_static_oracle["prefix_source"] = "static_attention"
        same_static_oracle["information_pattern"] = (
            "exact_postshock_oracle_same_realized_prefix"
        )
        same_static_oracle["rollouts"] = "NA"
        rows.append(same_static_oracle)

        same_c_prefix_oracle = solve_adaptive_gurobi(
            instance,
            scenario,
            method_c_prefix,
            time_limit_seconds=a.gurobi_reopt_time_limit,
            mip_gap=a.gurobi_mip_gap,
            threads=a.gurobi_threads,
            verbose=a.verbose_gurobi,
            output_directory=output / "gurobi_same_method_c_prefix",
            model_label="same_method_c_anticipatory_prefix",
        )
        same_c_prefix_oracle["method"] = (
            "adaptive_gurobi_same_method_c_prefix"
        )
        same_c_prefix_oracle["comparison_group"] = "end_to_end_online"
        same_c_prefix_oracle["prefix_source"] = (
            "method_c_anticipatory_no_future_shock_info"
        )
        same_c_prefix_oracle["information_pattern"] = (
            "exact_postshock_oracle_same_method_c_realized_prefix"
        )
        same_c_prefix_oracle["rollouts"] = "NA"
        rows.append(same_c_prefix_oracle)

        two_stage = solve_two_stage_adaptive_gurobi(
            instance,
            scenario,
            initial_time_limit_seconds=a.gurobi_initial_time_limit,
            reoptimization_time_limit_seconds=a.gurobi_reopt_time_limit,
            mip_gap=a.gurobi_mip_gap,
            threads=a.gurobi_threads,
            verbose=a.verbose_gurobi,
            output_directory=output / "gurobi_two_stage",
        )
        (output / "gurobi_two_stage.json").write_text(
            json.dumps(two_stage, indent=2) + "\n",
            encoding="utf-8",
        )

        initial_static = two_stage["initial_static_solution"]
        no_replan_row = {
            "method": "gurobi_initial_static_plan_no_replan",
            "comparison_group": "end_to_end_online",
            "prefix_source": "initial_static_gurobi",
            "objective": float(
                two_stage[
                    "static_plan_adaptive_objective_without_replanning"
                ]
            ),
            "runtime_seconds": float(
                two_stage["initial_planning_runtime_seconds"]
            ),
            "rollouts": "NA",
            "proven_optimal": bool(initial_static["proven_optimal"]),
            "optimality_gap_percent": float(
                initial_static["optimality_gap_percent"]
            ),
            "information_pattern": (
                "static_day0_plan_no_future_shock_info_no_replanning"
            ),
        }
        rows.append(no_replan_row)

        reopt = two_stage["reoptimized_solution"]
        two_stage_row = {
            "method": "gurobi_two_stage_replanned",
            "comparison_group": "end_to_end_online",
            "prefix_source": "initial_static_gurobi",
            "objective": float(two_stage["final_objective"]),
            "runtime_seconds": float(
                two_stage["reoptimization_runtime_seconds"]
            ),
            "rollouts": "NA",
            "proven_optimal": bool(reopt["proven_optimal"]),
            "optimality_gap_percent": float(
                reopt["optimality_gap_percent"]
            ),
            "information_pattern": (
                "static_exact_day0_plan_then_exact_reoptimization_after_shock"
            ),
            "initial_planning_runtime_seconds": float(
                two_stage["initial_planning_runtime_seconds"]
            ),
            "total_planning_runtime_seconds": float(
                two_stage["total_planning_runtime_seconds"]
            ),
            "reoptimization_gain_percent": float(
                two_stage["reoptimization_gain_percent"]
            ),
            "future_actions_changed": int(
                two_stage["future_actions_changed"]
            ),
        }
        rows.append(two_stage_row)

        clairvoyant = solve_clairvoyant_adaptive_gurobi(
            instance,
            scenario,
            time_limit_seconds=a.gurobi_clairvoyant_time_limit,
            mip_gap=a.gurobi_mip_gap,
            threads=a.gurobi_threads,
            verbose=a.verbose_gurobi,
            output_directory=output / "gurobi_clairvoyant_upper_bound",
        )
        clairvoyant["comparison_group"] = "perfect_information_upper_bound"
        clairvoyant["prefix_source"] = "none"
        clairvoyant["rollouts"] = "NA"
        rows.append(clairvoyant)

        # ----------------------------------------------------------
        # Gap annotations.
        # ----------------------------------------------------------
        same_static_obj = float(same_static_oracle["objective"])
        same_c_obj = float(same_c_prefix_oracle["objective"])
        two_stage_obj = float(two_stage["final_objective"])
        clair_obj = float(clairvoyant["objective"])

        for row in rows:
            value = float(row["objective"])
            method = str(row["method"])

            if row.get("comparison_group") == "same_static_attention_prefix":
                row["gap_to_group_oracle_percent"] = _gap_below(
                    same_static_obj,
                    value,
                )
            elif method in {
                "C_e2e_anticipatory_greedy",
                f"C_e2e_anticipatory_postshock_best_of_{a.final_rollouts}",
                "adaptive_gurobi_same_method_c_prefix",
            }:
                row["gap_to_group_oracle_percent"] = _gap_below(
                    same_c_obj,
                    value,
                )
            else:
                row["gap_to_group_oracle_percent"] = "NA"

            row["gap_to_two_stage_gurobi_percent"] = _gap_below(
                two_stage_obj,
                value,
            )
            row["gap_to_clairvoyant_upper_bound_percent"] = _gap_below(
                clair_obj,
                value,
            )

    # Save complete JSON first (includes itineraries and extra diagnostics).
    (output / "comparison_final.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    # Compact CSV removes large itineraries/prefixes but keeps all metrics.
    compact_rows = []
    for row in rows:
        compact_rows.append({
            k: v
            for k, v in row.items()
            if k not in {"itinerary", "prefix"}
        })
    _write_csv(output / "comparison_final.csv", compact_rows)

    # ==============================================================
    # Human-readable summary.
    # ==============================================================
    print("\n=== FINAL ADAPTIVE METHOD C EXPERIMENT ===")
    print(f"instance={instance.instance_id}")
    print(
        f"shock after {scenario.switch_day} completed days; "
        f"first post-shock decision is day {scenario.switch_day + 1}"
    )
    print(f"device={device}")
    print(
        "Information safety: no future shock identity is visible to Method C "
        "before the shock; B/D use strict static parallel observation envs."
    )

    print("\n--- Same static-Attention prefix ---")
    for row in rows:
        if row.get("comparison_group") == "same_static_attention_prefix":
            print(
                f"{row['method']}: objective={float(row['objective']):.3f} "
                f"time={float(row['runtime_seconds']):.3f}s"
            )

    print("\n--- End-to-end online planning ---")
    for row in rows:
        if row.get("comparison_group") == "end_to_end_online":
            print(
                f"{row['method']}: objective={float(row['objective']):.3f} "
                f"time={float(row['runtime_seconds']):.3f}s"
            )

    if not a.skip_gurobi:
        c_obj = float(c_e2e["objective"])
        two_stage_obj = float(two_stage["final_objective"])
        diff_pct = (
            100.0 * (c_obj - two_stage_obj) / abs(two_stage_obj)
            if two_stage_obj != 0.0 else 0.0
        )
        print(
            "\nMAIN TEST: Method C E2E vs two-stage Gurobi = "
            f"{diff_pct:+.3f}%"
        )
        print(
            "Clairvoyant Gurobi is reported only as a perfect-information "
            "upper bound, not as an operational online baseline."
        )

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
