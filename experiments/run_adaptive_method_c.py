"""FINAL V6 RL experiment for adaptive CIPP.

V6 has two distinct adaptive RL approaches:

METHOD C — GENERALIST ANTICIPATORY POLICY
    Warm-start from the static Attention checkpoint and train offline over a
    rich curriculum of shocks. Before a future shock is realized, its identity
    is hidden. The policy can learn option value / flexibility from the shock
    distribution. After the shock, the new reward context becomes observable.

METHOD R — SHOCK-TIME RESIDUAL RETRAINING
    Follow the static RL route until the shock. Freeze every pre-shock decision.
    Then train a new RL optimization phase only on the remaining suffix of the
    realized scenario. V6 evaluates both:
      R-static-init: warm-start residual training from static RL weights.
      R-scratch: train a fresh randomly initialized residual policy after shock.

The final held-out suite contains deterministic single-shock scenarios and is
saved to JSON so local Gurobi can solve EXACTLY the same realizations later.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced.features import StructuredFeatureBuilder
from src.advanced.ppo import AdvancedPPOAgent
from src.adaptive import (
    AdaptiveFeatureBuilder,
    AdaptiveTrainingConfig,
    MultiShockScenario,
    ResidualRetrainingConfig,
    RICH_CURRICULUM_DESCRIPTION,
    anticipatory_then_greedy,
    anticipatory_then_residual_best_of_k,
    generate_heldout_single_shock_suite,
    residual_best_of_k,
    sample_training_scenario,
    residual_greedy,
    scenario_to_dict,
    train_fixed_prefix_residual,
    train_method_c,
)
from src.envs import CIPPEnv
from src.utils import load_professor_benchmark, parse_professor_instance_id


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="D_14S_30P")
    p.add_argument("--data-directory", type=Path, default=Path("."))
    p.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/AdaptiveV6"),
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--static-checkpoint", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--skip-gurobi", action="store_true", help=argparse.SUPPRESS)

    # Generalist Method C training.
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--episodes-per-update", type=int, default=96)
    p.add_argument("--validation-interval", type=int, default=10)
    p.add_argument("--validation-scenarios", type=int, default=16)
    p.add_argument("--validation-rollouts", type=int, default=16)
    p.add_argument("--early-stopping-warmup", type=int, default=120)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--early-stopping-min-delta", type=float, default=2.0)
    p.add_argument("--adaptive-lr-scale", type=float, default=0.10)
    p.add_argument("--adaptive-update-epochs", type=int, default=2)

    # Main pilot and held-out suite.
    p.add_argument("--main-rollouts", type=int, default=512)
    p.add_argument("--suite-count", type=int, default=100)
    p.add_argument("--suite-rollouts", type=int, default=64)
    p.add_argument("--heldout-seed", type=int, default=260824)

    # User-proposed post-shock RL retraining baseline on the main pilot.
    p.add_argument("--skip-residual-retraining", action="store_true")
    p.add_argument("--residual-updates", type=int, default=120)
    p.add_argument("--residual-episodes", type=int, default=64)
    p.add_argument("--residual-validation-interval", type=int, default=5)
    p.add_argument("--residual-validation-rollouts", type=int, default=32)
    p.add_argument("--residual-warmup", type=int, default=20)
    p.add_argument("--residual-patience", type=int, default=10)
    p.add_argument("--residual-min-delta", type=float, default=1.0)
    p.add_argument("--residual-lr-scale", type=float, default=0.20)
    p.add_argument("--residual-update-epochs", type=int, default=2)

    # Small representative subset for repeated online retraining.
    p.add_argument("--retrain-suite-count", type=int, default=12)
    p.add_argument("--suite-retrain-updates", type=int, default=50)
    p.add_argument("--suite-retrain-episodes", type=int, default=32)
    p.add_argument("--suite-retrain-rollouts", type=int, default=64)
    return p.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_static_checkpoint(instance_id: str, seed: int) -> Path:
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


def _static_prefix(
    agent: AdvancedPPOAgent,
    *,
    switch_day: int,
    seed: int,
) -> tuple[int, ...]:
    env = CIPPEnv(agent.instance, seed=seed)
    env.reset(seed=seed)
    prefix: list[int] = []
    while env.day < switch_day:
        state = agent.feature_builder.build(env)
        action, _, _ = agent.select_action(state, deterministic=True)
        env.step(int(action))
        prefix.append(int(action))
    return tuple(prefix)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, object]], *, method_key: str = "method") -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        method = str(row[method_key])
        grouped.setdefault(method, []).append(float(row["objective"]))
    result = {}
    for method, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        result[method] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return result


def _load_agent(
    checkpoint: Path,
    builder,
    *,
    device: torch.device,
) -> AdvancedPPOAgent:
    agent, _ = AdvancedPPOAgent.load(
        checkpoint,
        feature_builder=builder,
        device=device,
        load_optimizer=False,
    )
    return agent


def _residual_config(a: argparse.Namespace, *, suite: bool = False) -> ResidualRetrainingConfig:
    if suite:
        return ResidualRetrainingConfig(
            updates=a.suite_retrain_updates,
            episodes_per_update=a.suite_retrain_episodes,
            validation_interval=max(5, min(10, a.suite_retrain_updates)),
            validation_rollouts=min(16, a.suite_retrain_rollouts),
            early_stopping_patience=4,
            early_stopping_warmup_updates=min(15, a.suite_retrain_updates),
            early_stopping_min_delta=1.0,
            learning_rate_scale=a.residual_lr_scale,
            update_epochs=a.residual_update_epochs,
        )
    return ResidualRetrainingConfig(
        updates=a.residual_updates,
        episodes_per_update=a.residual_episodes,
        validation_interval=a.residual_validation_interval,
        validation_rollouts=a.residual_validation_rollouts,
        early_stopping_patience=a.residual_patience,
        early_stopping_warmup_updates=a.residual_warmup,
        early_stopping_min_delta=a.residual_min_delta,
        learning_rate_scale=a.residual_lr_scale,
        update_epochs=a.residual_update_epochs,
    )



def _audit_curriculum(instance, *, seed: int, draws: int = 5000) -> dict[str, object]:
    """Empirically summarize the V6 training distribution without using test RNG."""
    rng = np.random.default_rng(seed)
    families: dict[str, int] = {}
    switch_days: list[int] = []
    affected_counts: list[int] = []
    boost_counts: list[int] = []
    suppress_counts: list[int] = []
    multi_change_counts: list[int] = []

    for i in range(draws):
        progress = (i + 0.5) / draws
        scenario = sample_training_scenario(
            instance,
            rng,
            progress=progress,
            allow_multi_shock=True,
        )
        families[scenario.family] = families.get(scenario.family, 0) + 1
        switch_days.append(int(scenario.switch_day))

        if isinstance(scenario, MultiShockScenario):
            multi_change_counts.append(len(scenario.change_days))
            multiplier = scenario.post_multipliers[0]
        else:
            multiplier = scenario.post_multiplier

        affected = np.abs(multiplier - 1.0) > 1e-12
        affected_counts.append(int(np.count_nonzero(affected)))
        boost_counts.append(int(np.count_nonzero(multiplier > 1.0 + 1e-12)))
        suppress_counts.append(int(np.count_nonzero(multiplier < 1.0 - 1e-12)))

    return {
        "draws": int(draws),
        "family_counts": families,
        "family_fractions": {
            k: float(v / draws) for k, v in sorted(families.items())
        },
        "switch_day": {
            "min": int(min(switch_days)),
            "max": int(max(switch_days)),
            "mean": float(np.mean(switch_days)),
            "unique": sorted(set(switch_days)),
        },
        "affected_locations": {
            "min": int(min(affected_counts)),
            "max": int(max(affected_counts)),
            "mean": float(np.mean(affected_counts)),
        },
        "boosted_locations_mean": float(np.mean(boost_counts)),
        "suppressed_locations_mean": float(np.mean(suppress_counts)),
        "multi_shock_change_count_mean": (
            float(np.mean(multi_change_counts))
            if multi_change_counts
            else 0.0
        ),
    }



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

    curriculum_audit = _audit_curriculum(
        instance,
        seed=a.seed + 880_000_000,
        draws=5000,
    )
    (output / "curriculum_audit.json").write_text(
        json.dumps(curriculum_audit, indent=2) + "\n",
        encoding="utf-8",
    )

    static_checkpoint = (
        a.static_checkpoint
        if a.static_checkpoint is not None
        else _default_static_checkpoint(instance.instance_id, a.seed)
    )
    if not static_checkpoint.exists():
        raise FileNotFoundError(
            f"Static Attention checkpoint not found: {static_checkpoint}"
        )

    static_builder = StructuredFeatureBuilder(instance)
    adaptive_builder = AdaptiveFeatureBuilder(instance)
    static_agent = _load_agent(static_checkpoint, static_builder, device=device)

    # --------------------------------------------------------------
    # Train generalist Method C V6.
    # --------------------------------------------------------------
    method_c_dir = output / "method_c_rich_generalist"
    method_c_checkpoint = method_c_dir / "checkpoint_best.pt"

    if not a.eval_only:
        initial_agent = _load_agent(
            static_checkpoint,
            AdaptiveFeatureBuilder(instance),
            device=device,
        )
        config = AdaptiveTrainingConfig(
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
            initial_agent,
            config=config,
            output_directory=method_c_dir,
            seed=a.seed,
        )

    if not method_c_checkpoint.exists():
        raise FileNotFoundError(
            f"Method C V6 checkpoint not found: {method_c_checkpoint}"
        )
    method_c_agent = _load_agent(
        method_c_checkpoint,
        AdaptiveFeatureBuilder(instance),
        device=device,
    )

    # --------------------------------------------------------------
    # Freeze and save the 100-scenario held-out suite.
    # Scenario 0 is the original pilot rank-flip.
    # --------------------------------------------------------------
    suite = generate_heldout_single_shock_suite(
        instance,
        count=a.suite_count,
        seed=a.heldout_seed,
    )
    suite_payload = {
        "instance": instance.instance_id,
        "seed": int(a.heldout_seed),
        "count": len(suite),
        "training_curriculum": RICH_CURRICULUM_DESCRIPTION,
        "scenarios": [scenario_to_dict(s) for s in suite],
    }
    (output / "heldout_suite.json").write_text(
        json.dumps(suite_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    main_scenario = suite[0]
    static_main_prefix = _static_prefix(
        static_agent,
        switch_day=main_scenario.switch_day,
        seed=a.seed,
    )

    # --------------------------------------------------------------
    # Main pilot RL comparison.
    # --------------------------------------------------------------
    rows_main: list[dict[str, object]] = []

    b = residual_greedy(
        static_agent,
        scenario=main_scenario,
        prefix=static_main_prefix,
        seed=a.seed + 10_000,
        context_aware=False,
    )
    rows_main.append({
        "method": "B_frozen_static_greedy",
        "objective": float(b["objective"]),
        "runtime_seconds": float(b["runtime_seconds"]),
        "rollouts": 1,
        "prefix": list(static_main_prefix),
    })

    d = residual_best_of_k(
        static_agent,
        scenario=main_scenario,
        prefix=static_main_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 20_000,
        context_aware=False,
    )
    rows_main.append({
        "method": f"D_frozen_static_best_of_{a.main_rollouts}",
        "objective": float(d["objective"]),
        "runtime_seconds": float(d["runtime_seconds"]),
        "rollouts": a.main_rollouts,
        "prefix": list(static_main_prefix),
    })

    c_same = residual_best_of_k(
        method_c_agent,
        scenario=main_scenario,
        prefix=static_main_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 30_000,
        context_aware=True,
    )
    rows_main.append({
        "method": f"C_rich_generalist_static_prefix_best_of_{a.main_rollouts}",
        "objective": float(c_same["objective"]),
        "runtime_seconds": float(c_same["runtime_seconds"]),
        "rollouts": a.main_rollouts,
        "prefix": list(static_main_prefix),
    })

    c_greedy = anticipatory_then_greedy(
        method_c_agent,
        scenario=main_scenario,
        seed=a.seed + 40_000,
        deterministic_prefix=True,
    )
    rows_main.append({
        "method": "C_rich_generalist_e2e_greedy",
        "objective": float(c_greedy["objective"]),
        "runtime_seconds": float(c_greedy["runtime_seconds"]),
        "rollouts": 1,
        "prefix": c_greedy["prefix"],
    })

    c_e2e = anticipatory_then_residual_best_of_k(
        method_c_agent,
        scenario=main_scenario,
        rollouts=a.main_rollouts,
        seed=a.seed + 50_000,
        deterministic_prefix=True,
    )
    rows_main.append({
        "method": f"C_rich_generalist_e2e_best_of_{a.main_rollouts}",
        "objective": float(c_e2e["objective"]),
        "runtime_seconds": float(c_e2e["runtime_seconds"]),
        "rollouts": a.main_rollouts,
        "prefix": c_e2e["prefix"],
    })

    # --------------------------------------------------------------
    # User-proposed Method R: fixed static prefix, then re-train after shock.
    # We compare warm-start and true scratch initialization.
    # --------------------------------------------------------------
    if not a.skip_residual_retraining:
        residual_cfg = _residual_config(a, suite=False)

        warm_dir = output / "residual_retrain_main" / "static_init"
        warm_ckpt = warm_dir / "checkpoint_best.pt"
        if not a.eval_only:
            warm_agent = _load_agent(
                static_checkpoint,
                AdaptiveFeatureBuilder(instance),
                device=device,
            )
            train_fixed_prefix_residual(
                warm_agent,
                scenario=main_scenario,
                prefix=static_main_prefix,
                config=residual_cfg,
                output_directory=warm_dir,
                seed=a.seed + 60_000,
                label="R_static_init_fixed_prefix",
            )
        warm_summary = json.loads(
            (warm_dir / "training_summary.json").read_text(encoding="utf-8")
        )
        warm_agent = _load_agent(
            warm_ckpt,
            AdaptiveFeatureBuilder(instance),
            device=device,
        )
        warm_eval = residual_best_of_k(
            warm_agent,
            scenario=main_scenario,
            prefix=static_main_prefix,
            rollouts=a.main_rollouts,
            seed=a.seed + 61_000,
            context_aware=True,
        )
        rows_main.append({
            "method": f"R_static_init_shock_retrain_best_of_{a.main_rollouts}",
            "objective": float(warm_eval["objective"]),
            "runtime_seconds": float(warm_eval["runtime_seconds"]),
            "online_training_seconds": float(warm_summary["elapsed_seconds"]),
            "rollouts": a.main_rollouts,
            "prefix": list(static_main_prefix),
        })

        scratch_dir = output / "residual_retrain_main" / "scratch"
        scratch_ckpt = scratch_dir / "checkpoint_best.pt"
        if not a.eval_only:
            scratch_agent = AdvancedPPOAgent(
                feature_builder=AdaptiveFeatureBuilder(instance),
                network_config=static_agent.network_config,
                ppo_config=static_agent.config,
                seed=a.seed + 70_000,
                device=str(device),
            )
            train_fixed_prefix_residual(
                scratch_agent,
                scenario=main_scenario,
                prefix=static_main_prefix,
                config=residual_cfg,
                output_directory=scratch_dir,
                seed=a.seed + 70_000,
                label="R_scratch_fixed_prefix",
            )
        scratch_summary = json.loads(
            (scratch_dir / "training_summary.json").read_text(encoding="utf-8")
        )
        scratch_agent = _load_agent(
            scratch_ckpt,
            AdaptiveFeatureBuilder(instance),
            device=device,
        )
        scratch_eval = residual_best_of_k(
            scratch_agent,
            scenario=main_scenario,
            prefix=static_main_prefix,
            rollouts=a.main_rollouts,
            seed=a.seed + 71_000,
            context_aware=True,
        )
        rows_main.append({
            "method": f"R_scratch_shock_retrain_best_of_{a.main_rollouts}",
            "objective": float(scratch_eval["objective"]),
            "runtime_seconds": float(scratch_eval["runtime_seconds"]),
            "online_training_seconds": float(scratch_summary["elapsed_seconds"]),
            "rollouts": a.main_rollouts,
            "prefix": list(static_main_prefix),
        })

    (output / "main_rl_comparison.json").write_text(
        json.dumps(rows_main, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / "main_rl_comparison.csv",
        [{k: v for k, v in row.items() if k != "prefix"} for row in rows_main],
    )

    # --------------------------------------------------------------
    # Full held-out suite: B / D / C generalist.
    # --------------------------------------------------------------
    suite_rows: list[dict[str, object]] = []
    for i, scenario in enumerate(suite):
        prefix_static = _static_prefix(
            static_agent,
            switch_day=scenario.switch_day,
            seed=a.seed + 100_000 + i,
        )
        c_suite = anticipatory_then_residual_best_of_k(
            method_c_agent,
            scenario=scenario,
            rollouts=a.suite_rollouts,
            seed=a.seed + 200_000 + i * 10_003,
            deterministic_prefix=True,
        )
        b_suite = residual_greedy(
            static_agent,
            scenario=scenario,
            prefix=prefix_static,
            seed=a.seed + 300_000 + i,
            context_aware=False,
        )
        d_suite = residual_best_of_k(
            static_agent,
            scenario=scenario,
            prefix=prefix_static,
            rollouts=a.suite_rollouts,
            seed=a.seed + 400_000 + i * 10_003,
            context_aware=False,
        )

        common = {
            "scenario_index": i,
            "scenario_id": scenario.scenario_id,
            "family": scenario.family,
            "switch_day": int(scenario.switch_day),
        }
        suite_rows.extend([
            {
                **common,
                "method": "B_frozen_static_greedy",
                "objective": float(b_suite["objective"]),
                "runtime_seconds": float(b_suite["runtime_seconds"]),
                "prefix": list(prefix_static),
            },
            {
                **common,
                "method": f"D_frozen_static_best_of_{a.suite_rollouts}",
                "objective": float(d_suite["objective"]),
                "runtime_seconds": float(d_suite["runtime_seconds"]),
                "prefix": list(prefix_static),
            },
            {
                **common,
                "method": f"C_rich_generalist_e2e_best_of_{a.suite_rollouts}",
                "objective": float(c_suite["objective"]),
                "runtime_seconds": float(c_suite["runtime_seconds"]),
                "prefix": c_suite["prefix"],
            },
        ])

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[suite] completed {i + 1}/{len(suite)} scenarios",
                flush=True,
            )

    (output / "suite_rl_results.json").write_text(
        json.dumps(suite_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / "suite_rl_results.csv",
        [{k: v for k, v in row.items() if k != "prefix"} for row in suite_rows],
    )
    suite_summary = {
        "overall": _summary(suite_rows),
        "by_family": {},
    }
    families = sorted({str(row["family"]) for row in suite_rows})
    for family in families:
        suite_summary["by_family"][family] = _summary(
            [row for row in suite_rows if row["family"] == family]
        )
    (output / "suite_rl_summary.json").write_text(
        json.dumps(suite_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Representative held-out subset for Method R repeated retraining.
    # Warm-start version only, because this is already an expensive online method.
    # --------------------------------------------------------------
    retrain_suite_rows: list[dict[str, object]] = []
    if (
        not a.skip_residual_retraining
        and a.retrain_suite_count > 0
        and not a.eval_only
    ):
        count = min(a.retrain_suite_count, len(suite))
        selected = np.unique(
            np.linspace(0, len(suite) - 1, num=count, dtype=int)
        ).tolist()
        cfg_suite = _residual_config(a, suite=True)

        for rank, i in enumerate(selected):
            scenario = suite[i]
            prefix_static = _static_prefix(
                static_agent,
                switch_day=scenario.switch_day,
                seed=a.seed + 500_000 + i,
            )
            retrain_dir = (
                output
                / "residual_retrain_suite"
                / f"{i:03d}_{scenario.scenario_id}"
            )
            agent = _load_agent(
                static_checkpoint,
                AdaptiveFeatureBuilder(instance),
                device=device,
            )
            ckpt = train_fixed_prefix_residual(
                agent,
                scenario=scenario,
                prefix=prefix_static,
                config=cfg_suite,
                output_directory=retrain_dir,
                seed=a.seed + 600_000 + i * 1009,
                label="R_static_init_suite",
            )
            training_summary = json.loads(
                (retrain_dir / "training_summary.json").read_text(encoding="utf-8")
            )
            agent = _load_agent(
                ckpt,
                AdaptiveFeatureBuilder(instance),
                device=device,
            )
            result = residual_best_of_k(
                agent,
                scenario=scenario,
                prefix=prefix_static,
                rollouts=a.suite_retrain_rollouts,
                seed=a.seed + 700_000 + i * 1009,
                context_aware=True,
            )
            retrain_suite_rows.append({
                "scenario_index": i,
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "switch_day": int(scenario.switch_day),
                "method": f"R_static_init_retrain_best_of_{a.suite_retrain_rollouts}",
                "objective": float(result["objective"]),
                "inference_runtime_seconds": float(result["runtime_seconds"]),
                "online_training_seconds": float(training_summary["elapsed_seconds"]),
                "prefix": list(prefix_static),
            })
            print(
                f"[retrain-suite] completed {rank + 1}/{len(selected)} "
                f"scenario={i}",
                flush=True,
            )

        (output / "retrain_suite_results.json").write_text(
            json.dumps(retrain_suite_rows, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_csv(
            output / "retrain_suite_results.csv",
            [{k: v for k, v in row.items() if k != "prefix"} for row in retrain_suite_rows],
        )

    # --------------------------------------------------------------
    # Storage cleanup: retain only checkpoints needed for later evaluation.
    # Suite-specific Method R checkpoints are disposable after their objective
    # has been recorded; checkpoint_last files are also unnecessary.
    # --------------------------------------------------------------
    for last_checkpoint in output.rglob("checkpoint_last.pt"):
        try:
            last_checkpoint.unlink()
        except OSError:
            pass
    suite_retrain_root = output / "residual_retrain_suite"
    if suite_retrain_root.exists():
        for best_checkpoint in suite_retrain_root.rglob("checkpoint_best.pt"):
            try:
                best_checkpoint.unlink()
            except OSError:
                pass

    metadata = {
        "version": "V6",
        "instance": instance.instance_id,
        "seed": int(a.seed),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "static_checkpoint": str(static_checkpoint),
        "curriculum": RICH_CURRICULUM_DESCRIPTION,
        "heldout_suite_count": len(suite),
        "heldout_seed": int(a.heldout_seed),
        "main_rollouts": int(a.main_rollouts),
        "suite_rollouts": int(a.suite_rollouts),
        "information_safety": {
            "future_shock_identity_hidden_before_realization": True,
            "no_pre_shock_best_of_k_hindsight": True,
            "frozen_B_D_use_parallel_static_observation_env": True,
            "multi_shocks_are_training_only_in_v6": True,
            "heldout_gurobi_suite_is_single_shock_only": True,
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== V6 MAIN RL RESULTS ===")
    for row in rows_main:
        extra = (
            f" train={row['online_training_seconds']:.3f}s"
            if "online_training_seconds" in row
            else ""
        )
        print(
            f"{row['method']}: objective={float(row['objective']):.3f} "
            f"inference={float(row['runtime_seconds']):.3f}s{extra}"
        )
    print("\n=== V6 SUITE SUMMARY ===")
    for method, stats in suite_summary["overall"].items():
        print(
            f"{method}: mean={stats['mean']:.3f} "
            f"median={stats['median']:.3f} std={stats['std']:.3f}"
        )
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
