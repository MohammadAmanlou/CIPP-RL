"""FINAL V7: union of all CIPP adaptive RL ideas.

This runner intentionally keeps every scientifically distinct idea from V3-V6
and adds the V7 recourse-aware/hybrid concepts.

RL-side methods produced on Kaggle
----------------------------------
Static / legacy
  B      : frozen static RL, greedy suffix
  D      : frozen static RL, best-of-K suffix search
  C-S    : rich generalist Method C after a fixed static-RL prefix
  C-C    : rich generalist Method C end-to-end

Recourse-aware prefix learning
  P-P    : recourse-aware prefix policy + its own adaptive suffix
  P-C    : recourse-aware prefix policy + frozen strong Method-C suffix
  I-I    : intensive instance-specific recourse-prefix policy + own suffix
  I-C    : instance-specific prefix + strong Method-C suffix

Online post-shock RL optimization
  S-R    : static-RL prefix -> shock-specific residual RL
  C-R    : Method-C anticipatory prefix -> residual RL
  P-R    : recourse-aware prefix -> residual RL
  I-R    : instance-specific recourse-aware prefix -> residual RL
  Scratch variants are retained on the main pilot as ablations.

Local Gurobi then adds
  G-G    : static exact Gurobi prefix -> exact Gurobi recourse
  S-G    : static-RL prefix -> exact Gurobi recourse
  C-G    : Method-C prefix -> exact Gurobi recourse
  P-G    : recourse-aware prefix -> exact Gurobi recourse
  I-G    : instance-specific prefix -> exact Gurobi recourse
  SG-G   : stochastic nonanticipative Gurobi prefix -> exact recourse
  Clairvoyant Gurobi upper bound

Information rule
----------------
No RL prefix sees the realized future shock identity before the shock. Best-of-K
selection is forbidden before the shock. Expensive shock-specific training is
allowed only after the prefix is fixed and the shock is realized.
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
    RecourseAwareTrainingConfig,
    ResidualRetrainingConfig,
    RICH_CURRICULUM_DESCRIPTION,
    anticipatory_then_greedy,
    anticipatory_then_residual_best_of_k,
    build_anticipatory_prefix,
    generate_heldout_single_shock_suite,
    residual_best_of_k,
    residual_greedy,
    scenario_to_dict,
    train_fixed_prefix_residual,
    train_method_c,
    train_multi_instance_method_c,
    train_recourse_aware_prefix,
)
from src.envs import CIPPEnv
from src.utils import load_professor_benchmark, parse_professor_instance_id


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="D_14S_30P")
    p.add_argument(
        "--training-instances",
        default=None,
        help=(
            "Comma-separated compatible instance IDs for true multi-instance "
            "generalist training. Default: target instance only."
        ),
    )
    p.add_argument("--data-directory", type=Path, default=Path("."))
    p.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/Adaptive"),
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--static-checkpoint", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")

    # C: broad shock-trained generalist.
    p.add_argument("--c-restarts", type=int, default=3)
    p.add_argument("--c-updates", type=int, default=500)
    p.add_argument("--c-episodes", type=int, default=128)
    p.add_argument("--c-validation-interval", type=int, default=10)
    p.add_argument("--c-validation-scenarios", type=int, default=24)
    p.add_argument("--c-validation-rollouts", type=int, default=32)
    p.add_argument("--c-warmup", type=int, default=150)
    p.add_argument("--c-patience", type=int, default=10)
    p.add_argument("--c-min-delta", type=float, default=1.5)
    p.add_argument("--c-lr-scale", type=float, default=0.08)
    p.add_argument("--c-epochs", type=int, default=2)

    # P: recourse-aware prefix learner using C as a strong frozen suffix teacher.
    p.add_argument("--p-restarts", type=int, default=2)
    p.add_argument("--p-updates", type=int, default=180)
    p.add_argument("--p-prefixes", type=int, default=16)
    p.add_argument("--p-teacher-rollouts", type=int, default=32)
    p.add_argument("--p-validation-interval", type=int, default=10)
    p.add_argument("--p-validation-scenarios", type=int, default=20)
    p.add_argument("--p-validation-rollouts", type=int, default=64)
    p.add_argument("--p-warmup", type=int, default=50)
    p.add_argument("--p-patience", type=int, default=8)
    p.add_argument("--p-min-delta", type=float, default=1.0)
    p.add_argument("--p-lr-scale", type=float, default=0.05)
    p.add_argument("--p-epochs", type=int, default=2)

    # I: heavier instance-specific prefix refinement on the target instance.
    p.add_argument("--i-restarts", type=int, default=2)
    p.add_argument("--i-updates", type=int, default=240)
    p.add_argument("--i-prefixes", type=int, default=24)
    p.add_argument("--i-teacher-rollouts", type=int, default=48)
    p.add_argument("--i-validation-interval", type=int, default=10)
    p.add_argument("--i-validation-scenarios", type=int, default=24)
    p.add_argument("--i-validation-rollouts", type=int, default=96)
    p.add_argument("--i-warmup", type=int, default=70)
    p.add_argument("--i-patience", type=int, default=10)
    p.add_argument("--i-min-delta", type=float, default=0.75)
    p.add_argument("--i-lr-scale", type=float, default=0.035)
    p.add_argument("--i-epochs", type=int, default=2)

    # Evaluation.
    p.add_argument("--main-rollouts", type=int, default=2048)
    p.add_argument("--suite-count", type=int, default=100)
    p.add_argument("--suite-rollouts", type=int, default=128)
    p.add_argument("--heldout-seed", type=int, default=260824)

    # Expensive online residual retraining.
    p.add_argument("--skip-residual-retraining", action="store_true")
    p.add_argument("--main-r-updates", type=int, default=250)
    p.add_argument("--main-r-episodes", type=int, default=96)
    p.add_argument("--main-r-validation-interval", type=int, default=5)
    p.add_argument("--main-r-validation-rollouts", type=int, default=64)
    p.add_argument("--main-r-warmup", type=int, default=40)
    p.add_argument("--main-r-patience", type=int, default=14)
    p.add_argument("--main-r-min-delta", type=float, default=0.5)
    p.add_argument("--main-r-lr-scale", type=float, default=0.18)
    p.add_argument("--main-r-epochs", type=int, default=3)

    p.add_argument(
        "--expensive-suite-count",
        type=int,
        default=24,
        help="Representative held-out scenarios receiving post-shock retraining.",
    )
    p.add_argument("--suite-r-updates", type=int, default=90)
    p.add_argument("--suite-r-episodes", type=int, default=48)
    p.add_argument("--suite-r-rollouts", type=int, default=128)
    return p.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_instance(instance_id: str, data_directory: Path):
    spec = parse_professor_instance_id(instance_id)
    benchmark = load_professor_benchmark(
        data_directory / f"CIPP-{spec.party}.xls",
        instance_id=spec.instance_id,
        objective_variant="professor_code",
        budget_mode="auto",
    )
    return benchmark.instance


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


def _load_agent(checkpoint: Path, builder, *, device: torch.device):
    agent, _ = AdvancedPPOAgent.load(
        checkpoint,
        feature_builder=builder,
        device=device,
        load_optimizer=False,
    )
    return agent


def _fresh_agent_like(
    base: AdvancedPPOAgent,
    builder,
    *,
    seed: int,
    device: torch.device,
):
    return AdvancedPPOAgent(
        feature_builder=builder,
        network_config=base.network_config,
        ppo_config=base.config,
        seed=seed,
        device=str(device),
    )


def _static_prefix(
    agent: AdvancedPPOAgent,
    *,
    switch_day: int,
    seed: int,
) -> tuple[int, ...]:
    env = CIPPEnv(agent.instance, seed=seed)
    env.reset(seed=seed)
    prefix = []
    while env.day < switch_day:
        state = agent.feature_builder.build(env)
        action, _, _ = agent.select_action(state, deterministic=True)
        env.step(int(action))
        prefix.append(int(action))
    return tuple(prefix)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    """Serialize filesystem/numpy values that appear in experiment metadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _selection_value(summary_path: Path) -> float:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("best_validation_recourse", "best_validation"):
        if key in payload:
            return float(payload[key])
    raise KeyError(f"no selection metric in {summary_path}")


def _select_best_restart(root: Path, count: int) -> tuple[Path, dict[str, object]]:
    candidates = []
    for j in range(count):
        run = root / f"restart_{j}"
        checkpoint = run / "checkpoint_best.pt"
        summary = run / "training_summary.json"
        if checkpoint.exists() and summary.exists():
            candidates.append(
                {
                    "restart": j,
                    "checkpoint": str(checkpoint),
                    "selection_value": _selection_value(summary),
                    "summary": json.loads(summary.read_text(encoding="utf-8")),
                }
            )
    if not candidates:
        raise FileNotFoundError(f"no completed restarts found under {root}")
    candidates.sort(key=lambda x: float(x["selection_value"]), reverse=True)
    best = candidates[0]
    _json(
        root / "restart_selection.json",
        {
            "best_restart": int(best["restart"]),
            "best_checkpoint": str(best["checkpoint"]),
            "best_selection_value": float(best["selection_value"]),
            "all_restarts": [
                {
                    "restart": int(x["restart"]),
                    "selection_value": float(x["selection_value"]),
                }
                for x in candidates
            ],
        },
    )
    return Path(str(best["checkpoint"])), best


def _adaptive_config_c(a) -> AdaptiveTrainingConfig:
    return AdaptiveTrainingConfig(
        updates=a.c_updates,
        episodes_per_update=a.c_episodes,
        validation_interval=a.c_validation_interval,
        validation_scenarios=a.c_validation_scenarios,
        validation_rollouts_per_scenario=a.c_validation_rollouts,
        early_stopping_patience=a.c_patience,
        early_stopping_warmup_updates=a.c_warmup,
        early_stopping_min_delta=a.c_min_delta,
        learning_rate_scale=a.c_lr_scale,
        update_epochs=a.c_epochs,
    )


def _recourse_config_p(a) -> RecourseAwareTrainingConfig:
    return RecourseAwareTrainingConfig(
        updates=a.p_updates,
        prefixes_per_update=a.p_prefixes,
        teacher_rollouts=a.p_teacher_rollouts,
        validation_interval=a.p_validation_interval,
        validation_scenarios=a.p_validation_scenarios,
        validation_rollouts=a.p_validation_rollouts,
        early_stopping_patience=a.p_patience,
        early_stopping_warmup_updates=a.p_warmup,
        early_stopping_min_delta=a.p_min_delta,
        learning_rate_scale=a.p_lr_scale,
        update_epochs=a.p_epochs,
    )


def _recourse_config_i(a) -> RecourseAwareTrainingConfig:
    return RecourseAwareTrainingConfig(
        updates=a.i_updates,
        prefixes_per_update=a.i_prefixes,
        teacher_rollouts=a.i_teacher_rollouts,
        validation_interval=a.i_validation_interval,
        validation_scenarios=a.i_validation_scenarios,
        validation_rollouts=a.i_validation_rollouts,
        early_stopping_patience=a.i_patience,
        early_stopping_warmup_updates=a.i_warmup,
        early_stopping_min_delta=a.i_min_delta,
        learning_rate_scale=a.i_lr_scale,
        update_epochs=a.i_epochs,
    )


def _residual_config_main(a) -> ResidualRetrainingConfig:
    return ResidualRetrainingConfig(
        updates=a.main_r_updates,
        episodes_per_update=a.main_r_episodes,
        validation_interval=a.main_r_validation_interval,
        validation_rollouts=a.main_r_validation_rollouts,
        early_stopping_patience=a.main_r_patience,
        early_stopping_warmup_updates=a.main_r_warmup,
        early_stopping_min_delta=a.main_r_min_delta,
        learning_rate_scale=a.main_r_lr_scale,
        update_epochs=a.main_r_epochs,
    )


def _residual_config_suite(a) -> ResidualRetrainingConfig:
    return ResidualRetrainingConfig(
        updates=a.suite_r_updates,
        episodes_per_update=a.suite_r_episodes,
        validation_interval=10,
        validation_rollouts=min(32, a.suite_r_rollouts),
        early_stopping_patience=6,
        early_stopping_warmup_updates=min(30, a.suite_r_updates),
        early_stopping_min_delta=0.75,
        learning_rate_scale=a.main_r_lr_scale,
        update_epochs=2,
    )


def _train_residual_once(
    *,
    init_checkpoint: Path | None,
    base_agent: AdvancedPPOAgent,
    builder: AdaptiveFeatureBuilder,
    scenario,
    prefix: tuple[int, ...],
    config: ResidualRetrainingConfig,
    output_directory: Path,
    seed: int,
    label: str,
    device: torch.device,
    evaluation_rollouts: int,
) -> dict[str, object]:
    if init_checkpoint is None:
        agent = _fresh_agent_like(
            base_agent,
            builder,
            seed=seed,
            device=device,
        )
        init_name = "scratch"
    else:
        agent = _load_agent(init_checkpoint, builder, device=device)
        init_name = str(init_checkpoint)

    checkpoint = train_fixed_prefix_residual(
        agent,
        scenario=scenario,
        prefix=prefix,
        config=config,
        output_directory=output_directory,
        seed=seed,
        label=label,
    )
    summary = json.loads(
        (output_directory / "training_summary.json").read_text(encoding="utf-8")
    )
    eval_agent = _load_agent(checkpoint, builder, device=device)
    result = residual_best_of_k(
        eval_agent,
        scenario=scenario,
        prefix=prefix,
        rollouts=evaluation_rollouts,
        seed=seed + 900_000_000,
        context_aware=True,
    )
    return {
        "objective": float(result["objective"]),
        "itinerary": result["itinerary"],
        "inference_runtime_seconds": float(result["runtime_seconds"]),
        "online_training_seconds": float(summary["elapsed_seconds"]),
        "init": init_name,
        "checkpoint": str(checkpoint),
    }


def _best_residual_from_inits(
    *,
    init_candidates: list[tuple[str, Path | None]],
    base_agent: AdvancedPPOAgent,
    builder: AdaptiveFeatureBuilder,
    scenario,
    prefix: tuple[int, ...],
    config: ResidualRetrainingConfig,
    output_root: Path,
    seed: int,
    label: str,
    device: torch.device,
    evaluation_rollouts: int,
) -> dict[str, object]:
    trials = []
    for j, (name, checkpoint) in enumerate(init_candidates):
        trial = _train_residual_once(
            init_checkpoint=checkpoint,
            base_agent=base_agent,
            builder=builder,
            scenario=scenario,
            prefix=prefix,
            config=config,
            output_directory=output_root / f"{j:02d}_{name}",
            seed=seed + j * 1_000_003,
            label=f"{label}_{name}",
            device=device,
            evaluation_rollouts=evaluation_rollouts,
        )
        trial["init_name"] = name
        trials.append(trial)
    best = max(trials, key=lambda x: float(x["objective"]))
    return {
        **best,
        "all_trials": trials,
        "total_online_training_seconds": float(
            sum(float(x["online_training_seconds"]) for x in trials)
        ),
    }


def _method_row(method: str, result, *, prefix, **extra):
    return {
        "method": method,
        "objective": float(result["objective"]),
        "runtime_seconds": float(
            result.get("runtime_seconds", result.get("inference_runtime_seconds", 0.0))
        ),
        "prefix": list(prefix),
        **extra,
    }


def main() -> None:
    a = _args()
    device = _device(a.device)
    target = _load_instance(a.instance, a.data_directory)
    builder_static = StructuredFeatureBuilder(target)
    builder_adaptive = AdaptiveFeatureBuilder(target)

    output = a.output_directory / target.instance_id / f"seed_{a.seed}"
    output.mkdir(parents=True, exist_ok=True)

    static_checkpoint = (
        a.static_checkpoint
        if a.static_checkpoint is not None
        else _default_static_checkpoint(target.instance_id, a.seed)
    )
    if not static_checkpoint.exists():
        raise FileNotFoundError(
            f"Static Attention checkpoint not found: {static_checkpoint}"
        )
    static_agent = _load_agent(
        static_checkpoint,
        builder_static,
        device=device,
    )

    training_ids = (
        [x.strip() for x in a.training_instances.split(",") if x.strip()]
        if a.training_instances
        else [target.instance_id]
    )
    training_instances = [
        _load_instance(x, a.data_directory) for x in training_ids
    ]
    if any(x.n != target.n for x in training_instances):
        raise ValueError(
            "V7 multi-instance generalist currently requires all training "
            "instances to have the same number of locations as the target."
        )
    training_builders = [
        AdaptiveFeatureBuilder(x) for x in training_instances
    ]

    # ==============================================================
    # C — high-compute rich-shock generalist, multiple restarts.
    # ==============================================================
    c_root = output / "C_generalist"
    if not a.eval_only:
        for j in range(a.c_restarts):
            run = c_root / f"restart_{j}"
            if (run / "checkpoint_best.pt").exists():
                continue
            agent = _load_agent(
                static_checkpoint,
                AdaptiveFeatureBuilder(target),
                device=device,
            )
            seed_j = a.seed + j * 10_000_019
            if len(training_builders) == 1:
                train_method_c(
                    agent,
                    config=_adaptive_config_c(a),
                    output_directory=run,
                    seed=seed_j,
                )
            else:
                train_multi_instance_method_c(
                    agent,
                    feature_builders=training_builders,
                    config=_adaptive_config_c(a),
                    output_directory=run,
                    seed=seed_j,
                )
    c_checkpoint, c_selection = _select_best_restart(
        c_root, a.c_restarts
    )
    c_agent = _load_agent(
        c_checkpoint,
        AdaptiveFeatureBuilder(target),
        device=device,
    )

    # ==============================================================
    # P — recourse-aware general prefix. Teacher = frozen best C.
    # ==============================================================
    p_root = output / "P_recourse_prefix"
    if not a.eval_only:
        for j in range(a.p_restarts):
            run = p_root / f"restart_{j}"
            if (run / "checkpoint_best.pt").exists():
                continue
            student = _load_agent(
                c_checkpoint,
                AdaptiveFeatureBuilder(target),
                device=device,
            )
            teacher = _load_agent(
                c_checkpoint,
                AdaptiveFeatureBuilder(target),
                device=device,
            )
            train_recourse_aware_prefix(
                student,
                teacher=teacher,
                config=_recourse_config_p(a),
                output_directory=run,
                seed=a.seed + 100_000_003 + j * 10_000_019,
                label="P_general_recourse_aware_prefix",
                recourse_solver="policy",
            )
    p_checkpoint, p_selection = _select_best_restart(
        p_root, a.p_restarts
    )
    p_agent = _load_agent(
        p_checkpoint,
        AdaptiveFeatureBuilder(target),
        device=device,
    )

    # ==============================================================
    # I — intensive instance-specific recourse-prefix refinement.
    # It uses a disjoint shock RNG and never sees held-out shock identities.
    # ==============================================================
    i_root = output / "I_instance_specific_prefix"
    if not a.eval_only:
        for j in range(a.i_restarts):
            run = i_root / f"restart_{j}"
            if (run / "checkpoint_best.pt").exists():
                continue
            student = _load_agent(
                p_checkpoint,
                AdaptiveFeatureBuilder(target),
                device=device,
            )
            teacher = _load_agent(
                c_checkpoint,
                AdaptiveFeatureBuilder(target),
                device=device,
            )
            train_recourse_aware_prefix(
                student,
                teacher=teacher,
                config=_recourse_config_i(a),
                output_directory=run,
                seed=a.seed + 300_000_007 + j * 10_000_019,
                label="I_instance_specific_recourse_prefix",
                recourse_solver="policy",
            )
    i_checkpoint, i_selection = _select_best_restart(
        i_root, a.i_restarts
    )
    i_agent = _load_agent(
        i_checkpoint,
        AdaptiveFeatureBuilder(target),
        device=device,
    )

    # ==============================================================
    # Frozen held-out shock suite.
    # ==============================================================
    suite = generate_heldout_single_shock_suite(
        target,
        count=a.suite_count,
        seed=a.heldout_seed,
    )
    _json(
        output / "heldout_suite.json",
        {
            "instance": target.instance_id,
            "heldout_seed": int(a.heldout_seed),
            "count": len(suite),
            "training_instances": training_ids,
            "training_curriculum": RICH_CURRICULUM_DESCRIPTION,
            "scenarios": [scenario_to_dict(s) for s in suite],
        },
    )

    # ==============================================================
    # Main pilot — every old/new RL idea at high compute.
    # ==============================================================
    main = suite[0]
    s_prefix = _static_prefix(
        static_agent,
        switch_day=main.switch_day,
        seed=a.seed,
    )
    c_prefix = build_anticipatory_prefix(
        c_agent,
        scenario=main,
        seed=a.seed + 1_000_001,
        deterministic=True,
    )
    p_prefix = build_anticipatory_prefix(
        p_agent,
        scenario=main,
        seed=a.seed + 2_000_003,
        deterministic=True,
    )
    i_prefix = build_anticipatory_prefix(
        i_agent,
        scenario=main,
        seed=a.seed + 3_000_007,
        deterministic=True,
    )

    prefixes_main = {
        "S_static_rl": list(s_prefix),
        "C_generalist": list(c_prefix),
        "P_recourse_aware": list(p_prefix),
        "I_instance_specific": list(i_prefix),
    }
    _json(output / "main_prefixes.json", prefixes_main)

    main_rows = []

    b = residual_greedy(
        static_agent,
        scenario=main,
        prefix=s_prefix,
        seed=a.seed + 10_000,
        context_aware=False,
    )
    main_rows.append(_method_row("B_frozen_static_greedy", b, prefix=s_prefix))

    d = residual_best_of_k(
        static_agent,
        scenario=main,
        prefix=s_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 20_000,
        context_aware=False,
    )
    main_rows.append(
        _method_row(
            f"D_frozen_static_best_of_{a.main_rollouts}",
            d,
            prefix=s_prefix,
            rollouts=a.main_rollouts,
        )
    )

    c_s = residual_best_of_k(
        c_agent,
        scenario=main,
        prefix=s_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 30_000,
        context_aware=True,
    )
    main_rows.append(
        _method_row(
            f"C_S_static_prefix_best_of_{a.main_rollouts}",
            c_s,
            prefix=s_prefix,
            rollouts=a.main_rollouts,
        )
    )

    c_greedy = anticipatory_then_greedy(
        c_agent,
        scenario=main,
        seed=a.seed + 40_000,
        deterministic_prefix=True,
    )
    main_rows.append(
        _method_row("C_C_e2e_greedy", c_greedy, prefix=c_prefix)
    )

    c_c = residual_best_of_k(
        c_agent,
        scenario=main,
        prefix=c_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 50_000,
        context_aware=True,
    )
    main_rows.append(
        _method_row(
            f"C_C_e2e_best_of_{a.main_rollouts}",
            c_c,
            prefix=c_prefix,
            rollouts=a.main_rollouts,
        )
    )

    p_p = residual_best_of_k(
        p_agent,
        scenario=main,
        prefix=p_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 60_000,
        context_aware=True,
    )
    main_rows.append(
        _method_row(
            f"P_P_recourse_prefix_own_suffix_best_of_{a.main_rollouts}",
            p_p,
            prefix=p_prefix,
            rollouts=a.main_rollouts,
        )
    )
    p_c = residual_best_of_k(
        c_agent,
        scenario=main,
        prefix=p_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 70_000,
        context_aware=True,
    )
    main_rows.append(
        _method_row(
            f"P_C_recourse_prefix_C_suffix_best_of_{a.main_rollouts}",
            p_c,
            prefix=p_prefix,
            rollouts=a.main_rollouts,
        )
    )

    i_i = residual_best_of_k(
        i_agent,
        scenario=main,
        prefix=i_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 80_000,
        context_aware=True,
    )
    main_rows.append(
        _method_row(
            f"I_I_instance_prefix_own_suffix_best_of_{a.main_rollouts}",
            i_i,
            prefix=i_prefix,
            rollouts=a.main_rollouts,
        )
    )
    i_c = residual_best_of_k(
        c_agent,
        scenario=main,
        prefix=i_prefix,
        rollouts=a.main_rollouts,
        seed=a.seed + 90_000,
        context_aware=True,
    )
    main_rows.append(
        _method_row(
            f"I_C_instance_prefix_C_suffix_best_of_{a.main_rollouts}",
            i_c,
            prefix=i_prefix,
            rollouts=a.main_rollouts,
        )
    )

    if not a.skip_residual_retraining:
        r_cfg = _residual_config_main(a)
        residual_specs = [
            (
                "S_R",
                s_prefix,
                [
                    ("static_init", static_checkpoint),
                    ("C_init", c_checkpoint),
                    ("scratch", None),
                ],
            ),
            (
                "C_R",
                c_prefix,
                [
                    ("C_init", c_checkpoint),
                    ("P_init", p_checkpoint),
                    ("scratch", None),
                ],
            ),
            (
                "P_R",
                p_prefix,
                [
                    ("C_init", c_checkpoint),
                    ("P_init", p_checkpoint),
                    ("I_init", i_checkpoint),
                    ("scratch", None),
                ],
            ),
            (
                "I_R",
                i_prefix,
                [
                    ("C_init", c_checkpoint),
                    ("P_init", p_checkpoint),
                    ("I_init", i_checkpoint),
                    ("scratch", None),
                ],
            ),
        ]
        for j, (name, prefix, inits) in enumerate(residual_specs):
            result = _best_residual_from_inits(
                init_candidates=inits,
                base_agent=static_agent,
                builder=AdaptiveFeatureBuilder(target),
                scenario=main,
                prefix=prefix,
                config=r_cfg,
                output_root=output / "main_residual" / name,
                seed=a.seed + 500_000_000 + j * 10_000_019,
                label=name,
                device=device,
                evaluation_rollouts=a.main_rollouts,
            )
            main_rows.append(
                {
                    "method": f"{name}_best_online_residual_RL",
                    "objective": float(result["objective"]),
                    "runtime_seconds": float(
                        result["inference_runtime_seconds"]
                    ),
                    "online_training_seconds": float(
                        result["online_training_seconds"]
                    ),
                    "total_search_training_seconds": float(
                        result["total_online_training_seconds"]
                    ),
                    "winning_init": str(result["init_name"]),
                    "prefix": list(prefix),
                }
            )
            _json(
                output / "main_residual" / name / "best_result.json",
                result,
            )

    _json(output / "main_rl_all_methods.json", main_rows)
    _write_csv(
        output / "main_rl_all_methods.csv",
        [{k: v for k, v in x.items() if k != "prefix"} for x in main_rows],
    )

    # ==============================================================
    # Full 100-scenario cheap/medium-cost RL suite.
    # ==============================================================
    suite_rows = []
    suite_prefixes = []
    for idx, scenario in enumerate(suite):
        s_pre = _static_prefix(
            static_agent,
            switch_day=scenario.switch_day,
            seed=a.seed + 1_000_000 + idx,
        )
        c_pre = build_anticipatory_prefix(
            c_agent,
            scenario=scenario,
            seed=a.seed + 2_000_000 + idx,
            deterministic=True,
        )
        p_pre = build_anticipatory_prefix(
            p_agent,
            scenario=scenario,
            seed=a.seed + 3_000_000 + idx,
            deterministic=True,
        )
        i_pre = build_anticipatory_prefix(
            i_agent,
            scenario=scenario,
            seed=a.seed + 4_000_000 + idx,
            deterministic=True,
        )
        suite_prefixes.append(
            {
                "scenario_index": idx,
                "scenario_id": scenario.scenario_id,
                "switch_day": int(scenario.switch_day),
                "S_static_rl": list(s_pre),
                "C_generalist": list(c_pre),
                "P_recourse_aware": list(p_pre),
                "I_instance_specific": list(i_pre),
            }
        )

        common = {
            "scenario_index": idx,
            "scenario_id": scenario.scenario_id,
            "family": scenario.family,
            "switch_day": int(scenario.switch_day),
        }

        eval_specs = [
            (
                "B_frozen_static_greedy",
                residual_greedy(
                    static_agent,
                    scenario=scenario,
                    prefix=s_pre,
                    seed=a.seed + 10_000_000 + idx,
                    context_aware=False,
                ),
                s_pre,
            ),
            (
                f"D_frozen_static_best_of_{a.suite_rollouts}",
                residual_best_of_k(
                    static_agent,
                    scenario=scenario,
                    prefix=s_pre,
                    rollouts=a.suite_rollouts,
                    seed=a.seed + 20_000_000 + idx * 1009,
                    context_aware=False,
                ),
                s_pre,
            ),
            (
                f"C_S_static_prefix_best_of_{a.suite_rollouts}",
                residual_best_of_k(
                    c_agent,
                    scenario=scenario,
                    prefix=s_pre,
                    rollouts=a.suite_rollouts,
                    seed=a.seed + 30_000_000 + idx * 1009,
                    context_aware=True,
                ),
                s_pre,
            ),
            (
                f"C_C_e2e_best_of_{a.suite_rollouts}",
                residual_best_of_k(
                    c_agent,
                    scenario=scenario,
                    prefix=c_pre,
                    rollouts=a.suite_rollouts,
                    seed=a.seed + 40_000_000 + idx * 1009,
                    context_aware=True,
                ),
                c_pre,
            ),
            (
                f"P_P_best_of_{a.suite_rollouts}",
                residual_best_of_k(
                    p_agent,
                    scenario=scenario,
                    prefix=p_pre,
                    rollouts=a.suite_rollouts,
                    seed=a.seed + 50_000_000 + idx * 1009,
                    context_aware=True,
                ),
                p_pre,
            ),
            (
                f"P_C_best_of_{a.suite_rollouts}",
                residual_best_of_k(
                    c_agent,
                    scenario=scenario,
                    prefix=p_pre,
                    rollouts=a.suite_rollouts,
                    seed=a.seed + 60_000_000 + idx * 1009,
                    context_aware=True,
                ),
                p_pre,
            ),
            (
                f"I_I_best_of_{a.suite_rollouts}",
                residual_best_of_k(
                    i_agent,
                    scenario=scenario,
                    prefix=i_pre,
                    rollouts=a.suite_rollouts,
                    seed=a.seed + 70_000_000 + idx * 1009,
                    context_aware=True,
                ),
                i_pre,
            ),
            (
                f"I_C_best_of_{a.suite_rollouts}",
                residual_best_of_k(
                    c_agent,
                    scenario=scenario,
                    prefix=i_pre,
                    rollouts=a.suite_rollouts,
                    seed=a.seed + 80_000_000 + idx * 1009,
                    context_aware=True,
                ),
                i_pre,
            ),
        ]

        for method, result, prefix in eval_specs:
            suite_rows.append(
                {
                    **common,
                    "method": method,
                    "objective": float(result["objective"]),
                    "runtime_seconds": float(result["runtime_seconds"]),
                    "prefix": list(prefix),
                }
            )

        if (idx + 1) % 10 == 0 or idx == 0:
            print(
                f"[V7 suite] completed {idx + 1}/{len(suite)}",
                flush=True,
            )

    _json(output / "suite_prefixes.json", suite_prefixes)
    _json(output / "suite_rl_all_methods.json", suite_rows)
    _write_csv(
        output / "suite_rl_all_methods.csv",
        [{k: v for k, v in x.items() if k != "prefix"} for x in suite_rows],
    )

    # ==============================================================
    # Expensive residual-RL suite on representative held-out shocks.
    # ==============================================================
    expensive_rows = []
    if (
        not a.skip_residual_retraining
        and a.expensive_suite_count > 0
        and not a.eval_only
    ):
        selected = np.unique(
            np.linspace(
                0,
                len(suite) - 1,
                num=min(a.expensive_suite_count, len(suite)),
                dtype=int,
            )
        ).tolist()
        r_cfg = _residual_config_suite(a)
        prefix_lookup = {
            int(x["scenario_index"]): x for x in suite_prefixes
        }

        for rank, idx in enumerate(selected):
            scenario = suite[idx]
            px = prefix_lookup[idx]
            variants = [
                (
                    "S_R",
                    tuple(px["S_static_rl"]),
                    static_checkpoint,
                ),
                (
                    "C_R",
                    tuple(px["C_generalist"]),
                    c_checkpoint,
                ),
                (
                    "P_R",
                    tuple(px["P_recourse_aware"]),
                    c_checkpoint,
                ),
                (
                    "I_R",
                    tuple(px["I_instance_specific"]),
                    c_checkpoint,
                ),
            ]
            for v, prefix, init_checkpoint in variants:
                result = _train_residual_once(
                    init_checkpoint=init_checkpoint,
                    base_agent=static_agent,
                    builder=AdaptiveFeatureBuilder(target),
                    scenario=scenario,
                    prefix=prefix,
                    config=r_cfg,
                    output_directory=(
                        output
                        / "expensive_residual_suite"
                        / f"{idx:03d}_{scenario.scenario_id}"
                        / v
                    ),
                    seed=a.seed + 800_000_000 + idx * 10_003 + len(expensive_rows),
                    label=f"{v}_suite",
                    device=device,
                    evaluation_rollouts=a.suite_r_rollouts,
                )
                expensive_rows.append(
                    {
                        "scenario_index": idx,
                        "scenario_id": scenario.scenario_id,
                        "family": scenario.family,
                        "switch_day": int(scenario.switch_day),
                        "method": v,
                        "objective": float(result["objective"]),
                        "inference_runtime_seconds": float(
                            result["inference_runtime_seconds"]
                        ),
                        "online_training_seconds": float(
                            result["online_training_seconds"]
                        ),
                        "prefix": list(prefix),
                    }
                )
            print(
                f"[V7 expensive R] {rank + 1}/{len(selected)} "
                f"scenario={idx}",
                flush=True,
            )

        _json(output / "expensive_residual_suite.json", expensive_rows)
        _write_csv(
            output / "expensive_residual_suite.csv",
            [{k: v for k, v in x.items() if k != "prefix"} for x in expensive_rows],
        )

    # ==============================================================
    # Aggregate RL-only summary.
    # ==============================================================
    summary = {"overall": {}, "by_family": {}}
    methods = sorted({str(x["method"]) for x in suite_rows})
    for method in methods:
        vals = np.asarray(
            [
                float(x["objective"])
                for x in suite_rows
                if x["method"] == method
            ],
            dtype=np.float64,
        )
        summary["overall"][method] = {
            "count": int(vals.size),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    families = sorted({str(x["family"]) for x in suite_rows})
    for family in families:
        summary["by_family"][family] = {}
        for method in methods:
            vals = np.asarray(
                [
                    float(x["objective"])
                    for x in suite_rows
                    if x["family"] == family and x["method"] == method
                ],
                dtype=np.float64,
            )
            if vals.size:
                summary["by_family"][family][method] = {
                    "count": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                }
    _json(output / "suite_rl_summary.json", summary)

    # Storage cleanup: keep only the selected C/P/I model checkpoints.
    selected_keep = {
        c_checkpoint.resolve(),
        p_checkpoint.resolve(),
        i_checkpoint.resolve(),
    }
    for q in output.rglob("checkpoint_best.pt"):
        try:
            if q.resolve() not in selected_keep:
                q.unlink()
        except OSError:
            pass

    # Drop disposable checkpoint_last files and expensive-suite checkpoints.
    for q in output.rglob("checkpoint_last.pt"):
        try:
            q.unlink()
        except OSError:
            pass
    expensive_root = output / "expensive_residual_suite"
    if expensive_root.exists():
        for q in expensive_root.rglob("checkpoint_best.pt"):
            try:
                q.unlink()
            except OSError:
                pass

    metadata = {
        "version": "V7_ALL_METHODS",
        "instance": target.instance_id,
        "training_instances": training_ids,
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
        "selected_checkpoints": {
            "C": str(c_checkpoint),
            "P": str(p_checkpoint),
            "I": str(i_checkpoint),
        },
        "selection": {
            "C": c_selection,
            "P": p_selection,
            "I": i_selection,
        },
        "heldout_suite_count": len(suite),
        "heldout_seed": int(a.heldout_seed),
        "information_safety": {
            "future_shock_identity_hidden_before_prefix_commitment": True,
            "no_pre_shock_best_of_k_hindsight": True,
            "B_D_static_parallel_observation_env": True,
            "P_training_recourse_revealed_only_after_prefix": True,
            "I_training_uses_disjoint_shock_rng_not_heldout_suite": True,
            "online_R_training_begins_only_after_realized_prefix_is_fixed": True,
        },
    }
    _json(output / "run_metadata.json", metadata)

    print("\n=== V7 MAIN ALL-RL METHODS ===")
    for row in sorted(main_rows, key=lambda x: float(x["objective"]), reverse=True):
        training = (
            f" train={row['online_training_seconds']:.1f}s"
            if "online_training_seconds" in row
            else ""
        )
        print(
            f"{row['method']}: objective={float(row['objective']):.3f} "
            f"runtime={float(row.get('runtime_seconds', 0.0)):.3f}s{training}"
        )
    print("\n=== V7 100-SCENARIO RL MEANS ===")
    for method, stats in sorted(
        summary["overall"].items(),
        key=lambda kv: float(kv[1]["mean"]),
        reverse=True,
    ):
        print(
            f"{method}: mean={stats['mean']:.3f} "
            f"median={stats['median']:.3f} std={stats['std']:.3f}"
        )
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
