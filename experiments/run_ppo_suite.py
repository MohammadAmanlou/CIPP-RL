"""Train and compare all paper-ready PPO variants on real CIPP instances."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.advanced import (
    AdvancedPPOAgent,
    AdvancedPPOConfig,
    ConstructionTrainingConfig,
    ImprovementPPOAgent,
    ImprovementPPOConfig,
    ImprovementTrainingConfig,
    NetworkConfig,
    PolicyEvaluation,
    StructuredFeatureBuilder,
    evaluate_best_of_k,
    evaluate_construction_policy,
    evaluate_rl_improvement,
    policy_beam_search,
    train_construction_agent,
    train_improvement_agent,
)
from src.baselines import run_greedy_policy
from src.core import evaluate_itinerary
from src.optimization import solve_cipp_gurobi
from src.utils import (
    ProfessorBenchmark,
    benchmark_from_instance,
    load_instance,
    load_professor_benchmark,
    parse_professor_instance_id,
)


CONSTRUCTION_METHODS = ("stable_mlp", "attention", "hierarchical", "hacipp")
ALL_METHODS = (*CONSTRUCTION_METHODS, "hacipp_rl_improve")


@dataclass(frozen=True, slots=True)
class Profile:
    construction: ConstructionTrainingConfig
    improvement_training: ImprovementTrainingConfig
    improvement: ImprovementPPOConfig
    final_rollouts: int
    beam_width: int
    beam_expansion: int


PROFILES = {
    "smoke": Profile(
        construction=ConstructionTrainingConfig(
            updates=2,
            episodes_per_update=4,
            pomo_group_size=2,
            validation_interval=1,
            validation_rollouts=2,
            self_imitation_interval=1,
            self_imitation_elites=1,
            early_stopping_patience=0,
            early_stopping_warmup_updates=0,
            history_flush_interval=1,
        ),
        improvement_training=ImprovementTrainingConfig(
            updates=1,
            episodes_per_update=2,
            construction_rollouts=1,
            validation_interval=1,
            validation_starts=1,
            validation_construction_rollouts=1,
            early_stopping_patience=0,
            history_flush_interval=1,
        ),
        improvement=ImprovementPPOConfig(max_candidates=16, max_moves=3, minibatch_size=16),
        final_rollouts=4,
        beam_width=4,
        beam_expansion=2,
    ),
    "quick": Profile(
        construction=ConstructionTrainingConfig(
            updates=50,
            episodes_per_update=16,
            pomo_group_size=8,
            validation_interval=5,
            validation_rollouts=10,
            self_imitation_interval=5,
            early_stopping_patience=5,
            early_stopping_min_delta=1.0,
            early_stopping_warmup_updates=20,
            history_flush_interval=5,
        ),
        improvement_training=ImprovementTrainingConfig(
            updates=30,
            episodes_per_update=4,
            construction_rollouts=2,
            validation_interval=5,
            validation_starts=2,
            validation_construction_rollouts=2,
            early_stopping_patience=4,
            early_stopping_min_delta=1.0,
            early_stopping_warmup_updates=10,
            history_flush_interval=5,
        ),
        improvement=ImprovementPPOConfig(max_candidates=64, max_moves=15),
        final_rollouts=30,
        beam_width=16,
        beam_expansion=3,
    ),
    "full": Profile(
        construction=ConstructionTrainingConfig(
            updates=400,
            episodes_per_update=64,
            pomo_group_size=16,
            validation_interval=20,
            validation_rollouts=30,
            self_imitation_interval=10,
            self_imitation_elites=8,
            early_stopping_patience=6,
            early_stopping_min_delta=1.0,
            early_stopping_warmup_updates=120,
            history_flush_interval=10,
        ),
        improvement_training=ImprovementTrainingConfig(
            updates=200,
            episodes_per_update=8,
            construction_rollouts=4,
            validation_interval=10,
            validation_starts=4,
            validation_construction_rollouts=4,
            early_stopping_patience=8,
            early_stopping_min_delta=1.0,
            early_stopping_warmup_updates=60,
            history_flush_interval=10,
        ),
        improvement=ImprovementPPOConfig(max_candidates=96, max_moves=30),
        final_rollouts=128,
        beam_width=64,
        beam_expansion=4,
    ),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        nargs="+",
        default=[],
        help=(
            "Professor instance IDs. 14S means 14 visitable states; Idle is "
            "an additional internal action. 30P means 30 periods."
        ),
    )
    parser.add_argument(
        "--instance-files",
        nargs="*",
        type=Path,
        default=[],
        help="JSON CIPP instances whose n and H fields define model dimensions.",
    )
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--objective-variant", choices=("professor_code", "paper_equation"), default="professor_code")
    parser.add_argument("--budget-mode", choices=("auto", "disabled", "paper"), default="auto")
    parser.add_argument("--active-search-mode", choices=("full", "eas"), default="full")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--data-directory", type=Path, default=Path("."))
    parser.add_argument("--output-directory", type=Path, default=Path("results/ppo_suite"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help=(
            "Never train. Require every requested checkpoint to exist and run "
            "only frozen-policy inference/baselines/Gurobi."
        ),
    )
    parser.add_argument("--run-gurobi", action="store_true")
    parser.add_argument("--run-gurobi-warm-start", action="store_true")
    parser.add_argument("--gurobi-time-limit", type=float, default=3600.0)
    parser.add_argument("--construction-updates", type=int)
    parser.add_argument("--episodes-per-update", type=int)
    parser.add_argument("--improvement-updates", type=int)
    parser.add_argument("--validation-metric", choices=("deterministic", "mean_stochastic", "best_of_k"))
    parser.add_argument("--construction-early-stopping-patience", type=int)
    parser.add_argument("--construction-early-stopping-min-delta", type=float)
    parser.add_argument("--construction-early-stopping-warmup-updates", type=int)
    parser.add_argument("--improvement-early-stopping-patience", type=int)
    parser.add_argument("--improvement-early-stopping-min-delta", type=float)
    parser.add_argument("--improvement-early-stopping-warmup-updates", type=int)
    parser.add_argument("--disable-vectorized-rollouts", action="store_true")
    parser.add_argument("--final-rollouts", type=int)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument(
        "--log-every-episodes",
        type=int,
        default=256,
        help=(
            "Print training metrics after approximately this many completed "
            "episodes, at the next PPO update boundary. Use 0 to disable."
        ),
    )
    return parser.parse_args()


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return requested


def _resolved_profile(arguments: argparse.Namespace) -> Profile:
    profile = PROFILES[arguments.profile]
    construction = profile.construction
    improvement_training = profile.improvement_training
    if arguments.construction_updates is not None:
        construction = replace(construction, updates=arguments.construction_updates)
    if arguments.episodes_per_update is not None:
        construction = replace(construction, episodes_per_update=arguments.episodes_per_update)
    if arguments.validation_metric is not None:
        construction = replace(construction, validation_metric=arguments.validation_metric)
    if arguments.construction_early_stopping_patience is not None:
        construction = replace(
            construction,
            early_stopping_patience=arguments.construction_early_stopping_patience,
        )
    if arguments.construction_early_stopping_min_delta is not None:
        construction = replace(
            construction,
            early_stopping_min_delta=arguments.construction_early_stopping_min_delta,
        )
    if arguments.construction_early_stopping_warmup_updates is not None:
        construction = replace(
            construction,
            early_stopping_warmup_updates=arguments.construction_early_stopping_warmup_updates,
        )
    if arguments.disable_vectorized_rollouts:
        construction = replace(construction, vectorized_rollouts=False)

    if arguments.improvement_updates is not None:
        improvement_training = replace(
            improvement_training, updates=arguments.improvement_updates
        )
    if arguments.improvement_early_stopping_patience is not None:
        improvement_training = replace(
            improvement_training,
            early_stopping_patience=arguments.improvement_early_stopping_patience,
        )
    if arguments.improvement_early_stopping_min_delta is not None:
        improvement_training = replace(
            improvement_training,
            early_stopping_min_delta=arguments.improvement_early_stopping_min_delta,
        )
    if arguments.improvement_early_stopping_warmup_updates is not None:
        improvement_training = replace(
            improvement_training,
            early_stopping_warmup_updates=arguments.improvement_early_stopping_warmup_updates,
        )

    return replace(
        profile,
        construction=construction,
        improvement_training=improvement_training,
        final_rollouts=(
            arguments.final_rollouts
            if arguments.final_rollouts is not None
            else profile.final_rollouts
        ),
        beam_width=(
            arguments.beam_width if arguments.beam_width is not None else profile.beam_width
        ),
    )


def _method_configs(
    method: str,
    instance_q: int,
    profile: Profile,
    active_search_mode: str,
) -> tuple[NetworkConfig, AdvancedPPOConfig, ConstructionTrainingConfig]:
    base_ppo = AdvancedPPOConfig(
        update_epochs=4,
        minibatch_size=512,
        gae_lambda=1.0,
        advantage_mode="gae",
        active_search_mode=active_search_mode,
    )
    training = replace(profile.construction, pomo_group_size=1, self_imitation_interval=0)
    if method == "stable_mlp":
        return NetworkConfig(architecture="stable_mlp", hidden_dim=256, max_visit_count=instance_q), base_ppo, training
    if method == "attention":
        return NetworkConfig(architecture="attention", hidden_dim=128, max_visit_count=instance_q), base_ppo, training
    if method == "hierarchical":
        return NetworkConfig(architecture="hierarchical_attention", hidden_dim=128, max_visit_count=instance_q), base_ppo, training
    if method == "hacipp":
        return (
            NetworkConfig(
                architecture="hierarchical_attention",
                hidden_dim=128,
                residual_marginal=True,
                count_planner=True,
                max_visit_count=instance_q,
            ),
            replace(base_ppo, advantage_mode="hybrid", group_advantage_weight=0.5),
            profile.construction,
        )
    raise ValueError(f"unknown construction method {method}")


def _greedy_evaluation(benchmark: ProfessorBenchmark) -> PolicyEvaluation:
    result = run_greedy_policy(benchmark.instance)
    evaluation = evaluate_itinerary(benchmark.instance, result.itinerary)
    total = max(float(np.sum(evaluation.visit_counts)), 1.0)
    return PolicyEvaluation(
        method="greedy_exact_increment",
        objective=result.objective,
        itinerary=result.itinerary,
        feasible=result.feasible,
        runtime_seconds=result.runtime_seconds,
        idle_days=result.idle_days,
        unique_locations=result.unique_locations,
        visit_hhi=float(np.sum((evaluation.visit_counts / total) ** 2)),
    )


def _row(
    benchmark: ProfessorBenchmark,
    evaluation: PolicyEvaluation,
    *,
    seed: int,
    source: str = "measured",
) -> dict[str, Any]:
    reference = benchmark.table5_reference
    bfs = reference.bfs if reference else None
    improvement = (
        100.0 * (evaluation.objective - bfs) / abs(bfs) if bfs not in (None, 0.0) else None
    )
    return {
        "instance_id": benchmark.instance.instance_id,
        "party": benchmark.party,
        "benchmark_state_count_excluding_idle": benchmark.state_count,
        "visit_location_count": benchmark.visit_location_count,
        "action_count_including_idle": benchmark.action_count,
        "idle_action": benchmark.idle_action,
        "horizon": benchmark.instance.H,
        "method": evaluation.method,
        "source": source,
        "seed": seed,
        "objective": evaluation.objective,
        "published_gurobi_bfs": bfs,
        "improvement_over_published_bfs_percent": improvement,
        "gap_to_published_bfs_percent": -improvement if improvement is not None else None,
        "published_gurobi_gap_percent": reference.optimality_gap_percent if reference else None,
        "measured_exact_objective": None,
        "gap_to_measured_exact_percent": None,
        "measured_exact_status": None,
        "feasible": evaluation.feasible,
        "runtime_seconds": evaluation.runtime_seconds,
        "idle_days": evaluation.idle_days,
        "unique_locations": evaluation.unique_locations,
        "visit_hhi": evaluation.visit_hhi,
        "start_generation_seconds": evaluation.start_generation_seconds,
        "improvement_seconds": evaluation.improvement_seconds,
        "itinerary": json.dumps(evaluation.itinerary),
    }


def _write_leaderboard(rows: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with (directory / "leaderboard.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    headers = ("instance_id", "method", "objective", "improvement_over_published_bfs_percent", "feasible", "runtime_seconds")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in sorted(rows, key=lambda item: (item["instance_id"], -float(item["objective"]))):
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    (directory / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "leaderboard.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _train_or_load(
    *,
    method: str,
    benchmark: ProfessorBenchmark,
    profile: Profile,
    output_directory: Path,
    seed: int,
    device: str,
    resume: bool,
    benchmark_only: bool,
    active_search_mode: str,
    initial_checkpoint: Path | None,
    log_every_episodes: int,
) -> AdvancedPPOAgent:
    feature_builder = StructuredFeatureBuilder(benchmark.instance)
    network_config, ppo_config, training_config = _method_configs(
        method, benchmark.instance.q, profile, active_search_mode
    )
    best_path = output_directory / "checkpoint_best.pt"
    if (resume or benchmark_only) and best_path.exists():
        print(
            f"[checkpoint:load] phase=construction instance={benchmark.instance.instance_id} "
            f"method={method} checkpoint={best_path}",
            flush=True,
        )
        agent, _ = AdvancedPPOAgent.load(
            best_path, feature_builder=feature_builder, device=device
        )
        return agent
    if benchmark_only:
        raise FileNotFoundError(
            f"--benchmark-only requires checkpoint: {best_path}"
        )
    agent = AdvancedPPOAgent(
        feature_builder=feature_builder,
        network_config=network_config,
        ppo_config=ppo_config,
        seed=seed,
        device=device,
    )
    if initial_checkpoint is not None:
        source, _ = AdvancedPPOAgent.load(
            initial_checkpoint,
            feature_builder=feature_builder,
            device=device,
            load_optimizer=False,
        )
        if source.network_config != network_config:
            raise ValueError(
                "initial checkpoint architecture does not match the requested method"
            )
        agent.network.load_state_dict(source.network.state_dict())
    best_path, _, _ = train_construction_agent(
        agent,
        config=training_config,
        output_directory=output_directory,
        seed=seed,
        method_name=method,
        log_every_episodes=log_every_episodes,
    )
    selected, _ = AdvancedPPOAgent.load(
        best_path, feature_builder=feature_builder, device=device
    )
    return selected


def main() -> None:
    arguments = _arguments()
    if arguments.log_every_episodes < 0:
        raise ValueError("--log-every-episodes must be non-negative")
    if not arguments.instances and not arguments.instance_files:
        arguments.instances = ["D_14S_30P"]
    device = _device(arguments.device)
    profile = _resolved_profile(arguments)
    print(
        f"[suite:start] profile={arguments.profile} device={device} "
        f"seed={arguments.seed} methods={','.join(arguments.methods)} "
        f"benchmark_only={arguments.benchmark_only} "
        f"log_every_episodes={arguments.log_every_episodes}",
        flush=True,
    )
    if arguments.active_search_mode == "eas" and arguments.initial_checkpoint is None:
        raise ValueError(
            "efficient active search requires --initial-checkpoint from a compatible "
            "pretrained/full-search model"
        )
    output_root = arguments.output_directory
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "arguments": vars(arguments),
        "resolved_device": device,
        "profile": {
            "construction": asdict(profile.construction),
            "improvement_training": asdict(profile.improvement_training),
            "improvement": asdict(profile.improvement),
            "final_rollouts": profile.final_rollouts,
            "beam_width": profile.beam_width,
            "beam_expansion": profile.beam_expansion,
        },
        "scientific_contract": {
            "idle_action": 0,
            "dimension_source": "each instance (instance ID or JSON n/H fields)",
            "S_semantics": "number_of_visitable_states_excluding_idle",
            "action_count_rule": "instance.n = S; total actions = S + 1; action 0 is Idle",
            "gurobi_role": "benchmark_only",
            "main_method_is_rl_only": True,
        },
        "instances": [],
    }
    manifest["arguments"] = {
        key: (
            [str(item) if isinstance(item, Path) else item for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in manifest["arguments"].items()
    }

    benchmarks: list[ProfessorBenchmark] = []
    seen_instance_ids: set[str] = set()
    for requested_id in arguments.instances:
        spec = parse_professor_instance_id(requested_id)
        benchmark = load_professor_benchmark(
            arguments.data_directory / f"CIPP-{spec.party}.xls",
            instance_id=spec.instance_id,
            objective_variant=arguments.objective_variant,
            budget_mode=arguments.budget_mode,
        )
        if benchmark.instance.instance_id in seen_instance_ids:
            raise ValueError(f"Duplicate instance: {benchmark.instance.instance_id}")
        seen_instance_ids.add(benchmark.instance.instance_id)
        benchmarks.append(benchmark)

    for instance_file in arguments.instance_files:
        benchmark = benchmark_from_instance(load_instance(instance_file))
        if benchmark.instance.instance_id in seen_instance_ids:
            raise ValueError(f"Duplicate instance: {benchmark.instance.instance_id}")
        seen_instance_ids.add(benchmark.instance.instance_id)
        benchmarks.append(benchmark)

    for benchmark in benchmarks:
        print(
            f"[instance:start] instance={benchmark.instance.instance_id} "
            f"locations={benchmark.visit_location_count} "
            f"actions={benchmark.action_count} horizon={benchmark.instance.H}",
            flush=True,
        )
        instance_directory = output_root / benchmark.instance.instance_id
        instance_directory.mkdir(parents=True, exist_ok=True)
        manifest["instances"].append(
            {
                "instance_id": benchmark.instance.instance_id,
                "benchmark_state_count_excluding_idle": benchmark.state_count,
                "visit_location_count": benchmark.visit_location_count,
                "action_count_including_idle": benchmark.action_count,
                "idle_action": benchmark.idle_action,
                "horizon": benchmark.instance.H,
                "table5_reference": asdict(benchmark.table5_reference) if benchmark.table5_reference else None,
            }
        )
        greedy = _greedy_evaluation(benchmark)
        rows.append(_row(benchmark, greedy, seed=arguments.seed))
        rl_candidates: list[PolicyEvaluation] = []

        construction_agents: dict[str, AdvancedPPOAgent] = {}
        requested_construction = {
            "hacipp" if method == "hacipp_rl_improve" else method
            for method in arguments.methods
        }
        for method in CONSTRUCTION_METHODS:
            if method not in requested_construction:
                continue
            total_training_episodes = (
                profile.construction.updates
                * _method_configs(
                    method,
                    benchmark.instance.q,
                    profile,
                    arguments.active_search_mode,
                )[2].episodes_per_update
            )
            print(
                f"[method:start] phase=construction "
                f"instance={benchmark.instance.instance_id} method={method} "
                f"training_episodes={total_training_episodes}",
                flush=True,
            )
            method_directory = instance_directory / method / f"seed_{arguments.seed}"
            agent = _train_or_load(
                method=method,
                benchmark=benchmark,
                profile=profile,
                output_directory=method_directory,
                seed=arguments.seed,
                device=device,
                resume=arguments.resume,
                benchmark_only=arguments.benchmark_only,
                active_search_mode=arguments.active_search_mode,
                initial_checkpoint=arguments.initial_checkpoint,
                log_every_episodes=arguments.log_every_episodes,
            )
            construction_agents[method] = agent
            deterministic = evaluate_construction_policy(
                agent,
                deterministic=True,
                seed=arguments.seed,
                method=f"{method}_greedy",
            )
            best_of_k = evaluate_best_of_k(
                agent,
                rollouts=profile.final_rollouts,
                seed=arguments.seed + 2_000_000,
                method=f"{method}_best_of_{profile.final_rollouts}",
            )
            beam = policy_beam_search(
                agent,
                beam_width=profile.beam_width,
                expansion_width=profile.beam_expansion,
                method=f"{method}_sgbs_{profile.beam_width}",
            )
            rows.extend(
                _row(benchmark, result, seed=arguments.seed)
                for result in (deterministic, best_of_k, beam)
            )
            rl_candidates.extend((deterministic, best_of_k, beam))
            print(
                f"[method:result] instance={benchmark.instance.instance_id} "
                f"method={method} greedy={deterministic.objective:.3f} "
                f"best_of_k={best_of_k.objective:.3f} sgbs={beam.objective:.3f}",
                flush=True,
            )

        if "hacipp_rl_improve" in arguments.methods:
            construction_agent = construction_agents["hacipp"]
            start_generation_started = time.perf_counter()
            starts = [
                evaluate_best_of_k(
                    construction_agent,
                    rollouts=profile.final_rollouts,
                    seed=arguments.seed + 3_000_000,
                    method="hacipp_start_best_of_k",
                ),
                policy_beam_search(
                    construction_agent,
                    beam_width=profile.beam_width,
                    expansion_width=profile.beam_expansion,
                    method="hacipp_start_sgbs",
                ),
            ]
            start_generation_seconds = float(time.perf_counter() - start_generation_started)
            start = max(starts, key=lambda result: result.objective)
            improvement_directory = instance_directory / "hacipp_rl_improve" / f"seed_{arguments.seed}"
            improvement_path = improvement_directory / "checkpoint_best.pt"
            if (arguments.resume or arguments.benchmark_only) and improvement_path.exists():
                print(
                    f"[checkpoint:load] phase=improvement "
                    f"instance={benchmark.instance.instance_id} "
                    f"method=hacipp_rl_improve checkpoint={improvement_path}",
                    flush=True,
                )
                improvement_agent, _ = ImprovementPPOAgent.load(
                    improvement_path, instance=benchmark.instance, device=device
                )
            elif arguments.benchmark_only:
                raise FileNotFoundError(
                    f"--benchmark-only requires checkpoint: {improvement_path}"
                )
            else:
                print(
                    f"[method:start] phase=improvement "
                    f"instance={benchmark.instance.instance_id} "
                    f"method=hacipp_rl_improve "
                    f"training_episodes="
                    f"{profile.improvement_training.updates * profile.improvement_training.episodes_per_update}",
                    flush=True,
                )
                improvement_agent = ImprovementPPOAgent(
                    benchmark.instance,
                    profile.improvement,
                    seed=arguments.seed,
                    device=device,
                )
                improvement_path = train_improvement_agent(
                    improvement_agent,
                    construction_agent,
                    config=profile.improvement_training,
                    output_directory=improvement_directory,
                    seed=arguments.seed,
                    log_every_episodes=arguments.log_every_episodes,
                )
                improvement_agent, _ = ImprovementPPOAgent.load(
                    improvement_path, instance=benchmark.instance, device=device
                )
            improved_core = evaluate_rl_improvement(
                improvement_agent,
                start,
                seed=arguments.seed + 4_000_000,
                method=f"hacipp_rl_improve_{profile.improvement.max_moves}",
            )
            improved = replace(
                improved_core,
                runtime_seconds=start_generation_seconds + improved_core.runtime_seconds,
                start_generation_seconds=start_generation_seconds,
                improvement_seconds=improved_core.runtime_seconds,
            )
            rows.append(_row(benchmark, improved, seed=arguments.seed))
            rl_candidates.append(improved)
            print(
                f"[method:result] instance={benchmark.instance.instance_id} "
                f"method=hacipp_rl_improve start={start.objective:.3f} "
                f"improved={improved.objective:.3f} "
                f"start_generation_seconds={start_generation_seconds:.3f} "
                f"improvement_seconds={improved.improvement_seconds:.3f} "
                f"total_runtime_seconds={improved.runtime_seconds:.3f}",
                flush=True,
            )

        if arguments.run_gurobi:
            gurobi = solve_cipp_gurobi(
                benchmark.instance,
                time_limit_seconds=arguments.gurobi_time_limit,
                output_directory=instance_directory / "gurobi",
                verbose=True,
            )
            evaluation = evaluate_itinerary(benchmark.instance, gurobi.itinerary)
            total = max(float(np.sum(evaluation.visit_counts)), 1.0)
            gurobi_evaluation = PolicyEvaluation(
                method=f"gurobi_{gurobi.status.lower()}",
                objective=gurobi.objective,
                itinerary=gurobi.itinerary,
                feasible=gurobi.feasible,
                runtime_seconds=gurobi.runtime_seconds,
                idle_days=evaluation.idle_days,
                unique_locations=int(np.count_nonzero(evaluation.visit_counts)),
                visit_hhi=float(np.sum((evaluation.visit_counts / total) ** 2)),
            )
            rows.append(_row(benchmark, gurobi_evaluation, seed=arguments.seed))
            exact_value = float(gurobi.objective)
            exact_status = str(gurobi.status)
            for row in rows:
                if row["instance_id"] != benchmark.instance.instance_id:
                    continue
                row["measured_exact_objective"] = exact_value
                row["measured_exact_status"] = exact_status
                row["gap_to_measured_exact_percent"] = (
                    100.0 * (exact_value - float(row["objective"])) / abs(exact_value)
                    if exact_value != 0.0
                    else None
                )

        if arguments.run_gurobi_warm_start:
            if not rl_candidates:
                raise ValueError(
                    "Gurobi warm start requires at least one trained RL method"
                )
            warm_start = max(rl_candidates, key=lambda result: result.objective)
            gurobi = solve_cipp_gurobi(
                benchmark.instance,
                time_limit_seconds=arguments.gurobi_time_limit,
                output_directory=instance_directory / "gurobi_rl_warm_start",
                verbose=True,
                mip_start_itinerary=warm_start.itinerary,
            )
            evaluation = evaluate_itinerary(benchmark.instance, gurobi.itinerary)
            total = max(float(np.sum(evaluation.visit_counts)), 1.0)
            hybrid_evaluation = PolicyEvaluation(
                method=f"rl_warm_start_gurobi_{gurobi.status.lower()}",
                objective=gurobi.objective,
                itinerary=gurobi.itinerary,
                feasible=gurobi.feasible,
                runtime_seconds=gurobi.runtime_seconds,
                idle_days=evaluation.idle_days,
                unique_locations=int(np.count_nonzero(evaluation.visit_counts)),
                visit_hhi=float(np.sum((evaluation.visit_counts / total) ** 2)),
            )
            rows.append(
                _row(
                    benchmark,
                    hybrid_evaluation,
                    seed=arguments.seed,
                    source="measured_hybrid",
                )
            )

        _write_leaderboard(rows, output_root)
        (output_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"[suite:done] completed={len(rows)} rows "
        f"output={output_root / 'leaderboard.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
