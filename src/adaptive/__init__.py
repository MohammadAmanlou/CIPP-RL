"""Adaptive CIPP components for Method C."""

from .context import (
    AdaptiveCIPPEnv,
    AdaptiveFeatureBuilder,
    ShockScenario,
    evaluate_adaptive_itinerary,
    rank_flip_scenario,
    rank_flip_scenario_at_day,
    replay_prefix,
    sample_training_scenario,
)
from .gurobi import (
    solve_adaptive_gurobi,
    solve_two_stage_adaptive_gurobi,
    solve_clairvoyant_adaptive_gurobi,
)
from .training import (
    AdaptiveTrainingConfig,
    residual_best_of_k,
    residual_greedy,
    build_anticipatory_prefix,
    anticipatory_then_residual_best_of_k,
    anticipatory_then_greedy,
    train_method_c,
)

__all__ = [
    "AdaptiveCIPPEnv",
    "AdaptiveFeatureBuilder",
    "ShockScenario",
    "evaluate_adaptive_itinerary",
    "rank_flip_scenario",
    "rank_flip_scenario_at_day",
    "replay_prefix",
    "sample_training_scenario",
    "solve_adaptive_gurobi",
    "solve_two_stage_adaptive_gurobi",
    "solve_clairvoyant_adaptive_gurobi",
    "AdaptiveTrainingConfig",
    "residual_best_of_k",
    "residual_greedy",
    "build_anticipatory_prefix",
    "anticipatory_then_residual_best_of_k",
    "anticipatory_then_greedy",
    "train_method_c",
]
