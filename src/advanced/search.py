"""Policy-only decoding methods for trained CIPP RL agents."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.advanced.ppo import AdvancedPPOAgent
from src.advanced.training import PolicyEvaluation
from src.core import evaluate_itinerary
from src.envs import CIPPEnv


@dataclass(frozen=True, slots=True)
class _BeamNode:
    prefix: tuple[int, ...]
    log_probability: float
    score: float


def _hhi(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    shares = counts.astype(np.float64) / total
    return float(np.sum(shares**2))


def _complete_prefixes_batched(
    agent: AdvancedPPOAgent,
    prefixes: list[tuple[int, ...]],
) -> list[float]:
    environments = [
        agent.feature_builder.environment_from_prefix(list(prefix)) for prefix in prefixes
    ]
    while True:
        active = [index for index, environment in enumerate(environments) if not environment.done]
        if not active:
            break
        states = [agent.feature_builder.build(environments[index]) for index in active]
        actions, _, _ = agent.batch_actions(states, deterministic=True)
        for local_index, environment_index in enumerate(active):
            environments[environment_index].step(int(actions[local_index]))
    return [float(environment.cumulative_reward) for environment in environments]


def policy_beam_search(
    agent: AdvancedPPOAgent,
    *,
    beam_width: int = 32,
    expansion_width: int = 4,
    objective_weight: float = 1.0,
    value_weight: float = 0.25,
    simulation_weight: float = 3.0,
    simulation_frequency: int = 7,
    method: str = "policy_beam_search",
) -> PolicyEvaluation:
    """Simulation-guided beam search with batched policy evaluation.

    The search budget and scoring rule are unchanged, but all beam nodes at a
    depth and all rollout completions are evaluated in GPU-friendly batches.
    """

    if beam_width < 1 or expansion_width < 1:
        raise ValueError("beam and expansion widths must be positive")
    started = time.perf_counter()
    beam = [_BeamNode(prefix=(), log_probability=0.0, score=0.0)]
    for depth in range(agent.instance.H):
        environments = [
            agent.feature_builder.environment_from_prefix(list(node.prefix)) for node in beam
        ]
        states = [
            agent.feature_builder.build(environment) for environment in environments
        ]
        probabilities_batch, values_batch = agent.batch_probabilities_and_values(states)
        children: list[_BeamNode] = []
        for index, node in enumerate(beam):
            state = states[index]
            probabilities = probabilities_batch[index]
            value = float(values_batch[index])
            valid = np.flatnonzero(state.action_mask)
            ordered = valid[np.argsort(probabilities[valid])[::-1]][:expansion_width]
            for action in ordered:
                probability = max(float(probabilities[action]), 1e-12)
                child_environment = agent.feature_builder.environment_from_prefix(
                    [*node.prefix, int(action)]
                )
                log_probability = node.log_probability + float(np.log(probability))
                normalized_prefix = (
                    child_environment.cumulative_reward / agent.feature_builder.scale.objective
                )
                score = (
                    log_probability / (depth + 1)
                    + objective_weight * normalized_prefix
                    + value_weight * value
                )
                children.append(
                    _BeamNode(
                        prefix=(*node.prefix, int(action)),
                        log_probability=log_probability,
                        score=score,
                    )
                )
        if not children:
            raise RuntimeError("beam search lost every feasible prefix")
        preselected = sorted(children, key=lambda node: node.score, reverse=True)[
            : max(beam_width * 2, beam_width)
        ]
        should_simulate = (
            simulation_frequency > 0
            and (depth + 1) < agent.instance.H
            and ((depth + 1) % simulation_frequency == 0)
        )
        if should_simulate:
            completed_objectives = _complete_prefixes_batched(
                agent, [node.prefix for node in preselected]
            )
            scale = max(max(completed_objectives, default=1.0), 1.0)
            preselected = [
                _BeamNode(
                    prefix=node.prefix,
                    log_probability=node.log_probability,
                    score=node.score + simulation_weight * objective / scale,
                )
                for node, objective in zip(preselected, completed_objectives)
            ]
        beam = sorted(preselected, key=lambda node: node.score, reverse=True)[:beam_width]

    evaluated = [(node, evaluate_itinerary(agent.instance, node.prefix)) for node in beam]
    node, result = max(evaluated, key=lambda item: item[1].objective)
    return PolicyEvaluation(
        method=method,
        objective=float(result.objective),
        itinerary=node.prefix,
        feasible=bool(result.feasible),
        runtime_seconds=float(time.perf_counter() - started),
        idle_days=int(result.idle_days),
        unique_locations=int(np.count_nonzero(result.visit_counts)),
        visit_hhi=_hhi(result.visit_counts),
    )
