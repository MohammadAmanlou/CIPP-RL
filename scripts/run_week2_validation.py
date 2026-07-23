"""Run Week 2 validation over masked random episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.core import (
    evaluate_itinerary,
)

from src.envs import (
    CIPPEnv,
)

from src.utils import (
    generate_cipp_instance,
)


def run_validation(
    episodes: int,
    output_path: Path,
) -> dict[str, object]:
    """Run masked episodes and return a result summary."""

    if episodes < 1:
        raise ValueError(
            "episodes must be positive."
        )

    started = time.perf_counter()

    completed = 0
    violations = 0
    dead_ends = 0
    reward_mismatches = 0
    total_steps = 0

    objectives: list[float] = []

    for episode in range(
        episodes
    ):
        instance = generate_cipp_instance(
            n=4 + episode % 11,
            H=12 + episode % 19,
            alpha=5,
            q=6,
            w=2,
            seed=100_000 + episode,
        )

        env = CIPPEnv(
            instance,
            seed=200_000 + episode,
        )

        env.reset()

        episode_failed = False

        while not env.done:
            mask = (
                env.get_action_mask()
            )

            if not bool(
                mask.any()
            ):
                dead_ends += 1
                episode_failed = True
                break

            action = (
                env.sample_viable_action()
            )

            env.step(
                action
            )

            total_steps += 1

        if episode_failed:
            continue

        result = evaluate_itinerary(
            instance,
            env.itinerary,
        )

        if not result.feasible:
            violations += 1

        reward_matches = np.isclose(
            env.cumulative_reward,
            result.objective,
            rtol=1e-10,
            atol=1e-8,
        )

        if not reward_matches:
            reward_mismatches += 1

        objectives.append(
            result.objective
        )

        completed += 1

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    summary: dict[str, object] = {
        "episodes_requested": episodes,
        "episodes_completed": completed,
        "total_steps": total_steps,
        "constraint_violations": violations,
        "dead_end_states": dead_ends,
        "reward_mismatches": (
            reward_mismatches
        ),
        "mean_objective": (
            float(
                np.mean(objectives)
            )
            if objectives
            else None
        ),
        "min_objective": (
            float(
                np.min(objectives)
            )
            if objectives
            else None
        ),
        "max_objective": (
            float(
                np.max(objectives)
            )
            if objectives
            else None
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "passed": (
            completed == episodes
            and violations == 0
            and dead_ends == 0
            and reward_mismatches == 0
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the deterministic "
            "CIPP Week 2 environment."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/week2_validation.json"
        ),
    )

    arguments = (
        parser.parse_args()
    )

    summary = run_validation(
        episodes=arguments.episodes,
        output_path=arguments.output,
    )

    print(
        "Episodes requested:",
        summary["episodes_requested"],
    )

    print(
        "Episodes completed:",
        summary["episodes_completed"],
    )

    print(
        "Total steps:",
        summary["total_steps"],
    )

    print(
        "Constraint violations:",
        summary[
            "constraint_violations"
        ],
    )

    print(
        "Dead-end states:",
        summary["dead_end_states"],
    )

    print(
        "Reward mismatches:",
        summary["reward_mismatches"],
    )

    print(
        "Mean objective:",
        summary["mean_objective"],
    )

    print(
        "Elapsed seconds:",
        round(
            float(
                summary[
                    "elapsed_seconds"
                ]
            ),
            3,
        ),
    )

    print(
        "Passed:",
        summary["passed"],
    )

    print(
        "Saved to:",
        arguments.output,
    )


if __name__ == "__main__":
    main()