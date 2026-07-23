"""Run the Stage 1 deterministic CIPP demonstration."""

from __future__ import annotations

import json
from pathlib import Path

from src.core import (
    CIPPInstance,
    evaluate_itinerary,
)


def main() -> None:
    """Evaluate the fixed golden itinerary."""

    instance = CIPPInstance(
        n=3,
        H=5,
        rewards=[100.0, 80.0, 60.0],
        costs=[10.0, 20.0, 15.0],
        budget=60.0,
        alpha=3,
        idle_requirements=[1, 1, 1],
        q=3,
        w=2,
        temporal_weights=[
            1.0,
            0.8,
            1.2,
            1.0,
            1.1,
        ],
        gamma=0.1,
        p=1,
        instance_id="golden-3S-5P",
    )

    itinerary = [1, 2, 0, 1, 3]

    result = evaluate_itinerary(
        instance,
        itinerary,
    )

    print(f"Instance: {instance.instance_id}")
    print(f"Itinerary: {itinerary}")
    print(f"Objective: {result.objective}")
    print(f"Total cost: {result.total_cost}")
    print(f"Idle days: {result.idle_days}")
    print(
        "Visit counts: "
        f"{result.visit_counts.tolist()}"
    )
    print(f"Feasible: {result.feasible}")
    print(
        "Violations: "
        f"{list(result.violations)}"
    )

    output_path = Path(
        "results/stage1_demo.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "instance_id": instance.instance_id,
        "itinerary": itinerary,
        **result.to_dict(),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()