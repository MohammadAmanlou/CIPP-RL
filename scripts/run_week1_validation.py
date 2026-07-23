"""Generate, save, load, and validate deterministic CIPP instances."""

from __future__ import annotations

import json
from pathlib import Path

from src.core import evaluate_itinerary

from src.utils import (
    generate_cipp_instance,
    load_instance,
    save_instance,
)


def main() -> None:
    """Run the complete Week 1 reproducibility check."""

    instance_path = Path(
        "data/instances/week1_seed42.json"
    )

    result_path = Path(
        "results/week1_validation.json"
    )

    instance = generate_cipp_instance(
        n=14,
        H=30,
        alpha=5,
        q=6,
        w=2,
        seed=42,
    )

    save_instance(
        instance,
        instance_path,
    )

    loaded_instance = load_instance(
        instance_path
    )

    all_idle_itinerary = (
        [0] * loaded_instance.H
    )

    evaluation = evaluate_itinerary(
        loaded_instance,
        all_idle_itinerary,
    )

    payload = {
        "instance_id": loaded_instance.instance_id,
        "instance_path": str(instance_path),
        "seed": 42,
        "n": loaded_instance.n,
        "H": loaded_instance.H,
        "evaluation": evaluation.to_dict(),
    }

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

        file.write("\n")

    print(
        f"Generated instance: "
        f"{loaded_instance.instance_id}"
    )

    print(
        f"Saved instance to: "
        f"{instance_path}"
    )

    print(
        f"All-idle feasible: "
        f"{evaluation.feasible}"
    )

    print(
        f"Objective: "
        f"{evaluation.objective}"
    )

    print(
        f"Violations: "
        f"{list(evaluation.violations)}"
    )

    print(
        f"Saved validation result to: "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()