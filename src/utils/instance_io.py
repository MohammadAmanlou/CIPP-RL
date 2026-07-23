"""JSON serialization helpers for CIPP instances."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.core import CIPPInstance


SCHEMA_VERSION = 1


def instance_to_dict(
    instance: CIPPInstance,
) -> dict[str, Any]:
    """Convert a CIPP instance to JSON-compatible values."""

    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance.instance_id,
        "n": instance.n,
        "H": instance.H,
        "rewards": instance.rewards.tolist(),
        "costs": instance.costs.tolist(),
        "budget": instance.budget,
        "alpha": instance.alpha,
        "idle_requirements": (
            instance.idle_requirements.tolist()
        ),
        "q": instance.q,
        "w": instance.w,
        "temporal_weights": (
            instance.temporal_weights.tolist()
        ),
        "gamma": instance.gamma,
        "p": instance.p,
    }


def instance_from_dict(
    data: Mapping[str, Any],
) -> CIPPInstance:
    """Construct and validate a CIPP instance from a mapping."""

    if not isinstance(data, Mapping):
        raise TypeError(
            "instance data must be a mapping."
        )

    schema_version = data.get(
        "schema_version",
        SCHEMA_VERSION,
    )

    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema_version}; "
            f"expected {SCHEMA_VERSION}."
        )

    required_fields = {
        "instance_id",
        "n",
        "H",
        "rewards",
        "costs",
        "budget",
        "alpha",
        "idle_requirements",
        "q",
        "w",
        "temporal_weights",
        "gamma",
        "p",
    }

    missing = sorted(
        required_fields.difference(
            data.keys()
        )
    )

    if missing:
        raise ValueError(
            "Missing required instance fields: "
            + ", ".join(missing)
        )

    return CIPPInstance(
        instance_id=data["instance_id"],
        n=data["n"],
        H=data["H"],
        rewards=data["rewards"],
        costs=data["costs"],
        budget=data["budget"],
        alpha=data["alpha"],
        idle_requirements=data[
            "idle_requirements"
        ],
        q=data["q"],
        w=data["w"],
        temporal_weights=data[
            "temporal_weights"
        ],
        gamma=data["gamma"],
        p=data["p"],
    )


def save_instance(
    instance: CIPPInstance,
    path: str | Path,
) -> Path:
    """Save an instance as readable deterministic JSON."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            instance_to_dict(instance),
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")

    return output_path


def load_instance(
    path: str | Path,
) -> CIPPInstance:
    """Load and validate a CIPP instance from JSON."""

    input_path = Path(path)

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return instance_from_dict(data)