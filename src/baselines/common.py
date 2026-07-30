"""Shared result structures for deterministic CIPP baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Evaluation summary produced by one construction policy run."""

    method: str
    instance_id: str
    itinerary: tuple[int, ...]
    objective: float
    total_cost: float
    idle_days: int
    unique_locations: int
    feasible: bool
    violations: tuple[str, ...]
    runtime_seconds: float
    rollouts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to JSON-compatible values."""

        return {
            "method": self.method,
            "instance_id": self.instance_id,
            "itinerary": list(self.itinerary),
            "objective": self.objective,
            "total_cost": self.total_cost,
            "idle_days": self.idle_days,
            "unique_locations": self.unique_locations,
            "feasible": self.feasible,
            "violations": list(self.violations),
            "runtime_seconds": self.runtime_seconds,
            "rollouts": self.rollouts,
            "metadata": self.metadata,
        }