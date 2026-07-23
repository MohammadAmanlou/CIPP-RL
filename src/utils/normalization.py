"""Training-only feature normalization for fixed-size DQN observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ObservationNormalizer:
    """Frozen per-feature observation normalization.

    Statistics must be computed using training observations only.
    The resulting mean and scale remain frozen during validation
    and testing.
    """

    mean: FloatArray
    scale: FloatArray
    epsilon: float = 1e-8

    @classmethod
    def fit(
        cls,
        observations: ArrayLike,
        *,
        epsilon: float = 1e-8,
    ) -> "ObservationNormalizer":
        """Fit normalization statistics."""

        values = np.asarray(
            observations,
            dtype=np.float64,
        )

        if (
            values.ndim != 2
            or values.shape[0] < 1
        ):
            raise ValueError(
                "observations must have shape "
                "(number_of_samples, observation_dim)."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "observations must contain only finite values."
            )

        if (
            not np.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError(
                "epsilon must be positive and finite."
            )

        mean = values.mean(
            axis=0
        )

        standard_deviation = values.std(
            axis=0,
            ddof=0,
        )

        scale = np.where(
            standard_deviation < epsilon,
            1.0,
            standard_deviation,
        )

        mean = mean.astype(
            np.float64,
            copy=True,
        )

        scale = scale.astype(
            np.float64,
            copy=True,
        )

        mean.setflags(
            write=False
        )

        scale.setflags(
            write=False
        )

        return cls(
            mean=mean,
            scale=scale,
            epsilon=float(epsilon),
        )

    @property
    def observation_dim(self) -> int:
        """Return observation vector size."""

        return int(
            self.mean.size
        )

    def transform(
        self,
        observation: ArrayLike,
    ) -> FloatArray:
        """Normalize one observation or a batch of observations."""

        values = np.asarray(
            observation,
            dtype=np.float64,
        )

        if values.shape[-1:] != (
            self.observation_dim,
        ):
            raise ValueError(
                "last observation dimension must be "
                f"{self.observation_dim}; "
                f"got shape {values.shape}."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "observation must contain only finite values."
            )

        normalized = (
            values - self.mean
        ) / self.scale

        return normalized.astype(
            np.float64
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert statistics to checkpoint-compatible values."""

        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "ObservationNormalizer":
        """Restore statistics from a checkpoint dictionary."""

        mean = np.asarray(
            payload["mean"],
            dtype=np.float64,
        )

        scale = np.asarray(
            payload["scale"],
            dtype=np.float64,
        )

        epsilon = float(
            payload.get(
                "epsilon",
                1e-8,
            )
        )

        if (
            mean.ndim != 1
            or scale.shape != mean.shape
        ):
            raise ValueError(
                "normalizer mean and scale must be "
                "one-dimensional and equal-sized."
            )

        if (
            np.any(scale <= 0)
            or not np.all(
                np.isfinite(scale)
            )
        ):
            raise ValueError(
                "normalizer scale values must be "
                "positive and finite."
            )

        mean = mean.copy()
        scale = scale.copy()

        mean.setflags(
            write=False
        )

        scale.setflags(
            write=False
        )

        return cls(
            mean=mean,
            scale=scale,
            epsilon=epsilon,
        )