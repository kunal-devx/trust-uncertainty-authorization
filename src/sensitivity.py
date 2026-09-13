"""Validated configuration and result containers for sensitivity analysis.

This module defines sensitivity dimensions but does not run simulations or
perform inferential testing.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import math


@dataclass(frozen=True)
class SensitivityConfig:
    """One validated sensitivity-analysis configuration."""

    alpha0: float = 1.0
    beta0: float = 1.0
    cold_start_threshold: int = 5
    z: float = 1.0
    p_catch: float = 0.80

    def __post_init__(self) -> None:
        if not isinstance(self.alpha0, (int, float)) or not math.isfinite(self.alpha0) or self.alpha0 <= 0:
            raise ValueError("alpha0 must be greater than 0")
        if not isinstance(self.beta0, (int, float)) or not math.isfinite(self.beta0) or self.beta0 <= 0:
            raise ValueError("beta0 must be greater than 0")
        if (
            isinstance(self.cold_start_threshold, bool)
            or not isinstance(self.cold_start_threshold, int)
            or self.cold_start_threshold < 0
        ):
            raise ValueError("cold_start_threshold must be a non-negative integer")
        if not isinstance(self.z, (int, float)) or not math.isfinite(self.z) or self.z < 0:
            raise ValueError("z must be greater than or equal to 0")
        if (
            not isinstance(self.p_catch, (int, float))
            or not math.isfinite(self.p_catch)
            or not 0.0 <= self.p_catch <= 1.0
        ):
            raise ValueError("p_catch must be in [0, 1]")

    @property
    def prior(self) -> Tuple[float, float]:
        """Return the Beta prior parameters."""
        return float(self.alpha0), float(self.beta0)

    def is_cold_start(self, evidence_count: int) -> bool:
        """Return whether an evidence count belongs to this cold-start region."""
        if isinstance(evidence_count, bool) or not isinstance(evidence_count, int):
            raise TypeError("evidence_count must be a non-negative integer")
        if evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")
        return evidence_count <= self.cold_start_threshold


PRIMARY_CONFIG = SensitivityConfig()


def primary_configuration() -> SensitivityConfig:
    """Return the frozen primary sensitivity configuration."""
    return PRIMARY_CONFIG


def generate_sensitivity_grid(
    priors: Sequence[Tuple[float, float]] = ((1.0, 1.0), (2.0, 2.0)),
    cold_start_thresholds: Sequence[int] = (2, 3, 5, 8),
    z_values: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    p_catch_values: Sequence[float] = (0.60, 0.80, 0.95),
) -> Tuple[SensitivityConfig, ...]:
    """Generate a deterministic, duplicate-free Cartesian sensitivity grid."""
    configurations = []
    seen = set()
    for (alpha0, beta0), cold_start, z, p_catch in product(
        priors,
        cold_start_thresholds,
        z_values,
        p_catch_values,
    ):
        configuration = SensitivityConfig(
            alpha0=alpha0,
            beta0=beta0,
            cold_start_threshold=cold_start,
            z=z,
            p_catch=p_catch,
        )
        if configuration not in seen:
            configurations.append(configuration)
            seen.add(configuration)
    return tuple(configurations)


@dataclass(frozen=True)
class SensitivityResult:
    """Structured storage for outputs associated with one configuration."""

    configuration: SensitivityConfig
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.configuration, SensitivityConfig):
            raise TypeError("configuration must be a SensitivityConfig")
        object.__setattr__(self, "metrics", dict(self.metrics))
