"""Paired seed-level statistical summaries for B1 versus B2.

All statistical inputs represent one aggregate metric per experimental seed.
Individual actions and agents must not be passed to these functions.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class StatsSummary:
    """Summary of paired seed-level unnecessary-intervention rates."""

    n_seeds: int
    mean_b1: float
    mean_b2: float
    mean_difference: float
    ci_lower: float
    ci_upper: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float


def _validate_paired_values(
    b1_values: Sequence[float],
    b2_values: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert paired numeric seed-level values without sorting."""
    if isinstance(b1_values, (str, bytes)) or isinstance(b2_values, (str, bytes)):
        raise TypeError("B1 and B2 values must be numeric sequences")

    try:
        b1 = np.asarray(b1_values, dtype=float)
        b2 = np.asarray(b2_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("B1 and B2 values must be numeric sequences") from error

    if b1.ndim != 1 or b2.ndim != 1:
        raise ValueError("B1 and B2 values must be one-dimensional sequences")
    if b1.size != b2.size:
        raise ValueError("B1 and B2 values must have equal length")
    if b1.size < 2:
        raise ValueError("At least 2 paired seeds are required")
    if not np.all(np.isfinite(b1)) or not np.all(np.isfinite(b2)):
        raise ValueError("B1 and B2 values must be finite and contain no NaN or inf")

    return b1, b2


def paired_differences(
    b1_values: Sequence[float],
    b2_values: Sequence[float],
) -> np.ndarray:
    """Return seed-wise paired differences ``B2_UI - B1_UI`` in input order."""
    b1, b2 = _validate_paired_values(b1_values, b2_values)
    return b2 - b1


def mean_paired_difference(
    b1_values: Sequence[float],
    b2_values: Sequence[float],
) -> float:
    """Return the unrounded mean of the paired B2-minus-B1 differences."""
    return float(np.mean(paired_differences(b1_values, b2_values)))


def paired_bootstrap_ci(
    b1_values: Sequence[float],
    b2_values: Sequence[float],
    rng: np.random.Generator,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Return a percentile bootstrap CI for the mean paired difference.

    Resampling is performed over paired seed-level differences, never over
    individual actions or agents. The caller must supply the RNG explicitly.
    """
    differences = paired_differences(b1_values, b2_values)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicitly supplied numpy.random.Generator")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise TypeError("n_bootstrap must be a positive integer")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if not isinstance(confidence_level, (int, float)):
        raise TypeError("confidence_level must be numeric")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    sample_indices = rng.integers(
        0,
        differences.size,
        size=(n_bootstrap, differences.size),
    )
    bootstrap_means = differences[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.percentile(bootstrap_means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lower), float(upper)


def wilcoxon_signed_rank(
    b1_values: Sequence[float],
    b2_values: Sequence[float],
) -> Tuple[float, float]:
    """Return the secondary paired Wilcoxon signed-rank statistic and p-value.

    The all-zero difference case is defined as statistic ``0.0`` and p-value
    ``1.0`` rather than delegating an avoidable exception to SciPy.
    """
    differences = paired_differences(b1_values, b2_values)
    if np.all(differences == 0.0):
        return 0.0, 1.0

    result = scipy_stats.wilcoxon(
        differences,
        alternative="two-sided",
        zero_method="wilcox",
    )
    return float(result.statistic), float(result.pvalue)


def summarize_paired_ui(
    b1_values: Sequence[float],
    b2_values: Sequence[float],
    rng: np.random.Generator,
    n_bootstrap: int = 10_000,
) -> StatsSummary:
    """Build the complete paired seed-level UI summary."""
    b1, b2 = _validate_paired_values(b1_values, b2_values)
    ci_lower, ci_upper = paired_bootstrap_ci(
        b1,
        b2,
        rng=rng,
        n_bootstrap=n_bootstrap,
    )
    statistic, pvalue = wilcoxon_signed_rank(b1, b2)
    return StatsSummary(
        n_seeds=int(b1.size),
        mean_b1=float(np.mean(b1)),
        mean_b2=float(np.mean(b2)),
        mean_difference=float(np.mean(b2 - b1)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        wilcoxon_statistic=statistic,
        wilcoxon_pvalue=pvalue,
    )


def interpret_ui_difference(ci_lower: float, ci_upper: float) -> str:
    """Interpret the direction of a UI difference CI without claiming equivalence."""
    if not isinstance(ci_lower, (int, float)) or not isinstance(ci_upper, (int, float)):
        raise TypeError("CI bounds must be numeric")
    if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        raise ValueError("CI bounds must be finite")
    if ci_lower > ci_upper:
        raise ValueError("ci_lower must be less than or equal to ci_upper")
    if ci_upper < 0:
        return "B2 lower UI"
    if ci_lower > 0:
        return "B1 lower UI"
    return "No statistically significant directional difference detected"
