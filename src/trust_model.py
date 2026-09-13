"""
Beta-Bernoulli trust model for graduated LLM-agent authorization.

This module implements a Bayesian trust estimation model based on the Beta-Bernoulli
conjugate prior, with posterior trust point estimates and uncertainty quantification.

Mathematical foundation:
- Prior: Beta(alpha0, beta0)
- Posterior: Beta(alpha0 + b, beta0 + h) where b is benign count, h is harmful count
- Trust point estimate: T = alpha / (alpha + beta)
- Posterior uncertainty: U = T(1-T) / (n_posterior + 1)
- Lower confidence bound: LCB = T - z * sqrt(U)
"""

import math
from typing import Tuple


def posterior_parameters(
    benign_count: int,
    harmful_count: int,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute posterior Beta parameters given observation counts.

    Args:
        benign_count: Number of benign observations (must be non-negative integer).
        harmful_count: Number of harmful observations (must be non-negative integer).
        alpha0: Prior alpha parameter (must be > 0).
        beta0: Prior beta parameter (must be > 0).

    Returns:
        Tuple of (alpha, beta) posterior parameters.

    Raises:
        TypeError: If counts are not integers.
        ValueError: If counts are negative or prior parameters are non-positive.
    """
    if not isinstance(benign_count, int):
        raise TypeError(f"benign_count must be an integer, got {type(benign_count).__name__}")
    if not isinstance(harmful_count, int):
        raise TypeError(f"harmful_count must be an integer, got {type(harmful_count).__name__}")

    if benign_count < 0:
        raise ValueError(f"benign_count must be non-negative, got {benign_count}")
    if harmful_count < 0:
        raise ValueError(f"harmful_count must be non-negative, got {harmful_count}")

    if alpha0 <= 0:
        raise ValueError(f"alpha0 must be > 0, got {alpha0}")
    if beta0 <= 0:
        raise ValueError(f"beta0 must be > 0, got {beta0}")

    alpha = alpha0 + benign_count
    beta = beta0 + harmful_count

    return alpha, beta


def trust_mean(alpha: float, beta: float) -> float:
    """
    Compute the trust point estimate (mean) of a Beta distribution.

    Args:
        alpha: Shape parameter alpha (must be > 0).
        beta: Shape parameter beta (must be > 0).

    Returns:
        Trust point estimate T = alpha / (alpha + beta).

    Raises:
        ValueError: If alpha or beta are non-positive.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")

    return alpha / (alpha + beta)


def trust_variance(alpha: float, beta: float) -> float:
    """
    Compute the posterior trust-estimation uncertainty (variance).

    Uses the formula:
        U = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))

    Which is equivalent to:
        U = T * (1 - T) / (n_posterior + 1)
        where T = alpha / (alpha + beta) and n_posterior = alpha + beta

    Args:
        alpha: Shape parameter alpha (must be > 0).
        beta: Shape parameter beta (must be > 0).

    Returns:
        Posterior uncertainty U.

    Raises:
        ValueError: If alpha or beta are non-positive.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")

    n_posterior = alpha + beta
    return (alpha * beta) / ((n_posterior ** 2) * (n_posterior + 1))


def lcb_score(
    trust: float,
    variance: float,
    z: float = 1.0,
    clip: bool = True,
) -> float:
    """
    Compute a lower confidence bound (LCB) score for trust with uncertainty penalty.

    The LCB score is:
        LCB = T - z * sqrt(U)

    where T is the trust point estimate and U is the posterior uncertainty.

    Args:
        trust: Trust point estimate T (typically in [0, 1]).
        variance: Posterior uncertainty U (must be non-negative).
        z: Standard deviation multiplier for uncertainty (must be >= 0).
        clip: If True, clip result to [0, 1]. If False, return raw value.

    Returns:
        LCB score, optionally clipped to [0, 1].

    Raises:
        ValueError: If variance is negative or z is negative.
    """
    if variance < 0:
        raise ValueError(f"variance must be non-negative, got {variance}")
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z}")

    lcb = trust - z * math.sqrt(variance)

    if clip:
        lcb = max(0.0, min(1.0, lcb))

    return lcb
