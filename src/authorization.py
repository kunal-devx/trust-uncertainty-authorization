"""
Authorization decision logic for graduated LLM-agent access control.

This module provides deterministic authorization decisions based on trust scores
and configurable thresholds.

Authorization outcomes:
- ALLOW: score >= theta_verify
- VERIFY: theta_reject <= score < theta_verify
- REJECT: score < theta_reject
"""


def authorize(
    score: float,
    theta_reject: float,
    theta_verify: float,
) -> str:
    """
    Make an authorization decision based on a trust score and thresholds.

    The decision is deterministic and depends only on the score and thresholds.
    No randomness, no external state, no side effects.

    Args:
        score: Trust score in [0, 1].
        theta_reject: Lower threshold in [0, 1]. Scores below this are rejected.
        theta_verify: Upper threshold in [0, 1]. Scores at or above this are allowed.
                      Must be strictly greater than theta_reject.

    Returns:
        One of: "ALLOW", "VERIFY", "REJECT"

    Raises:
        ValueError: If any input is invalid.
        TypeError: If any input has an invalid type.

    Authorization rules:
        - ALLOW if score >= theta_verify
        - VERIFY if theta_reject <= score < theta_verify
        - REJECT if score < theta_reject
    """
    # Validate input types
    if not isinstance(score, (int, float)):
        raise TypeError(f"score must be a number, got {type(score).__name__}")
    if not isinstance(theta_reject, (int, float)):
        raise TypeError(f"theta_reject must be a number, got {type(theta_reject).__name__}")
    if not isinstance(theta_verify, (int, float)):
        raise TypeError(f"theta_verify must be a number, got {type(theta_verify).__name__}")

    # Validate score range
    if score < 0.0 or score > 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")

    # Validate theta_reject range
    if theta_reject < 0.0 or theta_reject > 1.0:
        raise ValueError(f"theta_reject must be in [0, 1], got {theta_reject}")

    # Validate theta_verify range
    if theta_verify < 0.0 or theta_verify > 1.0:
        raise ValueError(f"theta_verify must be in [0, 1], got {theta_verify}")

    # Validate threshold ordering
    if theta_reject >= theta_verify:
        raise ValueError(
            f"theta_reject must be strictly less than theta_verify, "
            f"got theta_reject={theta_reject}, theta_verify={theta_verify}"
        )

    # Make decision
    if score >= theta_verify:
        return "ALLOW"
    elif score >= theta_reject:
        return "VERIFY"
    else:
        return "REJECT"
