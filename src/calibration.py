"""
Threshold calibration for B1 and B2 methods.

Calibrates threshold pairs (theta_reject, theta_verify) to match a target
harmful action execution rate (HAER) using disjoint calibration seeds.

HAER = harmful actions ultimately executed / all harmful actions proposed

Uses the same threshold grid for B1 and B2.
Tie-breaking is deterministic and based only on calibration quantities.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.population import generate_population, ALPHA0, BETA0
from src.trust_model import trust_mean, trust_variance, lcb_score
from src.authorization import authorize
from src.verify_sim import verify_action


def calibrate_thresholds(
    method: str,
    target_haer: float,
    calibration_seeds: List[int],
    theta_grid: Optional[List[float]] = None,
    z: float = 1.0,
    p_catch: float = 0.8,
) -> Tuple[float, float]:
    """
    Calibrate threshold pair to match target HAER.

    Args:
        method: "B1" (point estimate) or "B2" (LCB)
        target_haer: Target harmful action execution rate [0, 1]
        calibration_seeds: List of RNG seeds for calibration populations
        theta_grid: Candidate threshold values (default: [0.1, 0.2, ..., 0.9])
        z: Confidence level parameter for B2 LCB (default 1.0)
        p_catch: VERIFY catch probability (default 0.8)

    Returns:
        Tuple of (theta_reject, theta_verify)

    Raises:
        ValueError: If inputs are invalid or target is unreachable
        TypeError: If types are incorrect
    """
    if not isinstance(method, str) or method not in ("B1", "B2"):
        raise ValueError(f"method must be 'B1' or 'B2', got {method}")

    if not isinstance(target_haer, (int, float)):
        raise TypeError(f"target_haer must be numeric, got {type(target_haer).__name__}")
    if target_haer < 0.0 or target_haer > 1.0:
        raise ValueError(f"target_haer must be in [0, 1], got {target_haer}")

    if not isinstance(calibration_seeds, (list, tuple)):
        raise TypeError(f"calibration_seeds must be list/tuple, got {type(calibration_seeds).__name__}")
    if not calibration_seeds:
        raise ValueError("calibration_seeds must not be empty")

    if theta_grid is None:
        theta_grid = [i / 10.0 for i in range(1, 10)]  # 0.1 to 0.9

    candidate_pairs = [
        (theta_reject, theta_verify)
        for theta_reject in theta_grid
        for theta_verify in theta_grid
        if theta_reject < theta_verify
    ]
    candidate_haers = _evaluate_threshold_pairs(
        method=method,
        threshold_pairs=candidate_pairs,
        calibration_seeds=calibration_seeds,
        z=z,
        p_catch=p_catch,
    )

    if not candidate_haers:
        raise ValueError("No valid threshold pairs found with theta_reject < theta_verify")

    # Find candidates matching or closest to target
    best_pair, best_haer = _select_threshold_pair(
        candidate_haers=candidate_haers,
        target_haer=target_haer,
    )

    return best_pair


def _evaluate_threshold_pair(
    method: str,
    theta_reject: float,
    theta_verify: float,
    calibration_seeds: List[int],
    z: float,
    p_catch: float,
) -> float:
    """
    Evaluate a threshold pair on calibration seeds.

    Returns the aggregate HAER across all calibration populations.

    Args:
        method: "B1" or "B2"
        theta_reject: Lower threshold
        theta_verify: Upper threshold
        calibration_seeds: List of RNG seeds
        z: Confidence level for B2
        p_catch: VERIFY catch probability

    Returns:
        HAER value [0, 1]
    """
    candidate_haers = _evaluate_threshold_pairs(
        method=method,
        threshold_pairs=[(theta_reject, theta_verify)],
        calibration_seeds=calibration_seeds,
        z=z,
        p_catch=p_catch,
    )
    return candidate_haers[(theta_reject, theta_verify)]


def _evaluate_threshold_pairs(
    method: str,
    threshold_pairs: List[Tuple[float, float]],
    calibration_seeds: List[int],
    z: float,
    p_catch: float,
) -> Dict[Tuple[float, float], float]:
    """Evaluate all candidate pairs using shared calibration realizations."""
    if method not in ("B1", "B2"):
        raise ValueError(f"Unknown method: {method}")
    if not calibration_seeds:
        raise ValueError("calibration_seeds must not be empty")
    if not isinstance(p_catch, (int, float)):
        raise TypeError(f"p_catch must be numeric, got {type(p_catch).__name__}")
    if p_catch < 0.0 or p_catch > 1.0:
        raise ValueError(f"p_catch must be in [0, 1], got {p_catch}")

    totals = {pair: [0, 0] for pair in threshold_pairs}
    for seed in calibration_seeds:
        population = generate_population(np.random.default_rng(seed))
        verify_rng = np.random.default_rng(seed + 10000)

        for agent in population:
            alpha = agent["alpha"]
            beta = agent["beta"]
            if method == "B1":
                score = trust_mean(alpha, beta)
            else:
                score = lcb_score(
                    trust_mean(alpha, beta),
                    trust_variance(alpha, beta),
                    z=z,
                    clip=True,
                )

            verify_draws = [
                verify_rng.uniform() if is_harmful else None
                for is_harmful in agent["measurement_actions"]
            ]
            for pair in threshold_pairs:
                theta_reject, theta_verify = pair
                decision = authorize(score, theta_reject, theta_verify)
                for index, is_harmful in enumerate(agent["measurement_actions"]):
                    if not is_harmful:
                        continue
                    totals[pair][0] += 1
                    if decision == "ALLOW" or (
                        decision == "VERIFY" and verify_draws[index] >= p_catch
                    ):
                        totals[pair][1] += 1

    return {
        pair: (executed / harmful if harmful else 0.0)
        for pair, (harmful, executed) in totals.items()
    }


def _select_threshold_pair(
    candidate_haers: Dict[Tuple[float, float], float],
    target_haer: float,
) -> Tuple[Tuple[float, float], float]:
    """
    Select the best threshold pair with deterministic tie-breaking.

    Selection logic:
    1. Find candidates within acceptable range of target
    2. Among those, select closest to target
    3. Tie-break deterministically: prefer smaller theta_reject,
       then smaller theta_verify

    Args:
        candidate_haers: Dict mapping (theta_reject, theta_verify) -> haer
        target_haer: Target HAER value

    Returns:
        Tuple of (best_pair, best_haer)
    """
    if not candidate_haers:
        raise ValueError("No candidate threshold pairs provided")

    # Find distance from target for all candidates
    candidates_with_distance = []
    for pair, haer in candidate_haers.items():
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(value, (int, float)) for value in pair)
            or pair[0] >= pair[1]
        ):
            raise ValueError(f"Invalid threshold pair: {pair}")
        candidates_with_distance.append((pair, haer, abs(haer - target_haer)))

    # Sort by distance (closest first)
    # Then by theta_reject (smaller first) for deterministic tie-breaking
    # Then by theta_verify (smaller first) for secondary tie-breaking
    candidates_with_distance.sort(
        key=lambda x: (x[2], x[0][0], x[0][1])
    )

    best_pair, best_haer, _ = candidates_with_distance[0]
    return best_pair, best_haer


def create_threshold_grid(
    num_points: int = 10,
    exclude_boundaries: bool = True,
) -> List[float]:
    """
    Create a uniform threshold grid.

    Args:
        num_points: Number of points (including boundaries if included)
        exclude_boundaries: If True, excludes 0.0 and 1.0

    Returns:
        Sorted list of threshold values
    """
    if exclude_boundaries:
        grid = [i / (num_points + 1) for i in range(1, num_points + 1)]
    else:
        # Include 0 and 1
        grid = [i / (num_points - 1) for i in range(num_points)]

    return sorted(grid)
