"""Descriptive safety-utility operating curves for calibrated methods.

This module stores discrete calibrated operating points. It does not run
experiments, infer significance, or replace achieved HAER with a target value.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REQUIRED_FIELDS = (
    "method",
    "target_haer",
    "theta_reject",
    "theta_verify",
    "achieved_haer",
    "unnecessary_intervention",
    "false_verify",
    "false_reject",
    "legitimate_action_completion",
)


@dataclass(frozen=True)
class OperatingPoint:
    """One calibrated threshold pair and its measured aggregate metrics."""

    method: str
    target_haer: float
    theta_reject: float
    theta_verify: float
    achieved_haer: float
    unnecessary_intervention: float
    false_verify: float
    false_reject: float
    legitimate_action_completion: float

    def as_dict(self) -> Dict[str, Any]:
        """Return the point using the public curve field names."""
        return {
            field: getattr(self, field)
            for field in REQUIRED_FIELDS
        }


@dataclass(frozen=True)
class MatchedOperatingPoint:
    """A B1/B2 pair matched by target or achieved safety."""

    b1: OperatingPoint
    b2: OperatingPoint
    match_basis: str
    achieved_haer_difference: float


@dataclass(frozen=True)
class DominanceSummary:
    """Descriptive UI comparison across comparable operating points."""

    classification: str
    n_comparable: int
    ui_differences: Tuple[float, ...]


def _as_point(result: OperatingPoint | Mapping[str, Any]) -> OperatingPoint:
    if isinstance(result, OperatingPoint):
        point = result
    elif isinstance(result, Mapping):
        values = dict(result)
        if "legitimate_action_completion" not in values and "legitimate_completion" in values:
            values["legitimate_action_completion"] = values.pop("legitimate_completion")
        missing = [field for field in REQUIRED_FIELDS if field not in values]
        if missing:
            raise ValueError(f"Missing operating-point fields: {', '.join(missing)}")
        point = OperatingPoint(**{field: values[field] for field in REQUIRED_FIELDS})
    else:
        raise TypeError("Operating points must be OperatingPoint instances or mappings")

    if point.method not in ("B1", "B2"):
        raise ValueError("method must be 'B1' or 'B2'")
    for name in (
        "target_haer",
        "theta_reject",
        "theta_verify",
        "achieved_haer",
        "unnecessary_intervention",
        "false_verify",
        "false_reject",
        "legitimate_action_completion",
    ):
        value = getattr(point, name)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite numeric value")
    if not 0.0 <= point.target_haer <= 1.0:
        raise ValueError("target_haer must be in [0, 1]")
    if not 0.0 <= point.achieved_haer <= 1.0:
        raise ValueError("achieved_haer must be in [0, 1]")
    if not 0.0 <= point.theta_reject < point.theta_verify <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= theta_reject < theta_verify <= 1")
    for name in (
        "unnecessary_intervention",
        "false_verify",
        "false_reject",
        "legitimate_action_completion",
    ):
        if not 0.0 <= getattr(point, name) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    return point


def build_curve(
    results: Iterable[OperatingPoint | Mapping[str, Any]],
    method: Optional[str] = None,
) -> List[OperatingPoint]:
    """Validate and sort discrete operating points by target HAER.

    ``achieved_haer`` is preserved exactly as supplied. A curve cannot mix B1
    and B2 points; pass ``method`` to assert the expected method explicitly.
    """
    points = [_as_point(result) for result in results]
    if not points:
        raise ValueError("at least one operating point is required")
    methods = {point.method for point in points}
    if len(methods) != 1:
        raise ValueError("a curve cannot mix B1 and B2 operating points")
    actual_method = next(iter(methods))
    if method is not None and method != actual_method:
        raise ValueError(f"curve contains {actual_method} points, not {method}")
    return sorted(points, key=lambda point: (point.target_haer, point.achieved_haer))


def build_curve_from_calibrated_results(
    results: Iterable[OperatingPoint | Mapping[str, Any]],
    method: Optional[str] = None,
) -> List[OperatingPoint]:
    """Alias expressing that inputs are already calibrated results."""
    return build_curve(results, method=method)


def match_by_target_haer(
    b1_curve: Sequence[OperatingPoint | Mapping[str, Any]],
    b2_curve: Sequence[OperatingPoint | Mapping[str, Any]],
    tolerance: float = 0.0,
) -> List[MatchedOperatingPoint]:
    """Match B1/B2 points by retained target HAER labels.

    A pair is included only when target labels differ by at most ``tolerance``.
    Achieved HAER values remain visible in each returned point.
    """
    if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    b1_points = build_curve(b1_curve, method="B1")
    b2_points = build_curve(b2_curve, method="B2")
    matches: List[MatchedOperatingPoint] = []
    used_b2 = set()
    for b1 in b1_points:
        candidates = [
            (abs(b2.target_haer - b1.target_haer), index, b2)
            for index, b2 in enumerate(b2_points)
            if index not in used_b2 and abs(b2.target_haer - b1.target_haer) <= tolerance
        ]
        if candidates:
            _, index, b2 = min(candidates, key=lambda item: (item[0], item[1]))
            used_b2.add(index)
            matches.append(
                MatchedOperatingPoint(
                    b1=b1,
                    b2=b2,
                    match_basis="target_haer",
                    achieved_haer_difference=b2.achieved_haer - b1.achieved_haer,
                )
            )
    return matches


def match_by_achieved_haer(
    b1_curve: Sequence[OperatingPoint | Mapping[str, Any]],
    b2_curve: Sequence[OperatingPoint | Mapping[str, Any]],
    tolerance: float,
) -> List[MatchedOperatingPoint]:
    """Match points whose achieved HAER values are within a stated tolerance."""
    if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    b1_points = build_curve(b1_curve, method="B1")
    b2_points = build_curve(b2_curve, method="B2")
    matches: List[MatchedOperatingPoint] = []
    used_b2 = set()
    for b1 in b1_points:
        candidates = [
            (abs(b2.achieved_haer - b1.achieved_haer), index, b2)
            for index, b2 in enumerate(b2_points)
            if index not in used_b2 and abs(b2.achieved_haer - b1.achieved_haer) <= tolerance
        ]
        if candidates:
            _, index, b2 = min(candidates, key=lambda item: (item[0], item[1]))
            used_b2.add(index)
            matches.append(
                MatchedOperatingPoint(
                    b1=b1,
                    b2=b2,
                    match_basis="achieved_haer",
                    achieved_haer_difference=b2.achieved_haer - b1.achieved_haer,
                )
            )
    return matches


def compare_ui_dominance(
    matches: Sequence[MatchedOperatingPoint],
) -> DominanceSummary:
    """Classify descriptive UI differences without statistical inference."""
    if not matches:
        return DominanceSummary("insufficient comparable points", 0, ())
    differences = tuple(
        match.b2.unnecessary_intervention - match.b1.unnecessary_intervention
        for match in matches
    )
    if all(difference < 0 for difference in differences):
        classification = "B2 dominates"
    elif all(difference > 0 for difference in differences):
        classification = "B1 dominates"
    else:
        classification = "mixed/inconclusive"
    return DominanceSummary(classification, len(differences), differences)
