"""Fast synthetic tests for descriptive operating curves."""

import pytest

from src.curves import (
    DominanceSummary,
    OperatingPoint,
    build_curve,
    build_curve_from_calibrated_results,
    compare_ui_dominance,
    match_by_achieved_haer,
    match_by_target_haer,
)


def point(method, target, achieved, ui):
    return {
        "method": method,
        "target_haer": target,
        "theta_reject": 0.2,
        "theta_verify": 0.8,
        "achieved_haer": achieved,
        "unnecessary_intervention": ui,
        "false_verify": ui / 2,
        "false_reject": ui / 2,
        "legitimate_action_completion": 1.0 - ui,
    }


class TestCurveConstruction:
    def test_required_fields_and_method_labels(self):
        curve = build_curve([point("B1", 0.05, 0.06, 0.3)], method="B1")
        assert isinstance(curve[0], OperatingPoint)
        assert set(curve[0].as_dict()) == {
            "method", "target_haer", "theta_reject", "theta_verify",
            "achieved_haer", "unnecessary_intervention", "false_verify",
            "false_reject", "legitimate_action_completion",
        }

    def test_target_and_achieved_are_distinct_and_sorted(self):
        curve = build_curve([
            point("B1", 0.10, 0.13, 0.2),
            point("B1", 0.01, 0.02, 0.4),
        ])
        assert [item.target_haer for item in curve] == [0.01, 0.10]
        assert curve[1].achieved_haer == 0.13

    def test_calibrated_results_alias(self):
        assert build_curve_from_calibrated_results([point("B2", 0.05, 0.07, 0.2)], "B2")[0].method == "B2"

    def test_curve_cannot_mix_methods(self):
        with pytest.raises(ValueError, match="cannot mix"):
            build_curve([point("B1", 0.05, 0.05, 0.2), point("B2", 0.05, 0.04, 0.1)])

    def test_empty_curve(self):
        with pytest.raises(ValueError, match="at least one"):
            build_curve([])

    def test_invalid_values_and_thresholds(self):
        bad = point("B1", 0.05, 0.05, 0.2)
        bad["achieved_haer"] = 1.2
        with pytest.raises(ValueError, match="achieved_haer"):
            build_curve([bad])
        bad_threshold = point("B1", 0.05, 0.05, 0.2)
        bad_threshold["theta_reject"] = 0.9
        with pytest.raises(ValueError, match="thresholds"):
            build_curve([bad_threshold])


class TestMatching:
    def test_match_by_target_preserves_actual_safety(self):
        b1 = build_curve([point("B1", 0.05, 0.06, 0.4)])
        b2 = build_curve([point("B2", 0.05, 0.04, 0.2)])
        matches = match_by_target_haer(b1, b2)
        assert len(matches) == 1
        assert matches[0].b1.achieved_haer == 0.06
        assert matches[0].b2.achieved_haer == 0.04
        assert matches[0].match_basis == "target_haer"

    def test_target_tolerance_and_no_false_exact_match(self):
        b1 = build_curve([point("B1", 0.05, 0.06, 0.4)])
        b2 = build_curve([point("B2", 0.051, 0.04, 0.2)])
        assert match_by_target_haer(b1, b2) == []
        assert len(match_by_target_haer(b1, b2, tolerance=0.002)) == 1

    def test_match_by_achieved_safety_exposes_tolerance(self):
        b1 = build_curve([point("B1", 0.05, 0.06, 0.4)])
        b2 = build_curve([point("B2", 0.08, 0.061, 0.2)])
        matches = match_by_achieved_haer(b1, b2, tolerance=0.002)
        assert len(matches) == 1
        assert matches[0].match_basis == "achieved_haer"

    def test_matching_requires_distinct_methods(self):
        with pytest.raises(ValueError, match="not B2"):
            match_by_target_haer([point("B1", 0.05, 0.05, 0.2)], [point("B1", 0.05, 0.05, 0.1)])


class TestDominance:
    def test_b2_dominance(self):
        b1 = build_curve([point("B1", 0.01, 0.01, 0.4), point("B1", 0.05, 0.05, 0.5)])
        b2 = build_curve([point("B2", 0.01, 0.01, 0.2), point("B2", 0.05, 0.05, 0.3)])
        result = compare_ui_dominance(match_by_target_haer(b1, b2))
        assert result == DominanceSummary("B2 dominates", 2, (-0.2, -0.2))

    def test_b1_dominance(self):
        b1 = build_curve([point("B1", 0.01, 0.01, 0.2)])
        b2 = build_curve([point("B2", 0.01, 0.01, 0.4)])
        assert compare_ui_dominance(match_by_target_haer(b1, b2)).classification == "B1 dominates"

    def test_mixed_outcomes(self):
        b1 = build_curve([point("B1", 0.01, 0.01, 0.2), point("B1", 0.05, 0.05, 0.4)])
        b2 = build_curve([point("B2", 0.01, 0.01, 0.3), point("B2", 0.05, 0.05, 0.3)])
        assert compare_ui_dominance(match_by_target_haer(b1, b2)).classification == "mixed/inconclusive"

    def test_insufficient_points(self):
        result = compare_ui_dominance([])
        assert result.classification == "insufficient comparable points"
        assert result.n_comparable == 0

    def test_invalid_matching_tolerance(self):
        with pytest.raises(ValueError, match="tolerance"):
            match_by_target_haer([point("B1", 0.01, 0.01, 0.2)], [point("B2", 0.01, 0.01, 0.1)], -0.1)
