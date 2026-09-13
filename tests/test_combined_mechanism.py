"""Focused tests for the combined frozen-mechanism review."""

import hashlib
import json

from experiments.analyze_combined_mechanism import analyze


def test_combined_analysis_alignment_and_frozen_conclusion():
    result = analyze()
    assert result["targets"] if "targets" in result else True
    assert len(result["primary_target_comparison"]) == 7
    assert result["primary_conclusion"] == "INCONCLUSIVE"
    assert result["statistical_unit"] == "seed"
    assert result["inference_performed"] is False
    assert result["b2_ui_lower_targets"] == [0.01, 0.02, 0.03, 0.10]
    assert result["b2_ui_higher_targets"] == [0.05, 0.08, 0.15]


def test_math_and_threshold_recalibration_are_explicit():
    result = analyze()
    assert result["mathematics"]["pointwise_relationship"] == "B2 <= B1 at identical thresholds"
    assert "recalibration" in result["mathematics"]["recalibration_implication"]
    assert len(result["threshold_recalibration"]) == 7
    assert all(row["B1"]["theta_reject"] < row["B1"]["theta_verify"] for row in result["threshold_recalibration"])
    assert all(row["B2"]["theta_reject"] < row["B2"]["theta_verify"] for row in result["threshold_recalibration"])


def test_cold_start_and_b3_limits_are_not_fabricated():
    result = analyze()
    assert result["cold_start"]["evidence_counts"] == [0, 1, 2, 3, 5]
    assert any("B3" in limitation and "profile" in limitation for limitation in result["limitations"])
    assert result["B3_role"]["recommendation"] == "diagnostic/appendix secondary analysis"
    assert result["additional_production_diagnostic_required"] is False


def test_deterministic_and_inputs_unchanged():
    paths = [
        "results/primary/evaluation.json",
        "results/primary/calibration.json",
        "results/primary/analysis.json",
        "results/diagnostic/b3/b3_diagnostic.json",
        "results/diagnostic/b3/b3_analysis.json",
    ]
    before = {path: hashlib.sha256(open(path, "rb").read()).hexdigest() for path in paths}
    first = analyze()
    second = analyze()
    assert first == second
    after = {path: hashlib.sha256(open(path, "rb").read()).hexdigest() for path in paths}
    assert before == after
    serialized = json.dumps(first).lower()
    assert "pvalue" not in serialized
    assert "confidence_interval" not in serialized


def test_contribution_framing_avoids_forbidden_novelty_claims():
    result = analyze()
    forbidden = ("first", "novel uncertainty-aware authorization", "new trust model", "new bayesian trust", "new graduated authorization")
    for statement in result["contribution_statements"]:
        lowered = statement.lower()
        assert not any(term in lowered for term in forbidden)
