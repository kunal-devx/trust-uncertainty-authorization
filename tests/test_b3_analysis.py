"""Focused tests for deterministic downstream B3 analysis."""

import hashlib
import json

from experiments.analyze_b3 import analyze


def test_b3_analysis_has_all_targets_and_methods():
    result = analyze()
    assert result["targets"] == [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    assert result["methods"] == ["B1", "B2", "B3"]
    assert len(result["target_comparison"]) == 7
    assert result["statistical_unit"] == "seed"
    assert result["inference_performed"] is False


def test_b3_differences_and_calibration_errors_are_arithmetic():
    result = analyze()
    for row in result["target_comparison"]:
        assert row["B3_minus_B1"]["ui"] == row["B3"]["mean_ui"] - row["B1"]["mean_ui"]
        assert row["B3_minus_B2"]["haer"] == row["B3"]["mean_achieved_haer"] - row["B2"]["mean_achieved_haer"]
    for row in result["threshold_geometry"]:
        for method in ("B1", "B2", "B3"):
            assert row[method]["theta_reject"] < row[method]["theta_verify"]
            assert row[method]["absolute_calibration_error"] >= 0.0


def test_v_summary_and_limitation():
    result = analyze()
    summary = result["v_distribution"]
    assert summary["n"] == 18000
    assert 0.0 <= summary["minimum"] <= summary["maximum"] <= 1.0
    assert abs(summary["mean"] - 2.0 / 7.0) < 0.01
    assert summary["formal_goodness_of_fit_test"] is False
    assert any("profile" in limitation and "evidence" in limitation for limitation in result["limitations"])


def test_analysis_is_deterministic_and_does_not_modify_primary_inputs():
    evaluation_path = "results/primary/evaluation.json"
    calibration_path = "results/primary/calibration.json"
    analysis_path = "results/primary/analysis.json"
    before = {
        path: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for path in (evaluation_path, calibration_path, analysis_path)
    }
    first = analyze()
    second = analyze()
    assert first == second
    after = {
        path: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for path in (evaluation_path, calibration_path, analysis_path)
    }
    assert before == after
    assert "pvalue" not in json.dumps(first).lower()
    assert "confidence_interval" not in json.dumps(first).lower()
