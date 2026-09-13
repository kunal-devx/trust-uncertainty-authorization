"""Tiny invariant tests for decision decomposition."""

import copy
import json

from experiments.diagnose_decision_decomposition import (
    _empty_counts,
    _finalize_counts,
    build_diagnostic,
)


def test_count_conservation_and_metric_formulas():
    counts = _empty_counts()
    counts.update({
        "total_actions": 4,
        "benign_actions": 2,
        "harmful_actions": 2,
        "allow": 1,
        "verify": 2,
        "reject": 1,
        "benign_allow": 1,
        "benign_verify": 1,
        "benign_reject": 0,
        "harmful_allow": 0,
        "harmful_verify": 1,
        "harmful_reject": 1,
        "harmful_executed": 1,
        "harmful_blocked": 1,
        "benign_executed": 2,
        "benign_blocked": 0,
    })
    result = _finalize_counts(counts)
    assert result["allow"] + result["verify"] + result["reject"] == result["total_actions"]
    assert result["benign_allow"] + result["benign_verify"] + result["benign_reject"] == result["benign_actions"]
    assert result["harmful_allow"] + result["harmful_verify"] + result["harmful_reject"] == result["harmful_actions"]
    assert result["harmful_executed"] + result["harmful_blocked"] == result["harmful_actions"]
    assert result["ui"] == 0.5
    assert result["haer"] == 0.5


def tiny_inputs():
    config = {
        "K": 1,
        "M": 3,
        "profiles": {"reliable": 0.05, "malicious": 0.8},
        "evidence_counts": [0, 2],
        "alpha0": 1.0,
        "beta0": 1.0,
        "z": 1.0,
        "p_catch": 0.8,
        "evaluation_seeds": [101],
        "target_haer_grid": [0.05],
    }
    calibration = {
        "rows": [
            {"method": "B1", "target_haer": 0.05, "theta_reject": 0.2, "theta_verify": 0.8},
            {"method": "B2", "target_haer": 0.05, "theta_reject": 0.2, "theta_verify": 0.8},
        ]
    }
    evaluation = {
        "rows": [
            {"seed": 101, "target_haer": 0.05, "method": "B1"},
            {"seed": 101, "target_haer": 0.05, "method": "B2"},
        ]
    }
    return config, evaluation, calibration


def test_paired_realizations_and_trace_conservation():
    config, evaluation, calibration = tiny_inputs()
    payload = build_diagnostic(config, evaluation, calibration)
    assert payload["statistical_unit"] == "seed"
    assert payload["inference_performed"] is False
    assert len(payload["action_traces"]) == 8
    for row in payload["aggregate_seed_method_target"]:
        assert row["allow"] + row["verify"] + row["reject"] == row["total_actions"]
        assert row["benign_allow"] + row["benign_verify"] + row["benign_reject"] == row["benign_actions"]
        assert row["harmful_allow"] + row["harmful_verify"] + row["harmful_reject"] == row["harmful_actions"]
        assert row["harmful_executed"] + row["harmful_blocked"] == row["harmful_actions"]
    traces = payload["action_traces"]
    b1 = [trace for trace in traces if trace["method"] == "B1"]
    b2 = [trace for trace in traces if trace["method"] == "B2"]
    assert [(x["profile"], x["evidence_count"], x["is_harmful"]) for x in b1] == [
        (x["profile"], x["evidence_count"], x["is_harmful"]) for x in b2
    ]
    assert [x["verify_outcome"] for x in b1] == [x["verify_outcome"] for x in b2]


def test_repeated_build_is_deterministic_and_inputs_unchanged():
    config, evaluation, calibration = tiny_inputs()
    original_evaluation = copy.deepcopy(evaluation)
    first = build_diagnostic(config, evaluation, calibration)
    second = build_diagnostic(config, evaluation, calibration)
    assert first == second
    assert evaluation == original_evaluation
