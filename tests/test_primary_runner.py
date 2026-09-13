"""Tiny-config tests for primary calibration/evaluation runners."""

import copy

import pytest

from experiments.run_calibration import calibrate_targets, load_config, validate_config
from experiments.run_primary import evaluate_primary, validate_cached_calibration
from experiments.run_pilot import _seed_realization


def tiny_config():
    config = load_config()
    config = copy.deepcopy(config)
    config["K"] = 1
    config["M"] = 2
    config["calibration_seeds"] = [101, 102]
    config["evaluation_seeds"] = [201, 202]
    config["target_haer_grid"] = [0.05, 0.10]
    config["threshold_grid"] = [0.2, 0.5, 0.8]
    return config


def test_primary_config_loading_and_validation():
    config = load_config()
    validate_config(config)
    assert config["K"] == 30
    assert config["M"] == 50
    assert len(config["calibration_seeds"]) == 10
    assert len(config["evaluation_seeds"]) == 20


def test_tiny_validation_and_seed_disjointness():
    config = tiny_config()
    validate_config(config, require_primary_counts=False)
    invalid = copy.deepcopy(config)
    invalid["evaluation_seeds"] = [config["calibration_seeds"][0]]
    with pytest.raises(ValueError, match="disjoint"):
        validate_config(invalid, require_primary_counts=False)


def test_seed_sequence_reproducibility_and_population_size():
    config = tiny_config()
    first, first_verify_ss = _seed_realization(config, 201)
    second, second_verify_ss = _seed_realization(config, 201)
    assert first == second
    assert first_verify_ss.generate_state(4).tolist() == second_verify_ss.generate_state(4).tolist()
    assert len(first) == config["K"] * len(config["profiles"]) * len(config["evidence_counts"])
    assert all(len(agent["measurement_actions"]) == config["M"] for agent in first)


def test_multiple_targets_and_calibration_evaluation_separation():
    config = tiny_config()
    calibration = calibrate_targets(config)
    assert calibration["target_haer_grid"] == config["target_haer_grid"]
    assert set(config["calibration_seeds"]).isdisjoint(config["evaluation_seeds"])
    assert len(calibration["rows"]) == len(config["target_haer_grid"]) * 2
    assert all(row["theta_reject"] < row["theta_verify"] for row in calibration["rows"])
    assert all(row["method"] in {"B1", "B2"} for row in calibration["rows"])
    assert all(row["target_haer"] in config["target_haer_grid"] for row in calibration["rows"])


def test_primary_evaluation_schema_and_cold_start_retention():
    config = tiny_config()
    calibration = calibrate_targets(config)
    payload = evaluate_primary(config, calibration)
    assert payload["statistical_unit"] == "seed"
    assert payload["inference_performed"] is False
    assert len(payload["rows"]) == len(config["evaluation_seeds"]) * 2 * len(config["target_haer_grid"])
    required = {
        "seed", "method", "target_haer", "calibration_achieved_haer",
        "evaluation_achieved_haer", "theta_reject", "theta_verify",
        "unnecessary_intervention_rate", "false_verify_rate",
        "false_reject_rate", "legitimate_action_completion", "cell_metrics",
    }
    assert all(required <= set(row) for row in payload["rows"])
    assert all(row["theta_reject"] < row["theta_verify"] for row in payload["rows"])
    assert any(
        row["target_haer"] != row["evaluation_achieved_haer"]
        for row in payload["rows"]
    )
    assert all(row["cold_start_evidence_counts"] == [0, 1, 2, 3, 5] for row in payload["rows"])
    assert all({cell["evidence_count"] for cell in row["cell_metrics"]} == set(config["evidence_counts"])
               for row in payload["rows"])


def test_calibration_does_not_use_evaluation_seed_data():
    config = tiny_config()
    calibration = calibrate_targets(config)
    assert set(calibration["calibration_seeds"]) == set(config["calibration_seeds"])
    assert set(calibration["calibration_seeds"]).isdisjoint(set(calibration["evaluation_seeds"]))
    assert all(row["method"] in ("B1", "B2") for row in calibration["rows"])


def test_matching_calibration_metadata_is_accepted():
    config = tiny_config()
    calibration = calibrate_targets(config)
    validate_cached_calibration(calibration, config)


def test_mismatched_z_is_rejected():
    config = tiny_config()
    calibration = calibrate_targets(config)
    calibration["calibration_metadata"]["z"] = 2.0
    with pytest.raises(ValueError, match="does not match"):
        validate_cached_calibration(calibration, config)


def test_mismatched_prior_is_rejected():
    config = tiny_config()
    calibration = calibrate_targets(config)
    calibration["calibration_metadata"]["alpha0"] = 2.0
    with pytest.raises(ValueError, match="does not match"):
        validate_cached_calibration(calibration, config)


def test_mismatched_target_grid_is_rejected():
    config = tiny_config()
    calibration = calibrate_targets(config)
    calibration["calibration_metadata"]["target_haer_grid"] = [0.5]
    with pytest.raises(ValueError, match="does not match"):
        validate_cached_calibration(calibration, config)


def test_mismatched_calibration_seeds_are_rejected():
    config = tiny_config()
    calibration = calibrate_targets(config)
    calibration["calibration_metadata"]["calibration_seeds"] = [999, 1000]
    with pytest.raises(ValueError, match="does not match"):
        validate_cached_calibration(calibration, config)


def test_missing_target_record_is_detected():
    config = tiny_config()
    calibration = calibrate_targets(config)
    calibration["rows"] = [
        row for row in calibration["rows"]
        if not (row["method"] == "B2" and row["target_haer"] == 0.10)
    ]
    with pytest.raises(ValueError, match="exactly one calibration row"):
        from experiments.run_primary import _calibration_row
        _calibration_row(calibration, "B2", 0.10)


def test_missing_method_record_is_detected():
    config = tiny_config()
    calibration = calibrate_targets(config)
    calibration["rows"] = [
        row for row in calibration["rows"]
        if row["method"] != "B1"
    ]
    with pytest.raises(ValueError, match="exactly one calibration row"):
        from experiments.run_primary import _calibration_row
        _calibration_row(calibration, "B1", 0.05)


def test_missing_calibration_metadata_is_rejected():
    config = tiny_config()
    calibration = calibrate_targets(config)
    del calibration["calibration_metadata"]
    with pytest.raises(ValueError, match="no complete configuration metadata"):
        validate_cached_calibration(calibration, config)
