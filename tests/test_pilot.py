"""Lightweight tests for the pilot execution pipeline."""

import json

import numpy as np
import pytest

from experiments.run_pilot import (
    _calibration_thresholds,
    _evaluate_method,
    _paired_verify_outcomes,
    _seed_realization,
    load_config,
    run_pilot,
    validate_config,
)


@pytest.fixture
def pilot_config():
    return load_config()


def test_config_loading_and_validation(pilot_config):
    assert pilot_config["K"] == 2
    assert pilot_config["M"] == 5
    assert pilot_config["p_catch"] == 0.80
    validate_config(pilot_config)


def test_config_rejects_overlapping_seed_sets(pilot_config):
    invalid = dict(pilot_config)
    invalid["evaluation_seeds"] = [pilot_config["calibration_seeds"][0]]
    with pytest.raises(ValueError, match="disjoint"):
        validate_config(invalid)


def test_population_size_and_measurement_count(pilot_config):
    population, _ = _seed_realization(pilot_config, pilot_config["evaluation_seeds"][0])
    expected_size = pilot_config["K"] * len(pilot_config["profiles"]) * len(pilot_config["evidence_counts"])
    assert len(population) == expected_size
    assert all(len(agent["measurement_actions"]) == pilot_config["M"] for agent in population)


def test_same_seed_reproduces_population_and_actions(pilot_config):
    first, first_verify_ss = _seed_realization(pilot_config, 2001)
    second, second_verify_ss = _seed_realization(pilot_config, 2001)
    assert first == second
    first_outcomes = _paired_verify_outcomes(first, np.random.default_rng(first_verify_ss), 0.80)
    second_outcomes = _paired_verify_outcomes(second, np.random.default_rng(second_verify_ss), 0.80)
    assert first_outcomes == second_outcomes


def test_b1_b2_share_paired_realization(pilot_config):
    population, verify_ss = _seed_realization(pilot_config, 2001)
    outcomes = _paired_verify_outcomes(population, np.random.default_rng(verify_ss), 0.80)
    thresholds = _calibration_thresholds(pilot_config)
    b1_records = _evaluate_method(population, outcomes, "B1", thresholds["B1"], pilot_config, 2001)
    b2_records = _evaluate_method(population, outcomes, "B2", thresholds["B2"], pilot_config, 2001)
    assert [(record.agent_id, record.is_harmful) for record in b1_records] == [
        (record.agent_id, record.is_harmful) for record in b2_records
    ]


def test_calibration_uses_realizations_not_regeneration(pilot_config, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("calibration regenerated a population")

    monkeypatch.setattr("src.calibration.generate_population", fail_if_called)
    thresholds = _calibration_thresholds(pilot_config)
    assert all(theta_reject < theta_verify for theta_reject, theta_verify in thresholds.values())


def test_pilot_output_schema_and_target_achieved_separation(pilot_config, tmp_path):
    output_path = tmp_path / "pilot.json"
    payload, written_path = run_pilot(output_path=output_path)
    assert written_path == output_path
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == payload
    assert len(payload["rows"]) == len(pilot_config["evaluation_seeds"]) * 2
    required = {
        "seed",
        "method",
        "theta_reject",
        "theta_verify",
        "target_haer",
        "achieved_haer",
        "unnecessary_intervention_rate",
        "false_verify_rate",
        "false_reject_rate",
        "legitimate_action_completion",
    }
    assert all(required <= set(row) for row in payload["rows"])
    assert all(row["target_haer"] == pilot_config["target_haer"] for row in payload["rows"])
    assert all(row["theta_reject"] < row["theta_verify"] for row in payload["rows"])
    assert payload["b3_status"] == "B3 not implemented; excluded from pilot"


def test_repeated_pilot_execution_is_identical(pilot_config, tmp_path):
    output_path = tmp_path / "pilot.json"
    first_payload, _ = run_pilot(output_path=output_path)
    first_bytes = output_path.read_bytes()
    second_payload, _ = run_pilot(output_path=output_path)
    assert second_payload == first_payload
    assert output_path.read_bytes() == first_bytes
