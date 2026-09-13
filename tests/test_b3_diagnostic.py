"""Focused tests for the diagnostic-only B3 control."""

import json

import numpy as np
import pytest
import yaml

import experiments.run_b3_diagnostic as b3
from experiments.run_calibration import calibration_fingerprint, calibration_metadata
from experiments.run_b3_diagnostic import (
    _calibrate_b3,
    _evaluate_pair,
    _seed_realization,
    b3_score,
    validate_calibration_artifact,
    validate_evaluation_artifact,
    run_b3_diagnostic,
)


def tiny_config():
    return {
        "experiment_name": "b3_tiny",
        "base_seed": 999,
        "K": 1,
        "M": 3,
        "profiles": {"reliable": 0.05, "unreliable": 0.4, "malicious": 0.8},
        "evidence_counts": [0, 1, 2, 3, 5, 8, 12, 20, 40, 100],
        "alpha0": 1.0,
        "beta0": 1.0,
        "z": 1.0,
        "p_catch": 0.8,
        "calibration_seeds": list(range(101, 111)),
        "evaluation_seeds": list(range(201, 221)),
        "target_haer_grid": [0.05],
        "threshold_grid": [0.2, 0.5, 0.8],
    }


def test_v_reproducibility_one_per_agent_and_shared_actions():
    config = tiny_config()
    first = _seed_realization(config, 201)
    second = _seed_realization(config, 201)
    population, outcomes, values = first
    population2, outcomes2, values2 = second
    assert population == population2
    assert outcomes == outcomes2
    assert values == values2
    assert len(values) == len(population)
    assert all(len(agent["measurement_actions"]) == config["M"] for agent in population)


def test_v_is_independent_of_evidence_history():
    config = tiny_config()
    changed = dict(config)
    changed["alpha0"] = 2.0
    changed["beta0"] = 2.0
    _, _, first_values = _seed_realization(config, 201)
    _, _, second_values = _seed_realization(changed, 201)
    assert first_values == second_values


def test_b3_score_formula_and_clipping():
    agent = {"alpha": 3.0, "beta": 1.0}
    assert b3_score(agent, 0.25, 1.0) == np.clip(0.75 - np.sqrt(0.25), 0.0, 1.0)
    assert b3_score(agent, 1.0, 1.0) == 0.0
    assert 0.0 <= b3_score(agent, 0.0, 1.0) <= 1.0


def test_b3_calibration_is_independent_and_thresholds_ordered():
    selected, achieved = _calibrate_b3(tiny_config())
    assert set(selected) == {0.05}
    assert selected[0.05][0] < selected[0.05][1]
    assert 0.0 <= achieved[0.05] <= 1.0


def test_shared_population_and_verify_realization_for_b3_pair():
    config = tiny_config()
    population, outcomes, values = _seed_realization(config, 201)
    first = _evaluate_pair(population, outcomes, values, (0.2, 0.8), config)
    second = _evaluate_pair(population, outcomes, values, (0.2, 0.8), config)
    assert first == second


def valid_artifacts(config):
    metadata = calibration_metadata(config)
    calibration = {
        "calibration_metadata": metadata,
        "calibration_fingerprint": calibration_fingerprint(metadata),
        "rows": [
            {
                "method": method,
                "target_haer": target,
                "theta_reject": 0.2,
                "theta_verify": 0.8,
                "calibration_achieved_haer": target,
            }
            for target in config["target_haer_grid"]
            for method in ("B1", "B2")
        ],
    }
    evaluation = {
        key: config[key]
        for key in (
            "K", "M", "alpha0", "beta0", "z", "p_catch", "profiles",
            "evidence_counts", "evaluation_seeds", "target_haer_grid", "threshold_grid",
        )
    }
    evaluation["rows"] = [
        {
            "seed": seed,
            "method": method,
            "target_haer": target,
            "calibration_achieved_haer": target,
            "evaluation_achieved_haer": target,
            "theta_reject": 0.2,
            "theta_verify": 0.8,
            "unnecessary_intervention_rate": 0.0,
            "false_verify_rate": 0.0,
            "false_reject_rate": 0.0,
            "legitimate_action_completion": 1.0,
            "cell_metrics": [],
        }
        for seed in config["evaluation_seeds"]
        for target in config["target_haer_grid"]
        for method in ("B1", "B2")
    ]
    return evaluation, calibration


def test_valid_frozen_artifacts_pass():
    config = tiny_config()
    evaluation, calibration = valid_artifacts(config)
    validate_evaluation_artifact(evaluation, config)
    validate_calibration_artifact(calibration, config)


def test_evaluation_integrity_failures():
    config = tiny_config()
    evaluation, _ = valid_artifacts(config)
    cases = []
    cases.append(("missing row", lambda value: value["rows"].pop()))
    cases.append(("missing method", lambda value: value["rows"].pop(1)))
    cases.append(("missing target", lambda value: value["rows"].pop(2)))
    cases.append(("missing seed", lambda value: value["rows"].pop(0)))
    cases.append(("duplicate", lambda value: value["rows"].__setitem__(1, value["rows"][0])))
    cases.append(("metadata", lambda value: value.__setitem__("z", 2.0)))
    for label, mutate in cases:
        broken = json.loads(json.dumps(evaluation))
        mutate(broken)
        with pytest.raises(ValueError, match="evaluation"):
            validate_evaluation_artifact(broken, config)


def test_calibration_integrity_failures():
    config = tiny_config()
    _, calibration = valid_artifacts(config)
    cases = []
    cases.append(("missing metadata", lambda value: value.pop("calibration_metadata")))
    cases.append(("fingerprint", lambda value: value.__setitem__("calibration_fingerprint", "bad")))
    cases.append(("metadata", lambda value: value["calibration_metadata"].__setitem__("z", 2.0)))
    cases.append(("missing row", lambda value: value["rows"].pop()))
    cases.append(("method", lambda value: value["rows"][0].__setitem__("method", "B3")))
    cases.append(("ordering", lambda value: value["rows"][0].update(theta_reject=0.9, theta_verify=0.2)))
    for label, mutate in cases:
        broken = json.loads(json.dumps(calibration))
        mutate(broken)
        with pytest.raises(ValueError):
            validate_calibration_artifact(broken, config)


def test_evaluator_rejects_mismatched_sequence_lengths():
    config = tiny_config()
    population, outcomes, values = _seed_realization(config, 201)
    with pytest.raises(ValueError, match="equal length"):
        _evaluate_pair(population[:-1], outcomes, values, (0.2, 0.8), config)
    with pytest.raises(ValueError, match="equal length"):
        _evaluate_pair(population, outcomes, values[:-1], (0.2, 0.8), config)
    with pytest.raises(ValueError, match="lengths differ"):
        _evaluate_pair(population, [outcomes[0][:-1]] + outcomes[1:], values, (0.2, 0.8), config)


def test_output_schema_and_no_inference(tmp_path):
    config = tiny_config()
    config_path = tmp_path / "b3.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_path = tmp_path / "b3.json"
    summary_path = tmp_path / "b3.txt"
    evaluation, calibration = valid_artifacts(config)
    evaluation_path = tmp_path / "evaluation.json"
    calibration_path = tmp_path / "calibration.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(b3, "PRIMARY_EVALUATION", evaluation_path)
    monkeypatch.setattr(b3, "PRIMARY_CALIBRATION", calibration_path)
    payload = run_b3_diagnostic(config_path, output_path, summary_path)
    monkeypatch.undo()
    assert output_path.exists()
    assert summary_path.exists()
    assert payload["statistical_unit"] == "seed"
    assert payload["inference_performed"] is False
    assert len(payload["rows"]) == 20
    assert all(row["method"] == "B3" for row in payload["rows"])
    assert all(row["theta_reject"] < row["theta_verify"] for row in payload["rows"])
    assert payload["metadata"]["synthetic_uncertainty_distribution"] == {
        "name": "Beta", "alpha": 2.0, "beta": 5.0
    }
    assert len(payload["agent_v"]) == 20 * 30
    json.loads(output_path.read_text(encoding="utf-8"))
