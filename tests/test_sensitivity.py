"""Fast tests for sensitivity configuration handling."""

import pytest

from src.sensitivity import (
    PRIMARY_CONFIG,
    SensitivityConfig,
    SensitivityResult,
    generate_sensitivity_grid,
    primary_configuration,
)


class TestPrimaryConfiguration:
    def test_primary_configuration(self):
        configuration = primary_configuration()
        assert configuration == PRIMARY_CONFIG
        assert configuration.prior == (1.0, 1.0)
        assert configuration.cold_start_threshold == 5
        assert configuration.z == 1.0
        assert configuration.p_catch == 0.8

    def test_cold_start_classification(self):
        configuration = primary_configuration()
        assert configuration.is_cold_start(0)
        assert configuration.is_cold_start(5)
        assert not configuration.is_cold_start(6)

    def test_result_storage(self):
        result = SensitivityResult(primary_configuration(), {"mean_difference": -0.1})
        assert result.metrics["mean_difference"] == -0.1


class TestGrid:
    def test_all_sensitivity_values(self):
        grid = generate_sensitivity_grid()
        assert len(grid) == 2 * 4 * 4 * 3
        assert {configuration.prior for configuration in grid} == {(1.0, 1.0), (2.0, 2.0)}
        assert {configuration.cold_start_threshold for configuration in grid} == {2, 3, 5, 8}
        assert {configuration.z for configuration in grid} == {0.5, 1.0, 1.5, 2.0}
        assert {configuration.p_catch for configuration in grid} == {0.6, 0.8, 0.95}

    def test_deterministic_and_duplicate_free(self):
        first = generate_sensitivity_grid()
        second = generate_sensitivity_grid()
        assert first == second
        assert len(first) == len(set(first))


class TestValidation:
    @pytest.mark.parametrize("field", ["alpha0", "beta0"])
    def test_invalid_prior(self, field):
        values = {"alpha0": 1.0, "beta0": 1.0}
        values[field] = 0
        with pytest.raises(ValueError, match="greater than 0"):
            SensitivityConfig(**values)

    def test_invalid_cold_start_threshold(self):
        with pytest.raises(ValueError, match="cold_start_threshold"):
            SensitivityConfig(cold_start_threshold=-1)

    def test_invalid_z(self):
        with pytest.raises(ValueError, match="z"):
            SensitivityConfig(z=-0.1)

    def test_invalid_p_catch(self):
        with pytest.raises(ValueError, match="p_catch"):
            SensitivityConfig(p_catch=1.1)

    def test_invalid_evidence_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            primary_configuration().is_cold_start(-1)
        with pytest.raises(TypeError, match="evidence_count"):
            primary_configuration().is_cold_start(1.5)  # type: ignore[arg-type]

    def test_invalid_result_configuration(self):
        with pytest.raises(TypeError, match="SensitivityConfig"):
            SensitivityResult("bad", {})  # type: ignore[arg-type]


def test_runner_uses_seed_level_descriptive_output():
    from experiments.run_sensitivity import run_condition, threshold_grid

    config = {
        "K": 1,
        "M": 2,
        "profiles": {"reliable": 0.05, "unreliable": 0.4, "malicious": 0.8},
        "evidence_counts": [0, 1, 2],
        "calibration_seeds": [101, 102],
        "evaluation_seeds": [201, 202],
        "target_haer_grid": [0.05],
    }
    condition = {
        "id": "tiny",
        "factor": "z",
        "alpha0": 1.0,
        "beta0": 1.0,
        "z": 1.0,
        "p_catch": 0.8,
        "threshold_step": 0.25,
    }
    payload = run_condition(config, condition)
    assert threshold_grid(0.025)[-1] == 0.975
    assert payload["statistical_unit"] == "seed"
    assert payload["inference_performed"] is False
    assert len(payload["rows"]) == 4
    assert {row["method"] for row in payload["rows"]} == {"B1", "B2"}
    assert all(row["theta_reject"] < row["theta_verify"] for row in payload["rows"])


def test_runner_output_is_deterministic_and_separate():
    from experiments.run_sensitivity import run_condition

    config = {
        "K": 1,
        "M": 2,
        "profiles": {"reliable": 0.05, "unreliable": 0.4, "malicious": 0.8},
        "evidence_counts": [0, 1, 2],
        "calibration_seeds": [101, 102],
        "evaluation_seeds": [201, 202],
        "target_haer_grid": [0.05],
    }
    condition = {
        "id": "tiny",
        "factor": "z",
        "alpha0": 1.0,
        "beta0": 1.0,
        "z": 1.0,
        "p_catch": 0.8,
        "threshold_step": 0.25,
    }
    assert run_condition(config, condition) == run_condition(config, condition)


def _copy_condition_files(destination):
    import json
    from pathlib import Path

    source = Path("results/sensitivity")
    destination.mkdir()
    for condition_id in (
        "baseline", "prior_beta_2_2", "z_0_5", "z_1_5",
        "p_catch_0_6", "p_catch_0_95", "threshold_step_0_025",
    ):
        payload = json.loads((source / f"{condition_id}.json").read_text())
        (destination / f"{condition_id}.json").write_text(json.dumps(payload))


def test_analyzer_ignores_its_own_analysis_json(tmp_path):
    import json
    from experiments.analyze_sensitivity import analyze

    directory = tmp_path / "sensitivity"
    _copy_condition_files(directory)
    (directory / "analysis.json").write_text(json.dumps({"not": "a condition"}))
    result = analyze(directory)
    assert result["condition_ids"] == [
        "baseline", "p_catch_0_6", "p_catch_0_95", "prior_beta_2_2",
        "threshold_step_0_025", "z_0_5", "z_1_5",
    ]


def test_analyzer_rejects_duplicate_condition_rows(tmp_path):
    import json
    from experiments.analyze_sensitivity import analyze

    directory = tmp_path / "sensitivity"
    _copy_condition_files(directory)
    path = directory / "baseline.json"
    payload = json.loads(path.read_text())
    payload["rows"][1] = payload["rows"][0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        analyze(directory)


def test_analyzer_reports_descriptive_b2_minus_b1_values_and_cold_start_label():
    from experiments.analyze_sensitivity import analyze

    result = analyze()
    baseline = next(condition for condition in result["conditions"] if condition["condition"]["id"] == "baseline")
    row = baseline["target_summary"][0]
    assert row["b2_minus_b1_ui"] == pytest.approx(row["b2_mean_ui"] - row["b1_mean_ui"])
    assert "derived from retained primary cell metrics".lower() in result["cold_start"]["note"].lower()
    assert result["overall_assessment"] == "MIXED / SENSITIVITY-DEPENDENT"
