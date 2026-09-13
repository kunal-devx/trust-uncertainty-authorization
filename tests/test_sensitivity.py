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
