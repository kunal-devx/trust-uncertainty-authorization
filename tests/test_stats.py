"""Fast tests for paired seed-level statistics."""

import numpy as np
import pytest

from src.stats import (
    StatsSummary,
    interpret_ui_difference,
    mean_paired_difference,
    paired_bootstrap_ci,
    paired_differences,
    summarize_paired_ui,
    wilcoxon_signed_rank,
)


class TestPairedStatistics:
    def test_normal_paired_data_preserves_order(self):
        assert np.allclose(
            paired_differences([0.2, 0.4, 0.6], [0.1, 0.5, 0.9]),
            np.array([-0.1, 0.1, 0.3]),
        )

    def test_mean_difference(self):
        assert mean_paired_difference([0.4, 0.5], [0.2, 0.7]) == pytest.approx(0.0)

    def test_b2_clearly_better(self):
        summary = summarize_paired_ui(
            [0.6, 0.6, 0.6, 0.6],
            [0.1, 0.1, 0.1, 0.1],
            rng=np.random.default_rng(1),
            n_bootstrap=500,
        )
        assert summary.mean_difference < 0
        assert summary.ci_upper < 0

    def test_b1_clearly_better(self):
        summary = summarize_paired_ui(
            [0.1, 0.1, 0.1, 0.1],
            [0.6, 0.6, 0.6, 0.6],
            rng=np.random.default_rng(1),
            n_bootstrap=500,
        )
        assert summary.mean_difference > 0
        assert summary.ci_lower > 0

    def test_identical_arrays_and_all_zero_wilcoxon(self):
        statistic, pvalue = wilcoxon_signed_rank([0.2, 0.2], [0.2, 0.2])
        assert (statistic, pvalue) == (0.0, 1.0)
        assert mean_paired_difference([0.2, 0.2], [0.2, 0.2]) == 0.0

    def test_mixed_differences(self):
        differences = paired_differences([0.2, 0.8, 0.4], [0.4, 0.6, 0.5])
        assert np.allclose(differences, np.array([0.2, -0.2, 0.1]))
        assert np.isfinite(wilcoxon_signed_rank([0.2, 0.8, 0.4], [0.4, 0.6, 0.5])[1])

    def test_exactly_two_seeds(self):
        lower, upper = paired_bootstrap_ci(
            [0.2, 0.4], [0.1, 0.3], np.random.default_rng(3), n_bootstrap=100
        )
        assert lower <= upper

    def test_summary_structure(self):
        summary = summarize_paired_ui(
            [0.2, 0.3], [0.1, 0.4], np.random.default_rng(4), n_bootstrap=100
        )
        assert isinstance(summary, StatsSummary)
        assert summary.n_seeds == 2


class TestInputValidation:
    @pytest.mark.parametrize("b1, b2", [([], []), ([0.1], [0.2])])
    def test_empty_or_too_short_input(self, b1, b2):
        with pytest.raises(ValueError, match="At least 2 paired seeds"):
            paired_differences(b1, b2)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_differences([0.1, 0.2], [0.1])

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_values(self, bad_value):
        with pytest.raises(ValueError, match="finite"):
            paired_differences([0.1, bad_value], [0.2, 0.3])

    def test_non_numeric_values(self):
        with pytest.raises(TypeError, match="numeric"):
            paired_differences([0.1, "bad"], [0.2, 0.3])

    def test_invalid_bootstrap_count(self):
        with pytest.raises(ValueError, match="positive integer"):
            paired_bootstrap_ci([0.1, 0.2], [0.2, 0.3], np.random.default_rng(1), 0)

    def test_bootstrap_requires_explicit_generator(self):
        with pytest.raises(TypeError, match="explicitly supplied"):
            paired_bootstrap_ci([0.1, 0.2], [0.2, 0.3], None)  # type: ignore[arg-type]

    def test_same_rng_seed_is_deterministic(self):
        args = ([0.1, 0.3, 0.4], [0.2, 0.2, 0.5])
        first = paired_bootstrap_ci(*args, np.random.default_rng(10), n_bootstrap=300)
        second = paired_bootstrap_ci(*args, np.random.default_rng(10), n_bootstrap=300)
        assert first == second

    def test_ci_order(self):
        lower, upper = paired_bootstrap_ci(
            [0.1, 0.2, 0.3], [0.2, 0.3, 0.4], np.random.default_rng(2), n_bootstrap=200
        )
        assert lower <= upper


class TestInterpretation:
    def test_b2_lower(self):
        assert interpret_ui_difference(-0.4, -0.1) == "B2 lower UI"

    def test_contains_zero(self):
        assert "No statistically significant" in interpret_ui_difference(-0.1, 0.1)

    def test_b1_lower(self):
        assert interpret_ui_difference(0.1, 0.4) == "B1 lower UI"

    def test_invalid_interpretation_bounds(self):
        with pytest.raises(ValueError, match="less than or equal"):
            interpret_ui_difference(0.2, 0.1)
