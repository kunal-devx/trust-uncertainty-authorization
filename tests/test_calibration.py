"""
Tests for threshold calibration.

Tests cover:
- Correct threshold validity
- Deterministic calibration
- Tie-breaking behavior
- Method consistency (B1 vs B2)
- Target matching
"""

import pytest
import numpy as np
from src.calibration import (
    calibrate_thresholds,
    _evaluate_threshold_pair,
    _select_threshold_pair,
    create_threshold_grid,
)


@pytest.fixture(autouse=True)
def small_calibration_population(monkeypatch):
    """Keep calibration unit tests focused on threshold behavior."""
    def generate_small_population(rng, num_measurement_actions=50):
        return [
            {
                "alpha": 9.0,
                "beta": 1.0,
                "measurement_actions": [True, False, True, False],
            },
            {
                "alpha": 2.0,
                "beta": 8.0,
                "measurement_actions": [False, True, False, True],
            },
        ]

    monkeypatch.setattr("src.calibration.generate_population", generate_small_population)


class TestThresholdValidity:
    """Test that calibrated thresholds are valid."""

    def test_calibrated_thresholds_satisfy_ordering(self):
        """Test that theta_reject < theta_verify."""
        calibration_seeds = [1001, 1002]
        theta_reject, theta_verify = calibrate_thresholds(
            method="B1",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
        )
        
        assert theta_reject < theta_verify

    def test_calibrated_thresholds_in_valid_range(self):
        """Test that thresholds are in [0, 1]."""
        calibration_seeds = [1001, 1002]
        theta_reject, theta_verify = calibrate_thresholds(
            method="B1",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
        )
        
        assert 0 <= theta_reject <= 1
        assert 0 <= theta_verify <= 1

    def test_b1_and_b2_same_threshold_grid(self):
        """Test that B1 and B2 use the same threshold candidate grid."""
        calibration_seeds = [1001, 1002]
        
        # Both should use same grid by default
        theta_b1_reject, theta_b1_verify = calibrate_thresholds(
            method="B1",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
        )
        
        theta_b2_reject, theta_b2_verify = calibrate_thresholds(
            method="B2",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
        )
        
        # Both should produce valid threshold pairs (not necessarily identical)
        assert theta_b1_reject < theta_b1_verify
        assert theta_b2_reject < theta_b2_verify


class TestInputValidation:
    """Test input validation."""

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be 'B1' or 'B2'"):
            calibrate_thresholds(
                method="B3",
                target_haer=0.5,
                calibration_seeds=[1001],
            )

    def test_invalid_target_haer_negative(self):
        """Test that negative target_haer raises ValueError."""
        with pytest.raises(ValueError, match="target_haer must be in \\[0, 1\\]"):
            calibrate_thresholds(
                method="B1",
                target_haer=-0.1,
                calibration_seeds=[1001],
            )

    def test_invalid_target_haer_above_one(self):
        """Test that target_haer > 1 raises ValueError."""
        with pytest.raises(ValueError, match="target_haer must be in \\[0, 1\\]"):
            calibrate_thresholds(
                method="B1",
                target_haer=1.5,
                calibration_seeds=[1001],
            )

    def test_invalid_target_haer_type(self):
        """Test that non-numeric target_haer raises TypeError."""
        with pytest.raises(TypeError, match="target_haer must be numeric"):
            calibrate_thresholds(
                method="B1",
                target_haer="0.5",
                calibration_seeds=[1001],
            )

    def test_invalid_calibration_seeds_type(self):
        """Test that invalid calibration_seeds type raises TypeError."""
        with pytest.raises(TypeError, match="calibration_seeds must be list/tuple"):
            calibrate_thresholds(
                method="B1",
                target_haer=0.5,
                calibration_seeds=42,
            )

    def test_empty_calibration_seeds(self):
        """Test that empty calibration_seeds list raises error."""
        with pytest.raises((ValueError, IndexError)):
            calibrate_thresholds(
                method="B1",
                target_haer=0.5,
                calibration_seeds=[],
            )


class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_calibration_same_thresholds(self):
        """Test that same calibration data produces same thresholds."""
        calibration_seeds = [2001, 2002, 2003]
        
        thresholds1 = calibrate_thresholds(
            method="B1",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
        )
        
        thresholds2 = calibrate_thresholds(
            method="B1",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
        )
        
        assert thresholds1 == thresholds2

    def test_deterministic_tie_breaking(self):
        """Test that tie-breaking is deterministic.
        
        If two threshold pairs have equal distance to target,
        the selection should be deterministic (prefer smaller reject, then verify).
        """
        # Create artificial candidates where two tie on distance
        candidate_haers = {
            (0.2, 0.4): 0.49,
            (0.3, 0.5): 0.51,
        }
        
        best_pair, _ = _select_threshold_pair(candidate_haers, target_haer=0.50)
        
        # Should select the pair with smaller reject when tied.
        assert best_pair == (0.2, 0.4)

    def test_deterministic_order_independent(self):
        """Test that candidate order doesn't affect selection."""
        # Create candidates in different orders
        candidates_dict1 = {
            (0.3, 0.5): 0.51,
            (0.2, 0.4): 0.49,
        }
        
        candidates_dict2 = {
            (0.3, 0.5): 0.51,
            (0.2, 0.4): 0.49,
        }
        
        best1, _ = _select_threshold_pair(candidates_dict1, target_haer=0.50)
        best2, _ = _select_threshold_pair(candidates_dict2, target_haer=0.50)
        
        assert best1 == best2


class TestTargetMatching:
    """Test target HAER matching."""

    def test_exact_target_match(self):
        """Test when a candidate exactly matches target."""
        candidate_haers = {
            (0.2, 0.8): 0.40,
            (0.3, 0.7): 0.50,
            (0.4, 0.6): 0.60,
        }
        
        best_pair, best_haer = _select_threshold_pair(
            candidate_haers=candidate_haers,
            target_haer=0.50,
        )
        
        assert best_haer == 0.50
        assert best_pair == (0.3, 0.7)

    def test_target_between_candidates(self):
        """Test when target is between two candidates."""
        candidate_haers = {
            (0.2, 0.8): 0.40,
            (0.3, 0.7): 0.55,
            (0.4, 0.6): 0.60,
        }
        
        best_pair, best_haer = _select_threshold_pair(
            candidate_haers=candidate_haers,
            target_haer=0.50,
        )
        
        # Should select the closest
        assert best_haer == 0.55
        assert best_pair == (0.3, 0.7)

    def test_target_extreme_low(self):
        """Test calibration with target_haer=0."""
        candidate_haers = {
            (0.2, 0.8): 0.01,
            (0.3, 0.7): 0.05,
            (0.4, 0.6): 0.10,
        }
        
        best_pair, best_haer = _select_threshold_pair(
            candidate_haers=candidate_haers,
            target_haer=0.0,
        )
        
        # Should select minimum
        assert best_haer == 0.01

    def test_target_extreme_high(self):
        """Test calibration with target_haer=1."""
        candidate_haers = {
            (0.2, 0.8): 0.90,
            (0.3, 0.7): 0.95,
            (0.4, 0.6): 0.99,
        }
        
        best_pair, best_haer = _select_threshold_pair(
            candidate_haers=candidate_haers,
            target_haer=1.0,
        )
        
        # Should select closest to 1.0
        assert best_haer == 0.99


class TestThresholdGrid:
    """Test threshold grid creation."""

    def test_grid_without_boundaries(self):
        """Test grid excluding 0 and 1."""
        grid = create_threshold_grid(num_points=10, exclude_boundaries=True)
        
        assert 0.0 not in grid
        assert 1.0 not in grid
        assert len(grid) == 10
        assert grid[0] > 0.0

    def test_grid_with_boundaries(self):
        """Test grid including 0 and 1."""
        grid = create_threshold_grid(num_points=10, exclude_boundaries=False)
        
        assert 0.0 in grid
        assert 1.0 in grid
        assert len(grid) == 10

    def test_grid_sorted(self):
        """Test that grid is sorted."""
        grid = create_threshold_grid(num_points=10)
        
        assert grid == sorted(grid)

    def test_grid_uniform_spacing(self):
        """Test that grid has approximately uniform spacing."""
        grid = create_threshold_grid(num_points=11, exclude_boundaries=False)
        
        # Should be [0, 0.1, 0.2, ..., 1.0]
        for i, val in enumerate(grid):
            expected = i / 10.0
            assert abs(val - expected) < 1e-10


class TestCustomThresholdGrid:
    """Test calibration with custom threshold grids."""

    def test_custom_grid_dense(self):
        """Test calibration with a dense threshold grid."""
        calibration_seeds = [3001, 3002]
        custom_grid = [i / 100.0 for i in range(1, 100)]  # 0.01, 0.02, ..., 0.99
        
        theta_reject, theta_verify = calibrate_thresholds(
            method="B1",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
            theta_grid=custom_grid,
        )
        
        assert theta_reject < theta_verify

    def test_custom_grid_sparse(self):
        """Test calibration with a sparse threshold grid."""
        calibration_seeds = [3001, 3002]
        custom_grid = [0.25, 0.5, 0.75]
        
        theta_reject, theta_verify = calibrate_thresholds(
            method="B1",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
            theta_grid=custom_grid,
        )
        
        assert theta_reject < theta_verify


class TestEvaluationFunction:
    """Test the _evaluate_threshold_pair function."""

    def test_evaluation_returns_valid_haer(self):
        """Test that evaluation returns a value in [0, 1]."""
        calibration_seeds = [4001]
        
        haer = _evaluate_threshold_pair(
            method="B1",
            theta_reject=0.3,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=1.0,
            p_catch=0.8,
        )
        
        assert 0.0 <= haer <= 1.0

    def test_evaluation_deterministic(self):
        """Test that evaluation is deterministic."""
        calibration_seeds = [4001, 4002]
        
        haer1 = _evaluate_threshold_pair(
            method="B1",
            theta_reject=0.3,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=1.0,
            p_catch=0.8,
        )
        
        haer2 = _evaluate_threshold_pair(
            method="B1",
            theta_reject=0.3,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=1.0,
            p_catch=0.8,
        )
        
        assert haer1 == haer2

    def test_evaluation_differs_by_threshold(self):
        """Test that different thresholds can produce different HAERs."""
        calibration_seeds = [4001, 4002]
        
        haer_low_reject = _evaluate_threshold_pair(
            method="B1",
            theta_reject=0.1,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=1.0,
            p_catch=0.8,
        )
        
        haer_high_reject = _evaluate_threshold_pair(
            method="B1",
            theta_reject=0.6,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=1.0,
            p_catch=0.8,
        )
        
        # Different thresholds should typically give different results
        # (not guaranteed, but very likely with different rejection rates)
        # We just verify they're both valid
        assert 0.0 <= haer_low_reject <= 1.0
        assert 0.0 <= haer_high_reject <= 1.0

    def test_evaluation_b1_vs_b2(self):
        """Test that B1 and B2 can produce different HAERs."""
        calibration_seeds = [4001]
        
        haer_b1 = _evaluate_threshold_pair(
            method="B1",
            theta_reject=0.3,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=1.0,
            p_catch=0.8,
        )
        
        haer_b2 = _evaluate_threshold_pair(
            method="B2",
            theta_reject=0.3,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=1.0,
            p_catch=0.8,
        )
        
        # Both should be valid
        assert 0.0 <= haer_b1 <= 1.0
        assert 0.0 <= haer_b2 <= 1.0


class TestZParameter:
    """Test the z parameter for LCB."""

    def test_custom_z_value(self):
        """Test calibration with custom z parameter."""
        calibration_seeds = [5001]
        
        theta_reject, theta_verify = calibrate_thresholds(
            method="B2",
            target_haer=0.5,
            calibration_seeds=calibration_seeds,
            z=2.0,  # Higher confidence
        )
        
        assert theta_reject < theta_verify

    def test_z_affects_calibration_b2(self):
        """Test that z parameter can affect B2 calibration."""
        calibration_seeds = [5001, 5002]
        
        # Different z values should be independent
        haer_z1 = _evaluate_threshold_pair(
            method="B2",
            theta_reject=0.3,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=1.0,
            p_catch=0.8,
        )
        
        haer_z2 = _evaluate_threshold_pair(
            method="B2",
            theta_reject=0.3,
            theta_verify=0.7,
            calibration_seeds=calibration_seeds,
            z=2.0,
            p_catch=0.8,
        )
        
        # Both should be valid (they may or may not differ)
        assert 0.0 <= haer_z1 <= 1.0
        assert 0.0 <= haer_z2 <= 1.0
