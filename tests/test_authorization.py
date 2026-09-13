"""
Comprehensive tests for the authorization decision logic.

Tests cover:
- ALLOW, VERIFY, REJECT decisions at all boundary conditions
- Input validation for score and thresholds
- Threshold ordering validation
- Deterministic behavior
"""

import pytest
from src.authorization import authorize


class TestAllowDecision:
    """Test ALLOW authorization outcome."""

    def test_allow_above_theta_verify(self):
        """Test ALLOW when score is well above theta_verify."""
        result = authorize(score=0.9, theta_reject=0.3, theta_verify=0.7)
        assert result == "ALLOW"

    def test_allow_at_theta_verify(self):
        """Test ALLOW exactly at theta_verify boundary."""
        result = authorize(score=0.7, theta_reject=0.3, theta_verify=0.7)
        assert result == "ALLOW"

    def test_allow_at_theta_verify_zero(self):
        """Test ALLOW when theta_verify is close to 0."""
        result = authorize(score=0.001, theta_reject=0.0, theta_verify=0.001)
        assert result == "ALLOW"

    def test_allow_at_theta_verify_one(self):
        """Test ALLOW when theta_verify is 1 and score is 1."""
        result = authorize(score=1.0, theta_reject=0.5, theta_verify=1.0)
        assert result == "ALLOW"

    def test_allow_between_verify_one(self):
        """Test ALLOW for various scores between theta_verify and 1."""
        test_cases = [
            (0.71, 0.3, 0.7),
            (0.8, 0.3, 0.7),
            (0.99, 0.3, 0.7),
            (1.0, 0.5, 0.9),
        ]
        for score, theta_reject, theta_verify in test_cases:
            result = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            assert result == "ALLOW", f"Failed for score={score}, theta_verify={theta_verify}"


class TestVerifyDecision:
    """Test VERIFY authorization outcome."""

    def test_verify_strictly_between_thresholds(self):
        """Test VERIFY when score is strictly between thresholds."""
        result = authorize(score=0.5, theta_reject=0.3, theta_verify=0.7)
        assert result == "VERIFY"

    def test_verify_at_theta_reject_lower(self):
        """Test VERIFY exactly at theta_reject (lower boundary)."""
        result = authorize(score=0.3, theta_reject=0.3, theta_verify=0.7)
        assert result == "VERIFY"

    def test_verify_close_below_theta_verify(self):
        """Test VERIFY just below theta_verify boundary."""
        result = authorize(score=0.6999, theta_reject=0.3, theta_verify=0.7)
        assert result == "VERIFY"

    def test_verify_various_midpoints(self):
        """Test VERIFY for various scores in the middle of the VERIFY zone."""
        test_cases = [
            (0.4, 0.3, 0.7),
            (0.5, 0.2, 0.8),
            (0.45, 0.1, 0.9),
            (0.99, 0.5, 1.0),
            (0.51, 0.5, 0.6),
        ]
        for score, theta_reject, theta_verify in test_cases:
            result = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            assert result == "VERIFY", f"Failed for score={score} between {theta_reject} and {theta_verify}"


class TestRejectDecision:
    """Test REJECT authorization outcome."""

    def test_reject_below_theta_reject(self):
        """Test REJECT when score is well below theta_reject."""
        result = authorize(score=0.1, theta_reject=0.3, theta_verify=0.7)
        assert result == "REJECT"

    def test_reject_at_zero_score(self):
        """Test REJECT when score is 0."""
        result = authorize(score=0.0, theta_reject=0.5, theta_verify=0.9)
        assert result == "REJECT"

    def test_reject_just_below_theta_reject(self):
        """Test REJECT just below theta_reject boundary."""
        result = authorize(score=0.2999, theta_reject=0.3, theta_verify=0.7)
        assert result == "REJECT"

    def test_reject_various_low_scores(self):
        """Test REJECT for various low scores."""
        test_cases = [
            (0.0, 0.1, 0.5),
            (0.05, 0.1, 0.9),
            (0.2, 0.5, 0.8),
            (0.1, 0.2, 0.9),
        ]
        for score, theta_reject, theta_verify in test_cases:
            result = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            assert result == "REJECT", f"Failed for score={score} below theta_reject={theta_reject}"


class TestBoundaryConditions:
    """Test edge cases and boundaries."""

    def test_narrow_verify_zone(self):
        """Test VERIFY zone that is very narrow."""
        # VERIFY zone is only [0.5, 0.500001)
        result = authorize(score=0.5000001, theta_reject=0.5, theta_verify=0.500001)
        assert result == "VERIFY"

    def test_wide_verify_zone(self):
        """Test VERIFY zone that spans most of the range."""
        # VERIFY zone is [0.01, 0.99)
        result = authorize(score=0.5, theta_reject=0.01, theta_verify=0.99)
        assert result == "VERIFY"

    def test_very_close_thresholds(self):
        """Test when thresholds are very close but still strictly ordered."""
        # VERIFY zone is [0.5, 0.5000001)
        result = authorize(score=0.5, theta_reject=0.5, theta_verify=0.5000001)
        assert result == "VERIFY"

    def test_score_exactly_zero(self):
        """Test score at minimum boundary."""
        result = authorize(score=0.0, theta_reject=0.1, theta_verify=0.9)
        assert result == "REJECT"

    def test_score_exactly_one(self):
        """Test score at maximum boundary."""
        result = authorize(score=1.0, theta_reject=0.1, theta_verify=0.9)
        assert result == "ALLOW"

    def test_all_thresholds_at_extremes(self):
        """Test with thresholds at [0, 1]."""
        result_allow = authorize(score=1.0, theta_reject=0.0, theta_verify=1.0)
        assert result_allow == "ALLOW"
        
        result_verify_low = authorize(score=0.0, theta_reject=0.0, theta_verify=1.0)
        assert result_verify_low == "VERIFY"
        
        result_verify_high = authorize(score=0.5, theta_reject=0.0, theta_verify=1.0)
        assert result_verify_high == "VERIFY"


class TestInputValidationScore:
    """Test validation of score parameter."""

    def test_invalid_score_negative(self):
        """Test that negative score raises ValueError."""
        with pytest.raises(ValueError, match="score must be in \\[0, 1\\]"):
            authorize(score=-0.1, theta_reject=0.3, theta_verify=0.7)

    def test_invalid_score_above_one(self):
        """Test that score > 1 raises ValueError."""
        with pytest.raises(ValueError, match="score must be in \\[0, 1\\]"):
            authorize(score=1.1, theta_reject=0.3, theta_verify=0.7)

    def test_invalid_score_far_negative(self):
        """Test that very negative score raises ValueError."""
        with pytest.raises(ValueError, match="score must be in \\[0, 1\\]"):
            authorize(score=-10.0, theta_reject=0.3, theta_verify=0.7)

    def test_invalid_score_far_above_one(self):
        """Test that score far above 1 raises ValueError."""
        with pytest.raises(ValueError, match="score must be in \\[0, 1\\]"):
            authorize(score=100.0, theta_reject=0.3, theta_verify=0.7)

    def test_invalid_score_type_string(self):
        """Test that string score raises TypeError."""
        with pytest.raises(TypeError, match="score must be a number"):
            authorize(score="0.5", theta_reject=0.3, theta_verify=0.7)

    def test_invalid_score_type_none(self):
        """Test that None score raises TypeError."""
        with pytest.raises(TypeError, match="score must be a number"):
            authorize(score=None, theta_reject=0.3, theta_verify=0.7)

    def test_invalid_score_type_list(self):
        """Test that list score raises TypeError."""
        with pytest.raises(TypeError, match="score must be a number"):
            authorize(score=[0.5], theta_reject=0.3, theta_verify=0.7)


class TestInputValidationThresholds:
    """Test validation of theta_reject and theta_verify parameters."""

    def test_invalid_theta_reject_negative(self):
        """Test that negative theta_reject raises ValueError."""
        with pytest.raises(ValueError, match="theta_reject must be in \\[0, 1\\]"):
            authorize(score=0.5, theta_reject=-0.1, theta_verify=0.7)

    def test_invalid_theta_reject_above_one(self):
        """Test that theta_reject > 1 raises ValueError."""
        with pytest.raises(ValueError, match="theta_reject must be in \\[0, 1\\]"):
            authorize(score=0.5, theta_reject=1.1, theta_verify=0.7)

    def test_invalid_theta_verify_negative(self):
        """Test that negative theta_verify raises ValueError."""
        with pytest.raises(ValueError, match="theta_verify must be in \\[0, 1\\]"):
            authorize(score=0.5, theta_reject=0.3, theta_verify=-0.1)

    def test_invalid_theta_verify_above_one(self):
        """Test that theta_verify > 1 raises ValueError."""
        with pytest.raises(ValueError, match="theta_verify must be in \\[0, 1\\]"):
            authorize(score=0.5, theta_reject=0.3, theta_verify=1.1)

    def test_invalid_theta_reject_type(self):
        """Test that non-numeric theta_reject raises TypeError."""
        with pytest.raises(TypeError, match="theta_reject must be a number"):
            authorize(score=0.5, theta_reject="0.3", theta_verify=0.7)

    def test_invalid_theta_verify_type(self):
        """Test that non-numeric theta_verify raises TypeError."""
        with pytest.raises(TypeError, match="theta_verify must be a number"):
            authorize(score=0.5, theta_reject=0.3, theta_verify="0.7")


class TestThresholdOrdering:
    """Test validation of threshold relationships."""

    def test_theta_reject_greater_than_theta_verify(self):
        """Test that theta_reject > theta_verify raises ValueError."""
        with pytest.raises(ValueError, match="theta_reject must be strictly less than theta_verify"):
            authorize(score=0.5, theta_reject=0.7, theta_verify=0.3)

    def test_theta_reject_much_greater_than_theta_verify(self):
        """Test that theta_reject much greater than theta_verify raises ValueError."""
        with pytest.raises(ValueError, match="theta_reject must be strictly less than theta_verify"):
            authorize(score=0.5, theta_reject=0.9, theta_verify=0.1)

    def test_theta_reject_equal_theta_verify_raises_error(self):
        """Test that theta_reject == theta_verify raises ValueError."""
        with pytest.raises(ValueError, match="theta_reject must be strictly less than theta_verify"):
            authorize(score=0.5, theta_reject=0.5, theta_verify=0.5)

    def test_theta_reject_equal_theta_verify_multiple_scores_raises_error(self):
        """Test that theta_reject == theta_verify raises ValueError regardless of score."""
        # Must reject equal thresholds even when comparing with different scores
        with pytest.raises(ValueError, match="theta_reject must be strictly less than theta_verify"):
            authorize(score=0.4999, theta_reject=0.5, theta_verify=0.5)


class TestDeterminism:
    """Test that authorization decisions are deterministic."""

    def test_repeated_calls_same_result(self):
        """Test that repeated calls with same inputs produce identical results."""
        test_cases = [
            (0.7, 0.3, 0.7),
            (0.5, 0.3, 0.7),
            (0.1, 0.3, 0.7),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.5, 0.49, 0.5),
        ]
        
        for score, theta_reject, theta_verify in test_cases:
            result1 = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            result2 = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            result3 = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            
            assert result1 == result2 == result3, \
                f"Non-deterministic results for score={score}, theta_reject={theta_reject}, theta_verify={theta_verify}"

    def test_many_repeated_calls_deterministic(self):
        """Test many repeated calls are all identical."""
        score, theta_reject, theta_verify = 0.65, 0.3, 0.7
        results = [authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify) for _ in range(100)]
        
        # All results should be identical
        assert len(set(results)) == 1, "Non-deterministic behavior detected"
        assert results[0] == "VERIFY"


class TestConsistency:
    """Test consistency of decision logic."""

    def test_no_gaps_in_decision_space(self):
        """Test that for any score, exactly one decision is made."""
        theta_reject = 0.3
        theta_verify = 0.7
        
        scores = [i / 1000.0 for i in range(0, 1001)]
        for score in scores:
            result = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            assert result in ["ALLOW", "VERIFY", "REJECT"]

    def test_mutually_exclusive_decisions(self):
        """Test that score regions don't overlap."""
        theta_reject = 0.3
        theta_verify = 0.7
        
        # Test various scores and verify no overlaps
        test_scores = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        for score in test_scores:
            result = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            
            # Verify consistency with definition
            if score >= theta_verify:
                assert result == "ALLOW"
            elif score >= theta_reject:
                assert result == "VERIFY"
            else:
                assert result == "REJECT"

    def test_monotonic_behavior_with_increasing_score(self):
        """Test that decision improves (or stays same) as score increases."""
        theta_reject = 0.3
        theta_verify = 0.7
        
        decision_order = {"REJECT": 0, "VERIFY": 1, "ALLOW": 2}
        
        prev_decision_rank = -1
        for score in [i / 100.0 for i in range(0, 101)]:
            result = authorize(score=score, theta_reject=theta_reject, theta_verify=theta_verify)
            current_rank = decision_order[result]
            
            # Decision should not get worse
            assert current_rank >= prev_decision_rank, \
                f"Decision degraded at score={score}: {result}"
            prev_decision_rank = current_rank
