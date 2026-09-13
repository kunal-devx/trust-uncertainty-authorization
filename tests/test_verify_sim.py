"""
Tests for VERIFY simulation.

Tests cover:
- Benign actions always execute
- Harmful action blocking probability
- RNG usage and reproducibility
- Input validation
"""

import pytest
import numpy as np
from src.verify_sim import verify_action


class TestBenignExecution:
    """Test that benign actions always execute."""

    def test_benign_always_executes(self):
        """Test that benign action always returns True."""
        rng = np.random.default_rng(42)
        result = verify_action(is_harmful=False, p_catch=0.8, rng=rng)
        assert result is True

    def test_benign_with_p_catch_zero(self):
        """Test benign with p_catch=0."""
        rng = np.random.default_rng(42)
        result = verify_action(is_harmful=False, p_catch=0.0, rng=rng)
        assert result is True

    def test_benign_with_p_catch_one(self):
        """Test benign with p_catch=1."""
        rng = np.random.default_rng(42)
        result = verify_action(is_harmful=False, p_catch=1.0, rng=rng)
        assert result is True

    def test_benign_no_randomness_consumed(self):
        """Test that benign action does not consume RNG randomness.
        
        Generate one random number before and after benign action.
        The sequence should be unaffected by the benign call.
        """
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        # Get a random number from rng1
        r1_before = rng1.uniform()
        
        # Call verify_action with benign (should not consume randomness)
        verify_action(is_harmful=False, p_catch=0.8, rng=rng1)
        
        # Get next random number from rng1
        r1_after = rng1.uniform()
        
        # Get same sequence from rng2 without verify_action call
        r2_before = rng2.uniform()
        r2_after = rng2.uniform()
        
        # Sequences should match
        assert r1_before == r2_before
        assert r1_after == r2_after


class TestHarmfulCatching:
    """Test harmful action catching behavior."""

    def test_harmful_p_catch_one_never_executes(self):
        """Test that harmful action with p_catch=1 never executes."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            result = verify_action(is_harmful=True, p_catch=1.0, rng=rng)
            assert result is False

    def test_harmful_p_catch_zero_always_executes(self):
        """Test that harmful action with p_catch=0 always executes."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            result = verify_action(is_harmful=True, p_catch=0.0, rng=rng)
            assert result is True

    def test_harmful_p_catch_half(self):
        """Test harmful action with p_catch=0.5 roughly 50% execution rate.
        
        With 100 samples, expect roughly 50 executions (some variance allowed).
        """
        rng = np.random.default_rng(42)
        executions = sum(
            verify_action(is_harmful=True, p_catch=0.5, rng=rng)
            for _ in range(100)
        )
        # With p_catch=0.5, expect ~50 executions
        # Allow range [40, 60] for stochastic variance
        assert 40 <= executions <= 60

    def test_harmful_p_catch_0_8_execution_rate(self):
        """Test harmful action with p_catch=0.8 has ~0.2 execution rate.
        
        With 500 samples and p_catch=0.8, expect roughly 100 executions (20%).
        """
        rng = np.random.default_rng(42)
        executions = sum(
            verify_action(is_harmful=True, p_catch=0.8, rng=rng)
            for _ in range(500)
        )
        # With p_catch=0.8, expect 20% execution rate
        # Allow range [70, 130] for stochastic variance
        assert 70 <= executions <= 130


class TestReproducibility:
    """Test reproducibility with same RNG seed."""

    def test_same_seed_produces_same_results(self):
        """Test that same RNG seed produces identical results."""
        sequence1 = []
        sequence2 = []
        
        for _ in range(50):
            rng1 = np.random.default_rng(12345)
            rng2 = np.random.default_rng(12345)
            
            for _ in range(10):
                r1 = verify_action(is_harmful=True, p_catch=0.8, rng=rng1)
                r2 = verify_action(is_harmful=True, p_catch=0.8, rng=rng2)
                assert r1 == r2

    def test_different_seeds_different_results(self):
        """Test that different seeds can produce different results.
        
        Note: This is probabilistic; different seeds SHOULD produce
        different sequences, but it's theoretically possible they could
        be identical by chance. This test uses seeds that typically diverge.
        """
        results_seed1 = []
        results_seed2 = []
        
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        
        for _ in range(100):
            results_seed1.append(verify_action(is_harmful=True, p_catch=0.8, rng=rng1))
            results_seed2.append(verify_action(is_harmful=True, p_catch=0.8, rng=rng2))
        
        # Sequences should differ
        assert results_seed1 != results_seed2


class TestInputValidationPCatch:
    """Test validation of p_catch parameter."""

    def test_invalid_p_catch_negative(self):
        """Test that negative p_catch raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="p_catch must be in \\[0, 1\\]"):
            verify_action(is_harmful=True, p_catch=-0.1, rng=rng)

    def test_invalid_p_catch_above_one(self):
        """Test that p_catch > 1 raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="p_catch must be in \\[0, 1\\]"):
            verify_action(is_harmful=True, p_catch=1.1, rng=rng)

    def test_invalid_p_catch_type_string(self):
        """Test that string p_catch raises TypeError."""
        rng = np.random.default_rng(42)
        with pytest.raises(TypeError, match="p_catch must be numeric"):
            verify_action(is_harmful=True, p_catch="0.8", rng=rng)

    def test_invalid_p_catch_type_none(self):
        """Test that None p_catch raises TypeError."""
        rng = np.random.default_rng(42)
        with pytest.raises(TypeError, match="p_catch must be numeric"):
            verify_action(is_harmful=True, p_catch=None, rng=rng)

    def test_p_catch_boundary_zero(self):
        """Test that p_catch=0 is valid."""
        rng = np.random.default_rng(42)
        result = verify_action(is_harmful=True, p_catch=0.0, rng=rng)
        assert result is True

    def test_p_catch_boundary_one(self):
        """Test that p_catch=1 is valid."""
        rng = np.random.default_rng(42)
        result = verify_action(is_harmful=True, p_catch=1.0, rng=rng)
        assert result is False


class TestInputValidationRNG:
    """Test validation of RNG parameter."""

    def test_invalid_rng_none(self):
        """Test that None RNG raises TypeError."""
        with pytest.raises(TypeError, match="rng must be np.random.Generator"):
            verify_action(is_harmful=True, p_catch=0.8, rng=None)

    def test_invalid_rng_type(self):
        """Test that invalid RNG type raises TypeError."""
        with pytest.raises(TypeError, match="rng must be np.random.Generator"):
            verify_action(is_harmful=True, p_catch=0.8, rng=42)

    def test_invalid_rng_string(self):
        """Test that string RNG raises TypeError."""
        with pytest.raises(TypeError, match="rng must be np.random.Generator"):
            verify_action(is_harmful=True, p_catch=0.8, rng="not a generator")

    def test_valid_rng_default(self):
        """Test that default_rng is accepted."""
        rng = np.random.default_rng(42)
        result = verify_action(is_harmful=True, p_catch=0.8, rng=rng)
        assert isinstance(result, (bool, np.bool_))


class TestLargeScaleExecution:
    """Test large-scale stochastic behavior."""

    def test_p_catch_0_6_large_sample(self):
        """Test p_catch=0.6 with large sample."""
        rng = np.random.default_rng(42)
        executions = sum(
            verify_action(is_harmful=True, p_catch=0.6, rng=rng)
            for _ in range(1000)
        )
        # With p_catch=0.6, expect 40% execution rate (~400 out of 1000)
        # Allow 10% margin for stochastic variance
        expected_min = int(1000 * 0.4 * 0.9)  # 360
        expected_max = int(1000 * 0.4 * 1.1)  # 440
        assert expected_min <= executions <= expected_max

    def test_p_catch_0_8_large_sample(self):
        """Test p_catch=0.8 with large sample."""
        rng = np.random.default_rng(42)
        executions = sum(
            verify_action(is_harmful=True, p_catch=0.8, rng=rng)
            for _ in range(1000)
        )
        # With p_catch=0.8, expect 20% execution rate (~200 out of 1000)
        # Allow 10% margin for stochastic variance
        expected_min = int(1000 * 0.2 * 0.9)  # 180
        expected_max = int(1000 * 0.2 * 1.1)  # 220
        assert expected_min <= executions <= expected_max

    def test_p_catch_0_95_large_sample(self):
        """Test p_catch=0.95 with large sample."""
        rng = np.random.default_rng(42)
        executions = sum(
            verify_action(is_harmful=True, p_catch=0.95, rng=rng)
            for _ in range(1000)
        )
        # With p_catch=0.95, expect 5% execution rate (~50 out of 1000)
        # Allow 20% margin for stochastic variance (small expected value)
        expected_min = int(1000 * 0.05 * 0.8)  # 40
        expected_max = int(1000 * 0.05 * 1.2)  # 60
        assert expected_min <= executions <= expected_max


class TestConsistency:
    """Test internal consistency and edge cases."""

    def test_repeated_calls_use_rng_appropriately(self):
        """Test that repeated calls maintain RNG state correctly."""
        rng = np.random.default_rng(42)
        
        # First sequence
        seq1 = [verify_action(is_harmful=True, p_catch=0.8, rng=rng) for _ in range(50)]
        
        # Reset RNG with same seed
        rng = np.random.default_rng(42)
        
        # Second sequence should match
        seq2 = [verify_action(is_harmful=True, p_catch=0.8, rng=rng) for _ in range(50)]
        
        assert seq1 == seq2

    def test_mixed_benign_harmful_sequence(self):
        """Test a mixed sequence of benign and harmful actions."""
        rng = np.random.default_rng(42)
        
        # Mix benign and harmful actions
        actions = [False, True, False, True, True, False]
        results = [
            verify_action(is_harmful=is_h, p_catch=0.8, rng=rng)
            for is_h in actions
        ]
        
        # Benign (False) should always be True
        assert results[0] is True  # benign
        assert results[2] is True  # benign
        assert results[5] is True  # benign
        
        # Harmful results depend on RNG but should be boolean
        for r in results:
            assert isinstance(r, (bool, np.bool_))
