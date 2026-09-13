"""
Comprehensive tests for the Beta-Bernoulli trust model.

Tests cover:
1. Posterior parameter computation
2. Trust mean calculation
3. Trust variance calculation
4. Equivalence of variance formulas
5. Prior-only boundary behavior
6. Large-evidence asymptotic behavior
7-10. LCB score calculation and clipping
11. Invalid input handling
12. Determinism
"""

import math
import pytest
import numpy as np

from src.trust_model import (
    posterior_parameters,
    trust_mean,
    trust_variance,
    lcb_score,
)


class TestPosteriorParameters:
    """Test posterior parameter calculation."""

    def test_posterior_counting_example(self):
        """Test hand-calculated posterior counting example.
        
        Given: b = 2, h = 3, alpha0 = 1, beta0 = 1
        Expected: alpha = 3, beta = 4
        """
        alpha, beta = posterior_parameters(2, 3, alpha0=1.0, beta0=1.0)
        assert alpha == 3.0
        assert beta == 4.0

    def test_posterior_no_observations(self):
        """Test that prior is returned when no observations."""
        alpha, beta = posterior_parameters(0, 0, alpha0=2.0, beta0=3.0)
        assert alpha == 2.0
        assert beta == 3.0

    def test_posterior_benign_only(self):
        """Test posterior with only benign observations."""
        alpha, beta = posterior_parameters(5, 0, alpha0=1.0, beta0=1.0)
        assert alpha == 6.0
        assert beta == 1.0

    def test_posterior_harmful_only(self):
        """Test posterior with only harmful observations."""
        alpha, beta = posterior_parameters(0, 5, alpha0=1.0, beta0=1.0)
        assert alpha == 1.0
        assert beta == 6.0

    def test_posterior_different_prior(self):
        """Test posterior with different prior parameters."""
        alpha, beta = posterior_parameters(10, 5, alpha0=2.0, beta0=3.0)
        assert alpha == 12.0
        assert beta == 8.0

    def test_invalid_benign_count_negative(self):
        """Test that negative benign count raises ValueError."""
        with pytest.raises(ValueError, match="benign_count must be non-negative"):
            posterior_parameters(-1, 0)

    def test_invalid_harmful_count_negative(self):
        """Test that negative harmful count raises ValueError."""
        with pytest.raises(ValueError, match="harmful_count must be non-negative"):
            posterior_parameters(0, -1)

    def test_invalid_benign_count_type(self):
        """Test that non-integer benign count raises TypeError."""
        with pytest.raises(TypeError, match="benign_count must be an integer"):
            posterior_parameters(1.5, 0)

    def test_invalid_harmful_count_type(self):
        """Test that non-integer harmful count raises TypeError."""
        with pytest.raises(TypeError, match="harmful_count must be an integer"):
            posterior_parameters(0, 1.5)

    def test_invalid_alpha0_zero(self):
        """Test that alpha0 <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            posterior_parameters(0, 0, alpha0=0.0)

    def test_invalid_alpha0_negative(self):
        """Test that alpha0 < 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            posterior_parameters(0, 0, alpha0=-1.0)

    def test_invalid_beta0_zero(self):
        """Test that beta0 <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            posterior_parameters(0, 0, beta0=0.0)

    def test_invalid_beta0_negative(self):
        """Test that beta0 < 0 raises ValueError."""
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            posterior_parameters(0, 0, beta0=-1.0)


class TestTrustMean:
    """Test trust mean (point estimate) calculation."""

    def test_hand_calculated_example(self):
        """Test hand-calculated trust mean.
        
        For: alpha = 3, beta = 2
        Expected: T = 0.6
        """
        t = trust_mean(3.0, 2.0)
        assert t == pytest.approx(0.6, abs=1e-9)

    def test_symmetric_beta(self):
        """Test mean of symmetric Beta (should be 0.5)."""
        t = trust_mean(5.0, 5.0)
        assert t == pytest.approx(0.5, abs=1e-9)

    def test_uniform_prior_beta_1_1(self):
        """Test Beta(1,1) uniform prior (should be 0.5)."""
        t = trust_mean(1.0, 1.0)
        assert t == pytest.approx(0.5, abs=1e-9)

    def test_trust_zero(self):
        """Test case where trust approaches zero (high beta)."""
        t = trust_mean(0.1, 100.0)
        assert t == pytest.approx(0.001, abs=1e-4)

    def test_trust_one(self):
        """Test case where trust approaches one (high alpha)."""
        t = trust_mean(100.0, 0.1)
        assert t == pytest.approx(0.999, abs=1e-4)

    def test_invalid_alpha_zero(self):
        """Test that alpha <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            trust_mean(0.0, 1.0)

    def test_invalid_beta_zero(self):
        """Test that beta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="beta must be > 0"):
            trust_mean(1.0, 0.0)

    def test_invalid_alpha_negative(self):
        """Test that alpha < 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            trust_mean(-1.0, 1.0)

    def test_invalid_beta_negative(self):
        """Test that beta < 0 raises ValueError."""
        with pytest.raises(ValueError, match="beta must be > 0"):
            trust_mean(1.0, -1.0)


class TestTrustVariance:
    """Test posterior uncertainty (variance) calculation."""

    def test_hand_calculated_example(self):
        """Test hand-calculated posterior variance.
        
        For: alpha = 3, beta = 2
        Expected: U = 0.04
        
        Calculation:
        U = 3 * 2 / ((3+2)^2 * (3+2+1))
          = 6 / (25 * 6)
          = 6 / 150
          = 0.04
        """
        u = trust_variance(3.0, 2.0)
        assert u == pytest.approx(0.04, abs=1e-9)

    def test_prior_only_beta_1_1(self):
        """Test prior-only / n=0 boundary for Beta(1,1).
        
        For Beta(1,1):
        T = 0.5
        U = 1/12 ~ 0.08333...
        """
        u = trust_variance(1.0, 1.0)
        expected = 1.0 / 12.0
        assert u == pytest.approx(expected, abs=1e-9)

    def test_symmetric_beta_variance(self):
        """Test variance of symmetric Beta (same alpha and beta)."""
        u = trust_variance(5.0, 5.0)
        # For symmetric Beta, T = 0.5, so U = 0.5 * 0.5 / (10 + 1) = 0.25 / 11
        expected = 0.25 / 11.0
        assert u == pytest.approx(expected, abs=1e-9)

    def test_large_evidence_approaches_zero(self):
        """Test that variance decreases as evidence increases.
        
        Verify that as alpha and beta become large, uncertainty approaches zero.
        """
        u_small = trust_variance(10.0, 10.0)
        u_large = trust_variance(1000.0, 1000.0)
        assert u_large < u_small
        assert u_large < 1e-3

    def test_variance_formula_equivalence(self):
        """Test equivalence of the two variance formulas using a grid.
        
        Verify that:
        U1 = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
        
        equals:
        U2 = T*(1-T) / (n_posterior+1)
        
        across a comprehensive grid of positive alpha and beta values using np.isclose.
        """
        # Create a meaningful grid of alpha and beta values spanning several orders of magnitude
        alpha_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
        beta_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
        
        # Test all combinations of alpha and beta
        for alpha in alpha_values:
            for beta in beta_values:
                u1 = trust_variance(alpha, beta)
                
                # Formula 2: U = T*(1-T) / (n_posterior+1)
                t = trust_mean(alpha, beta)
                n_posterior = alpha + beta
                u2 = t * (1 - t) / (n_posterior + 1)
                
                # Use np.isclose with tight tolerance
                assert np.isclose(u1, u2, rtol=1e-14, atol=1e-15), \
                    f"Mismatch for alpha={alpha}, beta={beta}: u1={u1}, u2={u2}"

    def test_invalid_alpha_zero(self):
        """Test that alpha <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            trust_variance(0.0, 1.0)

    def test_invalid_beta_zero(self):
        """Test that beta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="beta must be > 0"):
            trust_variance(1.0, 0.0)

    def test_invalid_alpha_negative(self):
        """Test that alpha < 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            trust_variance(-1.0, 1.0)

    def test_invalid_beta_negative(self):
        """Test that beta < 0 raises ValueError."""
        with pytest.raises(ValueError, match="beta must be > 0"):
            trust_variance(1.0, -1.0)


class TestLCBScore:
    """Test lower confidence bound (LCB) score calculation."""

    def test_lcb_z_zero(self):
        """Test LCB with z=0.
        
        Verify: lcb_score(T, U, z=0) == T
        """
        t = 0.7
        u = 0.05
        lcb = lcb_score(t, u, z=0.0)
        assert lcb == pytest.approx(t, abs=1e-9)

    def test_lcb_calculation_example(self):
        """Test LCB score calculation with known values.
        
        For T = 0.6, U = 0.04, z = 1.0:
        LCB = 0.6 - 1.0 * sqrt(0.04)
            = 0.6 - 1.0 * 0.2
            = 0.4
        """
        t = 0.6
        u = 0.04
        z = 1.0
        lcb = lcb_score(t, u, z=z, clip=False)
        assert lcb == pytest.approx(0.4, abs=1e-9)

    def test_lcb_lower_clipping(self):
        """Test LCB lower clipping behavior.
        
        Use a case where T - z*sqrt(U) < 0.
        Verify result is 0 when clip=True.
        """
        t = 0.1
        u = 0.1
        z = 1.0
        lcb = lcb_score(t, u, z=z, clip=True)
        # 0.1 - 1.0 * sqrt(0.1) ~ 0.1 - 0.316 = -0.216
        assert lcb == pytest.approx(0.0, abs=1e-9)

    def test_lcb_upper_clipping(self):
        """Test LCB upper clipping behavior.
        
        Verify the function never returns a value above 1 when clipping is enabled.
        Even with z=0 and trust > 1 (edge case), should be clipped.
        """
        t = 1.0
        u = 0.01
        z = 0.0
        lcb = lcb_score(t, u, z=z, clip=True)
        assert lcb <= 1.0
        assert lcb == pytest.approx(1.0, abs=1e-9)

    def test_lcb_no_clipping(self):
        """Test LCB without clipping.
        
        Verify that without clipping, negative values can be returned.
        """
        t = 0.1
        u = 0.1
        z = 1.0
        lcb = lcb_score(t, u, z=z, clip=False)
        assert lcb < 0.0

    def test_lcb_various_z_values(self):
        """Test LCB with various z multiplier values."""
        t = 0.7
        u = 0.04
        
        lcb_z0 = lcb_score(t, u, z=0.0, clip=False)
        lcb_z1 = lcb_score(t, u, z=1.0, clip=False)
        lcb_z2 = lcb_score(t, u, z=2.0, clip=False)
        
        # Higher z should give lower LCB
        assert lcb_z0 > lcb_z1
        assert lcb_z1 > lcb_z2

    def test_lcb_uncertainty_effect(self):
        """Test that higher uncertainty reduces LCB score.
        
        For the same trust, higher uncertainty should yield lower LCB.
        """
        t = 0.5
        u_low = 0.001
        u_high = 0.1
        z = 1.0
        
        lcb_low_unc = lcb_score(t, u_low, z=z, clip=False)
        lcb_high_unc = lcb_score(t, u_high, z=z, clip=False)
        
        assert lcb_low_unc > lcb_high_unc

    def test_lcb_default_z_value(self):
        """Test that default z=1.0 is used."""
        t = 0.6
        u = 0.04
        lcb_default = lcb_score(t, u, clip=False)
        lcb_explicit = lcb_score(t, u, z=1.0, clip=False)
        assert lcb_default == pytest.approx(lcb_explicit, abs=1e-9)

    def test_lcb_default_clip_true(self):
        """Test that default clip=True is used."""
        t = 0.1
        u = 0.1
        lcb_default = lcb_score(t, u, z=1.0)
        lcb_explicit = lcb_score(t, u, z=1.0, clip=True)
        assert lcb_default == pytest.approx(lcb_explicit, abs=1e-9)

    def test_invalid_variance_negative(self):
        """Test that negative variance raises ValueError."""
        with pytest.raises(ValueError, match="variance must be non-negative"):
            lcb_score(0.5, -0.01)

    def test_invalid_z_negative(self):
        """Test that negative z raises ValueError."""
        with pytest.raises(ValueError, match="z must be >= 0"):
            lcb_score(0.5, 0.01, z=-1.0)

    def test_lcb_boundary_zero_variance(self):
        """Test LCB with zero variance.
        
        When variance is zero, LCB should equal trust (no uncertainty penalty).
        """
        t = 0.7
        u = 0.0
        lcb = lcb_score(t, u, z=1.0, clip=False)
        assert lcb == pytest.approx(t, abs=1e-9)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_end_to_end_trust_calculation(self):
        """Test complete workflow from observations to LCB score.
        
        Given benign=2, harmful=3 with uniform prior:
        1. Posterior: alpha=3, beta=4
        2. Trust: T = 3/7 ~ 0.4286
        3. Uncertainty: U ~ 0.0308
        4. LCB (z=1): T - sqrt(U) ~ 0.276
        """
        b, h = 2, 3
        alpha, beta = posterior_parameters(b, h)
        
        t = trust_mean(alpha, beta)
        u = trust_variance(alpha, beta)
        lcb = lcb_score(t, u, z=1.0)
        
        assert alpha == pytest.approx(3.0)
        assert beta == pytest.approx(4.0)
        assert t == pytest.approx(3.0 / 7.0, abs=1e-9)
        assert u == pytest.approx((3.0 * 4.0) / (49.0 * 8.0), abs=1e-9)
        assert lcb > 0.0 and lcb < t

    def test_determinism(self):
        """Test that all calculations are deterministic.
        
        Verify that repeated calls produce identical results.
        """
        params = [(2, 3, 1.0, 1.0), (10, 5, 2.0, 3.0), (0, 0, 1.0, 1.0)]
        
        for b, h, a0, b0 in params:
            # Repeated posterior_parameters calls
            result1 = posterior_parameters(b, h, a0, b0)
            result2 = posterior_parameters(b, h, a0, b0)
            assert result1 == result2
            
            alpha, beta = result1
            
            # Repeated trust_mean calls
            t1 = trust_mean(alpha, beta)
            t2 = trust_mean(alpha, beta)
            assert t1 == t2
            
            # Repeated trust_variance calls
            u1 = trust_variance(alpha, beta)
            u2 = trust_variance(alpha, beta)
            assert u1 == u2
            
            # Repeated lcb_score calls
            lcb1 = lcb_score(t1, u1, z=1.0)
            lcb2 = lcb_score(t1, u1, z=1.0)
            assert lcb1 == lcb2

    def test_large_observation_convergence(self):
        """Test that with many observations, uncertainty decreases."""
        # Small evidence
        alpha_s, beta_s = posterior_parameters(1, 1)
        u_small = trust_variance(alpha_s, beta_s)
        
        # Large evidence (same ratio)
        alpha_l, beta_l = posterior_parameters(1000, 1000)
        u_large = trust_variance(alpha_l, beta_l)
        
        # Uncertainty should decrease with more evidence
        assert u_large < u_small
        assert u_large < 1e-3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_prior_parameters(self):
        """Test with very small prior parameters."""
        alpha, beta = posterior_parameters(0, 0, alpha0=0.01, beta0=0.01)
        assert alpha == pytest.approx(0.01)
        assert beta == pytest.approx(0.01)
        
        t = trust_mean(alpha, beta)
        assert t == pytest.approx(0.5)

    def test_very_large_observations(self):
        """Test with very large observation counts."""
        alpha, beta = posterior_parameters(1000000, 500000)
        t = trust_mean(alpha, beta)
        u = trust_variance(alpha, beta)
        
        # Sanity checks
        assert 0 < t < 1
        assert u >= 0

    def test_imbalanced_observations(self):
        """Test with heavily imbalanced observations."""
        # Mostly benign
        alpha1, beta1 = posterior_parameters(1000, 1)
        t1 = trust_mean(alpha1, beta1)
        assert t1 > 0.99
        
        # Mostly harmful
        alpha2, beta2 = posterior_parameters(1, 1000)
        t2 = trust_mean(alpha2, beta2)
        assert t2 < 0.01

    def test_lcb_boundaries(self):
        """Test LCB at various trust boundaries."""
        # LCB with trust=0
        lcb = lcb_score(0.0, 0.01, z=1.0, clip=True)
        assert lcb == pytest.approx(0.0)
        
        # LCB with trust=1
        lcb = lcb_score(1.0, 0.01, z=1.0, clip=True)
        assert lcb >= 0.9
        assert lcb <= 1.0

    def test_lcb_extreme_z_values(self):
        """Test LCB with extreme z values."""
        t = 0.5
        u = 0.01
        
        # Very small z (nearly zero penalty)
        lcb_small_z = lcb_score(t, u, z=0.01, clip=False)
        assert lcb_small_z > t - 0.01
        
        # Very large z (extreme penalty)
        lcb_large_z = lcb_score(t, u, z=100.0, clip=True)
        assert lcb_large_z == pytest.approx(0.0)
