"""
Tests for population generation.

Tests cover:
- Correct population structure and size
- Profile and evidence-level distribution
- Pre-observation evidence consistency
- Trust parameter calculation
- Measurement action generation
- Reproducibility
"""

import pytest
import numpy as np
from src.population import (
    generate_population,
    get_population_stats,
    PROFILES,
    EVIDENCE_COUNTS,
    AGENTS_PER_CELL,
    MEASUREMENT_ACTIONS_PER_AGENT,
    ALPHA0,
    BETA0,
)


class TestPopulationStructure:
    """Test basic population structure and size."""

    def test_population_total_size(self):
        """Test that population has exactly 900 agents."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        expected_size = len(PROFILES) * len(EVIDENCE_COUNTS) * AGENTS_PER_CELL
        assert len(population) == expected_size
        assert len(population) == 900

    def test_agents_per_profile(self):
        """Test correct agent count per profile."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        stats = get_population_stats(population)
        
        # Each profile should have 10 evidence levels × 30 agents = 300 agents
        expected_per_profile = len(EVIDENCE_COUNTS) * AGENTS_PER_CELL
        for profile in PROFILES.keys():
            assert stats["agents_per_profile"][profile] == expected_per_profile

    def test_agents_per_evidence_level(self):
        """Test correct agent count per evidence level."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        stats = get_population_stats(population)
        
        # Each evidence level should have 3 profiles × 30 agents = 90 agents
        expected_per_evidence = len(PROFILES) * AGENTS_PER_CELL
        for evidence in EVIDENCE_COUNTS:
            assert stats["agents_per_evidence_level"][evidence] == expected_per_evidence

    def test_agents_per_cell(self):
        """Test correct agent count per profile/evidence cell."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        stats = get_population_stats(population)
        
        # Each cell should have exactly 30 agents
        for profile in PROFILES.keys():
            for evidence in EVIDENCE_COUNTS:
                cell_key = (profile, evidence)
                assert stats["agents_per_cell"][cell_key] == AGENTS_PER_CELL

    def test_agent_ids_unique(self):
        """Test that all agent IDs are unique."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        agent_ids = [a["agent_id"] for a in population]
        assert len(set(agent_ids)) == len(agent_ids)
        assert len(agent_ids) == 900


class TestAgentAttributes:
    """Test attributes of individual agents."""

    def test_agent_has_required_fields(self):
        """Test that each agent has all required fields."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        required_fields = {
            "agent_id", "profile", "harmful_prob", "evidence_count",
            "benign_count", "harmful_count", "alpha", "beta", "measurement_actions"
        }
        
        for agent in population:
            assert set(agent.keys()) == required_fields

    def test_profile_is_valid(self):
        """Test that all agents have a valid profile."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        valid_profiles = set(PROFILES.keys())
        for agent in population:
            assert agent["profile"] in valid_profiles

    def test_harmful_prob_matches_profile(self):
        """Test that harmful_prob matches the agent's profile."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        for agent in population:
            profile = agent["profile"]
            expected_prob = PROFILES[profile]
            assert agent["harmful_prob"] == expected_prob

    def test_evidence_count_is_valid(self):
        """Test that all agents have a valid evidence count."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        valid_evidence = set(EVIDENCE_COUNTS)
        for agent in population:
            assert agent["evidence_count"] in valid_evidence

    def test_counts_sum_to_evidence(self):
        """Test that benign_count + harmful_count == evidence_count."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        for agent in population:
            total = agent["benign_count"] + agent["harmful_count"]
            assert total == agent["evidence_count"]

    def test_counts_non_negative(self):
        """Test that benign and harmful counts are non-negative."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        for agent in population:
            assert agent["benign_count"] >= 0
            assert agent["harmful_count"] >= 0

    def test_trust_parameters_computed_correctly(self):
        """Test that alpha and beta are computed correctly."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        for agent in population:
            expected_alpha = ALPHA0 + agent["benign_count"]
            expected_beta = BETA0 + agent["harmful_count"]
            
            assert agent["alpha"] == expected_alpha
            assert agent["beta"] == expected_beta


class TestMeasurementActions:
    """Test measurement action generation."""

    def test_measurement_actions_count(self):
        """Test that each agent has correct number of measurement actions."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        for agent in population:
            assert len(agent["measurement_actions"]) == MEASUREMENT_ACTIONS_PER_AGENT

    def test_measurement_actions_are_boolean(self):
        """Test that measurement actions are boolean (is_harmful)."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        for agent in population:
            for action in agent["measurement_actions"]:
                assert isinstance(action, (bool, np.bool_))

    def test_measurement_actions_distribution(self):
        """Test that measurement action proportions roughly match profile.
        
        For Reliable (p=0.05), most actions should be benign.
        For Malicious (p=0.80), most should be harmful.
        """
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        # Find one Reliable agent
        reliable_agent = next(a for a in population if a["profile"] == "Reliable")
        harmful_count = sum(reliable_agent["measurement_actions"])
        # Should be roughly 5% harmful (allow [0, 10] out of 50)
        assert harmful_count <= 10
        
        # Find one Malicious agent
        malicious_agent = next(a for a in population if a["profile"] == "Malicious")
        harmful_count = sum(malicious_agent["measurement_actions"])
        # Should be roughly 80% harmful (allow [35, 50] out of 50)
        assert harmful_count >= 35

    def test_custom_measurement_action_count(self):
        """Test generating population with custom action count."""
        rng = np.random.default_rng(42)
        custom_count = 100
        population = generate_population(rng, num_measurement_actions=custom_count)
        
        for agent in population:
            assert len(agent["measurement_actions"]) == custom_count

    def test_total_measurement_actions(self):
        """Test total measurement actions count."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        stats = get_population_stats(population)
        
        expected_total = len(population) * MEASUREMENT_ACTIONS_PER_AGENT
        assert stats["total_measurement_actions"] == expected_total


class TestReproducibility:
    """Test reproducibility with same RNG seed."""

    def test_same_seed_same_population(self):
        """Test that same RNG seed produces identical population."""
        rng1 = np.random.default_rng(12345)
        pop1 = generate_population(rng1)
        
        rng2 = np.random.default_rng(12345)
        pop2 = generate_population(rng2)
        
        # Compare key attributes
        assert len(pop1) == len(pop2)
        for a1, a2 in zip(pop1, pop2):
            assert a1["agent_id"] == a2["agent_id"]
            assert a1["profile"] == a2["profile"]
            assert a1["benign_count"] == a2["benign_count"]
            assert a1["harmful_count"] == a2["harmful_count"]
            assert a1["measurement_actions"] == a2["measurement_actions"]

    def test_different_seeds_different_evidence(self):
        """Test that different seeds can produce different evidence distributions."""
        rng1 = np.random.default_rng(1)
        pop1 = generate_population(rng1)
        
        rng2 = np.random.default_rng(2)
        pop2 = generate_population(rng2)
        
        # Find agents in same cell
        agent1 = next(a for a in pop1 if a["profile"] == "Reliable" and a["evidence_count"] == 20)
        agent2 = next(a for a in pop2 if a["profile"] == "Reliable" and a["evidence_count"] == 20)
        
        # Evidence counts or measurement actions could differ
        # (not guaranteed by different seeds, but typically will)
        # Just verify they are both valid
        assert agent1["benign_count"] + agent1["harmful_count"] == 20
        assert agent2["benign_count"] + agent2["harmful_count"] == 20


class TestInputValidation:
    """Test input validation."""

    def test_invalid_rng_none(self):
        """Test that None RNG raises TypeError."""
        with pytest.raises(TypeError, match="rng must be np.random.Generator"):
            generate_population(rng=None)

    def test_invalid_rng_type(self):
        """Test that invalid RNG type raises TypeError."""
        with pytest.raises(TypeError, match="rng must be np.random.Generator"):
            generate_population(rng=42)

    def test_custom_action_count_zero(self):
        """Test population with zero measurement actions."""
        rng = np.random.default_rng(42)
        population = generate_population(rng, num_measurement_actions=0)
        
        for agent in population:
            assert len(agent["measurement_actions"]) == 0

    def test_custom_action_count_large(self):
        """Test population with large measurement action count."""
        rng = np.random.default_rng(42)
        custom_count = 1000
        population = generate_population(rng, num_measurement_actions=custom_count)
        
        for agent in population:
            assert len(agent["measurement_actions"]) == custom_count


class TestEvidenceDistribution:
    """Test evidence count distribution consistency."""

    def test_zero_evidence_always_zero_counts(self):
        """Test that evidence_count=0 always gives zero benign and harmful counts."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        zero_evidence_agents = [a for a in population if a["evidence_count"] == 0]
        assert len(zero_evidence_agents) > 0
        
        for agent in zero_evidence_agents:
            assert agent["benign_count"] == 0
            assert agent["harmful_count"] == 0
            assert agent["alpha"] == ALPHA0
            assert agent["beta"] == BETA0

    def test_evidence_consistency_per_profile(self):
        """Test that evidence distribution is consistent across profiles."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        
        # For each evidence level, count how many harmful actions
        profile_evidence_harmful = {}
        for agent in population:
            profile = agent["profile"]
            evidence = agent["evidence_count"]
            
            key = (profile, evidence)
            if key not in profile_evidence_harmful:
                profile_evidence_harmful[key] = []
            
            profile_evidence_harmful[key].append(agent["harmful_count"])
        
        # Verify structure is correct
        for profile in PROFILES.keys():
            for evidence in EVIDENCE_COUNTS:
                key = (profile, evidence)
                assert key in profile_evidence_harmful
                assert len(profile_evidence_harmful[key]) == AGENTS_PER_CELL


class TestPopulationStats:
    """Test population statistics calculation."""

    def test_stats_all_profiles_present(self):
        """Test that stats include all profiles."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        stats = get_population_stats(population)
        
        for profile in PROFILES.keys():
            assert profile in stats["agents_per_profile"]

    def test_stats_all_evidence_present(self):
        """Test that stats include all evidence levels."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        stats = get_population_stats(population)
        
        for evidence in EVIDENCE_COUNTS:
            assert evidence in stats["agents_per_evidence_level"]

    def test_stats_consistency(self):
        """Test that stats totals are consistent."""
        rng = np.random.default_rng(42)
        population = generate_population(rng)
        stats = get_population_stats(population)
        
        # Sum of per-profile should equal total
        total_by_profile = sum(stats["agents_per_profile"].values())
        assert total_by_profile == stats["total_agents"]
        
        # Sum of per-evidence should equal total
        total_by_evidence = sum(stats["agents_per_evidence_level"].values())
        assert total_by_evidence == stats["total_agents"]
        
        # Sum of per-cell should equal total
        total_by_cell = sum(stats["agents_per_cell"].values())
        assert total_by_cell == stats["total_agents"]
