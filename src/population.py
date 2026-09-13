"""
Population generation for research evaluation.

Generates controlled agent populations with:
- 3 agent profiles (Reliable, Unreliable, Malicious)
- 10 evidence levels (0, 1, 2, 3, 5, 8, 12, 20, 40, 100)
- 30 agents per profile/evidence-level cell
- Pre-observation benign/harmful counts
- Trust parameters
- 50 ground-truth measurement actions per agent

Population is reproducible and supports paired evaluation design.
Uses explicitly supplied NumPy Generator; no global RNG state.
"""

from typing import List, Dict, Any, Tuple, Mapping
import numpy as np


# Frozen configuration
PROFILES = {
    "Reliable": 0.05,     # P(harmful action)
    "Unreliable": 0.40,
    "Malicious": 0.80,
}

EVIDENCE_COUNTS = [0, 1, 2, 3, 5, 8, 12, 20, 40, 100]

AGENTS_PER_CELL = 30
MEASUREMENT_ACTIONS_PER_AGENT = 50

# Frozen prior
ALPHA0 = 1.0
BETA0 = 1.0


def generate_population(
    rng: np.random.Generator,
    num_measurement_actions: int = MEASUREMENT_ACTIONS_PER_AGENT,
    agents_per_cell: int = AGENTS_PER_CELL,
    evidence_counts: List[int] = None,
    profiles: Mapping[str, float] = None,
    alpha0: float = ALPHA0,
    beta0: float = BETA0,
) -> List[Dict[str, Any]]:
    """
    Generate evaluation population with controlled structure.

    Args:
        rng: NumPy Generator for reproducibility.
        num_measurement_actions: Actions per agent (default 50).
            agents_per_cell: Agents per profile/evidence cell (default: AGENTS_PER_CELL).
            evidence_counts: Evidence levels (default: EVIDENCE_COUNTS).
            profiles: Mapping of profile names to harmful probabilities (default: PROFILES).
            alpha0: Beta prior alpha parameter (default: ALPHA0).
            beta0: Beta prior beta parameter (default: BETA0).

    Returns:
        List of agent dictionaries with:
            - agent_id: Unique identifier
            - profile: "Reliable", "Unreliable", or "Malicious"
            - harmful_prob: Ground-truth P(harmful)
            - evidence_count: Pre-observation evidence level
            - benign_count: Pre-observed benign actions
            - harmful_count: Pre-observed harmful actions
            - alpha: Posterior Beta alpha parameter
            - beta: Posterior Beta beta parameter
            - measurement_actions: List of (is_harmful, int) ground-truth labels
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be np.random.Generator, got {type(rng).__name__}")

    if isinstance(num_measurement_actions, bool) or not isinstance(num_measurement_actions, int):
        raise TypeError("num_measurement_actions must be a positive integer")
    if num_measurement_actions <= 0:
        raise ValueError("num_measurement_actions must be greater than 0")
    if isinstance(agents_per_cell, bool) or not isinstance(agents_per_cell, int):
        raise TypeError("agents_per_cell must be a positive integer")
    if agents_per_cell <= 0:
        raise ValueError("agents_per_cell must be greater than 0")

    if evidence_counts is None:
        evidence_counts = EVIDENCE_COUNTS
    if not isinstance(evidence_counts, (list, tuple)) or not evidence_counts:
        raise ValueError("evidence_counts must be a non-empty sequence")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in evidence_counts):
        raise ValueError("evidence_counts must contain non-negative integers")

    if profiles is None:
        profiles = PROFILES
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("profiles must be a non-empty mapping")
    if any(
        not isinstance(name, str)
        or not isinstance(probability, (int, float))
        or probability < 0.0
        or probability > 1.0
        for name, probability in profiles.items()
    ):
        raise ValueError("profiles must map names to probabilities in [0, 1]")

    if not isinstance(alpha0, (int, float)) or alpha0 <= 0:
        raise ValueError("alpha0 must be greater than 0")
    if not isinstance(beta0, (int, float)) or beta0 <= 0:
        raise ValueError("beta0 must be greater than 0")

    population = []
    agent_id = 0

    for profile_name, harmful_prob in profiles.items():
        for evidence_count in evidence_counts:
            for _ in range(agents_per_cell):
                # Generate pre-observation counts
                # Consistent with the profile's harmful probability
                benign_count, harmful_count = _generate_evidence_counts(
                    evidence_count=evidence_count,
                    harmful_prob=harmful_prob,
                    rng=rng,
                )

                # Calculate trust parameters
                alpha = alpha0 + benign_count
                beta = beta0 + harmful_count

                # Generate measurement actions
                measurement_actions = [
                    rng.uniform() < harmful_prob
                    for _ in range(num_measurement_actions)
                ]

                agent = {
                    "agent_id": agent_id,
                    "profile": profile_name,
                    "harmful_prob": harmful_prob,
                    "evidence_count": evidence_count,
                    "benign_count": benign_count,
                    "harmful_count": harmful_count,
                    "alpha": alpha,
                    "beta": beta,
                    "measurement_actions": measurement_actions,
                }

                population.append(agent)
                agent_id += 1

    return population


def _generate_evidence_counts(
    evidence_count: int,
    harmful_prob: float,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """
    Generate benign and harmful counts for an agent.

    Generates evidence consistent with the agent's harmful probability.
    If evidence_count is 0, returns (0, 0).
    Otherwise, distributes the evidence_count between benign and harmful
    according to the harmful probability.

    Args:
        evidence_count: Total pre-observation evidence level
        harmful_prob: P(harmful) for this agent
        rng: NumPy Generator

    Returns:
        Tuple of (benign_count, harmful_count) summing to evidence_count
    """
    if evidence_count == 0:
        return 0, 0

    # Distribute evidence_count according to harmful_prob
    harmful_count = rng.binomial(evidence_count, harmful_prob)
    benign_count = evidence_count - harmful_count

    return benign_count, harmful_count


def get_population_stats(population: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics for a population.

    Args:
        population: Population from generate_population()

    Returns:
        Dictionary with:
            - total_agents
            - agents_per_profile
            - agents_per_evidence_level
            - agents_per_cell (profile × evidence)
            - total_measurement_actions
    """
    stats = {
        "total_agents": len(population),
        "agents_per_profile": {},
        "agents_per_evidence_level": {},
        "agents_per_cell": {},
        "total_measurement_actions": sum(
            len(a["measurement_actions"]) for a in population
        ),
    }

    for agent in population:
        profile = agent["profile"]
        evidence = agent["evidence_count"]

        # Count per profile
        stats["agents_per_profile"][profile] = (
            stats["agents_per_profile"].get(profile, 0) + 1
        )

        # Count per evidence level
        stats["agents_per_evidence_level"][evidence] = (
            stats["agents_per_evidence_level"].get(evidence, 0) + 1
        )

        # Count per cell
        cell_key = (profile, evidence)
        stats["agents_per_cell"][cell_key] = stats["agents_per_cell"].get(cell_key, 0) + 1

    return stats
