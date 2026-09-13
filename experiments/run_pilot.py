"""Execute the small end-to-end pilot pipeline.

The pilot validates pairing, calibration/evaluation separation, and metric
plumbing. It is not a source of primary research results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.authorization import authorize
from src.calibration import CalibrationRealization, calibrate_thresholds
from src.metrics import ActionRecord, compute_metrics
from src.population import generate_population, get_population_stats
from src.trust_model import lcb_score, trust_mean, trust_variance
from src.verify_sim import verify_action


REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "base_seed",
    "calibration_seeds",
    "evaluation_seeds",
    "K",
    "M",
    "profiles",
    "evidence_counts",
    "alpha0",
    "beta0",
    "z",
    "p_catch",
    "target_haer",
    "threshold_grid",
}


def load_config(path: str | Path = ROOT / "configs" / "pilot.yaml") -> Dict[str, Any]:
    """Load and validate a pilot YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate all research-critical pilot configuration values."""
    if not isinstance(config, Mapping):
        raise TypeError("pilot configuration must be a mapping")
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"pilot configuration missing keys: {sorted(missing)}")
    if not isinstance(config["experiment_name"], str) or not config["experiment_name"]:
        raise ValueError("experiment_name must be a non-empty string")
    if not isinstance(config["base_seed"], int) or isinstance(config["base_seed"], bool):
        raise ValueError("base_seed must be an integer")
    for key in ("calibration_seeds", "evaluation_seeds"):
        seeds = config[key]
        if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
            raise ValueError(f"{key} must be a non-empty list of integers")
    if set(config["calibration_seeds"]) & set(config["evaluation_seeds"]):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    if not isinstance(config["K"], int) or config["K"] <= 0:
        raise ValueError("K must be a positive integer")
    if not isinstance(config["M"], int) or config["M"] <= 0:
        raise ValueError("M must be a positive integer")
    profiles = config["profiles"]
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("profiles must be a non-empty mapping")
    if any(not isinstance(name, str) or not 0.0 <= probability <= 1.0 for name, probability in profiles.items()):
        raise ValueError("profiles must map names to probabilities in [0, 1]")
    evidence_counts = config["evidence_counts"]
    if not isinstance(evidence_counts, list) or not evidence_counts:
        raise ValueError("evidence_counts must be a non-empty list")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in evidence_counts):
        raise ValueError("evidence_counts must contain non-negative integers")
    if not isinstance(config["alpha0"], (int, float)) or config["alpha0"] <= 0:
        raise ValueError("alpha0 must be greater than 0")
    if not isinstance(config["beta0"], (int, float)) or config["beta0"] <= 0:
        raise ValueError("beta0 must be greater than 0")
    if not isinstance(config["z"], (int, float)) or config["z"] < 0:
        raise ValueError("z must be non-negative")
    if config["p_catch"] != 0.80:
        raise ValueError("pilot p_catch must be 0.80")
    if not 0.0 <= config["target_haer"] <= 1.0:
        raise ValueError("target_haer must be in [0, 1]")
    grid = config["threshold_grid"]
    if not isinstance(grid, list) or len(grid) < 2:
        raise ValueError("threshold_grid must contain at least two values")
    if any(not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0 for value in grid):
        raise ValueError("threshold_grid values must be in [0, 1]")


def _seed_realization(config: Mapping[str, Any], seed: int) -> Tuple[List[Dict[str, Any]], np.random.SeedSequence]:
    """Generate one population and return the child seed for VERIFY streams."""
    seed_sequence = np.random.SeedSequence(seed)
    population_ss, verify_ss = seed_sequence.spawn(2)
    population_rng = np.random.default_rng(population_ss)
    population = generate_population(
        population_rng,
        num_measurement_actions=config["M"],
        agents_per_cell=config["K"],
        evidence_counts=config["evidence_counts"],
        profiles=config["profiles"],
        alpha0=config["alpha0"],
        beta0=config["beta0"],
    )
    return population, verify_ss.spawn(1)[0]


def _calibration_thresholds(config: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Calibrate B1 and B2 on identical per-seed realizations."""
    b1_realizations: List[CalibrationRealization] = []
    b2_realizations: List[CalibrationRealization] = []
    for seed in config["calibration_seeds"]:
        population, verify_ss = _seed_realization(config, seed)
        b1_realizations.append((population, np.random.default_rng(verify_ss)))
        b2_realizations.append((population, np.random.default_rng(verify_ss)))

    common_kwargs = {
        "target_haer": config["target_haer"],
        "theta_grid": config["threshold_grid"],
        "z": config["z"],
        "p_catch": config["p_catch"],
    }
    thresholds = {
        "B1": calibrate_thresholds(
            method="B1",
            calibration_realizations=b1_realizations,
            **common_kwargs,
        ),
        "B2": calibrate_thresholds(
            method="B2",
            calibration_realizations=b2_realizations,
            **common_kwargs,
        ),
    }
    for theta_reject, theta_verify in thresholds.values():
        if not theta_reject < theta_verify:
            raise AssertionError("calibrated thresholds must satisfy theta_reject < theta_verify")
    return thresholds


def _score(agent: Mapping[str, Any], method: str, z: float) -> float:
    """Compute the configured B1 or B2 score from unchanged posterior values."""
    trust = trust_mean(agent["alpha"], agent["beta"])
    if method == "B1":
        return trust
    if method == "B2":
        return lcb_score(trust, trust_variance(agent["alpha"], agent["beta"]), z=z, clip=True)
    raise ValueError(f"unsupported pilot method: {method}")


def _paired_verify_outcomes(
    population: Sequence[Mapping[str, Any]],
    verify_rng: np.random.Generator,
    p_catch: float,
) -> List[List[bool | None]]:
    """Generate one harmful-action VERIFY realization shared by both methods."""
    return [
        [
            verify_action(True, p_catch, verify_rng) if is_harmful else None
            for is_harmful in agent["measurement_actions"]
        ]
        for agent in population
    ]


def _evaluate_method(
    population: Sequence[Mapping[str, Any]],
    verify_outcomes: Sequence[Sequence[bool | None]],
    method: str,
    thresholds: Tuple[float, float],
    config: Mapping[str, Any],
    seed: int,
) -> List[ActionRecord]:
    """Evaluate one method against a shared population and outcome realization."""
    theta_reject, theta_verify = thresholds
    records = []
    for agent, outcomes in zip(population, verify_outcomes):
        decision = authorize(_score(agent, method, config["z"]), theta_reject, theta_verify)
        for is_harmful, verify_outcome in zip(agent["measurement_actions"], outcomes):
            if decision == "ALLOW":
                executed = True
            elif decision == "VERIFY":
                executed = bool(verify_outcome) if is_harmful else True
            else:
                executed = False
            records.append(
                ActionRecord(
                    seed=seed,
                    agent_id=agent["agent_id"],
                    profile=agent["profile"],
                    evidence_count=agent["evidence_count"],
                    is_harmful=bool(is_harmful),
                    decision=decision,
                    executed=executed,
                    method=method,
                )
            )
    return records


def _run_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run calibration and evaluation and return the serializable pilot payload."""
    thresholds = _calibration_thresholds(config)
    rows = []
    for seed in config["evaluation_seeds"]:
        population, verify_ss = _seed_realization(config, seed)
        verify_rng = np.random.default_rng(verify_ss)
        verify_outcomes = _paired_verify_outcomes(population, verify_rng, config["p_catch"])
        for method in ("B1", "B2"):
            records = _evaluate_method(
                population,
                verify_outcomes,
                method,
                thresholds[method],
                config,
                seed,
            )
            metrics = compute_metrics(records, group_by=["seed"])[seed]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "theta_reject": thresholds[method][0],
                    "theta_verify": thresholds[method][1],
                    "target_haer": config["target_haer"],
                    "achieved_haer": metrics["haer"],
                    "unnecessary_intervention_rate": metrics["unnecessary_intervention"],
                    "false_verify_rate": metrics["false_verify"],
                    "false_reject_rate": metrics["false_reject"],
                    "legitimate_action_completion": metrics["legitimate_completion"],
                }
            )
    expected_population_size = config["K"] * len(config["profiles"]) * len(config["evidence_counts"])
    return {
        "experiment_name": config["experiment_name"],
        "base_seed": config["base_seed"],
        "K": config["K"],
        "M": config["M"],
        "profiles": dict(config["profiles"]),
        "evidence_counts": list(config["evidence_counts"]),
        "alpha0": config["alpha0"],
        "beta0": config["beta0"],
        "z": config["z"],
        "p_catch": config["p_catch"],
        "target_haer": config["target_haer"],
        "threshold_grid": list(config["threshold_grid"]),
        "calibration_seeds": list(config["calibration_seeds"]),
        "evaluation_seeds": list(config["evaluation_seeds"]),
        "expected_population_size": expected_population_size,
        "b3_status": "B3 not implemented; excluded from pilot",
        "rows": rows,
    }


def write_results(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write deterministic JSON without silently replacing different output."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite different pilot result: {path}")
        return path
    path.write_text(content, encoding="utf-8")
    return path


def run_pilot(config_path: str | Path = ROOT / "configs" / "pilot.yaml", output_path: str | Path | None = None) -> Tuple[Dict[str, Any], Path]:
    """Run the pilot and write its deterministic machine-readable result."""
    config = load_config(config_path)
    payload = _run_payload(config)
    destination = output_path if output_path is not None else ROOT / config["output_file"]
    return payload, write_results(payload, destination)


def main() -> None:
    """Run the configured pilot and print a concise summary."""
    payload, output_path = run_pilot()
    methods = {
        method: [row for row in payload["rows"] if row["method"] == method]
        for method in ("B1", "B2")
    }
    print(f"Pilot: {payload['experiment_name']}")
    print(f"Output: {output_path}")
    print(f"Population size per seed: {payload['expected_population_size']}")
    for method, rows in methods.items():
        mean_haer = sum(row["achieved_haer"] for row in rows) / len(rows)
        mean_ui = sum(row["unnecessary_intervention_rate"] for row in rows) / len(rows)
        print(f"{method}: mean achieved HAER={mean_haer:.6f}, mean UI={mean_ui:.6f}")
    print(payload["b3_status"])


if __name__ == "__main__":
    main()
