"""Diagnostic-only B3 synthetic exogenous uncertainty control.

B3 is an illustrative synthetic control. Its per-agent ``v`` is sampled from
Beta(2,5), independent of evidence history, fixed across that agent's actions,
and never interpreted as posterior variance or calibrated model confidence.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_calibration import load_config
from experiments.run_primary import validate_cached_calibration
from src.authorization import authorize
from src.calibration import _select_threshold_pair
from src.population import generate_population
from src.trust_model import trust_mean
from src.verify_sim import verify_action

PRIMARY_EVALUATION = ROOT / "results" / "primary" / "evaluation.json"
PRIMARY_CALIBRATION = ROOT / "results" / "primary" / "calibration.json"
OUTPUT_DIR = ROOT / "results" / "diagnostic" / "b3"
OUTPUT_JSON = OUTPUT_DIR / "b3_diagnostic.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "b3_diagnostic_summary.txt"
B3_ALPHA = 2.0
B3_BETA = 5.0
EVALUATION_ROW_FIELDS = {
    "seed",
    "method",
    "target_haer",
    "calibration_achieved_haer",
    "evaluation_achieved_haer",
    "theta_reject",
    "theta_verify",
    "unnecessary_intervention_rate",
    "false_verify_rate",
    "false_reject_rate",
    "legitimate_action_completion",
    "cell_metrics",
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_metadata_fields(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: Sequence[str],
    label: str,
) -> None:
    for field in fields:
        if field not in actual:
            raise ValueError(f"{label} metadata missing required field: {field}")
        if actual[field] != expected[field]:
            raise ValueError(f"{label} metadata mismatch for field: {field}")


def validate_evaluation_artifact(
    evaluation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Validate frozen evaluation contents against the active configuration."""
    metadata_fields = (
        "K", "M", "alpha0", "beta0", "z", "p_catch", "profiles",
        "evidence_counts", "evaluation_seeds", "target_haer_grid", "threshold_grid",
    )
    _validate_metadata_fields(evaluation, config, metadata_fields, "evaluation")
    expected_keys = {
        (seed, target, method)
        for seed in config["evaluation_seeds"]
        for target in config["target_haer_grid"]
        for method in ("B1", "B2")
    }
    rows = evaluation.get("rows")
    if not isinstance(rows, list):
        raise ValueError("evaluation rows must be a list")
    if len(rows) != len(expected_keys):
        raise ValueError(
            f"evaluation row count mismatch: expected {len(expected_keys)}, got {len(rows)}"
        )
    actual_keys = []
    for index, row in enumerate(rows):
        missing = EVALUATION_ROW_FIELDS - set(row)
        if missing:
            raise ValueError(f"evaluation row {index} missing required fields: {sorted(missing)}")
        key = (row["seed"], row["target_haer"], row["method"])
        actual_keys.append(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected evaluation row combination: {key}")
    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError("duplicate evaluation seed-target-method combination")
    if set(actual_keys) != expected_keys:
        raise ValueError("evaluation seed-target-method combinations are incomplete")


def validate_calibration_artifact(
    calibration: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Validate frozen calibration metadata, fingerprint, and selected rows."""
    validate_cached_calibration(calibration, config)
    expected_targets = set(config["target_haer_grid"])
    rows = calibration.get("rows")
    if not isinstance(rows, list):
        raise ValueError("calibration rows must be a list")
    expected_keys = {(target, method) for target in expected_targets for method in ("B1", "B2")}
    actual_keys = []
    required_fields = {
        "method", "target_haer", "theta_reject", "theta_verify",
        "calibration_achieved_haer",
    }
    for index, row in enumerate(rows):
        missing = required_fields - set(row)
        if missing:
            raise ValueError(f"calibration row {index} missing required fields: {sorted(missing)}")
        key = (row["target_haer"], row["method"])
        actual_keys.append(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected calibration row combination: {key}")
        if row["theta_reject"] >= row["theta_verify"]:
            raise ValueError(f"calibration row {index} has invalid threshold ordering")
    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError("duplicate calibration target-method combination")
    if set(actual_keys) != expected_keys:
        raise ValueError("calibration target-method combinations are incomplete")


def validate_frozen_primary_artifacts(
    config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> None:
    """Fail closed unless both frozen primary artifacts match config."""
    validate_evaluation_artifact(evaluation, config)
    validate_calibration_artifact(calibration, config)


def _seed_realization(config: Mapping[str, Any], seed: int) -> Tuple[List[Dict[str, Any]], List[List[bool | None]], List[float]]:
    """Build shared population/VERIFY outcomes plus one fixed v per agent."""
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
    verify_child_ss, b3_child_ss = verify_ss.spawn(2)
    verify_rng = np.random.default_rng(verify_child_ss)
    b3_rng = np.random.default_rng(b3_child_ss)
    verify_outcomes = [
        [
            verify_action(True, config["p_catch"], verify_rng) if is_harmful else None
            for is_harmful in agent["measurement_actions"]
        ]
        for agent in population
    ]
    synthetic_v = [float(b3_rng.beta(B3_ALPHA, B3_BETA)) for _ in population]
    return population, verify_outcomes, synthetic_v


def b3_score(agent: Mapping[str, Any], v: float, z: float) -> float:
    """Compute clipped B3 score: T - z*sqrt(v)."""
    trust = trust_mean(agent["alpha"], agent["beta"])
    return float(np.clip(trust - z * np.sqrt(v), 0.0, 1.0))


def _evaluate_pair(
    population: Sequence[Mapping[str, Any]],
    verify_outcomes: Sequence[Sequence[bool | None]],
    synthetic_v: Sequence[float],
    thresholds: Tuple[float, float],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate one B3 threshold pair on caller-owned paired realizations."""
    if len(population) != len(verify_outcomes):
        raise ValueError("population and verify_outcomes must have equal length")
    if len(population) != len(synthetic_v):
        raise ValueError("population and synthetic_v must have equal length")
    for index, (agent, outcomes) in enumerate(zip(population, verify_outcomes)):
        if len(agent["measurement_actions"]) != len(outcomes):
            raise ValueError(
                f"measurement action and VERIFY outcome lengths differ for agent index {index}"
            )
    theta_reject, theta_verify = thresholds
    counts = {
        "total_actions": 0,
        "benign_actions": 0,
        "harmful_actions": 0,
        "allow": 0,
        "verify": 0,
        "reject": 0,
        "benign_allow": 0,
        "benign_verify": 0,
        "benign_reject": 0,
        "harmful_allow": 0,
        "harmful_verify": 0,
        "harmful_reject": 0,
        "harmful_executed": 0,
        "harmful_blocked": 0,
        "benign_executed": 0,
        "benign_blocked": 0,
    }
    for agent, outcomes, v in zip(population, verify_outcomes, synthetic_v):
        decision = authorize(b3_score(agent, v, config["z"]), theta_reject, theta_verify)
        for is_harmful, verify_outcome in zip(agent["measurement_actions"], outcomes):
            executed = (
                True if decision == "ALLOW"
                else (bool(verify_outcome) if is_harmful else True)
                if decision == "VERIFY"
                else False
            )
            kind = "harmful" if is_harmful else "benign"
            counts["total_actions"] += 1
            counts[f"{kind}_actions"] += 1
            counts[decision.lower()] += 1
            counts[f"{kind}_{decision.lower()}"] += 1
            counts[f"{kind}_{'executed' if executed else 'blocked'}"] += 1
    benign = counts["benign_actions"]
    harmful = counts["harmful_actions"]
    return {
        **counts,
        "ui": (counts["benign_verify"] + counts["benign_reject"]) / benign if benign else 0.0,
        "haer": counts["harmful_executed"] / harmful if harmful else 0.0,
        "legitimate_completion": counts["benign_executed"] / benign if benign else 0.0,
        "false_verify_rate": counts["benign_verify"] / benign if benign else 0.0,
        "false_reject_rate": counts["benign_reject"] / benign if benign else 0.0,
    }


def _calibrate_b3(config: Mapping[str, Any]) -> Tuple[Dict[float, Tuple[float, float]], Dict[float, float]]:
    """Select B3 thresholds independently using only B3 calibration HAER."""
    selected = {}
    achieved = {}
    realizations = [
        _seed_realization(config, seed)
        for seed in config["calibration_seeds"]
    ]
    for target in config["target_haer_grid"]:
        candidate_haers = {}
        for theta_reject in config["threshold_grid"]:
            for theta_verify in config["threshold_grid"]:
                if theta_reject >= theta_verify:
                    continue
                harmful = 0
                executed = 0
                for population, verify_outcomes, synthetic_v in realizations:
                    result = _evaluate_pair(
                        population,
                        verify_outcomes,
                        synthetic_v,
                        (theta_reject, theta_verify),
                        config,
                    )
                    harmful += result["harmful_actions"]
                    executed += result["harmful_executed"]
                candidate_haers[(theta_reject, theta_verify)] = executed / harmful if harmful else 0.0
        pair, value = _select_threshold_pair(candidate_haers, target)
        selected[target] = pair
        achieved[target] = value
    return selected, achieved


def _metadata(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "alpha0": config["alpha0"],
        "beta0": config["beta0"],
        "z": config["z"],
        "p_catch": config["p_catch"],
        "K": config["K"],
        "M": config["M"],
        "profiles": dict(config["profiles"]),
        "evidence_counts": list(config["evidence_counts"]),
        "calibration_seeds": list(config["calibration_seeds"]),
        "evaluation_seeds": list(config["evaluation_seeds"]),
        "target_haer_grid": list(config["target_haer_grid"]),
        "threshold_grid": list(config["threshold_grid"]),
        "synthetic_uncertainty_distribution": {"name": "Beta", "alpha": B3_ALPHA, "beta": B3_BETA},
    }


def run_b3_diagnostic(
    config_path: str | Path = ROOT / "configs" / "primary.yaml",
    output_path: str | Path = OUTPUT_JSON,
    summary_path: str | Path = OUTPUT_SUMMARY,
) -> Dict[str, Any]:
    """Run B3 calibration/evaluation in the diagnostic output namespace."""
    config = load_config(config_path)
    if set(config["calibration_seeds"]) & set(config["evaluation_seeds"]):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    evaluation = json.loads(PRIMARY_EVALUATION.read_text(encoding="utf-8"))
    calibration = json.loads(PRIMARY_CALIBRATION.read_text(encoding="utf-8"))
    validate_frozen_primary_artifacts(config, evaluation, calibration)
    thresholds, calibration_achieved = _calibrate_b3(config)
    rows = []
    agent_v_rows = []
    for seed in config["evaluation_seeds"]:
        population, verify_outcomes, synthetic_v = _seed_realization(config, seed)
        agent_v_rows.extend({"seed": seed, "agent_id": agent["agent_id"], "v": v} for agent, v in zip(population, synthetic_v))
        for target in config["target_haer_grid"]:
            result = _evaluate_pair(
                population,
                verify_outcomes,
                synthetic_v,
                thresholds[target],
                config,
            )
            rows.append({
                "seed": seed,
                "method": "B3",
                "target_haer": target,
                "calibration_achieved_haer": calibration_achieved[target],
                "evaluation_achieved_haer": result["haer"],
                "theta_reject": thresholds[target][0],
                "theta_verify": thresholds[target][1],
                "unnecessary_intervention_rate": result["ui"],
                "false_verify_rate": result["false_verify_rate"],
                "false_reject_rate": result["false_reject_rate"],
                "legitimate_action_completion": result["legitimate_completion"],
            })
    metadata = _metadata(config)
    payload = {
        "diagnostic_experiment": "b3_synthetic_exogenous_uncertainty_control",
        "scientific_wording": "B3 is an illustrative synthetic control in which the evidence-based trust mean is penalized by a prespecified uncertainty-like quantity sampled independently of the agent's evidence history. It is not intended to represent empirically calibrated model confidence.",
        "configuration_source": str(Path(config_path).relative_to(ROOT)) if Path(config_path).is_relative_to(ROOT) else str(config_path),
        "primary_evaluation_sha256": sha256(PRIMARY_EVALUATION),
        "primary_calibration_sha256": sha256(PRIMARY_CALIBRATION),
        "metadata": metadata,
        "rng_architecture": "SeedSequence(seed).spawn(2); population child, VERIFY child plus dedicated B3 child from verify_ss.spawn(2)",
        "statistical_unit": "seed",
        "inference_performed": False,
        "thresholds": {str(target): thresholds[target] for target in config["target_haer_grid"]},
        "rows": rows,
        "agent_v": agent_v_rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        "B3 diagnostic: synthetic exogenous uncertainty control",
        f"Primary evaluation SHA-256: {payload['primary_evaluation_sha256']}",
        f"Primary calibration SHA-256: {payload['primary_calibration_sha256']}",
        "Inference performed: false; statistical unit: seed",
        "",
        "Calibration/evaluation rows",
    ]
    for row in rows:
        summary.append(
            f"seed={row['seed']} target={row['target_haer']:.2f} "
            f"thresholds=({row['theta_reject']:.2f},{row['theta_verify']:.2f}) "
            f"calibration_haer={row['calibration_achieved_haer']:.6f} "
            f"evaluation_haer={row['evaluation_achieved_haer']:.6f} "
            f"ui={row['unnecessary_intervention_rate']:.6f}"
        )
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text("\n".join(summary) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_b3_diagnostic()
    print(f"B3 diagnostic output: {OUTPUT_JSON}")
    print(f"Rows: {len(result['rows'])}")
