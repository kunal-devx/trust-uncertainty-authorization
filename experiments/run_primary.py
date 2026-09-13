"""Prepare seed-level primary evaluation results.

The module defines the full execution path but is intentionally not invoked by
this development task. It produces raw data only and performs no inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_calibration import (
    calibration_fingerprint,
    calibration_metadata,
    calibrate_targets,
    load_config,
    validate_config,
    write_json,
)
from experiments.run_pilot import (
    _evaluate_method,
    _paired_verify_outcomes,
    _seed_realization,
)
from src.metrics import compute_metrics


def _calibration_row(calibration: Mapping[str, Any], method: str, target: float) -> Mapping[str, Any]:
    """Find the calibration row for one method/target pair."""
    matches = [
        row for row in calibration["rows"]
        if row["method"] == method and row["target_haer"] == target
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one calibration row for {method} target {target}")
    return matches[0]


def validate_cached_calibration(
    calibration: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Reject cached calibration unless its complete metadata matches config."""
    expected_metadata = calibration_metadata(config)
    actual_metadata = calibration.get("calibration_metadata")
    actual_fingerprint = calibration.get("calibration_fingerprint")
    if not isinstance(actual_metadata, Mapping) or not isinstance(actual_fingerprint, str):
        raise ValueError(
            "cached calibration has no complete configuration metadata; "
            "run run_calibration.py to regenerate it"
        )
    if dict(actual_metadata) != expected_metadata:
        raise ValueError(
            "cached calibration configuration does not match the active primary.yaml; "
            "run run_calibration.py to regenerate it"
        )
    if actual_fingerprint != calibration_fingerprint(expected_metadata):
        raise ValueError(
            "cached calibration fingerprint does not match the active primary.yaml; "
            "run run_calibration.py to regenerate it"
        )


def evaluate_primary(
    config: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate every target/method/seed using paired realization streams."""
    validate_config(config, require_primary_counts=False)
    if set(calibration["calibration_seeds"]) & set(config["evaluation_seeds"]):
        raise ValueError("calibration and evaluation seeds must be disjoint")

    rows = []
    for seed in config["evaluation_seeds"]:
        population, verify_ss = _seed_realization(config, seed)
        verify_rng = np.random.default_rng(verify_ss)
        verify_outcomes = _paired_verify_outcomes(population, verify_rng, config["p_catch"])
        expected_size = config["K"] * len(config["profiles"]) * len(config["evidence_counts"])
        if len(population) != expected_size:
            raise AssertionError("population size does not match configuration")
        for target in config["target_haer_grid"]:
            for method in ("B1", "B2"):
                calibration_row = _calibration_row(calibration, method, target)
                thresholds = (calibration_row["theta_reject"], calibration_row["theta_verify"])
                records = _evaluate_method(
                    population,
                    verify_outcomes,
                    method,
                    thresholds,
                    config,
                    seed,
                )
                aggregate = compute_metrics(records, group_by=["seed"])[seed]
                cell_metrics = compute_metrics(records, group_by=["seed", "profile", "evidence_count"])
                rows.append({
                    "seed": seed,
                    "method": method,
                    "target_haer": target,
                    "calibration_achieved_haer": calibration_row["calibration_achieved_haer"],
                    "evaluation_achieved_haer": aggregate["haer"],
                    "theta_reject": thresholds[0],
                    "theta_verify": thresholds[1],
                    "unnecessary_intervention_rate": aggregate["unnecessary_intervention"],
                    "false_verify_rate": aggregate["false_verify"],
                    "false_reject_rate": aggregate["false_reject"],
                    "legitimate_action_completion": aggregate["legitimate_completion"],
                    "cold_start_evidence_counts": [
                        count for count in config["evidence_counts"] if count <= 5
                    ],
                    "cell_metrics": [
                        {
                            "seed": key[0],
                            "profile": key[1],
                            "evidence_count": key[2],
                            **metrics,
                        }
                        for key, metrics in sorted(cell_metrics.items(), key=lambda item: item[0])
                    ],
                })
    return {
        "experiment_name": config["experiment_name"],
        "K": config["K"],
        "M": config["M"],
        "profiles": dict(config["profiles"]),
        "evidence_counts": list(config["evidence_counts"]),
        "alpha0": config["alpha0"],
        "beta0": config["beta0"],
        "z": config["z"],
        "p_catch": config["p_catch"],
        "calibration_seeds": list(config["calibration_seeds"]),
        "evaluation_seeds": list(config["evaluation_seeds"]),
        "target_haer_grid": list(config["target_haer_grid"]),
        "threshold_grid": list(config["threshold_grid"]),
        "rows": rows,
        "statistical_unit": "seed",
        "inference_performed": False,
    }


def run_primary(
    config_path: str | Path = ROOT / "configs" / "primary.yaml",
    calibration_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[Dict[str, Any], Path]:
    """Load calibration data or compute it, then write primary raw results."""
    config = load_config(config_path)
    calibration_destination = Path(calibration_path or ROOT / config.get("calibration_output", "results/primary/calibration.json"))
    if calibration_destination.exists():
        calibration = json.loads(calibration_destination.read_text(encoding="utf-8"))
        validate_cached_calibration(calibration, config)
    else:
        calibration = calibrate_targets(config)
        write_json(calibration, calibration_destination)
    payload = evaluate_primary(config, calibration)
    destination = Path(output_path or ROOT / config.get("evaluation_output", "results/primary/evaluation.json"))
    write_json(payload, destination)
    return payload, destination


def main() -> None:
    """Execute the primary runner when explicitly requested."""
    payload, path = run_primary()
    print(f"Primary raw output: {path}")
    print(f"Rows: {len(payload['rows'])}")
    print("No statistical testing performed.")


if __name__ == "__main__":
    main()
