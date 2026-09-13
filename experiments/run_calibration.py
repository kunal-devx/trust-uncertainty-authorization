"""Prepare multi-target primary calibration results.

This module does not run evaluation. It is intended to be called explicitly
when the primary dataset is approved for generation.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import CalibrationRealization, _evaluate_threshold_pair, calibrate_thresholds
from experiments.run_pilot import _seed_realization


TARGET_KEY = "target_haer_grid"
REQUIRED_PRIMARY_KEYS = {
    "experiment_name", "base_seed", "calibration_seeds", "evaluation_seeds",
    "K", "M", "profiles", "evidence_counts", "alpha0", "beta0", "z",
    "p_catch", TARGET_KEY, "threshold_grid",
}

CALIBRATION_METADATA_KEYS = (
    "alpha0",
    "beta0",
    "z",
    "p_catch",
    "K",
    "M",
    "profiles",
    "evidence_counts",
    "calibration_seeds",
    "target_haer_grid",
    "threshold_grid",
)


def load_config(path: str | Path = ROOT / "configs" / "primary.yaml") -> Dict[str, Any]:
    """Load and validate the primary configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any], require_primary_counts: bool = True) -> None:
    """Validate primary dimensions and research-critical separation rules."""
    if not isinstance(config, Mapping):
        raise TypeError("primary configuration must be a mapping")
    missing = REQUIRED_PRIMARY_KEYS - set(config)
    if missing:
        raise ValueError(f"primary configuration missing keys: {sorted(missing)}")
    if not isinstance(config["K"], int) or config["K"] <= 0:
        raise ValueError("K must be a positive integer")
    if not isinstance(config["M"], int) or config["M"] <= 0:
        raise ValueError("M must be a positive integer")
    calibration_seeds = config["calibration_seeds"]
    evaluation_seeds = config["evaluation_seeds"]
    if not isinstance(calibration_seeds, list) or (require_primary_counts and len(calibration_seeds) < 10):
        raise ValueError("at least 10 calibration seeds are required")
    if not isinstance(evaluation_seeds, list) or (require_primary_counts and len(evaluation_seeds) < 20):
        raise ValueError("at least 20 evaluation seeds are required")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in calibration_seeds + evaluation_seeds):
        raise ValueError("seeds must be integers")
    if len(set(calibration_seeds)) != len(calibration_seeds) or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("seed lists must not contain duplicates")
    if set(calibration_seeds) & set(evaluation_seeds):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    if config["z"] != 1.0:
        raise ValueError("primary z must be fixed at 1.0")
    if config["p_catch"] != 0.80:
        raise ValueError("primary p_catch must be 0.80")
    profiles = config["profiles"]
    if profiles != {"reliable": 0.05, "unreliable": 0.40, "malicious": 0.80}:
        raise ValueError("primary profiles must match the frozen profiles")
    if config["evidence_counts"] != [0, 1, 2, 3, 5, 8, 12, 20, 40, 100]:
        raise ValueError("primary evidence_counts must match the frozen evidence levels")
    if config["alpha0"] != 1.0 or config["beta0"] != 1.0:
        raise ValueError("primary prior must be Beta(1,1)")
    for key in (TARGET_KEY, "threshold_grid"):
        values = config[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a non-empty list")
        if any(not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{key} values must be in [0, 1]")
    if any(config["threshold_grid"][i] >= config["threshold_grid"][i + 1] for i in range(len(config["threshold_grid"]) - 1)):
        raise ValueError("threshold_grid must be strictly increasing")


def calibration_metadata(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the canonical configuration subset that defines calibration."""
    return {
        key: config[TARGET_KEY] if key == TARGET_KEY else list(config[key]) if key in ("calibration_seeds", "evidence_counts", "threshold_grid") else dict(config[key]) if key == "profiles" else config[key]
        for key in CALIBRATION_METADATA_KEYS
    }


def calibration_fingerprint(metadata: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint of canonical metadata."""
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _calibration_realizations(config: Mapping[str, Any]) -> List[CalibrationRealization]:
    """Create one caller-owned population/VERIFY realization per calibration seed."""
    realizations = []
    for seed in config["calibration_seeds"]:
        population, verify_ss = _seed_realization(config, seed)
        # Rebuild the same explicit generator state for every method/target.
        realizations.append((population, np.random.default_rng(verify_ss)))
    return realizations


def _fresh_realizations(config: Mapping[str, Any]) -> List[CalibrationRealization]:
    """Create fresh equivalent realizations for each calibration call."""
    realizations = []
    for seed in config["calibration_seeds"]:
        population, verify_ss = _seed_realization(config, seed)
        realizations.append((population, np.random.default_rng(verify_ss)))
    return realizations


def calibrate_targets(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Calibrate B1/B2 independently across every configured target HAER."""
    validate_config(config, require_primary_counts=False)
    rows = []
    for target in config[TARGET_KEY]:
        for method in ("B1", "B2"):
            # Each call receives fresh explicit realization objects. The
            # population/action realization is deterministic and identical by seed.
            realizations = _fresh_realizations(config)
            thresholds = calibrate_thresholds(
                method=method,
                target_haer=target,
                calibration_realizations=realizations,
                theta_grid=config["threshold_grid"],
                z=config["z"],
                p_catch=config["p_catch"],
            )
            achieved = _evaluate_threshold_pair(
                method=method,
                theta_reject=thresholds[0],
                theta_verify=thresholds[1],
                calibration_seeds=None,
                calibration_realizations=_fresh_realizations(config),
                z=config["z"],
                p_catch=config["p_catch"],
            )
            rows.append({
                "method": method,
                "target_haer": target,
                "theta_reject": thresholds[0],
                "theta_verify": thresholds[1],
                "calibration_achieved_haer": achieved,
            })
    return {
        "experiment_name": config["experiment_name"],
        "calibration_metadata": calibration_metadata(config),
        "calibration_fingerprint": calibration_fingerprint(calibration_metadata(config)),
        "calibration_seeds": list(config["calibration_seeds"]),
        "evaluation_seeds": list(config["evaluation_seeds"]),
        "target_haer_grid": list(config[TARGET_KEY]),
        "threshold_grid": list(config["threshold_grid"]),
        "z": config["z"],
        "p_catch": config["p_catch"],
        "rows": rows,
    }


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write deterministic JSON and refuse conflicting overwrite."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite different result: {destination}")
    destination.write_text(content, encoding="utf-8")
    return destination


def main() -> None:
    """Prepare calibration output when explicitly invoked."""
    config = load_config()
    payload = calibrate_targets(config)
    path = write_json(payload, ROOT / config.get("calibration_output", "results/primary/calibration.json"))
    print(f"Calibration output: {path}")


if __name__ == "__main__":
    main()
