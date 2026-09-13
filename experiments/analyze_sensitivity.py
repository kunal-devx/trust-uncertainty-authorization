"""Descriptive analysis of one-factor sensitivity outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENSITIVITY_DIR = ROOT / "results/sensitivity"
PRIMARY_ANALYSIS = ROOT / "results/primary/analysis.json"
EXPECTED_CONDITION_IDS = {
    "baseline",
    "prior_beta_2_2",
    "z_0_5",
    "z_1_5",
    "p_catch_0_6",
    "p_catch_0_95",
    "threshold_step_0_025",
}
EXPECTED_TARGETS = {0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15}
EXPECTED_METHODS = {"B1", "B2"}
REQUIRED_ROW_FIELDS = {
    "seed", "target_haer", "method", "theta_reject", "theta_verify",
    "calibration_achieved_haer", "evaluation_achieved_haer",
    "unnecessary_intervention_rate", "legitimate_action_completion",
    "false_verify_rate", "false_reject_rate",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _condition_summary(payload):
    rows = payload["rows"]
    targets = sorted({row["target_haer"] for row in rows})
    summary = []
    for target in targets:
        method_rows = {method: [row for row in rows if row["target_haer"] == target and row["method"] == method] for method in ("B1", "B2")}
        b1, b2 = method_rows["B1"], method_rows["B2"]
        b1_ui = np.mean([r["unnecessary_intervention_rate"] for r in b1])
        b2_ui = np.mean([r["unnecessary_intervention_rate"] for r in b2])
        summary.append({
            "target_haer": target,
            "b1_mean_haer": float(np.mean([r["evaluation_achieved_haer"] for r in b1])),
            "b2_mean_haer": float(np.mean([r["evaluation_achieved_haer"] for r in b2])),
            "b1_mean_ui": float(b1_ui),
            "b2_mean_ui": float(b2_ui),
            "b2_minus_b1_ui": float(b2_ui - b1_ui),
            "b2_minus_b1_haer": float(np.mean([r["evaluation_achieved_haer"] for r in b2]) - np.mean([r["evaluation_achieved_haer"] for r in b1])),
            "b1_mean_legitimate_completion": float(np.mean([r["legitimate_action_completion"] for r in b1])),
            "b2_mean_legitimate_completion": float(np.mean([r["legitimate_action_completion"] for r in b2])),
            "b1_theta_reject": b1[0]["theta_reject"],
            "b1_theta_verify": b1[0]["theta_verify"],
            "b2_theta_reject": b2[0]["theta_reject"],
            "b2_theta_verify": b2[0]["theta_verify"],
            "b1_calibration_error": abs(b1[0]["calibration_achieved_haer"] - target),
            "b2_calibration_error": abs(b2[0]["calibration_achieved_haer"] - target),
        })
    signs = [np.sign(row["b2_minus_b1_ui"]) for row in summary if row["b2_minus_b1_ui"] != 0]
    classification = "UNIFORM DIRECTIONAL" if signs and all(sign == signs[0] for sign in signs) else "MIXED / SENSITIVITY-DEPENDENT" if signs else "NOT ASSESSABLE"
    return {
        "condition": payload["condition"],
        "fingerprint": payload["fingerprint"],
        "row_count": len(rows),
        "target_count": len(targets),
        "method_count": len(method_rows),
        "target_summary": summary,
        "qualitative_classification": classification,
    }


def _validate_condition_payload(payload, expected_id):
    condition = payload.get("condition")
    if not isinstance(condition, dict) or condition.get("id") != expected_id:
        raise ValueError(f"sensitivity condition id mismatch: expected {expected_id}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 280:
        raise ValueError(f"sensitivity condition {expected_id} must contain exactly 280 rows")
    keys = []
    for index, row in enumerate(rows):
        missing = REQUIRED_ROW_FIELDS - set(row)
        if missing:
            raise ValueError(f"condition {expected_id} row {index} missing fields: {sorted(missing)}")
        if row["method"] not in EXPECTED_METHODS or row["target_haer"] not in EXPECTED_TARGETS:
            raise ValueError(f"condition {expected_id} contains unexpected method or target")
        keys.append((row["seed"], row["target_haer"], row["method"]))
        if row["theta_reject"] >= row["theta_verify"]:
            raise ValueError(f"condition {expected_id} contains invalid threshold ordering")
    expected_seeds = set(range(20001, 20021))
    expected_keys = {
        (seed, target, method)
        for seed in expected_seeds
        for target in EXPECTED_TARGETS
        for method in EXPECTED_METHODS
    }
    if len(set(keys)) != len(keys):
        raise ValueError(f"condition {expected_id} contains duplicate seed-target-method rows")
    if set(keys) != expected_keys:
        raise ValueError(f"condition {expected_id} does not contain the complete expected seed-target-method grid")
    if payload.get("statistical_unit") != "seed" or payload.get("inference_performed") is not False:
        raise ValueError(f"condition {expected_id} must declare seed-level non-inferential output")


def analyze(directory=SENSITIVITY_DIR):
    directory = Path(directory)
    condition_payloads = {}
    for condition_id in sorted(EXPECTED_CONDITION_IDS):
        path = directory / f"{condition_id}.json"
        if not path.exists():
            raise ValueError(f"missing sensitivity condition file: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_condition_payload(payload, condition_id)
        condition_payloads[condition_id] = payload
    conditions = [_condition_summary(condition_payloads[condition_id]) for condition_id in sorted(condition_payloads)]
    primary = json.loads(PRIMARY_ANALYSIS.read_text(encoding="utf-8"))
    cold_start = {
        "source": "results/primary/analysis.json",
        "boundaries": [3, 5, 8],
        "note": "Derived from retained primary cell metrics; no sensitivity rerun was used for cold-start boundaries.",
        "primary_reference": primary["cold_start"],
    }
    return {
        "analysis": "one_factor_primary_robustness",
        "conditions": conditions,
        "condition_ids": sorted(condition_payloads),
        "input_files": [f"{condition_id}.json" for condition_id in sorted(condition_payloads)],
        "cold_start": cold_start,
        "statistical_unit": "seed",
        "inference_performed": False,
        "primary_conclusion": "INCONCLUSIVE",
        "overall_assessment": "MIXED / SENSITIVITY-DEPENDENT",
        "limitations": [
            "Sensitivity summaries are descriptive and use no p-values or confidence intervals.",
            "Cold-start boundaries are derived from frozen primary cell metrics; no new sensitivity simulation was run for that factor.",
            "MIXED / SENSITIVITY-DEPENDENT is a descriptive heuristic, not a formal statistical robustness test.",
            "Primary outputs remain the reference and are not overwritten.",
        ],
    }


def main():
    result = analyze()
    path = SENSITIVITY_DIR / "analysis.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = ["One-factor primary robustness analysis", "Primary conclusion: INCONCLUSIVE", "Overall assessment: MIXED / SENSITIVITY-DEPENDENT", ""]
    for condition in result["conditions"]:
        summary.append(f"{condition['condition']['id']}: {condition['qualitative_classification']}")
    summary.append(result["cold_start"]["note"])
    (SENSITIVITY_DIR / "analysis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
