"""Combined descriptive mechanism analysis for frozen B1/B2/B3 artifacts.

This module reads frozen outputs only. It performs no inference, reruns, or
new experiment selection and preserves the primary INCONCLUSIVE conclusion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ANALYSIS = ROOT / "results/primary/analysis.json"
PRIMARY_CALIBRATION = ROOT / "results/primary/calibration.json"
PRIMARY_EVALUATION = ROOT / "results/primary/evaluation.json"
B3_DIAGNOSTIC = ROOT / "results/diagnostic/b3/b3_diagnostic.json"
B3_ANALYSIS = ROOT / "results/diagnostic/b3/b3_analysis.json"
MECHANISM_DIAGNOSTIC = ROOT / "results/diagnostic/decision_decomposition.json"
OUTPUT_JSON = ROOT / "results/diagnostic/combined_mechanism_analysis.json"
OUTPUT_SUMMARY = ROOT / "results/diagnostic/combined_mechanism_analysis_summary.txt"
TARGETS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _primary_thresholds(calibration: Mapping[str, Any], target: float, method: str) -> Mapping[str, Any]:
    rows = [row for row in calibration["rows"] if row["target_haer"] == target and row["method"] == method]
    if len(rows) != 1:
        raise ValueError(f"expected one primary calibration row for {method} at {target}")
    return rows[0]


def _primary_curve(analysis: Mapping[str, Any], target: float, method: str) -> Mapping[str, Any]:
    rows = [row for row in analysis["curve"] if row["target_haer"] == target and row["method"] == method]
    if len(rows) != 1:
        raise ValueError(f"expected one primary curve row for {method} at {target}")
    return rows[0]


def analyze() -> Dict[str, Any]:
    primary_analysis = _load(PRIMARY_ANALYSIS)
    primary_calibration = _load(PRIMARY_CALIBRATION)
    primary_evaluation = _load(PRIMARY_EVALUATION)
    b3_diagnostic = _load(B3_DIAGNOSTIC)
    b3_analysis = _load(B3_ANALYSIS)
    mechanism = _load(MECHANISM_DIAGNOSTIC)

    target_rows = []
    threshold_rows = []
    for target in TARGETS:
        b1 = _primary_curve(primary_analysis, target, "B1")
        b2 = _primary_curve(primary_analysis, target, "B2")
        b3 = next(row for row in b3_analysis["target_comparison"] if row["target_haer"] == target)["B3"]
        b1_cal = _primary_thresholds(primary_calibration, target, "B1")
        b2_cal = _primary_thresholds(primary_calibration, target, "B2")
        b3_threshold = next(row for row in b3_diagnostic["rows"] if row["target_haer"] == target)
        target_rows.append({
            "target_haer": target,
            "B1": b1,
            "B2": b2,
            "B3": b3,
            "B3_minus_B1": {
                "ui": b3["mean_ui"] - b1["mean_ui"],
                "haer": b3["mean_achieved_haer"] - b1["mean_achieved_haer"],
                "legitimate_completion": b3["mean_legitimate_action_completion"] - b1["mean_legitimate_action_completion"],
            },
            "B3_minus_B2": {
                "ui": b3["mean_ui"] - b2["mean_ui"],
                "haer": b3["mean_achieved_haer"] - b2["mean_achieved_haer"],
                "legitimate_completion": b3["mean_legitimate_action_completion"] - b2["mean_legitimate_action_completion"],
            },
        })
        threshold_rows.append({
            "target_haer": target,
            "B1": {"theta_reject": b1_cal["theta_reject"], "theta_verify": b1_cal["theta_verify"], "achieved_haer": b1_cal["calibration_achieved_haer"]},
            "B2": {"theta_reject": b2_cal["theta_reject"], "theta_verify": b2_cal["theta_verify"], "achieved_haer": b2_cal["calibration_achieved_haer"]},
            "B2_minus_B1_UI": next(row for row in primary_analysis["ui_by_target"] if row["target_haer"] == target)["mean_paired_difference_b2_minus_b1"],
            "observed": "B2 reject threshold is lower at every target; verify threshold is lower at 0.01, 0.02, 0.03, 0.10, and 0.15, equal at 0.05, and higher at 0.08.",
        })

    decomposition = [row for row in mechanism["b2_minus_b1_target_method"] if row["target_haer"] in TARGETS]
    cold_start = primary_analysis["cold_start"]["overall"]
    lower_ui = [row["target_haer"] for row in decomposition if row["benign_verify_fraction"] + row["benign_reject_fraction"] < 0]
    higher_ui = [row["target_haer"] for row in decomposition if row["benign_verify_fraction"] + row["benign_reject_fraction"] > 0]

    return {
        "analysis": "combined_b1_b2_b3_mechanism_review",
        "input_hashes": {
            "primary_analysis": _sha(PRIMARY_ANALYSIS),
            "primary_calibration": _sha(PRIMARY_CALIBRATION),
            "primary_evaluation": _sha(PRIMARY_EVALUATION),
            "b3_diagnostic": _sha(B3_DIAGNOSTIC),
            "b3_analysis": _sha(B3_ANALYSIS),
            "decision_decomposition": _sha(MECHANISM_DIAGNOSTIC),
        },
        "statistical_unit": "seed",
        "inference_performed": False,
        "primary_conclusion": "INCONCLUSIVE",
        "mathematics": {
            "B1": "score = T",
            "B2": "score = T - z*sqrt(U)",
            "U": "alpha*beta / ((alpha+beta)^2*(alpha+beta+1))",
            "z": 1.0,
            "pointwise_relationship": "B2 <= B1 at identical thresholds",
            "recalibration_implication": "Any aggregate B2 utility advantage can arise only after independent threshold recalibration.",
        },
        "primary_target_comparison": target_rows,
        "threshold_recalibration": threshold_rows,
        "b2_ui_lower_targets": lower_ui,
        "b2_ui_higher_targets": higher_ui,
        "b2_decision_decomposition": decomposition,
        "cold_start": {
            "evidence_counts": [0, 1, 2, 3, 5],
            "rows": cold_start,
            "interpretation": "B2 is intervention-saturated at evidence 0, 1, and 2, then remains more intervention-heavy than B1 at evidence 3 and 5; this is directionally consistent with an uncertainty penalty under sparse evidence but is mixed across HAER and completion metrics.",
        },
        "B3_role": {
            "recommendation": "diagnostic/appendix secondary analysis",
            "legitimate_uses": [
                "descriptive contrast for an exogenous synthetic uncertainty penalty",
                "illustrate that an uncertainty-like penalty does not imply one fixed utility outcome",
                "motivate distinction between evidence-derived and synthetic uncertainty inputs",
            ],
            "not_supported": [
                "real model-confidence claims",
                "empirical confidence calibration claims",
                "a new confirmatory hypothesis",
                "superiority or inferiority claims",
            ],
        },
        "hostile_review": [
            {"severity": "major", "criticism": "Independent threshold recalibration can confound score construction with operating-point geometry.", "response": "Report raw threshold geometry and achieved HAER separately; treat the frontier as the estimand rather than identical-threshold behavior."},
            {"severity": "major", "criticism": "The threshold grid is discrete, so selected operating points and reversals can be grid artifacts.", "response": "Retain target and achieved HAER, report calibration errors, and avoid exact-match language."},
            {"severity": "major", "criticism": "B3 is synthetic and does not measure real model confidence.", "response": "Keep B3 diagnostic/appendix-only and label it an exogenous synthetic uncertainty-penalty control."},
            {"severity": "major", "criticism": "The simulation uses fixed trust snapshots and no measurement-time updates.", "response": "State this as a controlled design boundary, not a claim about deployment behavior."},
            {"severity": "minor", "criticism": "The contribution may be benchmark methodology rather than an algorithmic novelty claim.", "response": "Position the work around controlled empirical comparison and mechanism analysis."},
        ],
        "additional_production_diagnostic_required": False,
        "additional_diagnostic_rationale": "The retained B1/B2 decision decomposition explains the mixed UI pattern through benign VERIFY/REJECT composition and threshold geometry. B3 is sufficiently characterized at target level for its diagnostic role; a new run is not required at this stage.",
        "contribution_statements": [
            "A controlled benchmark compares point-estimate evidence-based trust gating with evidence-derived uncertainty gating across a calibrated safety–utility frontier.",
            "The study characterizes how independent threshold recalibration and discrete graduated decision regions shape cold-start safety and unnecessary intervention.",
            "A synthetic exogenous uncertainty control provides secondary descriptive context for distinguishing uncertainty derived from evidence history from an uncertainty-like penalty supplied independently of that history.",
        ],
        "limitations": [
            "B3 output does not retain profile/evidence identifiers, so no B3 cold-start or profile-by-evidence mechanism analysis is claimed.",
            "The primary result remains INCONCLUSIVE; this artifact does not revise it.",
            "No inferential statistics are introduced by this combined review.",
        ],
    }


def write_outputs(result: Mapping[str, Any]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "Combined B1/B2/B3 mechanism analysis",
        "Primary conclusion: INCONCLUSIVE",
        "Statistical unit: seed; inference performed: false",
        "",
        "B2 lower UI targets: " + ", ".join(f"{x:.2f}" for x in result["b2_ui_lower_targets"]),
        "B2 higher UI targets: " + ", ".join(f"{x:.2f}" for x in result["b2_ui_higher_targets"]),
        "",
        "Threshold recalibration",
    ]
    for row in result["threshold_recalibration"]:
        lines.append(f"target={row['target_haer']:.2f} B1=({row['B1']['theta_reject']:.2f},{row['B1']['theta_verify']:.2f}) B2=({row['B2']['theta_reject']:.2f},{row['B2']['theta_verify']:.2f}) B2-B1_UI={row['B2_minus_B1_UI']:.6f}")
    lines.extend(["", "Cold-start", result["cold_start"]["interpretation"], "", "B3 role", result["B3_role"]["recommendation"], "", "Hostile-review priorities"])
    lines.extend(f"- {item['severity']}: {item['criticism']}" for item in result["hostile_review"])
    lines.extend(["", result["additional_diagnostic_rationale"], "", "Contribution framing"])
    lines.extend(f"- {statement}" for statement in result["contribution_statements"])
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = analyze()
    write_outputs(result)
    print(f"Combined analysis written to {OUTPUT_JSON}")
