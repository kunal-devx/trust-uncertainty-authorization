"""Deterministic descriptive analysis of the frozen production B3 diagnostic.

B3 is treated only as a synthetic exogenous uncertainty-penalty control. This
module performs no inference and never modifies or reruns experiment outputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

B3_PATH = ROOT / "results/diagnostic/b3/b3_diagnostic.json"
PRIMARY_ANALYSIS_PATH = ROOT / "results/primary/analysis.json"
PRIMARY_EVALUATION_PATH = ROOT / "results/primary/evaluation.json"
PRIMARY_CALIBRATION_PATH = ROOT / "results/primary/calibration.json"
OUTPUT_JSON = ROOT / "results/diagnostic/b3/b3_analysis.json"
OUTPUT_SUMMARY = ROOT / "results/diagnostic/b3/b3_analysis_summary.txt"
FIGURE_DIR = ROOT / "figures/diagnostic/b3"
TARGETS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
REQUIRED_B3_FIELDS = {
    "seed", "method", "target_haer", "calibration_achieved_haer",
    "evaluation_achieved_haer", "theta_reject", "theta_verify",
    "unnecessary_intervention_rate", "legitimate_action_completion",
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean(values: Iterable[float]) -> float:
    return float(np.mean(np.asarray(list(values), dtype=float)))


def _load(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_b3(data: Mapping[str, Any]) -> None:
    rows = data.get("rows", [])
    keys = {(row["seed"], row["target_haer"], row["method"]) for row in rows}
    expected = {(seed, target, "B3") for seed in data["metadata"]["evaluation_seeds"] for target in data["metadata"]["target_haer_grid"]}
    if keys != expected or len(rows) != 140:
        raise ValueError("B3 output does not contain the expected 20 x 7 rows")
    for row in rows:
        missing = REQUIRED_B3_FIELDS - set(row)
        if missing:
            raise ValueError(f"B3 row missing fields: {sorted(missing)}")
        if row["theta_reject"] >= row["theta_verify"]:
            raise ValueError("B3 threshold ordering is invalid")
    if data.get("statistical_unit") != "seed" or data.get("inference_performed") is not False:
        raise ValueError("B3 output must declare seed-level descriptive analysis")


def _primary_curve(primary_analysis: Mapping[str, Any]) -> Dict[tuple[float, str], Mapping[str, Any]]:
    return {(row["target_haer"], row["method"]): row for row in primary_analysis["curve"]}


def _primary_thresholds(primary_evaluation: Mapping[str, Any]) -> Dict[tuple[float, str], Mapping[str, Any]]:
    result = {}
    for target in TARGETS:
        for method in ("B1", "B2"):
            rows = [row for row in primary_evaluation["rows"] if row["target_haer"] == target and row["method"] == method]
            if len(rows) != 20:
                raise ValueError(f"expected 20 primary rows for {method} target {target}")
            result[(target, method)] = rows[0]
    return result


def _b3_by_target(rows: Sequence[Mapping[str, Any]], target: float) -> Dict[str, float]:
    selected = [row for row in rows if row["target_haer"] == target]
    return {
        "mean_achieved_haer": _mean(row["evaluation_achieved_haer"] for row in selected),
        "mean_ui": _mean(row["unnecessary_intervention_rate"] for row in selected),
        "mean_legitimate_action_completion": _mean(row["legitimate_action_completion"] for row in selected),
    }


def _v_summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "quantiles": {
            "25": float(np.quantile(array, 0.25)),
            "50": float(np.quantile(array, 0.50)),
            "75": float(np.quantile(array, 0.75)),
            "95": float(np.quantile(array, 0.95)),
        },
        "theoretical_beta_2_5": {
            "mean": 2.0 / 7.0,
            "variance": 10.0 / 392.0,
            "standard_deviation": float(np.sqrt(10.0 / 392.0)),
        },
        "formal_goodness_of_fit_test": False,
    }


def analyze(
    b3_path: str | Path = B3_PATH,
    primary_analysis_path: str | Path = PRIMARY_ANALYSIS_PATH,
    primary_evaluation_path: str | Path = PRIMARY_EVALUATION_PATH,
) -> Dict[str, Any]:
    b3 = _load(b3_path)
    primary_analysis = _load(primary_analysis_path)
    primary_evaluation = _load(primary_evaluation_path)
    _validate_b3(b3)
    curve = _primary_curve(primary_analysis)
    thresholds = _primary_thresholds(primary_evaluation)
    b3_rows = b3["rows"]

    target_rows = []
    threshold_rows = []
    for target in TARGETS:
        b3_values = _b3_by_target(b3_rows, target)
        b1 = curve[(target, "B1")]
        b2 = curve[(target, "B2")]
        target_rows.append({
            "target_haer": target,
            "B1": {"mean_achieved_haer": b1["mean_achieved_haer"], "mean_ui": b1["mean_ui"], "mean_legitimate_action_completion": b1["mean_legitimate_action_completion"]},
            "B2": {"mean_achieved_haer": b2["mean_achieved_haer"], "mean_ui": b2["mean_ui"], "mean_legitimate_action_completion": b2["mean_legitimate_action_completion"]},
            "B3": b3_values,
            "B3_minus_B1": {
                "ui": b3_values["mean_ui"] - b1["mean_ui"],
                "haer": b3_values["mean_achieved_haer"] - b1["mean_achieved_haer"],
                "legitimate_completion": b3_values["mean_legitimate_action_completion"] - b1["mean_legitimate_action_completion"],
            },
            "B3_minus_B2": {
                "ui": b3_values["mean_ui"] - b2["mean_ui"],
                "haer": b3_values["mean_achieved_haer"] - b2["mean_achieved_haer"],
                "legitimate_completion": b3_values["mean_legitimate_action_completion"] - b2["mean_legitimate_action_completion"],
            },
        })
        b3_row = next(row for row in b3_rows if row["target_haer"] == target)
        threshold_rows.append({
            "target_haer": target,
            "B1": {"theta_reject": thresholds[(target, "B1")]["theta_reject"], "theta_verify": thresholds[(target, "B1")]["theta_verify"], "calibration_achieved_haer": thresholds[(target, "B1")]["calibration_achieved_haer"]},
            "B2": {"theta_reject": thresholds[(target, "B2")]["theta_reject"], "theta_verify": thresholds[(target, "B2")]["theta_verify"], "calibration_achieved_haer": thresholds[(target, "B2")]["calibration_achieved_haer"]},
            "B3": {"theta_reject": b3_row["theta_reject"], "theta_verify": b3_row["theta_verify"], "calibration_achieved_haer": b3_row["calibration_achieved_haer"]},
        })
        for method in ("B1", "B2", "B3"):
            threshold_rows[-1][method]["absolute_calibration_error"] = abs(threshold_rows[-1][method]["calibration_achieved_haer"] - target)

    errors = [(row["target_haer"], row["B3"]["absolute_calibration_error"]) for row in threshold_rows]
    v_summary = _v_summary([record["v"] for record in b3["agent_v"]])
    return {
        "analysis": "descriptive_b3_comparison",
        "input_hashes": {
            "b3": sha256(b3_path),
            "primary_analysis": sha256(primary_analysis_path),
            "primary_evaluation": sha256(primary_evaluation_path),
            "primary_calibration": sha256(PRIMARY_CALIBRATION_PATH),
        },
        "targets": TARGETS,
        "methods": ["B1", "B2", "B3"],
        "statistical_unit": "seed",
        "inference_performed": False,
        "target_comparison": target_rows,
        "threshold_geometry": threshold_rows,
        "b3_calibration": {
            "best_calibrated_target": min(errors, key=lambda item: item[1])[0],
            "best_calibration_error": min(errors, key=lambda item: item[1])[1],
            "largest_deviation_target": max(errors, key=lambda item: item[1])[0],
            "largest_calibration_error": max(errors, key=lambda item: item[1])[1],
            "wording": "closest selected grid point; no target is treated as an exact match",
        },
        "v_distribution": v_summary,
        "limitations": [
            "B3 is a synthetic exogenous uncertainty-penalty control, not real or empirically calibrated model confidence.",
            "B3 agent_v records contain seed, agent_id, and v only; profile/evidence and per-cell B3 outcomes are not retained.",
            "Therefore no B3 profile, evidence, cold-start, or profile-by-evidence analysis is performed.",
            "All comparisons are descriptive; no p-values, confidence intervals, or hypothesis tests are calculated.",
            "Target HAER and achieved HAER are retained as separate quantities.",
        ],
        "interpretation": "B3 adds descriptive context about a synthetic exogenous uncertainty penalty. It does not alter the frozen primary INCONCLUSIVE conclusion and does not establish superiority or inferiority.",
    }


def _write_figures(result: Mapping[str, Any], directory: str | Path = FIGURE_DIR) -> List[str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = result["target_comparison"]
    paths = []
    style = {"B1": "#1f4e79", "B2": "#b24745", "B3": "#4c956c"}
    for filename, ylabel, field, title in (
        ("b1_b2_b3_safety_utility_curve.png", "Unnecessary intervention rate", "mean_ui", "Descriptive B1/B2/B3 safety–utility curve"),
        ("b1_b2_b3_ui_by_target.png", "Unnecessary intervention rate", "mean_ui", "Descriptive UI by target HAER"),
        ("b1_b2_b3_legitimate_completion.png", "Legitimate-action completion", "mean_legitimate_action_completion", "Descriptive legitimate-action completion by target"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in ("B1", "B2", "B3"):
            x = [row["B3"]["mean_achieved_haer"] if field == "mean_ui" and filename.endswith("curve.png") and method == "B3" else row[method]["mean_achieved_haer"] if field == "mean_ui" and filename.endswith("curve.png") else row["target_haer"] for row in rows]
            y = [row[method][field] for row in rows]
            ax.plot(x, y, "o-", label=method, color=style[method])
        ax.set_xlabel("Mean achieved HAER" if filename.endswith("curve.png") else "Target HAER")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = directory / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _summary(result: Mapping[str, Any]) -> str:
    lines = [
        "B3 descriptive analysis",
        f"B3 input SHA-256: {result['input_hashes']['b3']}",
        f"Primary evaluation SHA-256: {result['input_hashes']['primary_evaluation']}",
        "Statistical unit: seed; inference performed: false",
        "",
        "Target comparison",
    ]
    for row in result["target_comparison"]:
        lines.append(f"target={row['target_haer']:.2f} B1_UI={row['B1']['mean_ui']:.6f} B2_UI={row['B2']['mean_ui']:.6f} B3_UI={row['B3']['mean_ui']:.6f} B3-B1_UI={row['B3_minus_B1']['ui']:.6f} B3-B2_UI={row['B3_minus_B2']['ui']:.6f}")
    lines.extend(["", "Calibration", json.dumps(result["b3_calibration"], sort_keys=True), "", "v distribution", json.dumps(result["v_distribution"], sort_keys=True), "", "Limitations"])
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(["", result["interpretation"]])
    return "\n".join(lines) + "\n"


def run_analysis() -> Dict[str, Any]:
    result = analyze()
    result["figure_paths"] = _write_figures(result)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUTPUT_SUMMARY.write_text(_summary(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run_analysis()
    print(f"B3 analysis written to {OUTPUT_JSON}")
    print(f"B3 summary written to {OUTPUT_SUMMARY}")
