"""Deterministic downstream analysis of the frozen primary evaluation dataset.

This module never reruns an experiment or mutates evaluation.json. Inferential
comparisons use one aggregate value per evaluation seed; cell and curve tables
are descriptive summaries.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curves import OperatingPoint, match_by_achieved_haer
from src.stats import interpret_ui_difference, summarize_paired_ui

DEFAULT_DATASET = ROOT / "results" / "primary" / "evaluation.json"
DEFAULT_ANALYSIS = ROOT / "results" / "primary" / "analysis.json"
DEFAULT_SUMMARY = ROOT / "results" / "primary" / "analysis_summary.txt"
DEFAULT_FIGURES = ROOT / "figures" / "primary"
DEFAULT_ANALYSIS_SEED = 20260913
DEFAULT_BOOTSTRAP = 10_000
MATCH_TOLERANCE = 0.005


def dataset_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of the raw dataset bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path: str | Path = DEFAULT_DATASET) -> Dict[str, Any]:
    """Load the frozen dataset without modifying it."""
    with Path(path).open("r", encoding="utf-8") as stream:
        dataset = json.load(stream)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    """Validate the raw primary schema and seed-level pairing."""
    required = {"rows", "evaluation_seeds", "target_haer_grid", "statistical_unit"}
    missing = required - set(dataset)
    if missing:
        raise ValueError(f"dataset missing required fields: {sorted(missing)}")
    if dataset["statistical_unit"] != "seed":
        raise ValueError("primary analysis requires seed as the statistical unit")
    rows = dataset["rows"]
    seeds = dataset["evaluation_seeds"]
    targets = dataset["target_haer_grid"]
    expected = {(seed, target, method) for seed in seeds for target in targets for method in ("B1", "B2")}
    actual = {(row["seed"], row["target_haer"], row["method"]) for row in rows}
    if actual != expected:
        raise ValueError("dataset does not contain exactly one B1 and B2 row per seed and target")
    for row in rows:
        if row["method"] not in ("B1", "B2"):
            raise ValueError(f"unsupported method in dataset: {row['method']}")
        if row["target_haer"] != row["target_haer"] or row["evaluation_achieved_haer"] != row["evaluation_achieved_haer"]:
            raise ValueError("dataset contains NaN values")
        if row["theta_reject"] >= row["theta_verify"]:
            raise ValueError("dataset contains invalid threshold ordering")
        if len(row.get("cell_metrics", [])) == 0:
            raise ValueError("dataset row has no cell metrics")


def _rows(dataset: Mapping[str, Any], target: float, method: str) -> List[Mapping[str, Any]]:
    """Return rows in configured seed order for one target/method."""
    by_seed = {
        row["seed"]: row
        for row in dataset["rows"]
        if row["target_haer"] == target and row["method"] == method
    }
    return [by_seed[seed] for seed in dataset["evaluation_seeds"]]


def _mean(values: Iterable[float]) -> float:
    return float(np.mean(np.asarray(list(values), dtype=float)))


def _paired_ui_analysis(
    dataset: Mapping[str, Any],
    analysis_seed: int,
    n_bootstrap: int,
) -> List[Dict[str, Any]]:
    """Compute paired seed-level UI statistics for every target."""
    target_results = []
    target_sequences = np.random.SeedSequence(analysis_seed).spawn(len(dataset["target_haer_grid"]))
    for target, target_sequence in zip(dataset["target_haer_grid"], target_sequences):
        b1 = _rows(dataset, target, "B1")
        b2 = _rows(dataset, target, "B2")
        summary = summarize_paired_ui(
            [row["unnecessary_intervention_rate"] for row in b1],
            [row["unnecessary_intervention_rate"] for row in b2],
            rng=np.random.default_rng(target_sequence),
            n_bootstrap=n_bootstrap,
        )
        target_results.append({
            "target_haer": target,
            "n_seeds": summary.n_seeds,
            "b1_mean_ui": summary.mean_b1,
            "b2_mean_ui": summary.mean_b2,
            "mean_paired_difference_b2_minus_b1": summary.mean_difference,
            "bootstrap_ci_lower": summary.ci_lower,
            "bootstrap_ci_upper": summary.ci_upper,
            "wilcoxon_statistic": summary.wilcoxon_statistic,
            "wilcoxon_pvalue": summary.wilcoxon_pvalue,
            "direction": interpret_ui_difference(summary.ci_lower, summary.ci_upper),
        })
    return target_results


def _safety_analysis(dataset: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Summarize achieved HAER by target and method at seed level."""
    results = []
    for target in dataset["target_haer_grid"]:
        for method in ("B1", "B2"):
            rows = _rows(dataset, target, method)
            achieved = [row["evaluation_achieved_haer"] for row in rows]
            mean_achieved = _mean(achieved)
            results.append({
                "target_haer": target,
                "method": method,
                "n_seeds": len(rows),
                "mean_achieved_haer": mean_achieved,
                "absolute_deviation_from_target": abs(mean_achieved - target),
                "ci_supported": False,
                "ci_note": "No generic seed-level mean bootstrap utility exists in src.stats.py; no safety CI was introduced here.",
            })
    return results


def _curve_analysis(dataset: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create descriptive operating points from seed-level means."""
    results = []
    for target in dataset["target_haer_grid"]:
        for method in ("B1", "B2"):
            rows = _rows(dataset, target, method)
            results.append({
                "target_haer": target,
                "method": method,
                "mean_achieved_haer": _mean(row["evaluation_achieved_haer"] for row in rows),
                "mean_ui": _mean(row["unnecessary_intervention_rate"] for row in rows),
                "mean_legitimate_action_completion": _mean(row["legitimate_action_completion"] for row in rows),
                "mean_false_verify": _mean(row["false_verify_rate"] for row in rows),
                "mean_false_reject": _mean(row["false_reject_rate"] for row in rows),
            })
    return results


def _matched_safety(curve_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Match observed mean operating points by achieved HAER within tolerance."""
    points = [
        OperatingPoint(
            method=row["method"],
            target_haer=row["target_haer"],
            theta_reject=0.0,
            theta_verify=1.0,
            achieved_haer=row["mean_achieved_haer"],
            unnecessary_intervention=row["mean_ui"],
            false_verify=row["mean_false_verify"],
            false_reject=row["mean_false_reject"],
            legitimate_action_completion=row["mean_legitimate_action_completion"],
        )
        for row in curve_rows
    ]
    b1 = [point for point in points if point.method == "B1"]
    b2 = [point for point in points if point.method == "B2"]
    matches = match_by_achieved_haer(b1, b2, tolerance=MATCH_TOLERANCE)
    differences = [match.b2.unnecessary_intervention - match.b1.unnecessary_intervention for match in matches]
    if not matches:
        classification = "inconclusive"
    elif all(difference < 0 for difference in differences):
        classification = "B2 lower UI at all matched observed points"
    elif all(difference > 0 for difference in differences):
        classification = "B1 lower UI at all matched observed points"
    else:
        classification = "mixed/inconclusive"
    return {
        "method": "observed-point achieved-HAER matching",
        "tolerance": MATCH_TOLERANCE,
        "interpolation": "none; only observed operating points within tolerance were matched",
        "n_matches": len(matches),
        "classification": classification,
        "matches": [
            {
                "target_b1": match.b1.target_haer,
                "target_b2": match.b2.target_haer,
                "achieved_haer_b1": match.b1.achieved_haer,
                "achieved_haer_b2": match.b2.achieved_haer,
                "ui_b1": match.b1.unnecessary_intervention,
                "ui_b2": match.b2.unnecessary_intervention,
                "ui_difference_b2_minus_b1": match.b2.unnecessary_intervention - match.b1.unnecessary_intervention,
            }
            for match in matches
        ],
    }


def _cold_start_analysis(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate descriptive cold-start cell metrics across seeds and targets."""
    cold_counts = [count for count in dataset["evidence_counts"] if count <= 5]
    rows = []
    profile_rows = []
    for method in ("B1", "B2"):
        for evidence_count in cold_counts:
            cells = [
                cell for row in dataset["rows"] if row["method"] == method
                for cell in row["cell_metrics"] if cell["evidence_count"] == evidence_count
            ]
            rows.append({
                "method": method,
                "evidence_count": evidence_count,
                "n_seed_target_cells": len(cells),
                "mean_ui": _mean(cell["unnecessary_intervention"] for cell in cells),
                "mean_haer": _mean(cell["haer"] for cell in cells),
                "mean_legitimate_action_completion": _mean(cell["legitimate_completion"] for cell in cells),
            })
            for profile in sorted({cell["profile"] for cell in cells}):
                profile_cells = [cell for cell in cells if cell["profile"] == profile]
                profile_rows.append({
                    "method": method,
                    "profile": profile,
                    "evidence_count": evidence_count,
                    "n_seed_target_cells": len(profile_cells),
                    "mean_ui": _mean(cell["unnecessary_intervention"] for cell in profile_cells),
                    "mean_haer": _mean(cell["haer"] for cell in profile_cells),
                    "mean_legitimate_action_completion": _mean(cell["legitimate_completion"] for cell in profile_cells),
                })
    return {"overall": rows, "by_profile": profile_rows, "cold_start_evidence_counts": cold_counts}


def _profile_evidence_analysis(dataset: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Produce descriptive method/profile/evidence mechanism summaries."""
    rows = []
    profiles = sorted(dataset["profiles"])
    for method in ("B1", "B2"):
        for profile in profiles:
            for evidence_count in dataset["evidence_counts"]:
                cells = [
                    cell for row in dataset["rows"] if row["method"] == method
                    for cell in row["cell_metrics"]
                    if cell["profile"] == profile and cell["evidence_count"] == evidence_count
                ]
                rows.append({
                    "method": method,
                    "profile": profile,
                    "evidence_count": evidence_count,
                    "n_seed_target_cells": len(cells),
                    "mean_haer": _mean(cell["haer"] for cell in cells),
                    "mean_ui": _mean(cell["unnecessary_intervention"] for cell in cells),
                    "mean_false_verify": _mean(cell["false_verify"] for cell in cells),
                    "mean_false_reject": _mean(cell["false_reject"] for cell in cells),
                    "mean_legitimate_completion": _mean(cell["legitimate_completion"] for cell in cells),
                })
    return rows


def _interpret(ui_results: Sequence[Mapping[str, Any]], matched: Mapping[str, Any]) -> str:
    """Apply the requested conservative hypothesis interpretation framework."""
    if matched["n_matches"] == 0:
        return "Inconclusive: matched-safety comparisons cannot be established reliably."
    if matched["classification"] == "B2 lower UI at all matched observed points" and all(row["bootstrap_ci_upper"] < 0 for row in ui_results):
        return "H1 supported: B2 has a statistically significant lower unnecessary-intervention rate than B1 at matched safety, with consistent curve-level evidence."
    if all(row["bootstrap_ci_lower"] <= 0 <= row["bootstrap_ci_upper"] for row in ui_results):
        return "H3/null-consistent: The paired B1/B2 difference is not statistically significant and the confidence interval includes zero."
    if matched["classification"] == "B1 lower UI at all matched observed points" and all(row["bootstrap_ci_lower"] > 0 for row in ui_results):
        return "Evidence against H1: B2 has significantly higher unnecessary intervention at matched safety."
    return "Inconclusive: confidence intervals or matched-safety comparisons do not support a definitive directional conclusion."


def analyze_dataset(
    dataset_path: str | Path = DEFAULT_DATASET,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
) -> Dict[str, Any]:
    """Run all downstream analyses and return a deterministic result object."""
    dataset = load_dataset(dataset_path)
    ui_results = _paired_ui_analysis(dataset, analysis_seed, n_bootstrap)
    safety_results = _safety_analysis(dataset)
    curve_results = _curve_analysis(dataset)
    matched = _matched_safety(curve_results)
    cold_start = _cold_start_analysis(dataset)
    profile_evidence = _profile_evidence_analysis(dataset)
    return {
        "dataset_sha256": dataset_sha256(dataset_path),
        "analysis_seed": analysis_seed,
        "bootstrap_n": n_bootstrap,
        "bootstrap_confidence_level": 0.95,
        "statistical_unit": "seed",
        "methods": ["B1", "B2"],
        "evaluation_seeds": list(dataset["evaluation_seeds"]),
        "targets": list(dataset["target_haer_grid"]),
        "ui_by_target": ui_results,
        "safety_by_target": safety_results,
        "curve": curve_results,
        "matched_safety": matched,
        "cold_start": cold_start,
        "profile_evidence": profile_evidence,
        "hypothesis_interpretation": _interpret(ui_results, matched),
        "limitations": [
            "Safety summaries do not include CIs because src.stats.py exposes only paired-difference bootstrap utilities.",
            "Matched safety uses observed mean operating points within a pre-specified 0.005 achieved-HAER tolerance and no interpolation.",
            "Cell and curve summaries are descriptive and do not treat cells or actions as independent inferential units.",
            "B2 utility effects are interpreted only after threshold recalibration; B2 is pointwise no greater than B1 at identical thresholds.",
        ],
    }


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write deterministic analysis JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_text(analysis: Mapping[str, Any]) -> str:
    """Render a concise human-readable analysis report."""
    lines = [
        "Primary dataset analysis",
        f"Dataset SHA-256: {analysis['dataset_sha256']}",
        f"Statistical unit: {analysis['statistical_unit']}",
        f"Bootstrap: paired seed differences, percentile 95% CI, n={analysis['bootstrap_n']}, seed={analysis['analysis_seed']}",
        "",
        "Per-target paired UI results",
    ]
    for row in analysis["ui_by_target"]:
        lines.append(
            f"target={row['target_haer']:.2f} n={row['n_seeds']} B1_UI={row['b1_mean_ui']:.6f} "
            f"B2_UI={row['b2_mean_ui']:.6f} diff={row['mean_paired_difference_b2_minus_b1']:.6f} "
            f"CI=[{row['bootstrap_ci_lower']:.6f}, {row['bootstrap_ci_upper']:.6f}] "
            f"Wilcoxon_p={row['wilcoxon_pvalue']:.6g}"
        )
    lines.extend(["", "Matched safety", json.dumps(analysis["matched_safety"], sort_keys=True), "", "Hypothesis interpretation", analysis["hypothesis_interpretation"], "", "Limitations"])
    lines.extend(f"- {limitation}" for limitation in analysis["limitations"])
    return "\n".join(lines) + "\n"


def _make_figures(analysis: Mapping[str, Any], directory: str | Path = DEFAULT_FIGURES) -> List[str]:
    """Create deterministic publication-style PNG figures."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "font.size": 10})
    paths = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, color in (("B1", "#1f4e79"), ("B2", "#b24745")):
        rows = [row for row in analysis["curve"] if row["method"] == method]
        ax.plot([row["mean_achieved_haer"] for row in rows], [row["mean_ui"] for row in rows], "o-", label=method, color=color)
    ax.set_xlabel("Mean achieved HAER")
    ax.set_ylabel("Mean unnecessary-intervention rate")
    ax.set_title("Primary safety–utility operating curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = directory / "safety_utility_curve.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    targets = [row["target_haer"] for row in analysis["ui_by_target"]]
    difference = [row["mean_paired_difference_b2_minus_b1"] for row in analysis["ui_by_target"]]
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(targets, difference, "o-", color="#6a3d9a")
    ax.set_xlabel("Target HAER")
    ax.set_ylabel("B2 UI − B1 UI")
    ax.set_title("Paired UI difference by target")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = directory / "ui_difference_by_target.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, color in (("B1", "#1f4e79"), ("B2", "#b24745")):
        rows = [row for row in analysis["cold_start"]["overall"] if row["method"] == method]
        ax.plot([row["evidence_count"] for row in rows], [row["mean_ui"] for row in rows], "o-", label=method, color=color)
    ax.set_xlabel("Evidence count (cold-start subset)")
    ax.set_ylabel("Mean unnecessary-intervention rate")
    ax.set_title("Cold-start UI comparison")
    ax.set_xticks([0, 1, 2, 3, 5])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = directory / "cold_start_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))
    return paths


def run_analysis(
    dataset_path: str | Path = DEFAULT_DATASET,
    analysis_path: str | Path = DEFAULT_ANALYSIS,
    summary_path: str | Path = DEFAULT_SUMMARY,
    figures_directory: str | Path = DEFAULT_FIGURES,
) -> Dict[str, Any]:
    """Run and write the complete deterministic downstream analysis."""
    analysis = analyze_dataset(dataset_path)
    analysis["figure_paths"] = _make_figures(analysis, figures_directory)
    _write_json(analysis, analysis_path)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(_summary_text(analysis), encoding="utf-8")
    return analysis


if __name__ == "__main__":
    result = run_analysis()
    print(f"Analysis written to {DEFAULT_ANALYSIS}")
    print(f"Summary written to {DEFAULT_SUMMARY}")
    print(result["hypothesis_interpretation"])
