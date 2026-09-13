"""Create publication figures from frozen downstream analysis outputs.

This script is analysis-only: it reads existing JSON artifacts, performs no
new statistical inference, and never modifies experiment results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ANALYSIS = ROOT / "results/primary/analysis.json"
SENSITIVITY_ANALYSIS = ROOT / "results/sensitivity/analysis.json"
OUTPUT_DIR = ROOT / "figures/paper"
TARGETS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
COLD_START_COUNTS = [0, 1, 2, 3, 5]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required analysis input does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON analysis input: {path}") from error


def _require_fields(mapping: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = set(fields) - set(mapping)
    if missing:
        raise ValueError(f"{label} missing required fields: {sorted(missing)}")


def _validate_primary(analysis: Mapping[str, Any]) -> None:
    _require_fields(analysis, ("curve", "ui_by_target", "cold_start"), "primary analysis")
    if analysis.get("statistical_unit") != "seed":
        raise ValueError("primary analysis must declare seed as the statistical unit")
    if len(analysis["targets"]) != len(TARGETS) or analysis["targets"] != TARGETS:
        raise ValueError("primary analysis target grid does not match the expected grid")
    curve_keys = {(row["target_haer"], row["method"]) for row in analysis["curve"]}
    if curve_keys != {(target, method) for target in TARGETS for method in ("B1", "B2")}:
        raise ValueError("primary curve does not contain the complete B1/B2 target grid")
    for row in analysis["ui_by_target"]:
        _require_fields(
            row,
            ("target_haer", "mean_paired_difference_b2_minus_b1", "bootstrap_ci_lower", "bootstrap_ci_upper"),
            "primary UI analysis row",
        )
    cold_rows = analysis["cold_start"].get("overall")
    if not isinstance(cold_rows, list):
        raise ValueError("primary cold-start analysis has no overall rows")
    if {row["evidence_count"] for row in cold_rows} != set(COLD_START_COUNTS):
        raise ValueError("primary cold-start analysis does not contain exactly 0, 1, 2, 3, 5")


def _validate_sensitivity(analysis: Mapping[str, Any]) -> None:
    _require_fields(analysis, ("conditions", "condition_ids"), "sensitivity analysis")
    expected_ids = {
        "baseline",
        "prior_beta_2_2",
        "z_0_5",
        "z_1_5",
        "p_catch_0_6",
        "p_catch_0_95",
        "threshold_step_0_025",
    }
    if set(analysis["condition_ids"]) != expected_ids:
        raise ValueError("sensitivity analysis does not contain the expected conditions")
    for condition in analysis["conditions"]:
        if len(condition["target_summary"]) != len(TARGETS):
            raise ValueError(f"sensitivity condition has incomplete target summary: {condition['condition']['id']}")
        for row in condition["target_summary"]:
            _require_fields(
                row,
                ("target_haer", "b2_minus_b1_ui"),
                f"sensitivity row for {condition['condition']['id']}",
            )


def _primary_curve_rows(analysis: Mapping[str, Any], method: str):
    rows = [row for row in analysis["curve"] if row["method"] == method]
    return sorted(rows, key=lambda row: row["target_haer"])


def _percent_axes(ax) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig1_safety_utility(analysis: Mapping[str, Any]) -> Path:
    """Plot discrete B1/B2 points using mean achieved HAER on the x-axis."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, marker in (("B1", "o"), ("B2", "s")):
        rows = _primary_curve_rows(analysis, method)
        ax.plot(
            [row["mean_achieved_haer"] for row in rows],
            [row["mean_ui"] for row in rows],
            marker=marker,
            linestyle="None",
            label=method,
        )
    ax.set_xlabel("Mean achieved HAER")
    ax.set_ylabel("Mean unnecessary-intervention rate")
    _percent_axes(ax)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig1_safety_utility_frontier.png"
    _save_figure(fig, path)
    return path


def make_fig2_ui_difference(analysis: Mapping[str, Any]) -> Path:
    """Plot B2-minus-B1 UI differences and existing paired bootstrap intervals."""
    rows = sorted(analysis["ui_by_target"], key=lambda row: row["target_haer"])
    x = [row["target_haer"] for row in rows]
    y = [row["mean_paired_difference_b2_minus_b1"] for row in rows]
    lower = [row["bootstrap_ci_lower"] for row in rows]
    upper = [row["bootstrap_ci_upper"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0.0, linewidth=0.8)
    ax.errorbar(
        x,
        y,
        yerr=[[(value - low) for value, low in zip(y, lower)], [(high - value) for value, high in zip(y, upper)]],
        fmt="o-",
        capsize=3,
        label="B2 − B1",
    )
    ax.set_xlabel("Target HAER")
    ax.set_ylabel("B2 − B1 unnecessary-intervention rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    _percent_axes(ax)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig2_ui_difference.png"
    _save_figure(fig, path)
    return path


def make_fig3_sparse_evidence(analysis: Mapping[str, Any]) -> Path:
    """Plot descriptive primary UI for the retained sparse-evidence subset."""
    rows = analysis["cold_start"]["overall"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, marker in (("B1", "o"), ("B2", "s")):
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["evidence_count"],
        )
        ax.plot(
            [row["evidence_count"] for row in method_rows],
            [row["mean_ui"] for row in method_rows],
            marker=marker,
            linestyle="None",
            label=method,
        )
    ax.set_xlabel("Evidence count (sparse-evidence / cold-start subset)")
    ax.set_ylabel("Mean unnecessary-intervention rate")
    ax.set_xticks(COLD_START_COUNTS)
    _percent_axes(ax)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig3_sparse_evidence.png"
    _save_figure(fig, path)
    return path


def make_fig4_sensitivity(sensitivity: Mapping[str, Any]) -> Path:
    """Plot descriptive B2-minus-B1 UI differences for all conditions."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for condition in sensitivity["conditions"]:
        rows = sorted(condition["target_summary"], key=lambda row: row["target_haer"])
        label = condition["condition"]["id"]
        ax.plot(
            [row["target_haer"] for row in rows],
            [row["b2_minus_b1_ui"] for row in rows],
            marker="o",
            linestyle="-",
            label=label,
        )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Target HAER")
    ax.set_ylabel("B2 − B1 unnecessary-intervention rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    _percent_axes(ax)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig4_sensitivity.png"
    _save_figure(fig, path)
    return path


def main() -> None:
    """Create all paper figures from existing frozen analysis artifacts."""
    primary = _load_json(PRIMARY_ANALYSIS)
    sensitivity = _load_json(SENSITIVITY_ANALYSIS)
    _validate_primary(primary)
    _validate_sensitivity(sensitivity)
    make_fig1_safety_utility(primary)
    make_fig2_ui_difference(primary)
    make_fig3_sparse_evidence(primary)
    make_fig4_sensitivity(sensitivity)


if __name__ == "__main__":
    main()
