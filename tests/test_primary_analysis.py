"""Tiny downstream analysis tests; no experiment execution."""

import copy
import json
from pathlib import Path

from experiments.analyze_primary import analyze_dataset, dataset_sha256


def synthetic_dataset():
    targets = [0.05, 0.10]
    seeds = [11, 12]
    rows = []
    profiles = ["malicious", "reliable"]
    evidence_counts = [0, 1, 5]
    for target in targets:
        for seed in seeds:
            for method, offset in (("B1", 0.0), ("B2", -0.05)):
                ui = 0.4 + (seed - 11) * 0.02 + offset
                cells = []
                for profile_index, profile in enumerate(profiles):
                    for evidence_count in evidence_counts:
                        cells.append({
                            "seed": seed,
                            "profile": profile,
                            "evidence_count": evidence_count,
                            "haer": 0.1 + profile_index * 0.1,
                            "unnecessary_intervention": ui,
                            "false_verify": ui / 2,
                            "false_reject": ui / 2,
                            "legitimate_completion": 1.0 - ui,
                        })
                rows.append({
                    "seed": seed,
                    "method": method,
                    "target_haer": target,
                    "calibration_achieved_haer": target + 0.001,
                    "evaluation_achieved_haer": target + 0.01 + offset / 10,
                    "theta_reject": 0.2,
                    "theta_verify": 0.8,
                    "unnecessary_intervention_rate": ui,
                    "false_verify_rate": ui / 2,
                    "false_reject_rate": ui / 2,
                    "legitimate_action_completion": 1.0 - ui,
                    "cold_start_evidence_counts": [0, 1, 5],
                    "cell_metrics": cells,
                })
    return {
        "experiment_name": "synthetic",
        "evaluation_seeds": seeds,
        "target_haer_grid": targets,
        "evidence_counts": evidence_counts,
        "profiles": {"malicious": 0.8, "reliable": 0.05},
        "rows": rows,
        "statistical_unit": "seed",
    }


def test_seed_observations_and_pair_alignment(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(synthetic_dataset()), encoding="utf-8")
    analysis = analyze_dataset(path, analysis_seed=7, n_bootstrap=100)
    assert all(row["n_seeds"] == 2 for row in analysis["ui_by_target"])
    assert all(row["n_seeds"] == 2 for row in analysis["safety_by_target"])
    assert all(row["mean_paired_difference_b2_minus_b1"] < 0 for row in analysis["ui_by_target"])


def test_target_achieved_and_cold_start_detail(tmp_path):
    path = tmp_path / "evaluation.json"
    dataset = synthetic_dataset()
    path.write_text(json.dumps(dataset), encoding="utf-8")
    analysis = analyze_dataset(path, analysis_seed=7, n_bootstrap=100)
    assert analysis["curve"][0]["target_haer"] != analysis["curve"][0]["mean_achieved_haer"]
    assert analysis["cold_start"]["cold_start_evidence_counts"] == [0, 1, 5]
    assert len(analysis["cold_start"]["by_profile"]) == 2 * 2 * 3
    assert len(analysis["profile_evidence"]) == 2 * 2 * 3


def test_analysis_is_deterministic_and_does_not_mutate_raw_dataset(tmp_path):
    path = tmp_path / "evaluation.json"
    dataset = synthetic_dataset()
    path.write_text(json.dumps(dataset, sort_keys=True), encoding="utf-8")
    before = path.read_bytes()
    first = analyze_dataset(path, analysis_seed=99, n_bootstrap=100)
    second = analyze_dataset(path, analysis_seed=99, n_bootstrap=100)
    assert first == second
    assert path.read_bytes() == before
    assert dataset_sha256(path) == __import__("hashlib").sha256(before).hexdigest()


def test_missing_seed_target_method_row_is_rejected(tmp_path):
    path = tmp_path / "evaluation.json"
    dataset = synthetic_dataset()
    dataset["rows"].pop()
    path.write_text(json.dumps(dataset), encoding="utf-8")
    try:
        analyze_dataset(path, n_bootstrap=100)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("missing paired row was not rejected")
