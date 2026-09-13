"""Integrity checks for the frozen primary evaluation output."""

import json
from collections import defaultdict
from pathlib import Path

import pytest


EVALUATION_PATH = Path(__file__).parents[1] / "results" / "primary" / "evaluation.json"


def load_evaluation():
    assert EVALUATION_PATH.exists()
    return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))


def grouped_rows(data):
    groups = defaultdict(list)
    for row in data["rows"]:
        groups[(row["method"], row["target_haer"])].append(row)
    return groups


def test_primary_evaluation_has_expected_row_count_and_methods():
    data = load_evaluation()

    assert len(data["rows"]) == 280
    assert {row["method"] for row in data["rows"]} == {"B1", "B2"}


def test_each_method_has_complete_targets_and_seed_counts():
    data = load_evaluation()
    groups = grouped_rows(data)
    expected_targets = set(data["target_haer_grid"])
    expected_seeds = set(data["evaluation_seeds"])

    for method in ("B1", "B2"):
        method_targets = {target for group_method, target in groups if group_method == method}
        assert method_targets == expected_targets
        assert len(method_targets) == 7

        for target in expected_targets:
            rows = groups[(method, target)]
            assert len(rows) == 20
            assert {row["seed"] for row in rows} == expected_seeds
            threshold_pairs = {
                (row["theta_reject"], row["theta_verify"])
                for row in rows
            }
            assert len(threshold_pairs) == 1
            theta_reject, theta_verify = next(iter(threshold_pairs))
            assert theta_reject < theta_verify


def test_target_values_match_top_level_grid():
    data = load_evaluation()
    assert {row["target_haer"] for row in data["rows"]} == set(data["target_haer_grid"])


@pytest.mark.parametrize(
    ("method", "target", "expected_mean"),
    [
        ("B1", 0.01, 0.9344627007050115),
        ("B1", 0.02, 0.7925631793648946),
        ("B1", 0.03, 0.9344627007050115),
        ("B1", 0.05, 0.7925631793648946),
        ("B2", 0.01, 0.91612328872619),
        ("B2", 0.05, 0.91612328872619),
        ("B2", 0.08, 0.91612328872619),
    ],
)
def test_repeated_ui_values_are_legitimate_aggregate_values(method, target, expected_mean):
    data = load_evaluation()
    rows = [
        row
        for row in data["rows"]
        if row["method"] == method and row["target_haer"] == target
    ]

    assert len(rows) == 20
    actual_mean = sum(row["unnecessary_intervention_rate"] for row in rows) / len(rows)
    assert actual_mean == pytest.approx(expected_mean)
