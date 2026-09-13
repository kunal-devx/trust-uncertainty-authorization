"""Descriptive decision decomposition for the frozen primary experiment.

This diagnostic reconstructs the frozen evaluation realizations and thresholds.
It does not recalibrate, rerun the primary experiment, or perform inference.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_calibration import load_config
from src.authorization import authorize
from src.population import generate_population
from src.trust_model import lcb_score, trust_mean, trust_variance
from src.verify_sim import verify_action

PRIMARY_EVALUATION = ROOT / "results" / "primary" / "evaluation.json"
PRIMARY_CALIBRATION = ROOT / "results" / "primary" / "calibration.json"
OUTPUT_JSON = ROOT / "results" / "diagnostic" / "decision_decomposition.json"
OUTPUT_SUMMARY = ROOT / "results" / "diagnostic" / "decision_decomposition_summary.txt"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _realization(config: Mapping[str, Any], seed: int) -> Tuple[List[Dict[str, Any]], List[List[bool | None]]]:
    """Reconstruct one frozen population and shared harmful VERIFY realization."""
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
    verify_child_ss = verify_ss.spawn(1)[0]
    verify_rng = np.random.default_rng(verify_child_ss)
    verify_outcomes = [
        [
            verify_action(True, config["p_catch"], verify_rng) if is_harmful else None
            for is_harmful in agent["measurement_actions"]
        ]
        for agent in population
    ]
    return population, verify_outcomes


def _thresholds_by_target(calibration: Mapping[str, Any], targets: Sequence[float]) -> Dict[Tuple[float, str], Tuple[float, float]]:
    """Index frozen calibration thresholds and reject missing rows."""
    result = {}
    for target in targets:
        for method in ("B1", "B2"):
            matches = [
                row for row in calibration["rows"]
                if row["target_haer"] == target and row["method"] == method
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one frozen calibration row for {method} target {target}")
            thresholds = (matches[0]["theta_reject"], matches[0]["theta_verify"])
            if thresholds[0] >= thresholds[1]:
                raise ValueError("frozen calibration contains invalid threshold ordering")
            result[(target, method)] = thresholds
    return result


def _decision_score(agent: Mapping[str, Any], method: str, z: float) -> Tuple[float, float, float | None]:
    trust = trust_mean(agent["alpha"], agent["beta"])
    if method == "B1":
        return trust, 0.0, None
    variance = trust_variance(agent["alpha"], agent["beta"])
    return lcb_score(trust, variance, z=z, clip=True), variance, lcb_score(trust, variance, z=z, clip=True)


def _empty_counts() -> Dict[str, int]:
    return {
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


def _finalize_counts(counts: Mapping[str, int]) -> Dict[str, Any]:
    result = dict(counts)
    total = counts["total_actions"]
    benign = counts["benign_actions"]
    harmful = counts["harmful_actions"]
    result.update({
        "allow_fraction": counts["allow"] / total if total else 0.0,
        "verify_fraction": counts["verify"] / total if total else 0.0,
        "reject_fraction": counts["reject"] / total if total else 0.0,
        "benign_allow_fraction": counts["benign_allow"] / benign if benign else 0.0,
        "benign_verify_fraction": counts["benign_verify"] / benign if benign else 0.0,
        "benign_reject_fraction": counts["benign_reject"] / benign if benign else 0.0,
        "harmful_allow_fraction": counts["harmful_allow"] / harmful if harmful else 0.0,
        "harmful_verify_fraction": counts["harmful_verify"] / harmful if harmful else 0.0,
        "harmful_reject_fraction": counts["harmful_reject"] / harmful if harmful else 0.0,
        "ui": (counts["benign_verify"] + counts["benign_reject"]) / benign if benign else 0.0,
        "haer": counts["harmful_executed"] / harmful if harmful else 0.0,
        "legitimate_completion": counts["benign_executed"] / benign if benign else 0.0,
        "harmful_execution_fraction": counts["harmful_executed"] / harmful if harmful else 0.0,
    })
    return result


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _trace_record(
    seed: int,
    method: str,
    target: float,
    profile: str,
    evidence_count: int,
    actions: Sequence[bool],
    decisions: Sequence[str],
    executed: Sequence[bool],
    verify_outcomes: Sequence[bool | None],
) -> Dict[str, Any]:
    """Store compact per-action traces; strings preserve every action outcome."""
    return {
        "seed": seed,
        "method": method,
        "target_haer": target,
        "profile": profile,
        "evidence_count": evidence_count,
        "is_harmful": "".join("1" if value else "0" for value in actions),
        "decision": "".join(decisions),
        "executed": "".join("1" if value else "0" for value in executed),
        "verify_outcome": "".join("." if value is None else "1" if value else "0" for value in verify_outcomes),
    }


def build_diagnostic(
    config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build diagnostic decompositions from frozen config, data, and thresholds."""
    targets = list(config["target_haer_grid"])
    thresholds = _thresholds_by_target(calibration, targets)
    seed_target_method: Dict[Tuple[int, float, str], Dict[str, int]] = {}
    seed_target_cell: Dict[Tuple[int, float, str, str, int], Dict[str, int]] = {}
    traces = []

    for seed in config["evaluation_seeds"]:
        population, verify_outcomes = _realization(config, seed)
        for target in targets:
            for method in ("B1", "B2"):
                theta_reject, theta_verify = thresholds[(target, method)]
                aggregate = _empty_counts()
                for agent, outcomes in zip(population, verify_outcomes):
                    trust = trust_mean(agent["alpha"], agent["beta"])
                    score = trust if method == "B1" else lcb_score(
                        trust,
                        trust_variance(agent["alpha"], agent["beta"]),
                        z=config["z"],
                        clip=True,
                    )
                    decision = authorize(score, theta_reject, theta_verify)
                    cell_key = (seed, target, method, agent["profile"], agent["evidence_count"])
                    cell = seed_target_cell.setdefault(cell_key, _empty_counts())
                    actions = agent["measurement_actions"]
                    decisions = []
                    action_executed = []
                    for is_harmful, verify_outcome in zip(actions, outcomes):
                        executed = (
                            True if decision == "ALLOW"
                            else (bool(verify_outcome) if decision == "VERIFY" and is_harmful else True)
                            if decision == "VERIFY"
                            else False
                        )
                        decisions.append(decision[0])
                        action_executed.append(executed)
                        for counts in (aggregate, cell):
                            counts["total_actions"] += 1
                            counts["harmful_actions" if is_harmful else "benign_actions"] += 1
                            counts[decision.lower()] += 1
                            counts[f"{'harmful' if is_harmful else 'benign'}_{decision.lower()}"] += 1
                            counts[f"{'harmful' if is_harmful else 'benign'}_{'executed' if executed else 'blocked'}"] += 1

                    traces.append(_trace_record(
                        seed, method, target, agent["profile"], agent["evidence_count"],
                        actions, decisions, action_executed,
                        outcomes,
                    ))
                seed_target_method[(seed, target, method)] = _finalize_counts(aggregate)

    cell_rows = []
    for key in sorted(seed_target_cell, key=lambda item: (item[0], item[1], item[2], item[3], item[4])):
        seed, target, method, profile, evidence_count = key
        cell_rows.append({
            "seed": seed,
            "target_haer": target,
            "method": method,
            "profile": profile,
            "evidence_count": evidence_count,
            **_finalize_counts(seed_target_cell[key]),
        })

    aggregate_rows = []
    for target in targets:
        for method in ("B1", "B2"):
            values = [seed_target_method[(seed, target, method)] for seed in config["evaluation_seeds"]]
            aggregate_rows.append({
                "target_haer": target,
                "method": method,
                "n_seeds": len(values),
                **{
                    metric: float(np.mean([value[metric] for value in values]))
                    for metric in (
                        "allow_fraction", "verify_fraction", "reject_fraction",
                        "benign_allow_fraction", "benign_verify_fraction", "benign_reject_fraction",
                        "harmful_allow_fraction", "harmful_verify_fraction", "harmful_reject_fraction",
                        "harmful_execution_fraction",
                    )
                },
            })

    differences = []
    for target in targets:
        b1 = next(row for row in aggregate_rows if row["target_haer"] == target and row["method"] == "B1")
        b2 = next(row for row in aggregate_rows if row["target_haer"] == target and row["method"] == "B2")
        differences.append({
            "target_haer": target,
            **{
                metric: b2[metric] - b1[metric]
                for metric in b1
                if metric not in ("target_haer", "method", "n_seeds")
            },
        })

    special_targets = []
    for target in targets:
        if target not in (0.01, 0.03, 0.05, 0.08, 0.10, 0.15):
            continue
        compact = {"target_haer": target, "thresholds": {}}
        for method in ("B1", "B2"):
            compact["thresholds"][method] = thresholds[(target, method)]
            compact[method] = {
                key: aggregate_rows[next(i for i, row in enumerate(aggregate_rows) if row["target_haer"] == target and row["method"] == method)][key]
                for key in (
                    "benign_allow_fraction", "benign_verify_fraction", "benign_reject_fraction",
                    "harmful_allow_fraction", "harmful_verify_fraction", "harmful_reject_fraction",
                    "harmful_execution_fraction",
                )
            }
        b1 = compact["B1"]
        b2 = compact["B2"]
        compact["B2_minus_B1"] = {
            key: b2[key] - b1[key]
            for key in b1
        }
        special_targets.append(compact)

    evaluation_keys = {(row["seed"], row["target_haer"], row["method"]) for row in evaluation["rows"]}
    cold_start_rows = []
    for method in ("B1", "B2"):
        for evidence_count in (0, 1, 2, 3, 5):
            cells = [
                row for row in cell_rows
                if row["method"] == method and row["evidence_count"] == evidence_count
            ]
            cold_start_rows.append({
                "method": method,
                "evidence_count": evidence_count,
                "ui": _mean_or_none([row["ui"] for row in cells]),
                "benign_allow_fraction": _mean_or_none([row["benign_allow_fraction"] for row in cells]),
                "benign_verify_fraction": _mean_or_none([row["benign_verify_fraction"] for row in cells]),
                "benign_reject_fraction": _mean_or_none([row["benign_reject_fraction"] for row in cells]),
                "harmful_execution_fraction": _mean_or_none([row["harmful_execution_fraction"] for row in cells]),
                "legitimate_completion": _mean_or_none([row["legitimate_completion"] for row in cells]),
            })
    return {
        "diagnostic_experiment": "primary_decision_decomposition",
        "configuration_source": "configs/primary.yaml",
        "evaluation_source": "results/primary/evaluation.json",
        "calibration_source": "results/primary/calibration.json",
        "evaluation_sha256": sha256(PRIMARY_EVALUATION),
        "calibration_sha256": sha256(PRIMARY_CALIBRATION),
        "evaluation_seeds": list(config["evaluation_seeds"]),
        "methods": ["B1", "B2"],
        "targets": targets,
        "thresholds": {f"{target:.2f}": {method: thresholds[(target, method)] for method in ("B1", "B2")} for target in targets},
        "K": config["K"],
        "M": config["M"],
        "profiles": dict(config["profiles"]),
        "evidence_counts": list(config["evidence_counts"]),
        "alpha0": config["alpha0"],
        "beta0": config["beta0"],
        "z": config["z"],
        "p_catch": config["p_catch"],
        "rng_architecture": "SeedSequence(seed).spawn(2); population child plus explicit VERIFY child SeedSequence",
        "trace_encoding": {
            "is_harmful": "1=harmful, 0=benign",
            "decision": "A=ALLOW, V=VERIFY, R=REJECT",
            "executed": "1=executed, 0=blocked",
            "verify_outcome": "1=VERIFY outcome executed, 0=VERIFY outcome blocked, .=benign/no VERIFY draw",
        },
        "statistical_unit": "seed",
        "inference_performed": False,
        "primary_row_keys_verified": len(evaluation_keys),
        "aggregate_seed_method_target": [
            {"seed": seed, "target_haer": target, "method": method, **seed_target_method[(seed, target, method)]}
            for seed in config["evaluation_seeds"] for target in targets for method in ("B1", "B2")
        ],
        "aggregate_seed_method_target_profile_evidence": cell_rows,
        "aggregate_target_method": aggregate_rows,
        "b2_minus_b1_target_method": differences,
        "cold_start": cold_start_rows,
        "special_targets": special_targets,
        "action_traces": traces,
    }


def _summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "Primary decision decomposition diagnostic",
        f"Evaluation SHA-256: {payload['evaluation_sha256']}",
        f"Calibration SHA-256: {payload['calibration_sha256']}",
        "Statistical unit: seed; inference performed: false",
        "",
        "Target-level decomposition (B2 minus B1 fractions)",
    ]
    for row in payload["b2_minus_b1_target_method"]:
        lines.append(
            f"target={row['target_haer']:.2f} benign_UI={row['benign_verify_fraction'] + row['benign_reject_fraction'] - 0:.6f} "
            f"harmful_execution={row['harmful_execution_fraction']:.6f} "
            f"benign_allow={row['benign_allow_fraction']:.6f} "
            f"benign_verify={row['benign_verify_fraction']:.6f} "
            f"benign_reject={row['benign_reject_fraction']:.6f}"
        )
    lines.extend(["", "Special target compact decompositions"])
    for item in payload["special_targets"]:
        lines.append(f"target={item['target_haer']:.2f} thresholds={item['thresholds']}")
        for method in ("B1", "B2"):
            lines.append(f"  {method}: {item[method]}")
        lines.append(f"  B2_minus_B1: {item['B2_minus_B1']}")
    lines.extend(["", "Cold-start descriptive decomposition"])
    for method in ("B1", "B2"):
        for evidence_count in [0, 1, 2, 3, 5]:
            row = next(item for item in payload["cold_start"] if item["method"] == method and item["evidence_count"] == evidence_count)
            lines.append(
                f"{method} n={evidence_count} UI={row['ui']} "
                f"benign_allow={row['benign_allow_fraction']} "
                f"benign_verify={row['benign_verify_fraction']} "
                f"benign_reject={row['benign_reject_fraction']} "
                f"harmful_execution={row['harmful_execution_fraction']} "
                f"legitimate_completion={row['legitimate_completion']}"
            )
    return "\n".join(lines) + "\n"


def run_diagnostic(
    evaluation_path: str | Path = PRIMARY_EVALUATION,
    calibration_path: str | Path = PRIMARY_CALIBRATION,
    output_path: str | Path = OUTPUT_JSON,
    summary_path: str | Path = OUTPUT_SUMMARY,
) -> Dict[str, Any]:
    """Run and write the diagnostic without touching primary output files."""
    config = load_config()
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    payload = build_diagnostic(config, evaluation, calibration)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(_summary(payload), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_diagnostic()
    print(f"Diagnostic output: {OUTPUT_JSON}")
    print(f"Summary output: {OUTPUT_SUMMARY}")
    print(f"Action traces: {len(result['action_traces'])}")
