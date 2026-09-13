"""One-factor-at-a-time robustness runner for frozen primary design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import yaml

from experiments.run_pilot import _paired_verify_outcomes
from src.authorization import authorize
from src.calibration import _select_threshold_pair
from src.metrics import ActionRecord, compute_metrics
from src.population import generate_population
from src.trust_model import lcb_score, trust_mean, trust_variance
from src.verify_sim import verify_action

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sensitivity.yaml"
PRIMARY_ANALYSIS = ROOT / "results/primary/analysis.json"
PRIMARY_EVALUATION = ROOT / "results/primary/evaluation.json"
PRIMARY_CALIBRATION = ROOT / "results/primary/calibration.json"


def load_config(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def threshold_grid(step: float):
    count = round(1.0 / step)
    return [round(index * step, 10) for index in range(1, count)]


def fingerprint(condition: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    payload = {
        "condition": dict(condition),
        "K": config["K"], "M": config["M"],
        "profiles": config["profiles"], "evidence_counts": config["evidence_counts"],
        "calibration_seeds": config["calibration_seeds"],
        "evaluation_seeds": config["evaluation_seeds"],
        "target_haer_grid": config["target_haer_grid"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _realization(config, seed, condition):
    ss = np.random.SeedSequence(seed)
    population_ss, verify_ss = ss.spawn(2)
    population = generate_population(
        np.random.default_rng(population_ss),
        num_measurement_actions=config["M"], agents_per_cell=config["K"],
        evidence_counts=config["evidence_counts"], profiles=config["profiles"],
        alpha0=condition["alpha0"], beta0=condition["beta0"],
    )
    verify_ss, = verify_ss.spawn(1)
    verify_rng = np.random.default_rng(verify_ss)
    outcomes = _paired_verify_outcomes(population, verify_rng, condition["p_catch"])
    return population, outcomes


def _score(agent, method, condition):
    trust = trust_mean(agent["alpha"], agent["beta"])
    if method == "B1":
        return trust
    return lcb_score(trust, trust_variance(agent["alpha"], agent["beta"]), z=condition["z"], clip=True)


def _evaluate(population, outcomes, method, thresholds, condition, seed):
    records = []
    for agent, agent_outcomes in zip(population, outcomes):
        decision = authorize(_score(agent, method, condition), *thresholds)
        for is_harmful, verify_outcome in zip(agent["measurement_actions"], agent_outcomes):
            executed = True if decision == "ALLOW" else (bool(verify_outcome) if decision == "VERIFY" and is_harmful else True) if decision == "VERIFY" else False
            records.append(ActionRecord(seed, agent["agent_id"], agent["profile"], agent["evidence_count"], bool(is_harmful), decision, executed, method))
    return compute_metrics(records, group_by=["seed"])[seed]


def _calibrate(config, condition, realizations):
    grid = threshold_grid(condition["threshold_step"])
    thresholds = {}
    achieved = {}
    prepared = {
        method: [
            [
                (
                    _score(agent, method, condition),
                    sum(agent["measurement_actions"]),
                    sum(bool(outcome) for harmful, outcome in zip(agent["measurement_actions"], agent_outcomes) if harmful),
                )
                for agent, agent_outcomes in zip(population, verify_outcomes)
            ]
            for population, verify_outcomes in realizations
        ]
        for method in ("B1", "B2")
    }
    for target in config["target_haer_grid"]:
        candidates = {}
        for reject in grid:
            for verify in grid:
                if reject >= verify:
                    continue
                candidates[(reject, verify)] = _candidate_haer(prepared["B1"], (reject, verify))
        thresholds[(target, "B1")], achieved[(target, "B1")] = _select_threshold_pair(candidates, target)
        for method in ("B2",):
            candidates = {(reject, verify): _candidate_haer(prepared[method], (reject, verify)) for reject in grid for verify in grid if reject < verify}
            thresholds[(target, method)], achieved[(target, method)] = _select_threshold_pair(candidates, target)
    return thresholds, achieved


def _candidate_haer(prepared_realizations, thresholds):
    harmful = executed = 0
    for realization in prepared_realizations:
        for score, harmful_count, verify_executed in realization:
            harmful += harmful_count
            decision = authorize(score, *thresholds)
            executed += harmful_count if decision == "ALLOW" else verify_executed if decision == "VERIFY" else 0
    return executed / harmful if harmful else 0.0


def run_condition(config, condition):
    realizations = [_realization(config, seed, condition) for seed in config["calibration_seeds"]]
    thresholds, calibration_achieved = _calibrate(config, condition, realizations)
    rows = []
    for seed in config["evaluation_seeds"]:
        population, outcomes = _realization(config, seed, condition)
        for target in config["target_haer_grid"]:
            for method in ("B1", "B2"):
                metrics = _evaluate(population, outcomes, method, thresholds[(target, method)], condition, seed)
                rows.append({"seed": seed, "target_haer": target, "method": method, "theta_reject": thresholds[(target, method)][0], "theta_verify": thresholds[(target, method)][1], "calibration_achieved_haer": calibration_achieved[(target, method)], "evaluation_achieved_haer": metrics["haer"], "unnecessary_intervention_rate": metrics["unnecessary_intervention"], "legitimate_action_completion": metrics["legitimate_completion"], "false_verify_rate": metrics["false_verify"], "false_reject_rate": metrics["false_reject"]})
    return {"condition": dict(condition), "fingerprint": fingerprint(condition, config), "statistical_unit": "seed", "inference_performed": False, "threshold_grid": threshold_grid(condition["threshold_step"]), "rows": rows}


def main():
    config = load_config()
    out = Path(config["output_directory"]); out.mkdir(parents=True, exist_ok=True)
    conditions = [{"id": "baseline", **config["baseline"]}] + config["sensitivity_conditions"]
    for condition in conditions:
        path = out / f"{condition['id']}.json"
        if not path.exists():
            path.write_text(json.dumps(run_condition(config, condition), indent=2, sort_keys=True) + "\n")
        print(path)


if __name__ == "__main__":
    main()
