"""
Research metrics for evaluation.

Computes:
1. HAER: harmful actions ultimately executed / all harmful actions
2. Raw unnecessary intervention: benign with VERIFY/REJECT / all benign
3. False-VERIFY: benign with VERIFY / all benign
4. False-REJECT: benign with REJECT / all benign
5. Legitimate-action completion: benign completed / all benign

Supports aggregation at seed, profile, evidence-level, and method levels.
Handles zero denominators explicitly.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class ActionRecord:
    """
    A single action record from evaluation.

    Attributes:
        seed: RNG seed used to generate population
        agent_id: Unique agent identifier
        profile: Agent profile (Reliable, Unreliable, Malicious)
        evidence_count: Agent evidence level
        is_harmful: Ground-truth: True if harmful, False if benign
        decision: Authorization decision: "ALLOW", "VERIFY", or "REJECT"
        executed: Whether action ultimately executed
        method: Method used: "B1" or "B2"
    """
    seed: int
    agent_id: int
    profile: str
    evidence_count: int
    is_harmful: bool
    decision: str
    executed: bool
    method: str


def compute_haer(records: List[ActionRecord]) -> float:
    """
    Compute harmful action execution rate.

    HAER = harmful actions ultimately executed / all harmful actions proposed

    Args:
        records: List of ActionRecord

    Returns:
        HAER value [0, 1], or None if no harmful actions

    Raises:
        ValueError: If records list is empty or malformed
    """
    if not records:
        raise ValueError("records list cannot be empty")

    harmful_records = [r for r in records if r.is_harmful]
    if not harmful_records:
        return None

    executed_count = sum(r.executed for r in harmful_records)
    return executed_count / len(harmful_records)


def compute_unnecessary_intervention_rate(records: List[ActionRecord]) -> float:
    """
    Compute raw unnecessary intervention rate.

    Raw unnecessary intervention = (benign with VERIFY or REJECT) / all benign

    Args:
        records: List of ActionRecord

    Returns:
        Rate value [0, 1], or None if no benign actions
    """
    if not records:
        raise ValueError("records list cannot be empty")

    benign_records = [r for r in records if not r.is_harmful]
    if not benign_records:
        return None

    intervened_count = sum(
        1 for r in benign_records
        if r.decision in ("VERIFY", "REJECT")
    )
    return intervened_count / len(benign_records)


def compute_false_verify_rate(records: List[ActionRecord]) -> float:
    """
    Compute false-VERIFY rate.

    False-VERIFY = benign actions receiving VERIFY / all benign actions

    Args:
        records: List of ActionRecord

    Returns:
        Rate value [0, 1], or None if no benign actions
    """
    if not records:
        raise ValueError("records list cannot be empty")

    benign_records = [r for r in records if not r.is_harmful]
    if not benign_records:
        return None

    verify_count = sum(1 for r in benign_records if r.decision == "VERIFY")
    return verify_count / len(benign_records)


def compute_false_reject_rate(records: List[ActionRecord]) -> float:
    """
    Compute false-REJECT rate.

    False-REJECT = benign actions receiving REJECT / all benign actions

    Args:
        records: List of ActionRecord

    Returns:
        Rate value [0, 1], or None if no benign actions
    """
    if not records:
        raise ValueError("records list cannot be empty")

    benign_records = [r for r in records if not r.is_harmful]
    if not benign_records:
        return None

    reject_count = sum(1 for r in benign_records if r.decision == "REJECT")
    return reject_count / len(benign_records)


def compute_legitimate_completion_rate(records: List[ActionRecord]) -> float:
    """
    Compute legitimate action completion rate.

    Legitimate completion = benign actions ultimately completed / all benign actions

    Args:
        records: List of ActionRecord

    Returns:
        Rate value [0, 1], or None if no benign actions
    """
    if not records:
        raise ValueError("records list cannot be empty")

    benign_records = [r for r in records if not r.is_harmful]
    if not benign_records:
        return None

    executed_count = sum(r.executed for r in benign_records)
    return executed_count / len(benign_records)


def aggregate_records(
    records: List[ActionRecord],
    group_by: Optional[List[str]] = None,
) -> Dict[Any, List[ActionRecord]]:
    """
    Aggregate records by specified attributes.

    Args:
        records: List of ActionRecord
        group_by: List of attribute names to group by.
                 Default ["seed"] groups by seed.
                 Can include: "seed", "method", "profile", "evidence_count"

    Returns:
        Dictionary mapping group key to list of records in that group.
        If group_by has one element, keys are scalar values.
        If group_by has multiple elements, keys are tuples.
    """
    if group_by is None:
        group_by = ["seed"]

    if not group_by:
        return {None: records}

    aggregated = {}
    for record in records:
        values = tuple(getattr(record, attr) for attr in group_by)
        if len(values) == 1:
            key = values[0]
        else:
            key = values

        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(record)

    return aggregated


def compute_metrics(
    records: List[ActionRecord],
    group_by: Optional[List[str]] = None,
) -> Dict[Any, Dict[str, Optional[float]]]:
    """
    Compute all five metrics, optionally grouped.

    Args:
        records: List of ActionRecord
        group_by: Grouping attributes (default ["seed"])

    Returns:
        Dictionary mapping group key to metrics dictionary containing:
            - haer: Harmful action execution rate
            - unnecessary_intervention: Raw unnecessary intervention rate
            - false_verify: False-VERIFY rate
            - false_reject: False-REJECT rate
            - legitimate_completion: Legitimate action completion rate
        Values are None if denominator is zero.
    """
    if group_by is None:
        group_by = ["seed"]

    aggregated = aggregate_records(records, group_by)
    metrics = {}

    for group_key, group_records in aggregated.items():
        metrics[group_key] = {
            "haer": compute_haer(group_records),
            "unnecessary_intervention": compute_unnecessary_intervention_rate(group_records),
            "false_verify": compute_false_verify_rate(group_records),
            "false_reject": compute_false_reject_rate(group_records),
            "legitimate_completion": compute_legitimate_completion_rate(group_records),
        }

    return metrics


def print_metrics_summary(
    metrics: Dict[Any, Dict[str, Optional[float]]],
    group_label: str = "Group",
) -> None:
    """
    Pretty-print metrics summary.

    Args:
        metrics: Output from compute_metrics()
        group_label: Label for the grouping attribute
    """
    print(f"\n{'Metrics Summary':=^80}")
    print(f"{group_label:20} | {'HAER':>10} | {'Unne. Int.':>10} | {'False-V':>10} | {'False-R':>10} | {'Legit. C.':>10}")
    print("-" * 80)

    for group_key, group_metrics in metrics.items():
        haer = group_metrics["haer"]
        unne = group_metrics["unnecessary_intervention"]
        fv = group_metrics["false_verify"]
        fr = group_metrics["false_reject"]
        lc = group_metrics["legitimate_completion"]

        haer_str = f"{haer:.4f}" if haer is not None else "N/A"
        unne_str = f"{unne:.4f}" if unne is not None else "N/A"
        fv_str = f"{fv:.4f}" if fv is not None else "N/A"
        fr_str = f"{fr:.4f}" if fr is not None else "N/A"
        lc_str = f"{lc:.4f}" if lc is not None else "N/A"

        print(f"{str(group_key):20} | {haer_str:>10} | {unne_str:>10} | {fv_str:>10} | {fr_str:>10} | {lc_str:>10}")
