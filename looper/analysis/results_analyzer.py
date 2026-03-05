"""Analyze experiment results into structured summaries.

Pure computation — no LLM calls, no I/O beyond what the caller provides.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from looper.models import ExperimentResult, TaskResult
from looper.evaluators.metrics import resolve_rate


@dataclass
class Phase1Analysis:
    """Structured analysis of a Phase 1 experiment."""

    base_resolve_rate: float
    adapted_resolve_rate: float
    forward_transfer: float
    per_task_comparison: list[dict]
    statistical_summary: dict[str, dict[str, float]]
    improvement_tasks: list[str]
    regression_tasks: list[str]


@dataclass
class ConditionComparison:
    """Comparison of metrics across conditions."""

    conditions: dict[str, dict[str, float]] = field(default_factory=dict)


def _split_by_condition(
    results: list[TaskResult],
) -> dict[str, list[TaskResult]]:
    by_cond: dict[str, list[TaskResult]] = {}
    for r in results:
        by_cond.setdefault(r.condition, []).append(r)
    return by_cond


def _stat_summary(results: list[TaskResult]) -> dict[str, float]:
    steps = [r.steps for r in results]
    tokens = [r.tokens for r in results]
    return {
        "mean_steps": statistics.mean(steps),
        "median_steps": statistics.median(steps),
        "std_steps": statistics.stdev(steps) if len(steps) > 1 else 0.0,
        "mean_tokens": statistics.mean(tokens),
        "median_tokens": statistics.median(tokens),
        "std_tokens": statistics.stdev(tokens) if len(tokens) > 1 else 0.0,
    }


def analyze_phase1(result: ExperimentResult) -> Phase1Analysis:
    """Analyze a Phase 1 experiment result.

    Expects task_results to contain at least two conditions. The first
    condition alphabetically is treated as "base" and the second as
    "adapted". Typically these are "base" and "base_lora".
    """
    by_cond = _split_by_condition(result.task_results)
    conditions_sorted = sorted(by_cond.keys())
    base_cond = conditions_sorted[0]
    adapted_cond = conditions_sorted[1] if len(conditions_sorted) > 1 else conditions_sorted[0]

    base_results = by_cond[base_cond]
    adapted_results = by_cond[adapted_cond]

    base_rr = resolve_rate(base_results)
    adapted_rr = resolve_rate(adapted_results)

    # Build lookup by task_id for each condition
    base_by_task = {r.task_id: r for r in base_results}
    adapted_by_task = {r.task_id: r for r in adapted_results}
    all_task_ids = sorted(set(base_by_task) | set(adapted_by_task))

    per_task: list[dict] = []
    improvement: list[str] = []
    regression: list[str] = []

    for tid in all_task_ids:
        b = base_by_task.get(tid)
        a = adapted_by_task.get(tid)
        entry: dict = {"task_id": tid}
        if b:
            entry["base_resolved"] = b.resolved
            entry["base_steps"] = b.steps
            entry["base_tokens"] = b.tokens
        if a:
            entry["adapted_resolved"] = a.resolved
            entry["adapted_steps"] = a.steps
            entry["adapted_tokens"] = a.tokens

        per_task.append(entry)

        if b and a:
            if not b.resolved and a.resolved:
                improvement.append(tid)
            elif b.resolved and not a.resolved:
                regression.append(tid)

    stat_summary = {cond: _stat_summary(rs) for cond, rs in by_cond.items()}

    return Phase1Analysis(
        base_resolve_rate=base_rr,
        adapted_resolve_rate=adapted_rr,
        forward_transfer=adapted_rr - base_rr,
        per_task_comparison=per_task,
        statistical_summary=stat_summary,
        improvement_tasks=improvement,
        regression_tasks=regression,
    )


def compare_conditions(results: list[TaskResult]) -> ConditionComparison:
    """Compare base vs adapted across multiple metrics."""
    by_cond = _split_by_condition(results)
    conditions: dict[str, dict[str, float]] = {}
    for cond, rs in by_cond.items():
        conditions[cond] = {
            "resolve_rate": resolve_rate(rs),
            "avg_steps": statistics.mean([r.steps for r in rs]),
            "avg_tokens": statistics.mean([r.tokens for r in rs]),
        }
    return ConditionComparison(conditions=conditions)
