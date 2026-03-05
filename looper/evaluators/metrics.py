"""Pure metric calculation functions for experiment evaluation.

No I/O, no model calls -- just math on TaskResult lists.
"""

from looper.models import TaskResult


def resolve_rate(results: list[TaskResult]) -> float:
    """Fraction of tasks where resolved=True. Returns 0.0 if empty."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.resolved) / len(results)


def avg_steps(results: list[TaskResult]) -> float:
    """Average number of steps across tasks. Returns 0.0 if empty."""
    if not results:
        return 0.0
    return sum(r.steps for r in results) / len(results)


def avg_tokens(results: list[TaskResult]) -> float:
    """Average token consumption across tasks. Returns 0.0 if empty."""
    if not results:
        return 0.0
    return sum(r.tokens for r in results) / len(results)


def forward_transfer(
    base_results: list[TaskResult],
    adapted_results: list[TaskResult],
) -> float:
    """Forward transfer metric.

    FT = adapted_resolve_rate - base_resolve_rate

    Positive FT means the adapter helped. Negative means it hurt.
    Both lists should contain results for the same tasks (test split).
    """
    return resolve_rate(adapted_results) - resolve_rate(base_results)


def compare_conditions(
    results_by_condition: dict[str, list[TaskResult]],
) -> dict[str, dict[str, float]]:
    """Compare metrics across conditions.

    Args:
        results_by_condition: {"base": [...], "base_lora": [...], ...}

    Returns dict like:
        {
            "base": {"resolve_rate": 0.4, "avg_steps": 15.2, "avg_tokens": 5000},
            "base_lora": {"resolve_rate": 0.6, "avg_steps": 12.1, "avg_tokens": 4200},
        }
    """
    summary: dict[str, dict[str, float]] = {}
    for condition, results in results_by_condition.items():
        summary[condition] = {
            "resolve_rate": resolve_rate(results),
            "avg_steps": avg_steps(results),
            "avg_tokens": avg_tokens(results),
        }
    return summary
