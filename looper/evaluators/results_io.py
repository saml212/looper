"""Save and load experiment results."""

import json
from pathlib import Path

from looper.models import ExperimentResult
from looper.evaluators.metrics import compare_conditions, forward_transfer


def save_results(result: ExperimentResult, path: Path) -> None:
    """Save experiment results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))


def load_results(path: Path) -> ExperimentResult:
    """Load experiment results from a JSON file."""
    return ExperimentResult.model_validate_json(path.read_text())


def results_summary(result: ExperimentResult) -> str:
    """Generate a human-readable summary of experiment results."""
    lines: list[str] = []
    lines.append(f"Experiment: {result.config.name} ({result.config.experiment_id})")
    lines.append(f"Model: {result.config.model_name}")
    lines.append(f"Repo: {result.config.repo}")
    lines.append(f"Tasks: {len(result.task_results)}")
    lines.append("")

    # Group results by condition and compute metrics
    by_condition: dict[str, list] = {}
    for tr in result.task_results:
        by_condition.setdefault(tr.condition, []).append(tr)

    if by_condition:
        comparison = compare_conditions(by_condition)
        for condition, metrics in comparison.items():
            lines.append(f"  {condition}:")
            lines.append(f"    Resolve rate: {metrics['resolve_rate']:.2%}")
            lines.append(f"    Avg steps:    {metrics['avg_steps']:.1f}")
            lines.append(f"    Avg tokens:   {metrics['avg_tokens']:.0f}")

    lines.append("")
    lines.append(f"Forward transfer: {result.forward_transfer:+.4f}")
    lines.append(f"Forgetting:       {result.forgetting:+.4f}")

    return "\n".join(lines)
