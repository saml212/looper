"""Generate draft sections for a research paper.

Pure string formatting — no LLM calls. Templates are filled with computed
values from Phase1Analysis and ExperimentConfig.
"""

from __future__ import annotations

from looper.analysis.results_analyzer import Phase1Analysis
from looper.models import ExperimentConfig


def generate_results_table(analysis: Phase1Analysis) -> str:
    """Generate a LaTeX table summarising per-condition metrics."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Phase 1 Results: Base vs.\ Adapted}",
        r"\label{tab:phase1}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Condition & Resolve Rate (\%) & Mean Steps & Mean Tokens \\",
        r"\midrule",
    ]
    for cond, stats in sorted(analysis.statistical_summary.items()):
        # Find resolve rate from per-condition stats (computed from analysis)
        rr = _resolve_rate_for(analysis, cond)
        lines.append(
            f"{cond} & {rr * 100:.1f} & {stats['mean_steps']:.1f} & {stats['mean_tokens']:.0f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _resolve_rate_for(analysis: Phase1Analysis, cond: str) -> float:
    """Return the resolve rate for the given condition name."""
    conditions_sorted = sorted(analysis.statistical_summary.keys())
    if cond == conditions_sorted[0]:
        return analysis.base_resolve_rate
    return analysis.adapted_resolve_rate


def generate_methodology_outline(config: ExperimentConfig) -> str:
    """Generate a bullet-point methodology outline."""
    lines = [
        "Methodology",
        "===========",
        "",
        f"- Experiment: {config.name} ({config.experiment_id})",
        f"- Model: {config.model_name}",
        f"- Repository: {config.repo}",
        f"- Anti-forgetting strategy: {config.strategy}",
        f"- LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}",
        f"- Training tasks: {len(config.train_task_ids)}",
        f"- Test tasks: {len(config.test_task_ids)}",
        f"- Random seed: {config.seed}",
    ]
    return "\n".join(lines)


def generate_results_narrative(analysis: Phase1Analysis) -> str:
    """Generate a narrative description of Phase 1 results."""
    base_pct = analysis.base_resolve_rate * 100
    adapted_pct = analysis.adapted_resolve_rate * 100
    ft = analysis.forward_transfer * 100

    parts: list[str] = []

    if ft > 0:
        parts.append(
            f"The adapted model showed an improvement in resolve rate, "
            f"increasing from {base_pct:.1f}% (base) to {adapted_pct:.1f}% "
            f"(adapted), a forward transfer of +{ft:.1f} percentage points."
        )
    elif ft < 0:
        parts.append(
            f"The adapted model showed a decrease in resolve rate, "
            f"dropping from {base_pct:.1f}% (base) to {adapted_pct:.1f}% "
            f"(adapted), a negative forward transfer of {ft:.1f} percentage points."
        )
    else:
        parts.append(
            f"The adapted model matched the base resolve rate at {base_pct:.1f}%."
        )

    n_improved = len(analysis.improvement_tasks)
    n_regressed = len(analysis.regression_tasks)
    if n_improved:
        parts.append(
            f"{n_improved} task(s) improved (resolved only by the adapted model): "
            f"{', '.join(analysis.improvement_tasks)}."
        )
    if n_regressed:
        parts.append(
            f"{n_regressed} task(s) regressed (resolved only by the base model): "
            f"{', '.join(analysis.regression_tasks)}."
        )

    return " ".join(parts)
