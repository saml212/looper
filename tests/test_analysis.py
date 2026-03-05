"""Tests for looper.analysis module — research paper analysis tools."""

import statistics
from pathlib import Path

import pytest

from looper.models import ExperimentConfig, ExperimentResult, TaskResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIG = ExperimentConfig(
    name="Phase 1 — Does it work at all?",
    experiment_id="exp-001",
    repo="django/django",
    model_name="codellama-7b",
    train_task_ids=["t1", "t2", "t3"],
    test_task_ids=["t4", "t5", "t6", "t7", "t8"],
    strategy="full_replay",
    lora_rank=16,
    lora_alpha=32,
    seed=42,
)


def _make_result() -> ExperimentResult:
    """Build a minimal ExperimentResult with base and base_lora conditions."""
    tasks = [
        # base condition — 2 of 5 resolved
        TaskResult(task_id="t4", condition="base", resolved=True, steps=20, tokens=5000, duration_seconds=60.0),
        TaskResult(task_id="t5", condition="base", resolved=False, steps=30, tokens=8000, duration_seconds=90.0),
        TaskResult(task_id="t6", condition="base", resolved=True, steps=15, tokens=4000, duration_seconds=50.0),
        TaskResult(task_id="t7", condition="base", resolved=False, steps=25, tokens=7000, duration_seconds=80.0),
        TaskResult(task_id="t8", condition="base", resolved=False, steps=35, tokens=9000, duration_seconds=100.0),
        # base_lora condition — 4 of 5 resolved (improvement on t5, t7, t8; regression nowhere new)
        TaskResult(task_id="t4", condition="base_lora", resolved=True, steps=12, tokens=3000, duration_seconds=40.0),
        TaskResult(task_id="t5", condition="base_lora", resolved=True, steps=18, tokens=5000, duration_seconds=55.0),
        TaskResult(task_id="t6", condition="base_lora", resolved=False, steps=22, tokens=6000, duration_seconds=70.0),
        TaskResult(task_id="t7", condition="base_lora", resolved=True, steps=14, tokens=3500, duration_seconds=45.0),
        TaskResult(task_id="t8", condition="base_lora", resolved=True, steps=16, tokens=4000, duration_seconds=48.0),
    ]
    return ExperimentResult(
        config=_CONFIG,
        task_results=tasks,
        forward_transfer=0.4,
        forgetting=0.0,
        started_at="2026-03-01T00:00:00Z",
        completed_at="2026-03-01T01:00:00Z",
    )


# ---------------------------------------------------------------------------
# results_analyzer tests
# ---------------------------------------------------------------------------


class TestAnalyzePhase1:
    def test_resolve_rates(self):
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        assert analysis.base_resolve_rate == pytest.approx(2 / 5)
        assert analysis.adapted_resolve_rate == pytest.approx(4 / 5)

    def test_forward_transfer(self):
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        assert analysis.forward_transfer == pytest.approx(4 / 5 - 2 / 5)

    def test_improvement_tasks(self):
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        # t5, t7, t8 went from not-resolved to resolved
        assert sorted(analysis.improvement_tasks) == ["t5", "t7", "t8"]

    def test_regression_tasks(self):
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        # t6 went from resolved to not-resolved
        assert analysis.regression_tasks == ["t6"]

    def test_per_task_comparison_length(self):
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        assert len(analysis.per_task_comparison) == 5

    def test_per_task_comparison_fields(self):
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        entry = next(e for e in analysis.per_task_comparison if e["task_id"] == "t4")
        assert entry["base_resolved"] is True
        assert entry["adapted_resolved"] is True
        assert entry["base_steps"] == 20
        assert entry["adapted_steps"] == 12

    def test_statistical_summary_keys(self):
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        for cond in ("base", "base_lora"):
            assert cond in analysis.statistical_summary
            s = analysis.statistical_summary[cond]
            for key in ("mean_steps", "median_steps", "std_steps", "mean_tokens", "median_tokens", "std_tokens"):
                assert key in s

    def test_statistical_summary_values(self):
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        base_steps = [20, 30, 15, 25, 35]
        s = analysis.statistical_summary["base"]
        assert s["mean_steps"] == pytest.approx(statistics.mean(base_steps))
        assert s["median_steps"] == pytest.approx(statistics.median(base_steps))
        assert s["std_steps"] == pytest.approx(statistics.stdev(base_steps))


class TestCompareConditions:
    def test_returns_both_conditions(self):
        from looper.analysis.results_analyzer import compare_conditions

        result = _make_result()
        comparison = compare_conditions(result.task_results)
        assert "base" in comparison.conditions
        assert "base_lora" in comparison.conditions

    def test_metrics_per_condition(self):
        from looper.analysis.results_analyzer import compare_conditions

        comparison = compare_conditions(_make_result().task_results)
        for cond in comparison.conditions:
            metrics = comparison.conditions[cond]
            assert "resolve_rate" in metrics
            assert "avg_steps" in metrics
            assert "avg_tokens" in metrics


# ---------------------------------------------------------------------------
# paper_sections tests
# ---------------------------------------------------------------------------


class TestGenerateResultsTable:
    def test_latex_table_has_required_structure(self):
        from looper.analysis.paper_sections import generate_results_table
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        latex = generate_results_table(analysis)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    def test_latex_table_contains_conditions(self):
        from looper.analysis.paper_sections import generate_results_table
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        latex = generate_results_table(analysis)
        assert "base" in latex.lower()

    def test_latex_table_contains_metrics(self):
        from looper.analysis.paper_sections import generate_results_table
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        latex = generate_results_table(analysis)
        # Should contain resolve rate values
        assert "40.0" in latex or "0.40" in latex  # base resolve rate


class TestGenerateMethodologyOutline:
    def test_mentions_model_and_repo(self):
        from looper.analysis.paper_sections import generate_methodology_outline

        outline = generate_methodology_outline(_CONFIG)
        assert "codellama-7b" in outline
        assert "django/django" in outline

    def test_mentions_strategy(self):
        from looper.analysis.paper_sections import generate_methodology_outline

        outline = generate_methodology_outline(_CONFIG)
        assert "full_replay" in outline


class TestGenerateResultsNarrative:
    def test_narrative_describes_improvement(self):
        from looper.analysis.paper_sections import generate_results_narrative
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        narrative = generate_results_narrative(analysis)
        # Should mention the improvement direction
        assert "improv" in narrative.lower() or "increase" in narrative.lower()

    def test_narrative_contains_numbers(self):
        from looper.analysis.paper_sections import generate_results_narrative
        from looper.analysis.results_analyzer import analyze_phase1

        analysis = analyze_phase1(_make_result())
        narrative = generate_results_narrative(analysis)
        # Should contain actual percentage values
        assert "40.0%" in narrative or "80.0%" in narrative


# ---------------------------------------------------------------------------
# related_work tests
# ---------------------------------------------------------------------------


class TestLoadRelatedWork:
    def test_loads_papers_from_landscape(self):
        from looper.analysis.related_work import load_related_work

        landscape = Path(__file__).parent.parent / "docs" / "research_landscape.md"
        papers = load_related_work(landscape)
        assert len(papers) > 0

    def test_paper_has_required_fields(self):
        from looper.analysis.related_work import load_related_work

        landscape = Path(__file__).parent.parent / "docs" / "research_landscape.md"
        papers = load_related_work(landscape)
        paper = papers[0]
        assert paper.title
        assert paper.year > 0
        assert paper.key_finding

    def test_finds_lora_paper(self):
        from looper.analysis.related_work import load_related_work

        landscape = Path(__file__).parent.parent / "docs" / "research_landscape.md"
        papers = load_related_work(landscape)
        titles = [p.title for p in papers]
        assert any("LoRA" in t for t in titles)

    def test_papers_have_categories(self):
        from looper.analysis.related_work import load_related_work

        landscape = Path(__file__).parent.parent / "docs" / "research_landscape.md"
        papers = load_related_work(landscape)
        categories = {p.category for p in papers}
        assert len(categories) > 1  # Should have multiple categories


class TestFindRelevantPapers:
    def test_finds_forgetting_papers(self):
        from looper.analysis.related_work import find_relevant_papers, load_related_work

        landscape = Path(__file__).parent.parent / "docs" / "research_landscape.md"
        papers = load_related_work(landscape)
        results = find_relevant_papers("forgetting", papers)
        assert len(results) > 0
        # All results should be related to forgetting
        for p in results:
            text = f"{p.title} {p.key_finding}".lower()
            assert "forget" in text

    def test_empty_topic_returns_empty(self):
        from looper.analysis.related_work import find_relevant_papers, load_related_work

        landscape = Path(__file__).parent.parent / "docs" / "research_landscape.md"
        papers = load_related_work(landscape)
        results = find_relevant_papers("xyznonexistent", papers)
        assert results == []
