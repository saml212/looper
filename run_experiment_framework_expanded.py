#!/usr/bin/env python3
"""Expanded Framework Fix Pilot — 12 SWE-Bench tasks with improved agent.

Builds on the 3-task pilot (run_experiment_framework_fix.py) which achieved:
  - 100% patch rate (3/3)
  - 33.3% resolve rate (1/3)
  - New pathology: verification-loop (model tries pip install after writing fix)

This run adds rule 8 ("skip verification") to the system prompt and tests on
12 new tasks (indices 6-17 from django sequence, skipping the 3 pilot tasks).
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.collectors.trajectory_store import collect_trajectories
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import ExperimentConfig, ExperimentResult, TaskResult
from looper.tasks.loader import get_repo_tasks, load_curriculum

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_framework_expanded_7b")

MAX_STEPS = 15
MAX_TOKENS = 4096

# Tasks 6-17 from django sequence (skipping pilot tasks at indices 3-5)
TARGET_TASKS = [
    "django__django-11099",
    "django__django-11119",
    "django__django-11133",
    "django__django-11163",
    "django__django-11179",
    "django__django-11239",
    "django__django-11299",
    "django__django-11433",
    "django__django-11451",
    "django__django-11490",
    "django__django-11555",
    "django__django-11603",
]


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "experiment.log"),
        ],
    )


def evaluate_trajectories(trajectories, tasks, workspace_root, condition):
    """Run FAIL_TO_PASS verification on trajectories."""
    task_map = {t.instance_id: t for t in tasks}
    results = []
    for traj in trajectories:
        task = task_map.get(traj.meta.task_id)
        resolved = False
        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(task, traj.generated_patch, workspace_root)
            resolved = vr["resolved"]
            logger.info(
                f"  Verify {traj.meta.task_id}: "
                f"{'PASS' if resolved else 'FAIL'} "
                f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
                + (f" error={vr['error']}" if vr["error"] else "")
            )
        else:
            logger.info(
                f"  Verify {traj.meta.task_id}: SKIP (no patch generated)"
            )
        results.append(
            TaskResult(
                task_id=traj.meta.task_id,
                condition=condition,
                resolved=resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            )
        )
    return results


def run_expanded_pilot():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    traj_dir = OUTPUT_DIR / "trajectories" / "base"
    traj_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Expanded Framework Fix Pilot — 7B with Agent Improvements")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Fixes: line-range read, context pruning, few-shot, loop detection, skip-verification")
    logger.info(f"Tasks: {len(TARGET_TASKS)} tasks")
    logger.info(f"Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}")
    logger.info("=" * 60)

    # Load tasks by index slice (6-17 = 12 tasks, skip pilot indices 3-5)
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    tasks = all_tasks[6:18]
    assert len(tasks) == 12, f"Expected 12 tasks, got {len(tasks)}"

    logger.info(f"Tasks ({len(tasks)}):")
    for t in tasks:
        logger.info(f"  {t.instance_id} (pos {t.sequence_position})")

    # Run base model via Ollama
    logger.info("")
    logger.info("Running 7B with framework fixes + skip-verification (Ollama)...")

    def on_complete(tid, traj):
        logger.info(
            f"  {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, "
            f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
        )

    trajectories = collect_trajectories(
        tasks=tasks,
        output_dir=traj_dir,
        workspace_root=WORKSPACE_ROOT,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        max_steps=MAX_STEPS,
        max_tokens=MAX_TOKENS,
        on_complete=on_complete,
    )

    # Evaluate with FAIL_TO_PASS verification
    logger.info("")
    logger.info("Evaluating with FAIL_TO_PASS verification...")
    results = evaluate_trajectories(
        trajectories, tasks, WORKSPACE_ROOT, "base_7b_framework_expanded"
    )

    # Compute metrics
    resolved_count = sum(1 for r in results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(results)
    avg_steps = sum(r.steps for r in results) / total if total else 0
    avg_tokens = sum(r.tokens for r in results) / total if total else 0

    # Save results
    config = ExperimentConfig(
        name="experiment_framework_expanded_7b",
        experiment_id="framework_expanded_7b_pilot",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[],
        test_task_ids=[t.instance_id for t in tasks],
        strategy="base_with_framework_fixes_v2",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=results,
        forward_transfer=0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "experiment_result.json")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPANDED FRAMEWORK FIX PILOT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info("")

    # Comparison
    logger.info("Comparison with prior results:")
    logger.info(f"  7B base (no fixes):    2/25 (8.0%) resolved")
    logger.info(f"  7B+fixes (3 tasks):    1/3  (33.3%) resolved")
    logger.info(f"  7B+fixes v2 (12 tasks): {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved")
    logger.info("")

    # Per-task detail
    logger.info("Per-task results:")
    for r, t in zip(results, trajectories):
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"steps={r.steps:>2} tokens={r.tokens:>6} "
            f"patch={'yes' if t.generated_patch.strip() else 'no':>3} "
            f"outcome={t.outcome}"
        )

    summary = results_summary(result)
    logger.info(f"\n{summary}")

    # Write REPORT.md
    write_report(results, trajectories, tasks, resolved_count, patch_count, total, avg_steps, avg_tokens)

    return result


def write_report(results, trajectories, tasks, resolved_count, patch_count, total, avg_steps, avg_tokens):
    """Write detailed REPORT.md to results directory."""
    pathologies = {"verification_loop": 0, "max_steps_no_patch": 0, "wrong_fix": 0}
    task_details = []

    for r, t in zip(results, trajectories):
        has_patch = bool(t.generated_patch.strip())
        detail = {
            "task_id": r.task_id,
            "resolved": r.resolved,
            "steps": r.steps,
            "tokens": r.tokens,
            "has_patch": has_patch,
            "outcome": t.outcome,
            "pathology": "-",
        }

        # Detect pathologies
        if has_patch and not r.resolved:
            pathologies["wrong_fix"] += 1
            detail["pathology"] = "wrong_fix"
        elif not has_patch and r.steps >= MAX_STEPS:
            pathologies["max_steps_no_patch"] += 1
            detail["pathology"] = "max_steps_no_patch"

        # Check for verification loop: pip/virtualenv/pytest attempts
        pip_steps = 0
        for step in t.steps:
            for tc in step.tool_calls:
                inp = str(tc.tool_input.get("input", ""))
                if any(kw in inp for kw in ["pip install", "virtualenv", "venv", "python -m pytest"]):
                    pip_steps += 1
        if pip_steps >= 2:
            pathologies["verification_loop"] += 1
            detail["pathology"] = "verification_loop"

        task_details.append(detail)

    report = f"""# Expanded Framework Fix Pilot — Results

## Summary

| Metric | Value |
|--------|-------|
| Tasks | {total} |
| Resolve rate | {resolved_count}/{total} ({resolved_count/total*100:.1f}%) |
| Patch rate | {patch_count}/{total} ({patch_count/total*100:.1f}%) |
| Avg steps | {avg_steps:.1f} |
| Avg tokens | {avg_tokens:.0f} |

## Comparison

| Condition | Resolve | Patch | Notes |
|-----------|---------|-------|-------|
| 7B base (no fixes) | 2/25 (8.0%) | 14/25 | Original Phase 1 |
| 7B+fixes (3 pilot) | 1/3 (33.3%) | 3/3 | Pilot with framework fixes |
| 7B+fixes v2 (12 expanded) | {resolved_count}/{total} ({resolved_count/total*100:.1f}%) | {patch_count}/{total} | This experiment |

## Pathologies

| Pathology | Count | Description |
|-----------|-------|-------------|
| verification_loop | {pathologies['verification_loop']} | Model tries pip install / pytest after writing fix |
| wrong_fix | {pathologies['wrong_fix']} | Patch generated but fails FAIL_TO_PASS tests |
| max_steps_no_patch | {pathologies['max_steps_no_patch']} | Hit max steps without generating a patch |

## Per-Task Results

| Task | Resolved | Steps | Tokens | Patch | Pathology |
|------|----------|-------|--------|-------|-----------|
"""
    for d in task_details:
        report += (
            f"| {d['task_id']} | {'YES' if d['resolved'] else 'NO'} | "
            f"{d['steps']} | {d['tokens']} | "
            f"{'yes' if d['has_patch'] else 'no'} | "
            f"{d['pathology']} |\n"
        )

    verif_status = "NOT" if pathologies["verification_loop"] == 0 else "STILL"
    report += f"""
## Key Finding

Verification-loop pathology was {verif_status} observed with the skip-verification prompt fix.

## Configuration

- Model: {OLLAMA_MODEL}
- Max steps: {MAX_STEPS}
- Max tokens: {MAX_TOKENS}
- Framework fixes: line-range read, context pruning, few-shot examples, loop detection, skip-verification prompt
- Tasks: indices 6-17 from django/django sequence
"""

    report_path = OUTPUT_DIR / "REPORT.md"
    report_path.write_text(report)
    logger.info(f"Report written to {report_path}")


if __name__ == "__main__":
    run_expanded_pilot()
