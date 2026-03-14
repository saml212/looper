#!/usr/bin/env python3
"""14B Framework Fix Pilot — 6 SWE-Bench tasks with improved agent.

Tests whether qwen2.5-coder:14b benefits from the same framework fixes that
doubled 7B resolve rate (8% -> 16.7%). Prior 14B run on original framework
scored 0% resolve.

Uses 6 tasks from the expanded 7B experiment for direct comparison:
  - 2 tasks 7B resolved (django-11099, django-11451)
  - 4 tasks 7B failed (django-11119, django-11163, django-11299, django-11433)
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
from looper.tasks.loader import get_repo_tasks, load_curriculum, get_task_by_id

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5-coder:14b"
OLLAMA_URL = "http://localhost:11434"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_pilot_v2")

MAX_STEPS = 15
MAX_TOKENS = 4096

# 6 tasks: 2 that 7B+fixes resolved + 4 it failed on
TARGET_TASKS = [
    "django__django-11099",   # 7B: RESOLVED (4 steps)
    "django__django-11119",   # 7B: FAILED (4 steps, wrong_fix)
    "django__django-11163",   # 7B: FAILED (4 steps, wrong_fix)
    "django__django-11299",   # 7B: FAILED (4 steps, wrong_fix)
    "django__django-11433",   # 7B: FAILED (5 steps, wrong_fix)
    "django__django-11451",   # 7B: RESOLVED (4 steps)
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


def run_pilot():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    traj_dir = OUTPUT_DIR / "trajectories" / "base"
    traj_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("14B Framework Fix Pilot — 14B with Agent Improvements")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Fixes: line-range read, context pruning, few-shot, loop detection, skip-verification")
    logger.info(f"Tasks: {len(TARGET_TASKS)} tasks")
    logger.info(f"Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}")
    logger.info("=" * 60)

    # Load specific tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    tasks = []
    for tid in TARGET_TASKS:
        task = get_task_by_id(all_tasks, tid)
        if task is None:
            logger.error(f"Task {tid} not found in curriculum!")
            sys.exit(1)
        tasks.append(task)

    logger.info(f"Tasks ({len(tasks)}):")
    for t in tasks:
        logger.info(f"  {t.instance_id} (pos {t.sequence_position})")

    # Run 14B model via Ollama
    logger.info("")
    logger.info("Running 14B with framework fixes (Ollama)...")

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
        trajectories, tasks, WORKSPACE_ROOT, "base_14b_framework_pilot"
    )

    # Compute metrics
    resolved_count = sum(1 for r in results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(results)
    avg_steps = sum(r.steps for r in results) / total if total else 0
    avg_tokens = sum(r.tokens for r in results) / total if total else 0

    # Save results
    config = ExperimentConfig(
        name="experiment_framework_14b_pilot_v2",
        experiment_id="framework_14b_pilot_v2",
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
    logger.info("14B FRAMEWORK FIX PILOT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info("")

    # Comparison
    logger.info("Comparison with prior results:")
    logger.info(f"  7B base (no fixes):     2/25 (8.0%) resolved")
    logger.info(f"  14B base (no fixes):    0/25 (0.0%) resolved")
    logger.info(f"  7B+fixes (same 6):      2/6  (33.3%) resolved")
    logger.info(f"  14B+fixes (this pilot):  {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved")
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

    # 7B results for comparison (from expanded experiment, same task IDs)
    seven_b = {
        "django__django-11099": {"resolved": True, "steps": 4, "tokens": 4635},
        "django__django-11119": {"resolved": False, "steps": 4, "tokens": 10611},
        "django__django-11163": {"resolved": False, "steps": 4, "tokens": 8597},
        "django__django-11299": {"resolved": False, "steps": 4, "tokens": 11783},
        "django__django-11433": {"resolved": False, "steps": 5, "tokens": 12366},
        "django__django-11451": {"resolved": True, "steps": 4, "tokens": 10718},
    }

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

        # Check for verification loop
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

    report = f"""# 14B Framework Fix Pilot — Results

**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Summary

| Metric | 14B+fixes (this) | 7B+fixes (same tasks) | 14B no fixes | 7B no fixes |
|--------|-------------------|----------------------|--------------|-------------|
| Tasks | {total} | 6 | 25 | 25 |
| Resolve rate | {resolved_count}/{total} ({resolved_count/total*100:.1f}%) | 2/6 (33.3%) | 0/25 (0.0%) | 2/25 (8.0%) |
| Patch rate | {patch_count}/{total} ({patch_count/total*100:.1f}%) | 6/6 (100%) | 11/25 (44%) | 14/25 (56%) |
| Avg steps | {avg_steps:.1f} | 4.2 | ~4 | ~8 |

## Head-to-Head: 14B vs 7B (same tasks, same framework fixes)

| Task | 7B Resolved | 14B Resolved | 7B Steps | 14B Steps | 7B Tokens | 14B Tokens |
|------|-------------|-------------|----------|----------|-----------|-----------|
"""
    for d in task_details:
        sb = seven_b.get(d["task_id"], {})
        report += (
            f"| {d['task_id']} | "
            f"{'YES' if sb.get('resolved') else 'NO'} | "
            f"{'YES' if d['resolved'] else 'NO'} | "
            f"{sb.get('steps', '?')} | {d['steps']} | "
            f"{sb.get('tokens', '?')} | {d['tokens']} |\n"
        )

    report += f"""
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

    fourteen_resolves = resolved_count
    seven_resolves = sum(1 for v in seven_b.values() if v["resolved"])
    fourteen_new = sum(
        1 for d in task_details
        if d["resolved"] and not seven_b.get(d["task_id"], {}).get("resolved", False)
    )
    fourteen_lost = sum(
        1 for d in task_details
        if not d["resolved"] and seven_b.get(d["task_id"], {}).get("resolved", False)
    )

    report += f"""
## Key Findings

- **14B resolve rate:** {resolved_count}/{total} ({resolved_count/total*100:.1f}%)
- **7B resolve rate (same tasks):** {seven_resolves}/6 (33.3%)
- **Tasks 14B resolved that 7B couldn't:** {fourteen_new}
- **Tasks 7B resolved that 14B couldn't:** {fourteen_lost}
- **Prior 14B (no fixes):** 0/25 (0.0%) — framework fixes {"helped" if resolved_count > 0 else "still not enough"}

## Configuration

- Model: {OLLAMA_MODEL}
- Max steps: {MAX_STEPS}
- Max tokens: {MAX_TOKENS}
- Framework fixes: line-range read, context pruning, few-shot examples, loop detection, skip-verification
- Inference: Ollama (not MLX — 14B MLX was too slow in prior test)
"""

    report_path = OUTPUT_DIR / "REPORT.md"
    report_path.write_text(report)
    logger.info(f"Report written to {report_path}")


if __name__ == "__main__":
    run_pilot()
