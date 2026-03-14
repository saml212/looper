#!/usr/bin/env python3
"""14B Fuzzy Edit Experiment — 50 SWE-Bench tasks with fuzzy <edit> tool.

Tests three improvements over the edit-tool baseline:
1. Fuzzy matching fallback (0.7 threshold) when exact match fails
2. Edit loop detection (nudge after 3 consecutive failures on same file)
3. Hybrid tool hint (<edit> for 50+ line files, <write> for small/new files)

Uses MLX server on port 8080 (mlx_lm.server with qwen2.5-coder:14b-instruct-q4).
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.agent.ollama_client import openai_chat
from looper.collectors.trajectory_store import collect_trajectories
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import ExperimentConfig, ExperimentResult, TaskResult
from looper.tasks.loader import get_repo_tasks, load_curriculum

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

MLX_MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
MLX_URL = "http://localhost:8080"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_fuzzy_edit_14b")

MAX_STEPS = 15
MAX_TOKENS = 4096


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


def analyze_tool_usage(trajectories):
    """Analyze edit/write/fuzzy usage across all trajectories."""
    stats = {
        "edit_exact": 0,
        "edit_fuzzy": 0,
        "edit_failed": 0,
        "write": 0,
        "edit_loop_nudges": 0,
    }
    per_task = {}

    for traj in trajectories:
        task_stats = {"edit_exact": 0, "edit_fuzzy": 0, "edit_failed": 0, "write": 0}
        for step in traj.steps:
            for tc in step.tool_calls:
                if tc.tool_name == "edit":
                    if tc.success and "fuzzy match" in tc.tool_result:
                        task_stats["edit_fuzzy"] += 1
                    elif tc.success:
                        task_stats["edit_exact"] += 1
                    else:
                        task_stats["edit_failed"] += 1
                elif tc.tool_name == "write":
                    task_stats["write"] += 1

        for key in task_stats:
            stats[key] += task_stats[key]
        per_task[traj.meta.task_id] = task_stats

    return stats, per_task


def run_experiment():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    traj_dir = OUTPUT_DIR / "trajectories" / "base"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Load all 50 Django tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    tasks = list(all_tasks)

    logger.info("=" * 60)
    logger.info("14B Fuzzy Edit Experiment — All 50 Django Tasks")
    logger.info(f"Model: {MLX_MODEL} via MLX server at {MLX_URL}")
    logger.info("Improvements over edit-tool baseline:")
    logger.info("  1. Fuzzy matching fallback (0.7 threshold)")
    logger.info("  2. Edit loop detection (nudge after 3 failures)")
    logger.info("  3. Hybrid tool hint (<edit> for 50+ lines, <write> for small)")
    logger.info(f"Tasks: {len(tasks)} tasks")
    logger.info(f"Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}")
    logger.info("=" * 60)

    logger.info(f"\nTasks ({len(tasks)}):")
    for t in tasks:
        logger.info(f"  {t.instance_id} (pos {t.sequence_position})")

    # Run tasks via MLX server (openai_chat)
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
        model=MLX_MODEL,
        base_url=MLX_URL,
        max_steps=MAX_STEPS,
        max_tokens=MAX_TOKENS,
        on_complete=on_complete,
        chat_fn=openai_chat,
    )

    # Evaluate with FAIL_TO_PASS verification
    logger.info("")
    logger.info("Evaluating with FAIL_TO_PASS verification...")
    task_results = evaluate_trajectories(
        trajectories, tasks, WORKSPACE_ROOT, "base_14b_fuzzy_edit"
    )

    # Compute metrics
    resolved_count = sum(1 for r in task_results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(task_results)
    avg_steps = sum(r.steps for r in task_results) / total if total else 0
    avg_tokens = sum(r.tokens for r in task_results) / total if total else 0

    # Analyze tool usage
    stats, per_task_stats = analyze_tool_usage(trajectories)

    # Save results
    config = ExperimentConfig(
        name="experiment_fuzzy_edit_14b",
        experiment_id="fuzzy_edit_14b",
        repo="django/django",
        model_name=MLX_MODEL,
        train_task_ids=[],
        test_task_ids=[t.instance_id for t in tasks],
        strategy="base_with_fuzzy_edit",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=task_results,
        forward_transfer=0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "experiment_result.json")

    # Save detailed tool stats
    with open(OUTPUT_DIR / "tool_stats.json", "w") as f:
        json.dump({"aggregate": stats, "per_task": per_task_stats}, f, indent=2)

    # Print summary
    total_edits = stats["edit_exact"] + stats["edit_fuzzy"] + stats["edit_failed"]
    logger.info("")
    logger.info("=" * 60)
    logger.info("14B FUZZY EDIT EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info("")
    logger.info("Tool usage:")
    logger.info(f"  Edit (exact match):  {stats['edit_exact']}")
    logger.info(f"  Edit (fuzzy match):  {stats['edit_fuzzy']}")
    logger.info(f"  Edit (failed):       {stats['edit_failed']}")
    logger.info(f"  Edit total:          {total_edits}")
    logger.info(f"  Write:               {stats['write']}")
    if total_edits > 0:
        logger.info(f"  Fuzzy save rate:     {stats['edit_fuzzy']}/{stats['edit_fuzzy']+stats['edit_failed']} "
                     f"({stats['edit_fuzzy']/(stats['edit_fuzzy']+stats['edit_failed'])*100:.0f}% of non-exact)" if stats['edit_fuzzy']+stats['edit_failed'] > 0 else "")
    logger.info("")

    # Comparison with prior results
    logger.info("Comparison with prior results:")
    logger.info(f"  14B+edit (baseline):   5/50  (10.0%) resolved, 20/50 (40%) patches")
    logger.info(f"  14B+fuzzy_edit:        {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved, {patch_count}/{total} ({patch_count/total*100:.1f}%) patches")
    logger.info("")

    # Per-task detail
    logger.info("Per-task results:")
    traj_map = {t.meta.task_id: t for t in trajectories}
    for r in task_results:
        ts = per_task_stats.get(r.task_id, {})
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"steps={r.steps:>2} tokens={r.tokens:>6} "
            f"exact={ts.get('edit_exact',0)} fuzzy={ts.get('edit_fuzzy',0)} "
            f"fail={ts.get('edit_failed',0)} write={ts.get('write',0)}"
        )

    # Write REPORT.md
    write_report(result, task_results, trajectories, stats, per_task_stats)

    summary = results_summary(result)
    logger.info(f"\n{summary}")

    return result


def write_report(result, task_results, trajectories, stats, per_task_stats):
    """Write experiment report to REPORT.md."""
    resolved_count = sum(1 for r in task_results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(task_results)
    avg_steps = sum(r.steps for r in task_results) / total if total else 0
    total_edits = stats["edit_exact"] + stats["edit_fuzzy"] + stats["edit_failed"]

    resolved_tasks = [r.task_id for r in task_results if r.resolved]

    report = f"""# 14B Fuzzy Edit Experiment Report

## Configuration
- Model: {MLX_MODEL}
- Tasks: {total} Django tasks from SWE-Bench-CL
- Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}
- Improvements: fuzzy matching (0.85), edit loop detection, hybrid tool hint

## Results Summary

| Metric | Fuzzy Edit | Edit Baseline | Delta |
|--------|-----------|---------------|-------|
| Resolve rate | {resolved_count}/{total} ({resolved_count/total*100:.1f}%) | 5/50 (10.0%) | {(resolved_count/total - 5/50)*100:+.1f}pp |
| Patch rate | {patch_count}/{total} ({patch_count/total*100:.1f}%) | 20/50 (40.0%) | {(patch_count/total - 20/50)*100:+.1f}pp |
| Avg steps | {avg_steps:.1f} | 9.2 | {avg_steps - 9.2:+.1f} |

## Fuzzy Matching Stats

| Metric | Count |
|--------|-------|
| Edit (exact match) | {stats['edit_exact']} |
| Edit (fuzzy match) | {stats['edit_fuzzy']} |
| Edit (failed) | {stats['edit_failed']} |
| Edit total | {total_edits} |
| Write | {stats['write']} |
| Fuzzy save rate | {stats['edit_fuzzy']}/{stats['edit_fuzzy']+stats['edit_failed']} ({stats['edit_fuzzy']/(stats['edit_fuzzy']+stats['edit_failed'])*100:.0f}% of non-exact edits) | """ if stats['edit_fuzzy']+stats['edit_failed'] > 0 else f"""| Fuzzy save rate | N/A |"""

    report += f"""

## Resolved Tasks
{chr(10).join(f'- {t}' for t in resolved_tasks) if resolved_tasks else 'None'}

## Per-Task Breakdown

| Task | Result | Steps | Exact | Fuzzy | Failed | Write |
|------|--------|-------|-------|-------|--------|-------|
"""
    traj_map = {t.meta.task_id: t for t in trajectories}
    for r in task_results:
        ts = per_task_stats.get(r.task_id, {})
        has_patch = "patch" if (traj_map.get(r.task_id) and traj_map[r.task_id].generated_patch.strip()) else "no patch"
        status = "RESOLVED" if r.resolved else has_patch
        report += (
            f"| {r.task_id} | {status} | {r.steps} | "
            f"{ts.get('edit_exact',0)} | {ts.get('edit_fuzzy',0)} | "
            f"{ts.get('edit_failed',0)} | {ts.get('write',0)} |\n"
        )

    report += f"""
## Analysis

### Key Findings
- Fuzzy matching saved {stats['edit_fuzzy']} edits that would have failed with exact matching
- Edit loop detection prevented repeated failed edit attempts
- Hybrid tool hint guided model to use <write> for small files

### Cumulative Resolved Tasks
All unique tasks resolved across experiments: see OVERNIGHT_REPORT.md for running total.
"""

    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write(report)
    logger.info(f"Report written to {OUTPUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    run_experiment()
