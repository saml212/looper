#!/usr/bin/env python3
"""Framework Fix Pilot — 3 SWE-Bench tasks with improved agent.

Tests whether the 4 agent framework fixes (line-range read, context pruning,
few-shot examples, loop detection) improve 7B resolve rate.

Uses the same 3 tasks as the 14B pilot for direct comparison:
  django__django-10914, django__django-10999, django__django-11066
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

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_framework_fix_7b")

MAX_STEPS = 15
MAX_TOKENS = 4096

TARGET_TASKS = [
    "django__django-10914",
    "django__django-10999",
    "django__django-11066",
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
    logger.info("Framework Fix Pilot — 7B with Agent Improvements")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Fixes: line-range read, context pruning, few-shot, loop detection")
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

    # Run base model via Ollama
    logger.info("")
    logger.info("Running 7B with framework fixes (Ollama)...")

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
        trajectories, tasks, WORKSPACE_ROOT, "base_7b_framework_fix"
    )

    # Compute metrics
    resolved_count = sum(1 for r in results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(results)
    avg_steps = sum(r.steps for r in results) / total if total else 0
    avg_tokens = sum(r.tokens for r in results) / total if total else 0

    # Save results
    config = ExperimentConfig(
        name="experiment_framework_fix_7b",
        experiment_id="framework_fix_7b_pilot",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[],
        test_task_ids=[t.instance_id for t in tasks],
        strategy="base_with_framework_fixes",
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
    logger.info("FRAMEWORK FIX PILOT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info("")

    # Comparison
    logger.info("Comparison with prior results:")
    logger.info(f"  7B base (no fixes):  2/25 (8.0%) resolved")
    logger.info(f"  14B base (3 tasks):  0/3  (0.0%) resolved")
    logger.info(f"  32B base (3 tasks):  0/3  (0.0%) resolved")
    logger.info(f"  7B+fixes (3 tasks):  {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved")
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

    return result


if __name__ == "__main__":
    run_pilot()
