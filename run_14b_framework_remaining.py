#!/usr/bin/env python3
"""14B Framework Remaining — 10 more SWE-Bench tasks to complete 25-task eval.

Extends the 15-task run (4/15 = 26.7%) to full 25 tasks for direct comparison
with original Phase 1 (25-task) results.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.collectors.trajectory_store import collect_trajectories
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import ExperimentConfig, ExperimentResult, TaskResult
from looper.tasks.loader import get_repo_tasks, load_curriculum, get_task_by_id

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen2.5-coder:14b"
OLLAMA_URL = "http://localhost:11434"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_remaining")

MAX_STEPS = 15
MAX_TOKENS = 4096

# Indices 18-27 from django sequence (completing the 25-task test set)
TARGET_TASKS = [
    "django__django-11820",
    "django__django-11880",
    "django__django-11951",
    "django__django-11964",
    "django__django-12125",
    "django__django-12193",
    "django__django-12209",
    "django__django-12276",
    "django__django-12304",
    "django__django-12308",
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
            logger.info(f"  Verify {traj.meta.task_id}: SKIP (no patch)")
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


def run():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    traj_dir = OUTPUT_DIR / "trajectories" / "base"
    traj_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("14B Framework Remaining — 10 tasks (indices 18-27)")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Tasks: {len(TARGET_TASKS)}")
    logger.info(f"Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}")
    logger.info("=" * 60)

    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    tasks = []
    for tid in TARGET_TASKS:
        task = get_task_by_id(all_tasks, tid)
        if task is None:
            logger.error(f"Task {tid} not found!")
            sys.exit(1)
        tasks.append(task)

    for t in tasks:
        logger.info(f"  {t.instance_id} (pos {t.sequence_position})")

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

    logger.info("\nEvaluating with FAIL_TO_PASS verification...")
    results = evaluate_trajectories(
        trajectories, tasks, WORKSPACE_ROOT, "base_14b_framework_remaining"
    )

    resolved_count = sum(1 for r in results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(results)
    avg_steps = sum(r.steps for r in results) / total if total else 0

    config = ExperimentConfig(
        name="experiment_framework_14b_remaining",
        experiment_id="framework_14b_remaining",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[],
        test_task_ids=TARGET_TASKS,
        strategy="base_with_framework_fixes_v3",
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

    # Combined 25-task summary
    prior_resolved = 4  # from 15-task run
    combined_resolved = prior_resolved + resolved_count
    combined_total = 15 + total

    logger.info("")
    logger.info("=" * 60)
    logger.info("14B REMAINING 10 TASKS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  This batch:  {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:  {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:   {avg_steps:.1f}")
    logger.info(f"  COMBINED 25: {combined_resolved}/{combined_total} ({combined_resolved/combined_total*100:.1f}%)")
    logger.info("")
    logger.info("  Comparison:")
    logger.info(f"    7B no fixes (25):  2/25 (8.0%)")
    logger.info(f"    14B no fixes (25): 0/25 (0.0%)")
    logger.info(f"    14B+fixes (25):    {combined_resolved}/{combined_total} ({combined_resolved/combined_total*100:.1f}%)")
    logger.info("")

    for r, t in zip(results, trajectories):
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"steps={r.steps:>2} "
            f"patch={'yes' if t.generated_patch.strip() else 'no':>3}"
        )

    return result


if __name__ == "__main__":
    run()
