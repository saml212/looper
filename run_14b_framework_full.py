#!/usr/bin/env python3
"""14B Framework Full Run — 15 SWE-Bench tasks with improved agent + code fence fix.

Extends the 6-task pilot (50% resolve) to the full 15-task set used for 7B
comparison (3 pilot + 12 expanded). Includes all framework fixes:
  - line-range read, context pruning, few-shot, loop detection
  - skip-verification prompt
  - code fence stripping (key fix for 14B)

7B baseline on same 15 tasks: 3/15 (20.0%) resolved.
14B pilot on 6 tasks: 3/6 (50.0%) resolved.
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
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_full")

MAX_STEPS = 15
MAX_TOKENS = 4096

# All 15 tasks: 3 from pilot + 12 from expanded (same as 7B total)
TARGET_TASKS = [
    # Pilot tasks (indices 3-5)
    "django__django-10914",
    "django__django-10999",
    "django__django-11066",
    # Expanded tasks (indices 6-17)
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


def run_full():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    traj_dir = OUTPUT_DIR / "trajectories" / "base"
    traj_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("14B Framework Full Run — 14B with All Fixes")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Fixes: line-range read, context pruning, few-shot, loop detection, skip-verification, code-fence-strip")
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

    # Skip tasks already done in pilot (reuse those results)
    pilot_results_path = Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_pilot_v2/experiment_result.json")
    pilot_task_ids = set()
    pilot_trajectories = []
    pilot_task_results = []
    if pilot_results_path.exists():
        pilot_data = json.loads(pilot_results_path.read_text())
        for tr in pilot_data["task_results"]:
            pilot_task_ids.add(tr["task_id"])
            pilot_task_results.append(tr)
        logger.info(f"Reusing {len(pilot_task_ids)} results from pilot: {pilot_task_ids}")
        # Load pilot trajectories
        pilot_traj_dir = Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_pilot_v2/trajectories/base")
        if pilot_traj_dir.exists():
            from looper.collectors.trajectory_store import load_trajectory
            for traj_file in sorted(pilot_traj_dir.glob("*.json")):
                traj = load_trajectory(traj_file)
                if traj.meta.task_id in pilot_task_ids:
                    pilot_trajectories.append(traj)

    remaining_tasks = [t for t in tasks if t.instance_id not in pilot_task_ids]
    logger.info(f"Running {len(remaining_tasks)} new tasks...")

    # Run remaining tasks via Ollama
    def on_complete(tid, traj):
        logger.info(
            f"  {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, "
            f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
        )

    new_trajectories = []
    if remaining_tasks:
        new_trajectories = collect_trajectories(
            tasks=remaining_tasks,
            output_dir=traj_dir,
            workspace_root=WORKSPACE_ROOT,
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            max_steps=MAX_STEPS,
            max_tokens=MAX_TOKENS,
            on_complete=on_complete,
        )

    # Combine pilot + new trajectories in task order
    traj_map = {}
    for t in pilot_trajectories:
        traj_map[t.meta.task_id] = t
    for t in new_trajectories:
        traj_map[t.meta.task_id] = t
    all_trajectories = [traj_map[tid] for tid in TARGET_TASKS if tid in traj_map]

    # Evaluate new trajectories
    logger.info("")
    logger.info("Evaluating new tasks with FAIL_TO_PASS verification...")
    new_results = evaluate_trajectories(
        new_trajectories,
        [t for t in tasks if t.instance_id not in pilot_task_ids],
        WORKSPACE_ROOT,
        "base_14b_framework_full",
    )

    # Combine all results in task order
    result_map = {}
    for tr in pilot_task_results:
        result_map[tr["task_id"]] = TaskResult(
            task_id=tr["task_id"],
            condition="base_14b_framework_full",
            resolved=tr["resolved"],
            steps=tr["steps"],
            tokens=tr["tokens"],
            duration_seconds=tr.get("duration_seconds", 0.0),
        )
    for tr in new_results:
        result_map[tr.task_id] = tr
    all_results = [result_map[tid] for tid in TARGET_TASKS if tid in result_map]

    # Compute metrics
    resolved_count = sum(1 for r in all_results if r.resolved)
    patch_count = sum(1 for t in all_trajectories if t.generated_patch.strip())
    total = len(all_results)
    avg_steps = sum(r.steps for r in all_results) / total if total else 0
    avg_tokens = sum(r.tokens for r in all_results) / total if total else 0

    # Save results
    config = ExperimentConfig(
        name="experiment_framework_14b_full",
        experiment_id="framework_14b_full",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[],
        test_task_ids=TARGET_TASKS,
        strategy="base_with_framework_fixes_v3",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=all_results,
        forward_transfer=0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "experiment_result.json")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("14B FRAMEWORK FULL RUN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info("")

    # Comparison
    logger.info("Comparison with prior results:")
    logger.info(f"  7B base (no fixes):      2/25 (8.0%) resolved")
    logger.info(f"  14B base (no fixes):     0/25 (0.0%) resolved")
    logger.info(f"  7B+fixes (15 tasks):     3/15 (20.0%) resolved")
    logger.info(f"  14B+fixes (6 pilot):     3/6  (50.0%) resolved")
    logger.info(f"  14B+fixes (15 full):     {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved")
    logger.info("")

    # Per-task detail
    logger.info("Per-task results:")
    for r in all_results:
        t = traj_map.get(r.task_id)
        has_patch = t.generated_patch.strip() if t else ""
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"steps={r.steps:>2} tokens={r.tokens:>6} "
            f"patch={'yes' if has_patch else 'no':>3}"
        )

    summary = results_summary(result)
    logger.info(f"\n{summary}")

    return result


if __name__ == "__main__":
    run_full()
