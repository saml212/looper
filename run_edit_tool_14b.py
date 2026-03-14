#!/usr/bin/env python3
"""14B Edit Tool Experiment — 50 SWE-Bench tasks with <edit> tool.

Tests the new <edit> tool (targeted find-replace) alongside the existing
framework fixes. Goal: collect 50+ resolved trajectories for future LoRA
training with diverse strategies.

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
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_edit_tool_14b")

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
    logger.info("14B Edit Tool Experiment — All 50 Django Tasks")
    logger.info(f"Model: {MLX_MODEL} via MLX server at {MLX_URL}")
    logger.info(f"New tool: <edit> (targeted find-replace)")
    logger.info(f"Fixes: line-range read, context pruning, few-shot, loop detection, skip-verification, code-fence-strip, EDIT TOOL")
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
        trajectories, tasks, WORKSPACE_ROOT, "base_14b_edit_tool"
    )

    # Compute metrics
    resolved_count = sum(1 for r in task_results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(task_results)
    avg_steps = sum(r.steps for r in task_results) / total if total else 0
    avg_tokens = sum(r.tokens for r in task_results) / total if total else 0

    # Count edit vs write tool usage
    edit_count = 0
    write_count = 0
    for traj in trajectories:
        for step in traj.steps:
            for tc in step.tool_calls:
                if tc.tool_name == "edit":
                    edit_count += 1
                elif tc.tool_name == "write":
                    write_count += 1

    # Save results
    config = ExperimentConfig(
        name="experiment_edit_tool_14b",
        experiment_id="edit_tool_14b",
        repo="django/django",
        model_name=MLX_MODEL,
        train_task_ids=[],
        test_task_ids=[t.instance_id for t in tasks],
        strategy="base_with_edit_tool",
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

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("14B EDIT TOOL EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info(f"  Edit tool uses:  {edit_count}")
    logger.info(f"  Write tool uses: {write_count}")
    logger.info(f"  Edit adoption:   {edit_count}/{edit_count+write_count} ({edit_count/(edit_count+write_count)*100:.0f}%)" if edit_count + write_count > 0 else "  No edits or writes")
    logger.info("")

    # Comparison with prior results
    logger.info("Comparison with prior results:")
    logger.info(f"  7B base (no fixes):       2/25  (8.0%) resolved")
    logger.info(f"  7B+fixes (15 tasks):      3/15  (20.0%) resolved")
    logger.info(f"  14B+fixes (15 tasks):     4/15  (26.7%) resolved")
    logger.info(f"  14B+edit (50 tasks):      {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved")
    logger.info("")

    # Per-task detail
    logger.info("Per-task results:")
    traj_map = {t.meta.task_id: t for t in trajectories}
    for r in task_results:
        t = traj_map.get(r.task_id)
        has_patch = t.generated_patch.strip() if t else ""
        # Count tool usage per task
        task_edits = 0
        task_writes = 0
        if t:
            for step in t.steps:
                for tc in step.tool_calls:
                    if tc.tool_name == "edit":
                        task_edits += 1
                    elif tc.tool_name == "write":
                        task_writes += 1
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"steps={r.steps:>2} tokens={r.tokens:>6} "
            f"patch={'yes' if has_patch else 'no':>3} "
            f"edit={task_edits} write={task_writes}"
        )

    summary = results_summary(result)
    logger.info(f"\n{summary}")

    return result


if __name__ == "__main__":
    run_experiment()
