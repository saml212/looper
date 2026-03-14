#!/usr/bin/env python3
"""32B Base Condition Pilot — 10 SWE-Bench tasks.

Tests whether qwen2.5-coder:32b has a high enough base resolve rate (>20%)
to serve as a foundation for future LoRA experiments. All prior experiments
(1-4, 6, 7, 9) converged on the finding that 7B's 8% base resolve rate
means 92% of training trajectories are failures, making any LoRA training
counterproductive.

This is BASE CONDITION ONLY — no LoRA training, no MLX. Uses Ollama only.
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
from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5-coder:32b"
OLLAMA_URL = "http://localhost:11434"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/phase1_32b_pilot")

# 32B should be smarter — give it room to work
MAX_STEPS = 15
MAX_TOKENS = 4096

# Use test tasks 26-35 (same slice as Experiments 3/4 for comparability)
NUM_TEST_TASKS = 10


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


def run_32b_pilot():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    traj_dir = OUTPUT_DIR / "trajectories" / "base"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks — use test tasks for evaluation (same split as all prior experiments)
    logger.info("=" * 60)
    logger.info("LOOPER Phase 1 — 32B Base Condition Pilot")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Tasks: {NUM_TEST_TASKS} test tasks (positions 26-35)")
    logger.info(f"Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}")
    logger.info("=" * 60)

    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    _, test_tasks_all = split_tasks(all_tasks, train_size=25, seed=None)
    test_tasks = test_tasks_all[:NUM_TEST_TASKS]

    logger.info(f"Test tasks ({len(test_tasks)}):")
    for t in test_tasks:
        logger.info(f"  {t.instance_id} (pos {t.sequence_position})")

    # Run base model via Ollama
    logger.info("")
    logger.info("Running base model (Ollama 32B)...")
    task_start_times = {}

    def on_complete(tid, traj):
        elapsed = time.time() - task_start_times.get(tid, time.time())
        logger.info(
            f"  {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, "
            f"patch={'yes' if traj.generated_patch.strip() else 'no'}, "
            f"{elapsed:.0f}s)"
        )

    # Track per-task timing
    original_collect = collect_trajectories

    trajectories = collect_trajectories(
        tasks=test_tasks,
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
        trajectories, test_tasks, WORKSPACE_ROOT, "base_32b"
    )

    # Compute metrics
    resolved_count = sum(1 for r in results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(results)
    avg_steps = sum(r.steps for r in results) / total if total else 0
    avg_tokens = sum(r.tokens for r in results) / total if total else 0

    # Save results
    config = ExperimentConfig(
        name="phase1_32b_pilot",
        experiment_id="phase1_32b_base_pilot",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="base_only",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=results,
        forward_transfer=0.0,  # No adapted condition
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "experiment_result.json")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("32B BASE CONDITION PILOT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info("")

    # Comparison with prior models
    logger.info("Comparison with prior results (same 10 tasks):")
    logger.info(f"  7B base:  1/10 (10%) resolved, 3/10 patches")
    logger.info(f"  14B base: 0/25 (0%)  resolved, 11/25 patches (full run)")
    logger.info(f"  32B base: {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved, {patch_count}/{total} patches")
    logger.info("")

    if resolved_count / total >= 0.20:
        logger.info("SUCCESS: 32B achieves >=20% resolve rate!")
        logger.info("This model is viable for LoRA experiments.")
    else:
        logger.info(f"32B resolve rate ({resolved_count/total*100:.1f}%) below 20% threshold.")
        logger.info("Consider: larger model, different prompts, or external gold trajectories.")

    # Per-task detail
    logger.info("")
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
    run_32b_pilot()
