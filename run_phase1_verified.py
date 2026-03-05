#!/usr/bin/env python3
"""Re-verify Phase 1 results using proper FAIL_TO_PASS test verification.

Reuses existing agent trajectories from the original Phase 1 run but
verifies patches by running the actual Django test suite. This gives us
trustworthy resolve rates without re-running the expensive agent loop.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/phase1_verified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "experiment.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    from looper.collectors.trajectory_store import load_trajectory, load_all_trajectories
    from looper.evaluators.metrics import forward_transfer
    from looper.evaluators.patch_verifier import verify_patch_tests
    from looper.evaluators.results_io import save_results, results_summary
    from looper.models import (
        ExperimentConfig,
        ExperimentResult,
        TaskResult,
    )
    from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks

    started_at = datetime.now(timezone.utc).isoformat()

    # Config
    curriculum_path = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
    workspace_root = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
    original_traj_dir = Path("/Volumes/1TB_SSD/looper/results/phase1/trajectories/base")

    verify_detail_dir = OUTPUT_DIR / "verification"
    verify_detail_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and split tasks
    logger.info("=" * 60)
    logger.info("LOOPER Phase 1 — Re-verification with FAIL_TO_PASS tests")
    logger.info("=" * 60)
    curriculum = load_curriculum(curriculum_path)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, test_tasks = split_tasks(all_tasks, train_size=25)
    task_map = {t.instance_id: t for t in all_tasks}
    logger.info(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Load all existing trajectories
    all_trajectories = load_all_trajectories(original_traj_dir)
    traj_map = {t.meta.task_id: t for t in all_trajectories}
    logger.info(f"Loaded {len(all_trajectories)} cached trajectories")

    # Step 2: Verify ALL tasks (both train and test) with real tests
    logger.info("Step 2: Verifying all trajectories with FAIL_TO_PASS tests...")
    all_results: list[TaskResult] = []
    all_details: list[dict] = []

    for i, task in enumerate(all_tasks):
        traj = traj_map.get(task.instance_id)
        if traj is None:
            logger.warning(f"  [{i+1}/{len(all_tasks)}] {task.instance_id} — NO TRAJECTORY")
            continue

        logger.info(
            f"  [{i+1}/{len(all_tasks)}] {task.instance_id} "
            f"(outcome={traj.outcome}, patch={len(traj.generated_patch)} chars)"
        )

        t0 = time.time()
        if traj.generated_patch.strip():
            vr = verify_patch_tests(
                task=task,
                generated_patch=traj.generated_patch,
                workspace_root=workspace_root,
                timeout=300,
            )
            resolved = vr["resolved"]
            status = "PASS" if resolved else "FAIL"
            logger.info(
                f"    [{status}] {vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']} tests passed"
                + (f" — {vr['error']}" if vr["error"] else "")
            )
        else:
            vr = {
                "resolved": False,
                "fail_to_pass_passed": 0,
                "fail_to_pass_total": len(task.fail_to_pass),
                "error": "No patch generated",
                "test_output": "",
            }
            resolved = False
            logger.info(f"    [FAIL] No patch generated")

        duration = time.time() - t0
        condition = "train" if task.instance_id in {t.instance_id for t in train_tasks} else "test"

        all_results.append(TaskResult(
            task_id=task.instance_id,
            condition=condition,
            resolved=resolved,
            steps=traj.meta.total_steps,
            tokens=traj.meta.total_tokens,
            duration_seconds=duration,
        ))

        all_details.append({
            "task_id": task.instance_id,
            "condition": condition,
            "agent_outcome": traj.outcome,
            "patch_length": len(traj.generated_patch),
            "verification": {k: v for k, v in vr.items() if k != "test_output"},
            "verify_duration_s": round(duration, 1),
        })

    # Save verification details
    detail_path = verify_detail_dir / "all_verification.json"
    detail_path.write_text(json.dumps(all_details, indent=2))

    # Step 3: Compute metrics
    train_results = [r for r in all_results if r.condition == "train"]
    test_results = [r for r in all_results if r.condition == "test"]

    train_resolved = sum(1 for r in train_results if r.resolved)
    test_resolved = sum(1 for r in test_results if r.resolved)

    logger.info("=" * 60)
    logger.info("VERIFIED RESULTS")
    logger.info("=" * 60)
    logger.info(f"Train: {train_resolved}/{len(train_results)} resolved ({train_resolved/max(len(train_results),1):.1%})")
    logger.info(f"Test:  {test_resolved}/{len(test_results)} resolved ({test_resolved/max(len(test_results),1):.1%})")
    logger.info(f"Total: {train_resolved+test_resolved}/{len(all_results)} resolved")
    logger.info("")

    for r in all_results:
        status = "PASS" if r.resolved else "FAIL"
        logger.info(f"  [{status}] [{r.condition:5s}] {r.task_id} ({r.steps} steps)")

    # Step 4: Save experiment result
    experiment_config = ExperimentConfig(
        name="phase1_verified",
        experiment_id="phase1_verified_base_only",
        repo="django/django",
        model_name="qwen2.5-coder:7b",
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="full_replay",
        lora_rank=16,
        seed=0,
    )

    result = ExperimentResult(
        config=experiment_config,
        task_results=all_results,
        forward_transfer=0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    save_results(result, OUTPUT_DIR / "experiment_result.json")
    summary = results_summary(result)
    logger.info(f"\n{summary}")

    logger.info("=" * 60)
    logger.info(f"Results saved to {OUTPUT_DIR / 'experiment_result.json'}")
    logger.info(f"Verification details saved to {detail_path}")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    main()
