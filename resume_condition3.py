#!/usr/bin/env python3
"""Resume Phase 1 Condition 3 (Base+LoRA) with robust error handling.

Starts the MLX server with the cached LoRA adapter, runs all 25 test tasks,
saves trajectories to base_lora/, then verifies and prints results.

Key improvements over original run:
- Per-task try/except so one failure doesn't kill the batch
- MLX server stderr/stdout captured to a log file for debugging
- Progress counter and ETA logging
- Graceful shutdown on KeyboardInterrupt
"""

import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --- Config (matches run_phase1_full.py) ---
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/phase1_full")
CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
CACHED_BASE_TRAJ = Path("/Volumes/1TB_SSD/looper/results/phase1/trajectories/base")
CACHED_ADAPTER = Path("/Volumes/1TB_SSD/looper/results/phase1/adapter")
HF_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MLX_PORT = 8080
MAX_STEPS = 15

# --- Logging ---
LOG_FILE = OUTPUT_DIR / "condition3_resume.log"
MLX_LOG = OUTPUT_DIR / "mlx_server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


def start_mlx_server(adapter_path):
    """Start MLX server with LoRA adapter, capturing output."""
    import httpx

    venv_bin = Path(sys.executable).parent
    mlx_server = str(venv_bin / "mlx_lm.server")
    cmd = [mlx_server, "--model", HF_MODEL, "--port", str(MLX_PORT)]
    if adapter_path:
        cmd += ["--adapter-path", str(adapter_path)]

    logger.info(f"Starting MLX server: {' '.join(cmd)}")
    mlx_log_fh = open(MLX_LOG, "w")
    proc = subprocess.Popen(cmd, stdout=mlx_log_fh, stderr=mlx_log_fh)

    for i in range(30):
        time.sleep(2)
        try:
            resp = httpx.get(f"http://127.0.0.1:{MLX_PORT}/v1/models", timeout=5.0)
            if resp.status_code == 200:
                logger.info(f"MLX server ready on port {MLX_PORT}")
                return proc, mlx_log_fh
        except Exception:
            continue

    proc.terminate()
    mlx_log_fh.close()
    raise RuntimeError(f"MLX server failed to start. Check {MLX_LOG}")


def stop_mlx_server(proc, log_fh):
    """Stop MLX server gracefully."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_fh:
        log_fh.close()


def verify_task(task, patch, workspace_root):
    """Verify a single patch with FAIL_TO_PASS tests."""
    from looper.evaluators.patch_verifier import verify_patch_tests
    if not patch.strip():
        return {"resolved": False, "fail_to_pass_passed": 0,
                "fail_to_pass_total": len(task.fail_to_pass),
                "error": "No patch", "test_output": ""}
    return verify_patch_tests(task, patch, workspace_root, timeout=300)


def main():
    from looper.agent.ollama_client import openai_chat
    from looper.agent.runner import run_agent
    from looper.collectors.trajectory_store import (
        load_trajectory, save_trajectory, load_all_trajectories,
    )
    from looper.models import TaskResult
    from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks

    started_at = datetime.now(timezone.utc).isoformat()

    # Load tasks (same split as run_phase1_full.py)
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    _, test_tasks = split_tasks(all_tasks, train_size=25)
    task_map = {t.instance_id: t for t in all_tasks}

    logger.info("=" * 60)
    logger.info("CONDITION 3 RESUME: Base + LoRA")
    logger.info(f"Tasks: {len(test_tasks)}, Max steps: {MAX_STEPS}")
    logger.info("=" * 60)

    traj_dir = OUTPUT_DIR / "trajectories" / "base_lora"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Check how many already exist (resume support)
    existing = list(traj_dir.glob("*.json"))
    logger.info(f"Existing trajectories: {len(existing)}/{len(test_tasks)}")

    # Start MLX server
    mlx_proc, mlx_log_fh = start_mlx_server(adapter_path=CACHED_ADAPTER)
    mlx_base_url = f"http://127.0.0.1:{MLX_PORT}"

    trajectories = []
    completed = len(existing)
    failed_tasks = []
    task_durations = []

    try:
        for i, task in enumerate(test_tasks):
            existing_path = traj_dir / f"{task.instance_id}.json"

            if existing_path.exists():
                logger.info(f"  [{i+1}/{len(test_tasks)}] SKIP (exists): {task.instance_id}")
                traj = load_trajectory(existing_path)
                trajectories.append(traj)
                continue

            logger.info(f"  [{i+1}/{len(test_tasks)}] RUNNING: {task.instance_id}")
            task_start = time.monotonic()

            try:
                traj = run_agent(
                    task=task,
                    workspace_root=WORKSPACE_ROOT,
                    model=HF_MODEL,
                    base_url=mlx_base_url,
                    max_steps=MAX_STEPS,
                    max_tokens=4096,
                    chat_fn=openai_chat,
                )
                save_trajectory(traj, traj_dir)
                trajectories.append(traj)
                completed += 1

                duration = time.monotonic() - task_start
                task_durations.append(duration)
                avg_dur = sum(task_durations) / len(task_durations)
                remaining = len(test_tasks) - (i + 1)
                eta_min = (avg_dur * remaining) / 60

                logger.info(
                    f"  [{i+1}/{len(test_tasks)}] DONE: {task.instance_id} "
                    f"-> {traj.outcome} ({traj.meta.total_steps} steps, "
                    f"{duration:.0f}s, ETA: {eta_min:.0f}min for {remaining} remaining)"
                )

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — stopping gracefully")
                raise
            except Exception as e:
                duration = time.monotonic() - task_start
                logger.error(
                    f"  [{i+1}/{len(test_tasks)}] FAILED: {task.instance_id} "
                    f"after {duration:.0f}s — {type(e).__name__}: {e}"
                )
                logger.error(traceback.format_exc())
                failed_tasks.append(task.instance_id)
                continue

    except KeyboardInterrupt:
        logger.info("Interrupted. Saving progress...")
    finally:
        stop_mlx_server(mlx_proc, mlx_log_fh)

    # Summary
    logger.info("=" * 60)
    logger.info("CONDITION 3 COLLECTION COMPLETE")
    logger.info(f"  Completed: {completed}/{len(test_tasks)}")
    logger.info(f"  Failed: {len(failed_tasks)}")
    if failed_tasks:
        logger.info(f"  Failed tasks: {failed_tasks}")
    logger.info("=" * 60)

    # Verify trajectories
    if trajectories:
        logger.info("Verifying trajectories...")
        results = []
        for traj in trajectories:
            task = task_map[traj.meta.task_id]
            vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
            status = "PASS" if vr["resolved"] else "FAIL"
            logger.info(f"  [{status}] {task.instance_id} ({traj.meta.total_steps} steps)")
            results.append(TaskResult(
                task_id=traj.meta.task_id, condition="base_lora",
                resolved=vr["resolved"], steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens, duration_seconds=0.0))

        resolved = sum(1 for r in results if r.resolved)
        logger.info(f"\n  Base+LoRA: {resolved}/{len(results)} resolved")

        # Save results JSON
        results_data = [r.model_dump() for r in results]
        (OUTPUT_DIR / "condition3_results.json").write_text(
            json.dumps(results_data, indent=2)
        )
        logger.info(f"Results saved to {OUTPUT_DIR / 'condition3_results.json'}")


if __name__ == "__main__":
    main()
