#!/usr/bin/env python3
"""Resume Phase 1 full experiment from Condition 3 (base + LoRA).

Conditions 1 (base) and 2 (base + RAG) are already complete.
This script:
1. Re-verifies conditions 1+2 from saved trajectories
2. Starts the MLX server with the LoRA adapter
3. Runs condition 3 (base + LoRA) — 25 test tasks
4. Runs condition 4 (base + LoRA + RAG) — 25 test tasks
5. Computes comparison metrics and saves results

Resume-safe: if a trajectory file already exists, it's loaded instead of re-run.
Can be run directly: cd looper && .venv/bin/python run_phase1_full_resume.py
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/phase1_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "experiment_resume.log"),
    ],
)
logger = logging.getLogger(__name__)

# Config
CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
CACHED_BASE_TRAJ = Path("/Volumes/1TB_SSD/looper/results/phase1/trajectories/base")
CACHED_ADAPTER = Path("/Volumes/1TB_SSD/looper/results/phase1/adapter")

MODEL = "qwen2.5-coder:7b"
HF_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MLX_PORT = 8080
MAX_STEPS = 15


def verify_task(task, patch, workspace_root):
    from looper.evaluators.patch_verifier import verify_patch_tests
    if not patch.strip():
        return {"resolved": False, "fail_to_pass_passed": 0,
                "fail_to_pass_total": len(task.fail_to_pass),
                "error": "No patch", "test_output": ""}
    return verify_patch_tests(task, patch, workspace_root, timeout=300)


def start_mlx_server():
    """Start MLX server with LoRA adapter. Returns the subprocess."""
    import httpx

    # Check if already running
    try:
        resp = httpx.get(f"http://127.0.0.1:{MLX_PORT}/v1/models", timeout=3.0)
        if resp.status_code == 200:
            logger.info(f"MLX server already running on port {MLX_PORT}")
            return None  # Externally managed
    except Exception:
        pass

    venv_bin = Path(sys.executable).parent
    mlx_server = str(venv_bin / "mlx_lm.server")
    cmd = [mlx_server, "--model", HF_MODEL,
           "--adapter-path", str(CACHED_ADAPTER),
           "--port", str(MLX_PORT)]

    logger.info(f"Starting MLX server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(30):
        time.sleep(2)
        try:
            httpx.get(f"http://127.0.0.1:{MLX_PORT}/v1/models", timeout=5.0)
            logger.info(f"MLX server ready on port {MLX_PORT} (PID {proc.pid})")
            return proc
        except Exception:
            continue

    proc.terminate()
    raise RuntimeError("MLX server failed to start")


def stop_mlx_server(proc):
    if proc:
        proc.terminate()
        proc.wait(timeout=10)
        logger.info("MLX server stopped")


def main():
    from looper.collectors.trajectory_store import (
        collect_trajectories, load_all_trajectories,
    )
    from looper.agent.ollama_client import openai_chat
    from looper.evaluators.metrics import forward_transfer, compare_conditions
    from looper.evaluators.rag import build_rag_index, retrieve_context
    from looper.evaluators.results_io import save_results, results_summary
    from looper.models import ExperimentConfig, ExperimentResult, TaskResult
    from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks

    started_at = datetime.now(timezone.utc).isoformat()

    # Load tasks (same split as original run)
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, test_tasks = split_tasks(all_tasks, train_size=25)
    task_map = {t.instance_id: t for t in all_tasks}

    logger.info("=" * 60)
    logger.info("LOOPER Phase 1 — RESUME (Conditions 3+4)")
    logger.info(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")
    logger.info("=" * 60)

    # ========================================
    # Condition 1: Base (re-verify from cache)
    # ========================================
    logger.info("=" * 60)
    logger.info("CONDITION 1: Base model (cached, re-verifying)")
    logger.info("=" * 60)

    all_trajs = load_all_trajectories(CACHED_BASE_TRAJ)
    traj_map = {t.meta.task_id: t for t in all_trajs}

    base_results = []
    for task in test_tasks:
        traj = traj_map.get(task.instance_id)
        if traj is None:
            base_results.append(TaskResult(
                task_id=task.instance_id, condition="base",
                resolved=False, steps=0, tokens=0, duration_seconds=0.0,
                error="No trajectory"))
            continue
        vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
        status = "PASS" if vr["resolved"] else "FAIL"
        logger.info(f"  [{status}] {task.instance_id} ({traj.meta.total_steps} steps)")
        base_results.append(TaskResult(
            task_id=task.instance_id, condition="base",
            resolved=vr["resolved"], steps=traj.meta.total_steps,
            tokens=traj.meta.total_tokens, duration_seconds=0.0))

    base_resolved = sum(1 for r in base_results if r.resolved)
    logger.info(f"  Base: {base_resolved}/{len(base_results)} resolved")

    # ========================================
    # Condition 2: Base + RAG (re-verify from saved trajectories)
    # ========================================
    logger.info("=" * 60)
    logger.info("CONDITION 2: Base + RAG (re-verifying saved trajectories)")
    logger.info("=" * 60)

    rag_traj_dir = OUTPUT_DIR / "trajectories" / "base_rag"
    rag_trajs = load_all_trajectories(rag_traj_dir)
    rag_traj_map = {t.meta.task_id: t for t in rag_trajs}

    rag_results = []
    for task in test_tasks:
        traj = rag_traj_map.get(task.instance_id)
        if traj is None:
            rag_results.append(TaskResult(
                task_id=task.instance_id, condition="base_rag",
                resolved=False, steps=0, tokens=0, duration_seconds=0.0,
                error="No trajectory"))
            continue
        vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
        status = "PASS" if vr["resolved"] else "FAIL"
        logger.info(f"  [{status}] {task.instance_id} ({traj.meta.total_steps} steps)")
        rag_results.append(TaskResult(
            task_id=task.instance_id, condition="base_rag",
            resolved=vr["resolved"], steps=traj.meta.total_steps,
            tokens=traj.meta.total_tokens, duration_seconds=0.0))

    rag_resolved = sum(1 for r in rag_results if r.resolved)
    logger.info(f"  Base+RAG: {rag_resolved}/{len(rag_results)} resolved")

    # ========================================
    # Start MLX server for LoRA conditions
    # ========================================
    mlx_proc = start_mlx_server()
    mlx_base_url = f"http://127.0.0.1:{MLX_PORT}"

    try:
        # ========================================
        # Condition 3: Base + LoRA
        # ========================================
        logger.info("=" * 60)
        logger.info("CONDITION 3: Base + LoRA")
        logger.info("=" * 60)

        lora_traj_dir = OUTPUT_DIR / "trajectories" / "base_lora"
        lora_traj_dir.mkdir(parents=True, exist_ok=True)

        lora_trajs = collect_trajectories(
            tasks=test_tasks, output_dir=lora_traj_dir, workspace_root=WORKSPACE_ROOT,
            model=HF_MODEL, base_url=mlx_base_url, max_steps=MAX_STEPS,
            on_complete=lambda tid, t: logger.info(f"  Agent: {tid} -> {t.outcome}"),
            chat_fn=openai_chat,
        )

        lora_results = []
        for traj in lora_trajs:
            task = task_map[traj.meta.task_id]
            vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
            status = "PASS" if vr["resolved"] else "FAIL"
            logger.info(f"  [{status}] {task.instance_id}")
            lora_results.append(TaskResult(
                task_id=traj.meta.task_id, condition="base_lora",
                resolved=vr["resolved"], steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens, duration_seconds=0.0))

        lora_resolved = sum(1 for r in lora_results if r.resolved)
        logger.info(f"  Base+LoRA: {lora_resolved}/{len(lora_results)} resolved")

        # ========================================
        # Condition 4: Base + LoRA + RAG
        # ========================================
        logger.info("=" * 60)
        logger.info("CONDITION 4: Base + LoRA + RAG")
        logger.info("=" * 60)

        train_trajs = load_all_trajectories(CACHED_BASE_TRAJ)
        train_traj_subset = [t for t in train_trajs
                             if t.meta.task_id in {tk.instance_id for tk in train_tasks}]
        rag_index = build_rag_index(train_traj_subset, train_tasks)
        logger.info(f"  RAG index: {len(train_traj_subset)} trajectories indexed")

        lora_rag_traj_dir = OUTPUT_DIR / "trajectories" / "base_lora_rag"
        lora_rag_traj_dir.mkdir(parents=True, exist_ok=True)

        rag_contexts = {}
        for task in test_tasks:
            ctx = retrieve_context(task, rag_index, train_tasks, top_k=3, max_context_chars=3000)
            rag_contexts[task.instance_id] = ctx

        lora_rag_trajs = collect_trajectories(
            tasks=test_tasks, output_dir=lora_rag_traj_dir, workspace_root=WORKSPACE_ROOT,
            model=HF_MODEL, base_url=mlx_base_url, max_steps=MAX_STEPS,
            on_complete=lambda tid, t: logger.info(f"  Agent: {tid} -> {t.outcome}"),
            chat_fn=openai_chat, rag_contexts=rag_contexts,
        )

        lora_rag_results = []
        for traj in lora_rag_trajs:
            task = task_map[traj.meta.task_id]
            vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
            status = "PASS" if vr["resolved"] else "FAIL"
            logger.info(f"  [{status}] {task.instance_id}")
            lora_rag_results.append(TaskResult(
                task_id=traj.meta.task_id, condition="base_lora_rag",
                resolved=vr["resolved"], steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens, duration_seconds=0.0))

        lora_rag_resolved = sum(1 for r in lora_rag_results if r.resolved)
        logger.info(f"  Base+LoRA+RAG: {lora_rag_resolved}/{len(lora_rag_results)} resolved")

    finally:
        stop_mlx_server(mlx_proc)

    # ========================================
    # Compute metrics and save
    # ========================================
    all_results = base_results + rag_results + lora_results + lora_rag_results

    results_by_condition = {}
    for r in all_results:
        results_by_condition.setdefault(r.condition, []).append(r)

    comparison = compare_conditions(results_by_condition)
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)
    for cond, metrics in comparison.items():
        logger.info(f"  {cond:15s}: resolve={metrics['resolve_rate']:.1%}, "
                     f"steps={metrics['avg_steps']:.1f}, tokens={metrics['avg_tokens']:.0f}")

    ft_lora = forward_transfer(base_results, lora_results)
    ft_rag = forward_transfer(base_results, rag_results)
    ft_both = forward_transfer(base_results, lora_rag_results)
    logger.info(f"\nForward Transfer:")
    logger.info(f"  LoRA:      {ft_lora:+.4f}")
    logger.info(f"  RAG:       {ft_rag:+.4f}")
    logger.info(f"  LoRA+RAG:  {ft_both:+.4f}")

    config = ExperimentConfig(
        name="phase1_full_4conditions",
        experiment_id="phase1_4cond",
        repo="django/django",
        model_name=MODEL,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="full_replay",
        lora_rank=16,
        seed=0,
    )

    result = ExperimentResult(
        config=config,
        task_results=all_results,
        forward_transfer=ft_lora,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    save_results(result, OUTPUT_DIR / "experiment_result.json")

    detail = {
        "comparison": comparison,
        "forward_transfer": {"lora": ft_lora, "rag": ft_rag, "lora_rag": ft_both},
        "per_task": [],
    }
    for task in test_tasks:
        row = {"task_id": task.instance_id}
        for r in all_results:
            if r.task_id == task.instance_id:
                row[r.condition] = {"resolved": r.resolved, "steps": r.steps}
        detail["per_task"].append(row)

    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(detail, indent=2))

    summary = results_summary(result)
    logger.info(f"\n{summary}")
    logger.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
