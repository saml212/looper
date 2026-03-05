#!/usr/bin/env python3
"""Run complete Phase 1 experiment with all 4 conditions and proper verification.

Conditions:
1. Base model — no adapter, no RAG
2. Base + RAG — no adapter, RAG context from train trajectories
3. Base + LoRA — adapted model, no RAG
4. Base + LoRA + RAG — adapted model + RAG context

Uses qwen2.5-coder:7b via Ollama for base conditions, MLX server for LoRA conditions.
Verifies all patches with FAIL_TO_PASS Django test suite.
"""

import json
import logging
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
        logging.FileHandler(OUTPUT_DIR / "experiment.log"),
    ],
)
logger = logging.getLogger(__name__)

# Config
CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
CACHED_BASE_TRAJ = Path("/Volumes/1TB_SSD/looper/results/phase1/trajectories/base")
CACHED_SYNTHESIS = Path("/Volumes/1TB_SSD/looper/results/phase1/synthesis")
CACHED_ADAPTER = Path("/Volumes/1TB_SSD/looper/results/phase1/adapter")

MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434"
HF_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MLX_PORT = 8080
MAX_STEPS = 15
LORA_RANK = 16
LORA_ITERS = 100


def verify_task(task, patch, workspace_root):
    """Verify a single patch with FAIL_TO_PASS tests."""
    from looper.evaluators.patch_verifier import verify_patch_tests
    if not patch.strip():
        return {"resolved": False, "fail_to_pass_passed": 0,
                "fail_to_pass_total": len(task.fail_to_pass),
                "error": "No patch", "test_output": ""}
    return verify_patch_tests(task, patch, workspace_root, timeout=300)


def run_condition_base(test_tasks, task_map):
    """Condition 1: Base model, no adapter, no RAG. Uses cached trajectories."""
    from looper.collectors.trajectory_store import load_all_trajectories
    from looper.models import TaskResult

    logger.info("=" * 60)
    logger.info("CONDITION 1: Base model (cached trajectories)")
    logger.info("=" * 60)

    all_trajs = load_all_trajectories(CACHED_BASE_TRAJ)
    traj_map = {t.meta.task_id: t for t in all_trajs}

    results = []
    for task in test_tasks:
        traj = traj_map.get(task.instance_id)
        if traj is None:
            logger.warning(f"  No trajectory for {task.instance_id}")
            results.append(TaskResult(
                task_id=task.instance_id, condition="base",
                resolved=False, steps=0, tokens=0, duration_seconds=0.0,
                error="No trajectory"))
            continue

        vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
        status = "PASS" if vr["resolved"] else "FAIL"
        logger.info(f"  [{status}] {task.instance_id} ({traj.meta.total_steps} steps)")
        results.append(TaskResult(
            task_id=task.instance_id, condition="base",
            resolved=vr["resolved"], steps=traj.meta.total_steps,
            tokens=traj.meta.total_tokens, duration_seconds=0.0))

    return results


def run_condition_rag(test_tasks, train_tasks, task_map, rag_index):
    """Condition 2: Base model + RAG context from train trajectories."""
    from looper.collectors.trajectory_store import collect_trajectories, save_trajectory
    from looper.evaluators.rag import retrieve_context
    from looper.models import TaskResult

    logger.info("=" * 60)
    logger.info("CONDITION 2: Base + RAG")
    logger.info("=" * 60)

    traj_dir = OUTPUT_DIR / "trajectories" / "base_rag"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Build RAG contexts for each test task
    rag_contexts = {}
    for task in test_tasks:
        ctx = retrieve_context(task, rag_index, train_tasks, top_k=3, max_context_chars=3000)
        rag_contexts[task.instance_id] = ctx
        if ctx:
            logger.info(f"  RAG for {task.instance_id}: {len(ctx)} chars")

    trajs = collect_trajectories(
        tasks=test_tasks, output_dir=traj_dir, workspace_root=WORKSPACE_ROOT,
        model=MODEL, base_url=OLLAMA_URL, max_steps=MAX_STEPS,
        on_complete=lambda tid, t: logger.info(f"  Agent: {tid} -> {t.outcome}"),
        rag_contexts=rag_contexts,
    )

    results = []
    for traj in trajs:
        task = task_map[traj.meta.task_id]
        vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
        status = "PASS" if vr["resolved"] else "FAIL"
        logger.info(f"  [{status}] {task.instance_id}")
        results.append(TaskResult(
            task_id=traj.meta.task_id, condition="base_rag",
            resolved=vr["resolved"], steps=traj.meta.total_steps,
            tokens=traj.meta.total_tokens, duration_seconds=0.0))

    return results


def run_condition_lora(test_tasks, task_map, mlx_base_url):
    """Condition 3: Base + LoRA adapter, no RAG."""
    from looper.collectors.trajectory_store import collect_trajectories
    from looper.agent.ollama_client import openai_chat
    from looper.models import TaskResult

    logger.info("=" * 60)
    logger.info("CONDITION 3: Base + LoRA")
    logger.info("=" * 60)

    traj_dir = OUTPUT_DIR / "trajectories" / "base_lora"
    traj_dir.mkdir(parents=True, exist_ok=True)

    trajs = collect_trajectories(
        tasks=test_tasks, output_dir=traj_dir, workspace_root=WORKSPACE_ROOT,
        model=HF_MODEL, base_url=mlx_base_url, max_steps=MAX_STEPS,
        on_complete=lambda tid, t: logger.info(f"  Agent: {tid} -> {t.outcome}"),
        chat_fn=openai_chat,
    )

    results = []
    for traj in trajs:
        task = task_map[traj.meta.task_id]
        vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
        status = "PASS" if vr["resolved"] else "FAIL"
        logger.info(f"  [{status}] {task.instance_id}")
        results.append(TaskResult(
            task_id=traj.meta.task_id, condition="base_lora",
            resolved=vr["resolved"], steps=traj.meta.total_steps,
            tokens=traj.meta.total_tokens, duration_seconds=0.0))

    return results


def run_condition_lora_rag(test_tasks, train_tasks, task_map, rag_index, mlx_base_url):
    """Condition 4: Base + LoRA adapter + RAG."""
    from looper.collectors.trajectory_store import collect_trajectories
    from looper.agent.ollama_client import openai_chat
    from looper.evaluators.rag import retrieve_context
    from looper.models import TaskResult

    logger.info("=" * 60)
    logger.info("CONDITION 4: Base + LoRA + RAG")
    logger.info("=" * 60)

    traj_dir = OUTPUT_DIR / "trajectories" / "base_lora_rag"
    traj_dir.mkdir(parents=True, exist_ok=True)

    rag_contexts = {}
    for task in test_tasks:
        ctx = retrieve_context(task, rag_index, train_tasks, top_k=3, max_context_chars=3000)
        rag_contexts[task.instance_id] = ctx

    trajs = collect_trajectories(
        tasks=test_tasks, output_dir=traj_dir, workspace_root=WORKSPACE_ROOT,
        model=HF_MODEL, base_url=mlx_base_url, max_steps=MAX_STEPS,
        on_complete=lambda tid, t: logger.info(f"  Agent: {tid} -> {t.outcome}"),
        chat_fn=openai_chat, rag_contexts=rag_contexts,
    )

    results = []
    for traj in trajs:
        task = task_map[traj.meta.task_id]
        vr = verify_task(task, traj.generated_patch, WORKSPACE_ROOT)
        status = "PASS" if vr["resolved"] else "FAIL"
        logger.info(f"  [{status}] {task.instance_id}")
        results.append(TaskResult(
            task_id=traj.meta.task_id, condition="base_lora_rag",
            resolved=vr["resolved"], steps=traj.meta.total_steps,
            tokens=traj.meta.total_tokens, duration_seconds=0.0))

    return results


def start_mlx_server(adapter_path=None):
    """Start MLX server for LoRA inference."""
    venv_bin = Path(sys.executable).parent
    mlx_server = str(venv_bin / "mlx_lm.server")
    cmd = [mlx_server, "--model", HF_MODEL, "--port", str(MLX_PORT)]
    if adapter_path:
        cmd += ["--adapter-path", str(adapter_path)]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import httpx
    for _ in range(30):
        time.sleep(2)
        try:
            httpx.get(f"http://127.0.0.1:{MLX_PORT}/v1/models", timeout=5.0)
            logger.info(f"  MLX server ready on port {MLX_PORT}")
            return proc
        except Exception:
            continue

    proc.terminate()
    raise RuntimeError("MLX server failed to start")


def stop_mlx_server(proc):
    """Stop MLX server."""
    if proc:
        proc.terminate()
        proc.wait(timeout=10)


def ensure_adapter():
    """Ensure LoRA adapter exists (use cached or retrain)."""
    adapter_file = CACHED_ADAPTER / "adapters.safetensors"
    if adapter_file.exists():
        logger.info("LoRA adapter already trained (cached)")
        return CACHED_ADAPTER

    # Need to train — use cached synthesis data
    training_jsonl = CACHED_SYNTHESIS / "training.jsonl"
    if not training_jsonl.exists():
        raise RuntimeError(f"No training data at {training_jsonl}")

    adapter_dir = OUTPUT_DIR / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training LoRA adapter...")
    train_script = f"""
import sys
sys.path.insert(0, '.')
from looper.trainers.full_replay import full_replay_train
from looper.trainers.lora_trainer import LoRAConfig
from looper.synthesizers.synthesizer import load_training_data
from pathlib import Path
import json

examples = load_training_data(Path('{training_jsonl}'))
config = LoRAConfig(rank={LORA_RANK}, iters={LORA_ITERS}, batch_size=1, max_seq_length=1024)
metrics = full_replay_train(examples, '{HF_MODEL}', Path('{adapter_dir}'), config)
print(json.dumps(metrics))
"""
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"LoRA training failed: {result.stderr[-500:]}")

    logger.info("  LoRA training complete")
    return adapter_dir


def main():
    from looper.collectors.trajectory_store import load_all_trajectories
    from looper.evaluators.metrics import forward_transfer, compare_conditions
    from looper.evaluators.rag import build_rag_index
    from looper.evaluators.results_io import save_results, results_summary
    from looper.models import ExperimentConfig, ExperimentResult
    from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks

    started_at = datetime.now(timezone.utc).isoformat()

    # Load and split tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, test_tasks = split_tasks(all_tasks, train_size=25)
    task_map = {t.instance_id: t for t in all_tasks}

    logger.info("=" * 60)
    logger.info("LOOPER Phase 1 — Full 4-Condition Experiment")
    logger.info(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")
    logger.info("=" * 60)

    # Build RAG index from train trajectories
    logger.info("Building RAG index from train trajectories...")
    train_trajs = load_all_trajectories(CACHED_BASE_TRAJ)
    train_traj_subset = [t for t in train_trajs if t.meta.task_id in {tk.instance_id for tk in train_tasks}]
    rag_index = build_rag_index(train_traj_subset, train_tasks)
    logger.info(f"  RAG index: {len(train_traj_subset)} trajectories indexed")

    # Condition 1: Base (use cached + verified)
    base_results = run_condition_base(test_tasks, task_map)

    # Condition 2: Base + RAG
    rag_results = run_condition_rag(test_tasks, train_tasks, task_map, rag_index)

    # Ensure LoRA adapter is trained
    adapter_dir = ensure_adapter()

    # Stop Ollama to free GPU for MLX
    logger.info("Stopping Ollama to free GPU for MLX server...")
    subprocess.run(["ollama", "stop", MODEL], capture_output=True)
    time.sleep(3)

    # Start MLX server with adapter
    mlx_proc = None
    try:
        logger.info("Starting MLX server with LoRA adapter...")
        mlx_proc = start_mlx_server(adapter_path=adapter_dir)
        mlx_base_url = f"http://127.0.0.1:{MLX_PORT}"

        # Condition 3: Base + LoRA
        lora_results = run_condition_lora(test_tasks, task_map, mlx_base_url)

        # Condition 4: Base + LoRA + RAG
        lora_rag_results = run_condition_lora_rag(
            test_tasks, train_tasks, task_map, rag_index, mlx_base_url)

    finally:
        stop_mlx_server(mlx_proc)
        # Restart Ollama
        subprocess.run(["ollama", "serve"], capture_output=True, timeout=5)

    # Compute metrics
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

    # Forward transfer: adapted vs base
    ft_lora = forward_transfer(base_results, lora_results)
    ft_rag = forward_transfer(base_results, rag_results)
    ft_both = forward_transfer(base_results, lora_rag_results)
    logger.info(f"\nForward Transfer:")
    logger.info(f"  LoRA:      {ft_lora:+.4f}")
    logger.info(f"  RAG:       {ft_rag:+.4f}")
    logger.info(f"  LoRA+RAG:  {ft_both:+.4f}")

    # Save results
    config = ExperimentConfig(
        name="phase1_full_4conditions",
        experiment_id="phase1_4cond",
        repo="django/django",
        model_name=MODEL,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="full_replay",
        lora_rank=LORA_RANK,
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

    # Save detailed comparison
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
