#!/usr/bin/env python3
"""Oracle SFT experiment: train on gold patches, evaluate on test tasks.

Tests Hypothesis 1 from DEEP_AUDIT.md:
  "Training on gold data produces measurable transfer"

Conditions:
  1. Base model (Ollama) on 25 test tasks
  2. Oracle-adapted model (MLX) on 25 test tasks

Each condition gets a clean workspace (workspace bug is fixed).
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project root on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from looper.agent.ollama_client import ChatMessage
from looper.collectors.trajectory_store import collect_trajectories, save_trajectory
from looper.evaluators.metrics import forward_transfer
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results
from looper.models import ExperimentConfig, ExperimentResult, TaskResult
from looper.tasks.loader import load_curriculum, get_repo_tasks, split_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# --- Config ---
CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
ORACLE_DATA = Path("/Volumes/1TB_SSD/looper/results/oracle_sft_test/data")
ORACLE_ADAPTER = Path("/Volumes/1TB_SSD/looper/results/oracle_sft_test/adapter")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/oracle_sft_test")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")

OLLAMA_MODEL = "qwen2.5-coder:7b"
HF_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
OLLAMA_URL = "http://localhost:11434"

MAX_STEPS = 15
MAX_TOKENS_BASE = 4096
MAX_TOKENS_ADAPTED = 512  # MLX memory constraint

LORA_RANK = 16
LORA_ITERS = 100


def train_oracle_adapter():
    """Train oracle LoRA adapter in a subprocess (frees GPU after)."""
    adapter_file = ORACLE_ADAPTER / "adapters.safetensors"
    if adapter_file.exists():
        log.info("Oracle adapter already trained, skipping.")
        return

    log.info("Training oracle LoRA adapter...")
    script = f"""
import sys, json
sys.path.insert(0, '.')
from looper.trainers.lora_trainer import train_lora, LoRAConfig
from pathlib import Path

config = LoRAConfig(rank={LORA_RANK}, iters={LORA_ITERS}, batch_size=1, max_seq_length=2048)
metrics = train_lora(
    model_name='{HF_MODEL}',
    data_dir=Path('{ORACLE_DATA}'),
    adapter_output_dir=Path('{ORACLE_ADAPTER}'),
    config=config,
)
print(json.dumps(metrics))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.returncode != 0:
        log.error(f"Training failed:\n{result.stderr[-1000:]}")
        raise RuntimeError("Oracle LoRA training failed")

    # Parse metrics from last JSON line
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            log.info(f"Training metrics: {metrics}")
            break
        except json.JSONDecodeError:
            continue


def evaluate_condition(
    name: str,
    tasks,
    traj_dir: Path,
    chat_fn=None,
    model: str = OLLAMA_MODEL,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS_BASE,
) -> list[TaskResult]:
    """Run agent on tasks, verify patches, return results."""
    log.info(f"--- Condition: {name} ({len(tasks)} tasks) ---")

    trajectories = collect_trajectories(
        tasks=tasks,
        output_dir=traj_dir,
        workspace_root=WORKSPACE_ROOT,
        model=model,
        base_url=OLLAMA_URL,
        max_steps=max_steps,
        max_tokens=max_tokens,
        chat_fn=chat_fn,
        on_complete=lambda tid, t: log.info(f"  {tid}: {t.outcome}"),
    )

    results = []
    for traj in trajectories:
        task = next((t for t in tasks if t.instance_id == traj.meta.task_id), None)
        resolved = False
        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
            resolved = vr["resolved"]
            log.info(
                f"  Verify {traj.meta.task_id}: "
                f"{'PASS' if resolved else 'FAIL'} "
                f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
            )
        results.append(TaskResult(
            task_id=traj.meta.task_id,
            condition=name,
            resolved=resolved,
            steps=traj.meta.total_steps,
            tokens=traj.meta.total_tokens,
            duration_seconds=0.0,
        ))

    resolved_count = sum(1 for r in results if r.resolved)
    log.info(f"  {name}: {resolved_count}/{len(results)} resolved")
    return results


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, test_tasks = split_tasks(all_tasks, train_size=25)
    log.info(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Step 1: Train oracle adapter (subprocess, frees GPU)
    train_oracle_adapter()

    # Step 2: Base condition (Ollama)
    base_traj_dir = OUTPUT_DIR / "trajectories" / "base"
    base_results = evaluate_condition(
        "base", test_tasks, base_traj_dir,
        max_steps=MAX_STEPS, max_tokens=MAX_TOKENS_BASE,
    )

    # Step 3: Stop Ollama, load adapted model
    log.info("Stopping Ollama for MLX inference...")
    subprocess.run(["pkill", "ollama"], capture_output=True)
    time.sleep(3)

    from looper.agent.ollama_client import load_mlx_model, mlx_chat
    log.info(f"Loading adapted model with oracle adapter: {ORACLE_ADAPTER}")
    load_mlx_model(HF_MODEL, adapter_path=str(ORACLE_ADAPTER))

    # Step 4: Oracle-adapted condition (MLX in-process)
    adapted_traj_dir = OUTPUT_DIR / "trajectories" / "oracle_adapted"
    adapted_results = evaluate_condition(
        "oracle_adapted", test_tasks, adapted_traj_dir,
        chat_fn=mlx_chat,
        model=HF_MODEL,
        max_steps=MAX_STEPS,
        max_tokens=MAX_TOKENS_ADAPTED,
    )

    # Step 5: Compute metrics and save
    ft = forward_transfer(base_results, adapted_results)
    log.info(f"Forward transfer: {ft:.4f}")

    config = ExperimentConfig(
        name="oracle_sft_eval",
        experiment_id="oracle_sft_h1",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="oracle_sft",
        lora_rank=LORA_RANK,
        seed=0,
    )

    result = ExperimentResult(
        config=config,
        task_results=base_results + adapted_results,
        forward_transfer=ft,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    results_path = OUTPUT_DIR / "results.json"
    save_results(result, results_path)
    log.info(f"Results saved to {results_path}")

    # Summary
    base_resolved = sum(1 for r in base_results if r.resolved)
    adapted_resolved = sum(1 for r in adapted_results if r.resolved)
    print(f"\n{'='*50}")
    print(f"Oracle SFT Evaluation Results")
    print(f"{'='*50}")
    print(f"Base:           {base_resolved}/{len(base_results)} resolved")
    print(f"Oracle-adapted: {adapted_resolved}/{len(adapted_results)} resolved")
    print(f"Forward transfer: {ft:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
