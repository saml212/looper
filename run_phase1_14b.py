#!/usr/bin/env python3
"""Phase 1 experiment with qwen2.5-coder:14b.

Key changes from 7B experiments:
- Uses 14B model (tests H3: capacity hypothesis from DEEP_AUDIT)
- Training data uses XML tool-call format (not Q&A pairs — fixes H1: format mismatch)
- Adapted inference via MLX server (OpenAI-compatible API on port 8080)
- Workspace reset between conditions (fixes workspace contamination bug)
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be importable from the project root
sys.path.insert(0, str(Path(__file__).parent))

from looper.agent.ollama_client import ChatMessage, openai_chat
from looper.collectors.trajectory_store import (
    collect_trajectories,
    load_all_trajectories,
)
from looper.evaluators.metrics import forward_transfer
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import (
    ExperimentConfig,
    ExperimentResult,
    TaskResult,
    TrainingExample,
)
from looper.synthesizers.trajectory_synthesizer import (
    trajectories_to_training_examples,
)
from looper.synthesizers.synthesizer import save_training_data, load_training_data
from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks
from looper.trainers.lora_trainer import LoRAConfig

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5-coder:14b"
OLLAMA_URL = "http://localhost:11434"
HF_MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
MLX_PORT = 8080

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")

MAX_STEPS = 15
MAX_TOKENS = 4096
# Adapted inference: 4096 tokens needed for full <write> calls, 5 steps limits
# context growth to avoid MLX OOM on 32GB (previous OOM at step 6 with 4096 tokens)
ADAPTED_MAX_TOKENS = 4096
ADAPTED_MAX_STEPS = 5

LORA_RANK = 16
LORA_ITERS = 100
LORA_MAX_SEQ = 1024  # Reduced for 14B memory constraints
LORA_BATCH = 1  # 14B 4-bit + LoRA is memory-tight


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


def start_mlx_server(model: str, port: int, adapter_path: str | None = None):
    """Start mlx_lm.server and wait for it to be ready."""
    import httpx

    venv_bin = Path(sys.executable).parent
    cmd = [str(venv_bin / "mlx_lm.server"), "--model", model, "--port", str(port)]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    logger.info(f"  Starting MLX server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(60):  # 14B takes longer to load
        time.sleep(2)
        try:
            httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
            logger.info(f"  MLX server ready on port {port}")
            return proc
        except Exception:
            continue

    proc.terminate()
    raise RuntimeError(f"MLX server failed to start on port {port}")


def stop_mlx_server(proc):
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=10)
        logger.info("  MLX server stopped")


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


def train_adapter(training_examples, output_dir, adapter_dir):
    """Train LoRA adapter in a subprocess to free GPU memory afterwards."""
    synthesis_dir = output_dir / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    training_jsonl = synthesis_dir / "training.jsonl"
    save_training_data(training_examples, training_jsonl)
    logger.info(f"  Saved {len(training_examples)} training examples")

    adapter_file = adapter_dir / "adapters.safetensors"
    if adapter_file.exists():
        logger.info("  Adapter already trained, skipping...")
        return {"cached": True}

    # Stop Ollama to free GPU memory
    logger.info("  Stopping Ollama to free GPU memory...")
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(3)

    train_script = f"""
import sys
sys.path.insert(0, '.')
from looper.trainers.full_replay import full_replay_train
from looper.trainers.lora_trainer import LoRAConfig
from looper.synthesizers.synthesizer import load_training_data
from pathlib import Path
import json

examples = load_training_data(Path('{training_jsonl}'))
config = LoRAConfig(
    rank={LORA_RANK},
    iters={LORA_ITERS},
    batch_size={LORA_BATCH},
    max_seq_length={LORA_MAX_SEQ},
)
metrics = full_replay_train(examples, '{HF_MODEL}', Path('{adapter_dir}'), config)
print(json.dumps(metrics))
"""
    logger.info("  Training LoRA adapter in subprocess...")
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
        timeout=3600,
    )
    if result.returncode != 0:
        logger.error(f"Training failed:\n{result.stderr[-1000:]}")
        raise RuntimeError("LoRA training failed")

    # Parse metrics from last JSON line
    metrics = {}
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    return metrics


def run_phase1_14b(
    train_size: int = 25,
    test_size: int = 25,
    adapted_test_size: int | None = None,
    output_dir: Path = Path("/Volumes/1TB_SSD/looper/results/phase1_14b"),
):
    """Run Phase 1 with qwen2.5-coder:14b.

    Args:
        train_size: Number of training tasks.
        test_size: Number of test tasks (train_size + test_size must be <= 50).
        adapted_test_size: Subset of test tasks for adapted eval (None = all).
        output_dir: Where to save results.
    """
    setup_logging(output_dir)
    started_at = datetime.now(timezone.utc).isoformat()

    base_traj_dir = output_dir / "trajectories" / "base"
    adapted_traj_dir = output_dir / "trajectories" / "adapted"
    adapter_dir = output_dir / "adapter"
    for d in [base_traj_dir, adapted_traj_dir, adapter_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Load tasks
    logger.info("=" * 60)
    logger.info(f"LOOPER Phase 1 — 14B ({train_size} train / {test_size} test)")
    logger.info("=" * 60)

    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, test_tasks = split_tasks(all_tasks, train_size, seed=None)

    if adapted_test_size is not None:
        test_tasks = test_tasks[:adapted_test_size]

    logger.info(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Step 2: Run base model on ALL tasks (train + test)
    logger.info("Step 2: Running base model (Ollama 14B) on all tasks...")
    all_tasks_to_run = train_tasks + test_tasks
    all_base_trajectories = collect_trajectories(
        tasks=all_tasks_to_run,
        output_dir=base_traj_dir,
        workspace_root=WORKSPACE_ROOT,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        max_steps=MAX_STEPS,
        max_tokens=MAX_TOKENS,
        on_complete=lambda tid, traj: logger.info(
            f"  Base: {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, patch={'yes' if traj.generated_patch.strip() else 'no'})"
        ),
    )

    # Separate train / test trajectories
    train_ids = {t.instance_id for t in train_tasks}
    test_ids = {t.instance_id for t in test_tasks}
    train_trajectories = [t for t in all_base_trajectories if t.meta.task_id in train_ids]
    test_base_trajectories = [t for t in all_base_trajectories if t.meta.task_id in test_ids]

    # Step 3: Evaluate base on test tasks
    logger.info("Step 3: Evaluating base model...")
    base_results = evaluate_trajectories(
        test_base_trajectories, test_tasks, WORKSPACE_ROOT, "base"
    )
    base_resolved = sum(1 for r in base_results if r.resolved)
    logger.info(f"  Base resolve rate: {base_resolved}/{len(base_results)}")

    # Step 4: Synthesize XML-format training data from train trajectories
    logger.info("Step 4: Converting trajectories to XML tool-call training examples...")
    training_jsonl = output_dir / "synthesis" / "training.jsonl"
    if training_jsonl.exists():
        training_examples = load_training_data(training_jsonl)
        logger.info(f"  Loaded {len(training_examples)} cached examples")
    else:
        training_examples = trajectories_to_training_examples(
            train_trajectories, train_tasks, only_successful=False,
            per_step=True,  # Per-step examples fit in 14B memory
        )
        if not training_examples:
            logger.error("No training examples generated! Aborting adapted eval.")
            # Still save base results
            _save_result(
                base_results, [], 0.0, train_tasks, test_tasks,
                started_at, output_dir, train_size,
            )
            return

    # Step 5: Train LoRA adapter
    logger.info(f"Step 5: Training LoRA adapter ({len(training_examples)} examples)...")
    train_metrics = train_adapter(training_examples, output_dir, adapter_dir)
    logger.info(f"  Training metrics: {train_metrics}")

    # Step 6: Run adapted model via MLX server
    logger.info("Step 6: Starting MLX server with adapter...")

    # Make sure Ollama is stopped (resource contention)
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(3)

    mlx_proc = start_mlx_server(HF_MODEL, MLX_PORT, str(adapter_dir))

    try:
        # Use openai_chat pointed at MLX server for adapted inference
        def adapted_chat_fn(messages, model="", base_url="", **kwargs):
            return openai_chat(
                messages,
                model=HF_MODEL,
                base_url=f"http://127.0.0.1:{MLX_PORT}",
                max_tokens=ADAPTED_MAX_TOKENS,  # Short to avoid MLX OOM
            )

        logger.info(
            f"  Running adapted model on {len(test_tasks)} test tasks "
            f"(max_steps={ADAPTED_MAX_STEPS}, max_tokens={ADAPTED_MAX_TOKENS})..."
        )
        adapted_trajectories = collect_trajectories(
            tasks=test_tasks,
            output_dir=adapted_traj_dir,
            workspace_root=WORKSPACE_ROOT,
            model=HF_MODEL,
            base_url=f"http://127.0.0.1:{MLX_PORT}",
            max_steps=ADAPTED_MAX_STEPS,
            max_tokens=ADAPTED_MAX_TOKENS,
            chat_fn=adapted_chat_fn,
            on_complete=lambda tid, traj: logger.info(
                f"  Adapted: {tid} -> {traj.outcome} "
                f"({len(traj.steps)} steps, patch={'yes' if traj.generated_patch.strip() else 'no'})"
            ),
        )
    finally:
        stop_mlx_server(mlx_proc)

    # Restart Ollama for future use
    logger.info("  Restarting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        env={**__import__("os").environ, "OLLAMA_MODELS": "/Volumes/1TB_SSD/looper/ollama_models"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Step 7: Evaluate adapted model
    logger.info("Step 7: Evaluating adapted model...")
    adapted_results = evaluate_trajectories(
        adapted_trajectories, test_tasks, WORKSPACE_ROOT, "adapted"
    )
    adapted_resolved = sum(1 for r in adapted_results if r.resolved)
    logger.info(f"  Adapted resolve rate: {adapted_resolved}/{len(adapted_results)}")

    # Compute forward transfer
    ft = forward_transfer(base_results, adapted_results)

    # Step 8: Save results
    _save_result(
        base_results, adapted_results, ft, train_tasks, test_tasks,
        started_at, output_dir, train_size,
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"  Base:    {base_resolved}/{len(base_results)} ({base_resolved/len(base_results)*100:.1f}%)")
    logger.info(f"  Adapted: {adapted_resolved}/{len(adapted_results)} ({adapted_resolved/len(adapted_results)*100:.1f}%)")
    logger.info(f"  Forward Transfer: {ft:.4f}")
    logger.info("=" * 60)

    return ft


def _save_result(base_results, adapted_results, ft, train_tasks, test_tasks,
                 started_at, output_dir, train_size):
    config = ExperimentConfig(
        name="phase1_14b",
        experiment_id="phase1_14b_xml_format",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="full_replay_xml_trajectory",
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
    save_results(result, output_dir / "experiment_result.json")
    summary = results_summary(result)
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 experiment with 14B model")
    parser.add_argument("--pilot", action="store_true", help="Run 3/3 pilot")
    parser.add_argument("--train-size", type=int, default=25)
    parser.add_argument("--test-size", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.pilot:
        output_dir = Path(args.output_dir or "/Volumes/1TB_SSD/looper/results/phase1_14b_pilot")
        run_phase1_14b(
            train_size=3,
            test_size=47,  # split_tasks takes first N for train
            adapted_test_size=3,
            output_dir=output_dir,
        )
    else:
        output_dir = Path(args.output_dir or "/Volumes/1TB_SSD/looper/results/phase1_14b")
        run_phase1_14b(
            train_size=args.train_size,
            test_size=50 - args.train_size,
            output_dir=output_dir,
        )
