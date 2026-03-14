#!/usr/bin/env python3
"""Experiment: LoRA trained on correct-format trajectories (XML tool-calls).

The DEEP_AUDIT found that format mismatch is the dominant failure mode for LoRA.
This experiment trains on ACTUAL RESOLVED trajectories — multi-turn XML tool-call
conversations matching exactly what the agent produces at inference time.

Training data: 5 resolved trajectories from 14B and 32B framework experiments.
Evaluation: Same 15 tasks as experiment_framework_14b_full, comparing base vs adapted.

Hypothesis: LoRA trained on correct-format data will improve (or at minimum not
regress) resolve rate compared to 14B base (4/15 = 26.7%).
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.collectors.trajectory_store import collect_trajectories, load_trajectory
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import (
    AgentTrajectory,
    ExperimentConfig,
    ExperimentResult,
    TaskResult,
    TrainingExample,
)
from looper.synthesizers.trajectory_synthesizer import (
    trajectory_to_training_example,
    trajectory_to_step_examples,
)
from looper.tasks.loader import get_repo_tasks, load_curriculum, get_task_by_id

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5-coder:14b"
OLLAMA_URL = "http://localhost:11434"
MLX_MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
MLX_PORT = 8080
MLX_URL = f"http://localhost:{MLX_PORT}"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_correct_format_lora")
ADAPTER_DIR = Path("/Volumes/1TB_SSD/looper/adapters/correct_format_v1")
DATASET_DIR = Path("/Volumes/1TB_SSD/looper/datasets/correct_format_trajectories")

MAX_STEPS = 15
MAX_TOKENS = 4096

# Same 15 tasks as 14B framework experiment
TARGET_TASKS = [
    "django__django-10914",
    "django__django-10999",
    "django__django-11066",
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

# Resolved trajectories to use as training data
RESOLVED_TRAJECTORIES = {
    # From 14B experiment
    "django__django-11066": "/Volumes/1TB_SSD/looper/results/experiment_framework_14b_full/trajectories/base/django__django-11066.json",
    # From 32B experiment (these also resolved on 14B but trajectories weren't saved)
    "django__django-11099": "/Volumes/1TB_SSD/looper/results/experiment_framework_32b/trajectories/base/django__django-11099.json",
    "django__django-11119": "/Volumes/1TB_SSD/looper/results/experiment_framework_32b/trajectories/base/django__django-11119.json",
    "django__django-11451": "/Volumes/1TB_SSD/looper/results/experiment_framework_32b/trajectories/base/django__django-11451.json",
    "django__django-11603": "/Volumes/1TB_SSD/looper/results/experiment_framework_32b/trajectories/base/django__django-11603.json",
}

# ── LoRA Training Config ──────────────────────────────────────────────

LORA_RANK = 16
LORA_NUM_LAYERS = 16
LORA_LR = 1e-4
LORA_ITERS = 200  # More iters for only 5 trajectories — need multiple passes
LORA_BATCH_SIZE = 1  # Small dataset, use batch=1
LORA_MAX_SEQ = 2048


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


# ── Step 1: Prepare Training Data ─────────────────────────────────────

def prepare_training_data():
    """Convert resolved trajectories to MLX training format."""
    logger.info("=" * 60)
    logger.info("Step 1: Preparing training data from resolved trajectories")
    logger.info("=" * 60)

    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    task_map = {t.instance_id: t for t in all_tasks}

    # Load trajectories and convert to training examples
    full_examples = []
    step_examples = []

    for task_id, traj_path in RESOLVED_TRAJECTORIES.items():
        task = task_map.get(task_id)
        if task is None:
            logger.warning(f"Task {task_id} not found in curriculum, skipping")
            continue

        traj = load_trajectory(Path(traj_path))
        logger.info(f"  {task_id}: {len(traj.steps)} steps, outcome={traj.outcome}")

        # Full-trajectory example (one multi-turn conversation per trajectory)
        full_ex = trajectory_to_training_example(traj, task)
        if full_ex:
            full_examples.append(full_ex)
            logger.info(f"    Full example: {len(full_ex.messages)} messages")

        # Per-step examples (one example per step)
        step_exs = trajectory_to_step_examples(traj, task)
        step_examples.extend(step_exs)
        logger.info(f"    Step examples: {len(step_exs)}")

    logger.info(f"\nTotal: {len(full_examples)} full examples, {len(step_examples)} step examples")

    # Use per-step format for training (shorter sequences, more examples)
    # Also include the full-trajectory examples for complete conversation learning
    all_examples = step_examples + full_examples
    logger.info(f"Combined: {len(all_examples)} training examples")

    # Write train/valid split (use 1 full trajectory for validation, rest for train)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Use last trajectory as validation
    val_examples = [full_examples[-1]] if full_examples else []
    train_examples = step_examples + full_examples[:-1]

    with open(DATASET_DIR / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")

    with open(DATASET_DIR / "valid.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")

    # Also save the raw dataset for inspection
    with open(DATASET_DIR / "all_examples.jsonl", "w") as f:
        for ex in all_examples:
            f.write(json.dumps({
                "messages": ex.messages,
                "source_pair_id": ex.source_pair_id,
            }) + "\n")

    logger.info(f"  Train: {len(train_examples)} examples -> {DATASET_DIR / 'train.jsonl'}")
    logger.info(f"  Valid: {len(val_examples)} examples -> {DATASET_DIR / 'valid.jsonl'}")

    # Print example stats
    for ex in all_examples[:3]:
        total_chars = sum(len(m["content"]) for m in ex.messages)
        logger.info(
            f"  Example {ex.source_pair_id}: "
            f"{len(ex.messages)} messages, "
            f"~{total_chars // 4} tokens"
        )

    return train_examples, val_examples


# ── Step 2: Train LoRA ─────────────────────────────────────────────────

def train_lora_adapter():
    """Train LoRA adapter in subprocess to isolate GPU memory."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Training LoRA adapter")
    logger.info("=" * 60)
    logger.info(f"  Model: {MLX_MODEL}")
    logger.info(f"  Rank: {LORA_RANK}, Layers: {LORA_NUM_LAYERS}")
    logger.info(f"  LR: {LORA_LR}, Iters: {LORA_ITERS}, Batch: {LORA_BATCH_SIZE}")
    logger.info(f"  Output: {ADAPTER_DIR}")

    # Run training in subprocess to fully free GPU memory before inference
    train_script = f"""
import sys
sys.path.insert(0, "{Path(__file__).parent}")
from pathlib import Path
from looper.trainers.lora_trainer import train_lora, LoRAConfig
import json

config = LoRAConfig(
    rank={LORA_RANK},
    num_layers={LORA_NUM_LAYERS},
    learning_rate={LORA_LR},
    iters={LORA_ITERS},
    batch_size={LORA_BATCH_SIZE},
    max_seq_length={LORA_MAX_SEQ},
)

metrics = train_lora(
    model_name="{MLX_MODEL}",
    data_dir=Path("{DATASET_DIR}"),
    adapter_output_dir=Path("{ADAPTER_DIR}"),
    config=config,
)

print("TRAINING_METRICS=" + json.dumps(metrics))
"""

    logger.info("  Launching training subprocess...")
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max
    )

    duration = time.monotonic() - start
    logger.info(f"  Training took {duration:.0f}s")

    if result.returncode != 0:
        logger.error(f"  Training failed! stderr:\n{result.stderr[-2000:]}")
        logger.error(f"  stdout:\n{result.stdout[-2000:]}")
        raise RuntimeError("LoRA training failed")

    # Extract metrics from stdout
    metrics = {}
    for line in result.stdout.split("\n"):
        if line.startswith("TRAINING_METRICS="):
            metrics = json.loads(line.split("=", 1)[1])
            break

    # Log full training output
    logger.info(f"  Training output:\n{result.stdout[-3000:]}")

    logger.info(f"  Final train loss: {metrics.get('final_train_loss', '?')}")
    logger.info(f"  Final val loss: {metrics.get('final_val_loss', '?')}")
    logger.info(f"  Adapter saved to: {ADAPTER_DIR}")

    # Save training config
    with open(ADAPTER_DIR / "training_config.json", "w") as f:
        json.dump({
            "model": MLX_MODEL,
            "rank": LORA_RANK,
            "num_layers": LORA_NUM_LAYERS,
            "lr": LORA_LR,
            "iters": LORA_ITERS,
            "batch_size": LORA_BATCH_SIZE,
            "max_seq_length": LORA_MAX_SEQ,
            "train_examples": len(list(open(DATASET_DIR / "train.jsonl"))),
            "val_examples": len(list(open(DATASET_DIR / "valid.jsonl"))),
            "metrics": metrics,
            "duration_seconds": duration,
        }, f, indent=2)

    return metrics


# ── Step 3: Evaluate ───────────────────────────────────────────────────

def start_mlx_server():
    """Start MLX server with the trained adapter."""
    logger.info("  Starting MLX inference server...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlx_lm.server",
            "--model", MLX_MODEL,
            "--adapter-path", str(ADAPTER_DIR),
            "--port", str(MLX_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    import urllib.request
    for i in range(60):
        try:
            urllib.request.urlopen(f"{MLX_URL}/v1/models", timeout=2)
            logger.info(f"  MLX server ready on port {MLX_PORT}")
            return proc
        except Exception:
            time.sleep(2)

    # If not ready after 2 minutes, check if process died
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"MLX server exited with code {proc.returncode}: {stderr[-1000:]}")
    raise RuntimeError("MLX server did not become ready in 2 minutes")


def evaluate_adapted(tasks):
    """Run evaluation with the LoRA-adapted model via MLX server."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: Evaluating adapted model")
    logger.info("=" * 60)

    from looper.agent.ollama_client import openai_chat

    traj_dir = OUTPUT_DIR / "trajectories" / "adapted"
    traj_dir.mkdir(parents=True, exist_ok=True)

    mlx_proc = start_mlx_server()

    try:
        def mlx_chat_fn(messages, model=None, base_url=None, max_tokens=4096):
            return openai_chat(messages, model=MLX_MODEL, base_url=MLX_URL, max_tokens=max_tokens)

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
            chat_fn=mlx_chat_fn,
            on_complete=on_complete,
        )
    finally:
        logger.info("  Stopping MLX server...")
        mlx_proc.terminate()
        mlx_proc.wait(timeout=10)

    return trajectories


def evaluate_trajectories(trajectories, tasks, condition):
    """Run FAIL_TO_PASS verification on trajectories."""
    task_map = {t.instance_id: t for t in tasks}
    results = []
    for traj in trajectories:
        task = task_map.get(traj.meta.task_id)
        resolved = False
        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
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


# ── Main ───────────────────────────────────────────────────────────────

def main():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("EXPERIMENT: LoRA on Correct-Format Trajectories")
    logger.info("=" * 60)
    logger.info(f"Model: {OLLAMA_MODEL} (base) / {MLX_MODEL} (adapted)")
    logger.info(f"Training: {len(RESOLVED_TRAJECTORIES)} resolved trajectories")
    logger.info(f"Evaluation: {len(TARGET_TASKS)} tasks")
    logger.info(f"Baseline: 14B+fixes = 4/15 (26.7%) resolved")
    logger.info("")

    # Load tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    tasks = []
    for tid in TARGET_TASKS:
        task = get_task_by_id(all_tasks, tid)
        if task is None:
            logger.error(f"Task {tid} not found!")
            sys.exit(1)
        tasks.append(task)

    # Step 1: Prepare training data
    train_examples, val_examples = prepare_training_data()

    # Step 2: Train LoRA
    metrics = train_lora_adapter()

    # Step 3: Evaluate adapted model
    adapted_trajectories = evaluate_adapted(tasks)

    # Step 4: Verify results
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: FAIL_TO_PASS Verification")
    logger.info("=" * 60)
    adapted_results = evaluate_trajectories(adapted_trajectories, tasks, "adapted_correct_format")

    # Compute metrics
    resolved_adapted = sum(1 for r in adapted_results if r.resolved)
    patch_adapted = sum(1 for t in adapted_trajectories if t.generated_patch.strip())
    total = len(adapted_results)
    avg_steps = sum(r.steps for r in adapted_results) / total if total else 0
    avg_tokens = sum(r.tokens for r in adapted_results) / total if total else 0

    # Save results
    config = ExperimentConfig(
        name="experiment_correct_format_lora",
        experiment_id="correct_format_lora",
        repo="django/django",
        model_name=f"{MLX_MODEL} + LoRA",
        train_task_ids=list(RESOLVED_TRAJECTORIES.keys()),
        test_task_ids=TARGET_TASKS,
        strategy="correct_format_lora_v1",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=adapted_results,
        forward_transfer=0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "experiment_result.json")

    # Print final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE: Correct-Format LoRA")
    logger.info("=" * 60)
    logger.info(f"  Adapted resolve:  {resolved_adapted}/{total} ({resolved_adapted/total*100:.1f}%)")
    logger.info(f"  Adapted patches:  {patch_adapted}/{total} ({patch_adapted/total*100:.1f}%)")
    logger.info(f"  Avg steps:        {avg_steps:.1f}")
    logger.info(f"  Avg tokens:       {avg_tokens:.0f}")
    logger.info("")
    logger.info("  Baseline comparison:")
    logger.info(f"    14B base (no LoRA):  4/15 (26.7%) resolved")
    logger.info(f"    14B + LoRA adapted:  {resolved_adapted}/{total} ({resolved_adapted/total*100:.1f}%) resolved")
    ft = (resolved_adapted - 4) / total
    logger.info(f"    Forward transfer:    {ft:+.4f}")
    logger.info("")

    # Per-task detail
    logger.info("Per-task results:")
    baseline_resolved = {"django__django-11066", "django__django-11099",
                         "django__django-11119", "django__django-11451"}
    for r in adapted_results:
        was_baseline = r.task_id in baseline_resolved
        t = {t.meta.task_id: t for t in adapted_trajectories}.get(r.task_id)
        has_patch = t.generated_patch.strip() if t else ""
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"(baseline={'RESOLVED' if was_baseline else 'FAILED':>8}) "
            f"steps={r.steps:>2} "
            f"patch={'yes' if has_patch else 'no':>3}"
        )

    # Training metrics
    logger.info("")
    logger.info("Training metrics:")
    logger.info(f"  Train loss: {metrics.get('final_train_loss', '?')}")
    logger.info(f"  Val loss:   {metrics.get('final_val_loss', '?')}")
    logger.info(f"  Train examples: {len(train_examples)}")
    logger.info(f"  Val examples:   {len(val_examples)}")

    return result


if __name__ == "__main__":
    main()
