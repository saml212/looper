#!/usr/bin/env python3
"""Trajectory Collection + LoRA Training Pipeline.

Phase 1: Run 3 attempts per task with temperature=1.0 to collect diverse resolved
          trajectories. Uses the best framework (cot_fewshot + fuzzy edit + all fixes).
Phase 2: Harvest all resolved trajectories (new + prior experiments).
Phase 3: Train LoRA on resolved trajectories.
Phase 4: Evaluate adapted model vs base.

Uses MLX server on port 8080 (mlx-community/Qwen2.5-Coder-14B-Instruct-4bit).
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.agent.ollama_client import openai_chat, ChatMessage, ChatResponse
from looper.agent.runner import run_agent
from looper.collectors.trajectory_store import save_trajectory, load_trajectory
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results
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
from looper.tasks.loader import get_repo_tasks, load_curriculum

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

MLX_MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
MLX_URL = "http://localhost:8080"
MLX_PORT = 8080

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_trajectory_collection")
ADAPTER_DIR = Path("/Volumes/1TB_SSD/looper/adapters/trajectory_collection_v1")
DATASET_DIR = Path("/Volumes/1TB_SSD/looper/datasets/trajectory_collection")

MAX_STEPS = 15
MAX_TOKENS = 4096
TEMPERATURE = 0.7
NUM_ATTEMPTS = 3

# LoRA hyperparams (from DEEP_AUDIT: avoid overfitting with larger dataset)
LORA_RANK = 8
LORA_NUM_LAYERS = 16
LORA_LR = 5e-5       # Lower LR for larger dataset
LORA_ITERS = 300     # More iters but with lower LR
LORA_BATCH_SIZE = 2
LORA_MAX_SEQ = 2048

# Prior experiment directories with resolved trajectories
PRIOR_EXPERIMENTS = {
    "cot_fewshot_14b": Path("/Volumes/1TB_SSD/looper/results/experiment_cot_fewshot_14b/trajectories/base"),
    "edit_tool_14b": Path("/Volumes/1TB_SSD/looper/results/experiment_edit_tool_14b/trajectories/base"),
    "fuzzy_edit_14b": Path("/Volumes/1TB_SSD/looper/results/experiment_fuzzy_edit_14b/trajectories/base"),
    "framework_14b_full": Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_full/trajectories/base"),
    "framework_14b_remaining": Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_remaining/trajectories/base"),
    "framework_expanded_7b": Path("/Volumes/1TB_SSD/looper/results/experiment_framework_expanded_7b/trajectories/base"),
    "framework_fix_7b": Path("/Volumes/1TB_SSD/looper/results/experiment_framework_fix_7b/trajectories/base"),
    "framework_32b": Path("/Volumes/1TB_SSD/looper/results/experiment_framework_32b/trajectories/base"),
}


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


def chat_with_temperature(
    messages: list[ChatMessage],
    model: str = MLX_MODEL,
    base_url: str = MLX_URL,
    max_tokens: int = 4096,
    **kwargs,
) -> ChatResponse:
    """Wrapper around openai_chat that injects temperature=1.0 for diversity."""
    return openai_chat(
        messages,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
    )


# ── Phase 1: Multi-attempt trajectory collection ──────────────────────


def collect_multi_attempt(tasks, traj_dir: Path):
    """Run NUM_ATTEMPTS attempts per task with temperature diversity.

    Saves as {task_id}_attempt{N}.json. Resumes by skipping existing files.
    Returns list of (task_id, attempt, trajectory) tuples.
    """
    traj_dir.mkdir(parents=True, exist_ok=True)
    all_trajectories = []
    total_runs = len(tasks) * NUM_ATTEMPTS
    completed = 0

    for task in tasks:
        for attempt in range(1, NUM_ATTEMPTS + 1):
            filename = f"{task.instance_id}_attempt{attempt}.json"
            filepath = traj_dir / filename

            if filepath.exists():
                traj = load_trajectory(filepath)
                all_trajectories.append((task.instance_id, attempt, traj))
                completed += 1
                logger.info(
                    f"  [{completed}/{total_runs}] {task.instance_id} attempt {attempt}: "
                    f"LOADED ({traj.outcome}, "
                    f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
                )
                continue

            logger.info(
                f"  [{completed + 1}/{total_runs}] {task.instance_id} attempt {attempt}..."
            )
            start = time.monotonic()

            try:
                traj = run_agent(
                    task=task,
                    workspace_root=WORKSPACE_ROOT,
                    model=MLX_MODEL,
                    base_url=MLX_URL,
                    max_steps=MAX_STEPS,
                    max_tokens=MAX_TOKENS,
                    chat_fn=chat_with_temperature,
                )
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                completed += 1
                continue

            duration = time.monotonic() - start

            # Save with attempt-specific filename
            traj_dir.mkdir(parents=True, exist_ok=True)
            filepath.write_text(traj.model_dump_json(indent=2))

            all_trajectories.append((task.instance_id, attempt, traj))
            completed += 1

            has_patch = bool(traj.generated_patch.strip())
            logger.info(
                f"    -> {traj.outcome} ({len(traj.steps)} steps, "
                f"patch={'yes' if has_patch else 'no'}, {duration:.0f}s)"
            )

    return all_trajectories


# ── Phase 2: Harvest resolved trajectories ────────────────────────────


def verify_and_harvest(all_trajectories, tasks):
    """Verify trajectories with FAIL_TO_PASS and return resolved ones.

    Returns:
        resolved: list of (source, task_id, trajectory) for resolved
        results: list of TaskResult for reporting
    """
    task_map = {t.instance_id: t for t in tasks}
    resolved = []
    results = []

    for source, task_id_or_attempt, traj in all_trajectories:
        task_id = traj.meta.task_id
        task = task_map.get(task_id)
        is_resolved = False

        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
            is_resolved = vr["resolved"]
            logger.info(
                f"  {source}/{task_id} attempt {task_id_or_attempt}: "
                f"{'RESOLVED' if is_resolved else 'FAIL'} "
                f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
                + (f" error={vr['error']}" if vr.get("error") else "")
            )
            if is_resolved:
                resolved.append((source, task_id, traj))
        else:
            logger.info(f"  {source}/{task_id}: SKIP (no patch)")

        results.append(
            TaskResult(
                task_id=task_id,
                condition=f"{source}_attempt{task_id_or_attempt}",
                resolved=is_resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            )
        )

    return resolved, results


def harvest_prior_experiments(tasks):
    """Load and verify resolved trajectories from prior experiments.

    Returns list of (source, task_id, trajectory) for resolved.
    """
    task_map = {t.instance_id: t for t in tasks}
    resolved = []

    for exp_name, traj_dir in PRIOR_EXPERIMENTS.items():
        if not traj_dir.exists():
            logger.info(f"  {exp_name}: directory not found, skipping")
            continue

        traj_files = list(traj_dir.glob("*.json"))
        logger.info(f"  {exp_name}: {len(traj_files)} trajectory files")

        for traj_file in traj_files:
            try:
                traj = load_trajectory(traj_file)
            except Exception as e:
                logger.warning(f"    Failed to load {traj_file}: {e}")
                continue

            task_id = traj.meta.task_id
            task = task_map.get(task_id)

            if not task or not traj.generated_patch.strip():
                continue

            vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
            if vr["resolved"]:
                resolved.append((exp_name, task_id, traj))
                logger.info(f"    RESOLVED: {task_id}")

    return resolved


# ── Phase 3: Train LoRA ───────────────────────────────────────────────


def prepare_training_data(resolved_trajectories, tasks):
    """Convert resolved trajectories to MLX training format.

    Uses both full-trajectory and per-step examples for comprehensive learning.
    Deduplicates by keeping the best trajectory per unique task (fewest steps).
    """
    task_map = {t.instance_id: t for t in tasks}

    # Deduplicate: keep shortest trajectory per task
    best_per_task: dict[str, AgentTrajectory] = {}
    for source, task_id, traj in resolved_trajectories:
        existing = best_per_task.get(task_id)
        if existing is None or len(traj.steps) < len(existing.steps):
            best_per_task[task_id] = traj
            logger.info(
                f"  Best for {task_id}: {len(traj.steps)} steps (from {source})"
            )

    logger.info(f"\n  Unique resolved tasks for training: {len(best_per_task)}")

    full_examples = []
    step_examples = []

    for task_id, traj in best_per_task.items():
        task = task_map.get(task_id)
        if task is None:
            continue

        full_ex = trajectory_to_training_example(traj, task)
        if full_ex:
            full_examples.append(full_ex)

        step_exs = trajectory_to_step_examples(traj, task)
        step_examples.extend(step_exs)

        logger.info(
            f"  {task_id}: {len(traj.steps)} steps -> "
            f"1 full + {len(step_exs)} step examples"
        )

    # Combine: use per-step examples for training, full for validation
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Use 80/20 split: hold out ~20% of full examples for validation
    n_val = max(1, len(full_examples) // 5)
    val_examples = full_examples[-n_val:]
    train_full = full_examples[:-n_val]
    train_examples = step_examples + train_full

    logger.info(f"\n  Train: {len(train_examples)} examples ({len(step_examples)} step + {len(train_full)} full)")
    logger.info(f"  Valid: {len(val_examples)} examples (full trajectories)")

    with open(DATASET_DIR / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")

    with open(DATASET_DIR / "valid.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")

    # Save metadata
    with open(DATASET_DIR / "metadata.json", "w") as f:
        json.dump({
            "unique_tasks": list(best_per_task.keys()),
            "num_train": len(train_examples),
            "num_val": len(val_examples),
            "num_step_examples": len(step_examples),
            "num_full_examples": len(full_examples),
        }, f, indent=2)

    return train_examples, val_examples, best_per_task


def train_lora_adapter():
    """Train LoRA adapter in subprocess to isolate GPU memory."""
    logger.info("  Launching training subprocess...")
    start = time.monotonic()

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

    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True,
        text=True,
        timeout=3600,
    )

    duration = time.monotonic() - start
    logger.info(f"  Training took {duration:.0f}s")

    if result.returncode != 0:
        logger.error(f"  Training failed! stderr:\n{result.stderr[-2000:]}")
        logger.error(f"  stdout:\n{result.stdout[-2000:]}")
        raise RuntimeError("LoRA training failed")

    metrics = {}
    for line in result.stdout.split("\n"):
        if line.startswith("TRAINING_METRICS="):
            metrics = json.loads(line.split("=", 1)[1])
            break

    logger.info(f"  Training output:\n{result.stdout[-3000:]}")
    logger.info(f"  Final train loss: {metrics.get('final_train_loss', '?')}")
    logger.info(f"  Final val loss: {metrics.get('final_val_loss', '?')}")

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
            "metrics": metrics,
            "duration_seconds": duration,
        }, f, indent=2)

    return metrics


# ── Phase 4: Evaluate ─────────────────────────────────────────────────


def _kill_mlx_server():
    """Kill any existing MLX server on the target port."""
    import signal
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{MLX_PORT}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid.strip():
                logger.info(f"  Killing existing process on port {MLX_PORT}: PID {pid}")
                os.kill(int(pid.strip()), signal.SIGTERM)
        if pids and pids[0]:
            time.sleep(3)  # Wait for port to free up
    except Exception as e:
        logger.warning(f"  Could not kill existing server: {e}")


def start_mlx_server_with_adapter():
    """Start MLX server with the trained adapter (kills existing server first)."""
    _kill_mlx_server()
    logger.info("  Starting MLX inference server with LoRA adapter...")
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

    import urllib.request
    for i in range(60):
        try:
            urllib.request.urlopen(f"{MLX_URL}/v1/models", timeout=2)
            logger.info(f"  MLX server ready on port {MLX_PORT}")
            return proc
        except Exception:
            time.sleep(2)

    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"MLX server exited: {stderr[-1000:]}")
    raise RuntimeError("MLX server did not become ready in 2 minutes")


def evaluate_adapted(test_tasks, train_task_ids: set):
    """Run evaluation with the LoRA-adapted model.

    Tests on tasks NOT in the training set (forward transfer test).
    Also re-runs training tasks to check for regression.
    """
    from looper.collectors.trajectory_store import collect_trajectories

    # Filter to tasks not in training set (forward transfer)
    novel_tasks = [t for t in test_tasks if t.instance_id not in train_task_ids]
    train_tasks = [t for t in test_tasks if t.instance_id in train_task_ids]

    logger.info(f"  Novel tasks (not in training): {len(novel_tasks)}")
    logger.info(f"  Training tasks (sanity check): {len(train_tasks)}")

    traj_dir = OUTPUT_DIR / "trajectories" / "adapted"
    traj_dir.mkdir(parents=True, exist_ok=True)

    mlx_proc = start_mlx_server_with_adapter()

    try:
        def on_complete(tid, traj):
            logger.info(
                f"  {tid} -> {traj.outcome} "
                f"({len(traj.steps)} steps, "
                f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
            )

        # Run on all tasks (novel + train for sanity check)
        all_eval_tasks = novel_tasks + train_tasks
        trajectories = collect_trajectories(
            tasks=all_eval_tasks,
            output_dir=traj_dir,
            workspace_root=WORKSPACE_ROOT,
            model=MLX_MODEL,
            base_url=MLX_URL,
            max_steps=MAX_STEPS,
            max_tokens=MAX_TOKENS,
            chat_fn=openai_chat,  # temp=0.0 for deterministic eval
            on_complete=on_complete,
        )
    finally:
        logger.info("  Stopping MLX server...")
        mlx_proc.terminate()
        mlx_proc.wait(timeout=10)

    return trajectories, novel_tasks, train_tasks


# ── Main ──────────────────────────────────────────────────────────────


def main():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 70)
    logger.info("TRAJECTORY COLLECTION + LoRA TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Model: {MLX_MODEL} via MLX server at {MLX_URL}")
    logger.info(f"Temperature: {TEMPERATURE} ({NUM_ATTEMPTS} attempts/task)")
    logger.info(f"LoRA: rank={LORA_RANK}, lr={LORA_LR}, iters={LORA_ITERS}")
    logger.info("")

    # Load all 50 Django tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = list(get_repo_tasks(curriculum, "django/django"))
    logger.info(f"Loaded {len(all_tasks)} tasks")

    # ── Phase 1: Multi-attempt collection ──────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: Multi-attempt trajectory collection")
    logger.info(f"  {len(all_tasks)} tasks x {NUM_ATTEMPTS} attempts = {len(all_tasks) * NUM_ATTEMPTS} runs")
    logger.info(f"  Temperature: {TEMPERATURE}")
    logger.info("=" * 70)

    traj_dir = OUTPUT_DIR / "trajectories" / "collection"
    new_trajectories = collect_multi_attempt(all_tasks, traj_dir)

    # Quick stats
    new_with_patch = sum(1 for _, _, t in new_trajectories if t.generated_patch.strip())
    logger.info(f"\nPhase 1 complete: {len(new_trajectories)} runs, {new_with_patch} with patches")

    # ── Phase 2: Verify and harvest ────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: Verify + harvest resolved trajectories")
    logger.info("=" * 70)

    # Verify new trajectories
    logger.info("\nVerifying new trajectories...")
    new_resolved, new_results = verify_and_harvest(
        [("collection", attempt, traj) for tid, attempt, traj in new_trajectories],
        all_tasks,
    )
    logger.info(f"  New resolved: {len(new_resolved)}")

    # Harvest from prior experiments
    logger.info("\nHarvesting from prior experiments...")
    prior_resolved = harvest_prior_experiments(all_tasks)
    logger.info(f"  Prior resolved: {len(prior_resolved)}")

    # Combine all resolved
    all_resolved = new_resolved + prior_resolved
    unique_resolved_tasks = set(tid for _, tid, _ in all_resolved)
    logger.info(f"\nTotal resolved trajectories: {len(all_resolved)}")
    logger.info(f"Unique resolved tasks: {len(unique_resolved_tasks)}")
    for tid in sorted(unique_resolved_tasks):
        sources = [s for s, t, _ in all_resolved if t == tid]
        logger.info(f"  {tid}: {len(sources)} trajectories ({', '.join(sources[:3])})")

    # Save harvest summary
    with open(OUTPUT_DIR / "harvest_summary.json", "w") as f:
        json.dump({
            "new_resolved": len(new_resolved),
            "prior_resolved": len(prior_resolved),
            "total_resolved": len(all_resolved),
            "unique_tasks": sorted(unique_resolved_tasks),
            "new_results_total": len(new_results),
            "new_with_patch": new_with_patch,
        }, f, indent=2)

    if len(unique_resolved_tasks) < 5:
        logger.warning(
            f"Only {len(unique_resolved_tasks)} unique resolved tasks. "
            "LoRA training may not be effective. Consider running more attempts."
        )

    # ── Phase 3: Train LoRA ────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: Train LoRA on resolved trajectories")
    logger.info("=" * 70)

    train_examples, val_examples, best_per_task = prepare_training_data(
        all_resolved, all_tasks
    )
    train_task_ids = set(best_per_task.keys())

    logger.info(f"\nTraining LoRA:")
    logger.info(f"  Rank: {LORA_RANK}, LR: {LORA_LR}, Iters: {LORA_ITERS}")
    logger.info(f"  Training examples: {len(train_examples)}")
    logger.info(f"  Validation examples: {len(val_examples)}")

    metrics = train_lora_adapter()

    # ── Phase 4: Evaluate ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: Evaluate adapted model vs base")
    logger.info("=" * 70)

    # NOTE: The MLX server must be restarted with the adapter.
    # The user should stop the current MLX server before this phase.
    logger.info("")
    logger.info("  ** Stopping any existing MLX server on port 8080... **")
    logger.info("  ** Starting MLX server with LoRA adapter... **")

    adapted_trajectories, novel_tasks, sanity_tasks = evaluate_adapted(
        all_tasks, train_task_ids
    )

    # Verify adapted results
    logger.info("\nVerifying adapted model results...")
    task_map = {t.instance_id: t for t in all_tasks}
    adapted_results = []
    adapted_resolved_novel = 0
    adapted_resolved_train = 0

    for traj in adapted_trajectories:
        task = task_map.get(traj.meta.task_id)
        resolved = False
        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
            resolved = vr["resolved"]
        is_novel = traj.meta.task_id not in train_task_ids
        if resolved and is_novel:
            adapted_resolved_novel += 1
        elif resolved:
            adapted_resolved_train += 1

        adapted_results.append(
            TaskResult(
                task_id=traj.meta.task_id,
                condition="adapted" if is_novel else "adapted_sanity",
                resolved=resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            )
        )

    # ── Final Report ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)

    n_novel = len(novel_tasks)
    n_train = len(sanity_tasks)

    logger.info(f"\n  Collection phase:")
    logger.info(f"    Runs: {len(new_trajectories)}")
    logger.info(f"    New resolved: {len(new_resolved)}")
    logger.info(f"    Prior resolved: {len(prior_resolved)}")
    logger.info(f"    Total unique resolved tasks: {len(unique_resolved_tasks)}")

    logger.info(f"\n  Training:")
    logger.info(f"    Examples: {len(train_examples)} train, {len(val_examples)} val")
    logger.info(f"    Train loss: {metrics.get('final_train_loss', '?')}")
    logger.info(f"    Val loss: {metrics.get('final_val_loss', '?')}")

    logger.info(f"\n  Evaluation (adapted model):")
    logger.info(f"    Novel tasks: {adapted_resolved_novel}/{n_novel} ({adapted_resolved_novel/n_novel*100:.1f}%)" if n_novel else "    Novel tasks: N/A")
    logger.info(f"    Train tasks (sanity): {adapted_resolved_train}/{n_train} ({adapted_resolved_train/n_train*100:.1f}%)" if n_train else "    Train tasks: N/A")

    # Forward transfer
    if n_novel:
        # Compare against base rate on novel tasks (from collection phase)
        base_novel_resolved = sum(
            1 for _, tid, _ in new_resolved if tid not in train_task_ids
        )
        base_novel_attempts = sum(
            1 for tid, _, traj in new_trajectories if tid not in train_task_ids
        )
        logger.info(f"\n  Forward transfer:")
        logger.info(f"    Base (best of {NUM_ATTEMPTS}): {base_novel_resolved} resolved on novel tasks")
        logger.info(f"    Adapted: {adapted_resolved_novel} resolved on novel tasks")
        ft = (adapted_resolved_novel - base_novel_resolved) / n_novel if n_novel else 0
        logger.info(f"    FT = {ft:+.4f}")

    # Save final experiment result
    config = ExperimentConfig(
        name="experiment_trajectory_collection",
        experiment_id="trajectory_collection_lora",
        repo="django/django",
        model_name=f"{MLX_MODEL} + LoRA (rank={LORA_RANK})",
        train_task_ids=sorted(train_task_ids),
        test_task_ids=[t.instance_id for t in all_tasks],
        strategy="trajectory_collection_lora_v1",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=adapted_results + new_results,
        forward_transfer=ft if n_novel else 0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "experiment_result.json")

    # Write summary report
    write_report(
        unique_resolved_tasks, len(new_resolved), len(prior_resolved),
        train_examples, val_examples, metrics,
        adapted_resolved_novel, n_novel, adapted_resolved_train, n_train,
        adapted_results, train_task_ids,
    )

    logger.info(f"\nResults saved to {OUTPUT_DIR}")
    return result


def write_report(
    unique_tasks, n_new, n_prior, train_ex, val_ex, metrics,
    novel_resolved, n_novel, train_resolved, n_train,
    adapted_results, train_task_ids,
):
    """Write experiment report."""
    report = f"""# Trajectory Collection + LoRA Training Report

## Collection Results
- New resolved (temp={TEMPERATURE}, {NUM_ATTEMPTS} attempts): {n_new}
- Prior resolved (from {len(PRIOR_EXPERIMENTS)} experiments): {n_prior}
- **Unique resolved tasks: {len(unique_tasks)}**
- Tasks: {', '.join(sorted(unique_tasks))}

## Training
- Model: {MLX_MODEL}
- LoRA: rank={LORA_RANK}, lr={LORA_LR}, iters={LORA_ITERS}
- Train examples: {len(train_ex)} ({len(train_ex) - len(val_ex)} step + full)
- Val examples: {len(val_ex)}
- Final train loss: {metrics.get('final_train_loss', '?')}
- Final val loss: {metrics.get('final_val_loss', '?')}

## Evaluation
- Novel tasks (not in training): {novel_resolved}/{n_novel} ({novel_resolved/n_novel*100:.1f}% resolved)
- Train tasks (sanity check): {train_resolved}/{n_train} ({train_resolved/n_train*100:.1f}% resolved)

## Per-Task Results (Adapted Model)

| Task | Type | Resolved | Steps | Tokens |
|------|------|----------|-------|--------|
"""
    for r in sorted(adapted_results, key=lambda x: x.task_id):
        task_type = "train" if r.task_id in train_task_ids else "novel"
        report += f"| {r.task_id} | {task_type} | {'YES' if r.resolved else 'no'} | {r.steps} | {r.tokens} |\n"

    report += f"""
## Core Thesis Test
Does LoRA-encoded skill learning improve agent performance on NOVEL tasks?
- Forward transfer = (adapted_novel - base_novel) / n_novel
- Result: See final log output above.

Generated at {datetime.now(timezone.utc).isoformat()}
"""
    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
