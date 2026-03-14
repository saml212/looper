#!/usr/bin/env python3
"""Self-Play Trajectory Collection + LoRA Training (14B).

Hypothesis: The base 14B model already generates XML tool-call formatted
trajectories. If we collect resolved trajectories from the base model itself
(self-play), train LoRA on those, and eval — does it improve?

This is cleaner than trajectory_collection because we skip the format mismatch:
- Collection: temp=0.3 (maximize resolve rate)
- Training: resolved trajectories only (correct XML tool-call format)
- Eval: temp=0.0 (deterministic) on held-out tasks

Design:
- Pilot: tasks 1-25 for collection
- If >3 resolved: train LoRA, eval on tasks 26-50
- If pilot succeeds: expand collection to all 50, retrain, re-eval

Uses MLX server on port 8080 (mlx-community/Qwen2.5-Coder-14B-Instruct-4bit).
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.agent.ollama_client import openai_chat, ChatMessage, ChatResponse
from looper.agent.runner import run_agent
from looper.collectors.trajectory_store import save_trajectory, load_trajectory, collect_trajectories
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
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_selfplay_14b")
ADAPTER_DIR = Path("/Volumes/1TB_SSD/looper/adapters/selfplay_14b_v1")
DATASET_DIR = Path("/Volumes/1TB_SSD/looper/datasets/selfplay_14b")

MAX_STEPS = 15
MAX_TOKENS = 4096
COLLECTION_TEMP = 0.3  # More deterministic for collection
EVAL_TEMP = 0.0        # Deterministic for eval

# LoRA hyperparams — rank 16 for more capacity than trajectory_collection's rank 8
LORA_RANK = 16
LORA_NUM_LAYERS = 16
LORA_LR = 5e-5
LORA_ITERS = 200
LORA_BATCH_SIZE = 2
LORA_MAX_SEQ = 2048

# Minimum resolved trajectories to proceed with training
MIN_RESOLVED = 3


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


def chat_with_collection_temp(
    messages: list[ChatMessage],
    model: str = MLX_MODEL,
    base_url: str = MLX_URL,
    max_tokens: int = 4096,
    **kwargs,
) -> ChatResponse:
    """Wrapper that injects collection temperature."""
    return openai_chat(
        messages,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=COLLECTION_TEMP,
    )


# ── MLX Server Management ─────────────────────────────────────────────


def kill_mlx_server():
    """Kill any existing MLX server on the target port."""
    for lsof_cmd in ["/usr/sbin/lsof", "lsof"]:
        try:
            result = subprocess.run(
                [lsof_cmd, "-ti", f":{MLX_PORT}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid.strip():
                    logger.info(f"  Killing process on port {MLX_PORT}: PID {pid}")
                    os.kill(int(pid.strip()), signal.SIGTERM)
            if pids and pids[0]:
                time.sleep(3)
            return
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"  Could not kill server: {e}")
            return


def start_mlx_server(adapter_path: str | None = None):
    """Start MLX server, optionally with adapter. Returns Popen."""
    kill_mlx_server()

    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", MLX_MODEL,
        "--port", str(MLX_PORT),
    ]
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])
        logger.info(f"  Starting MLX server with adapter: {adapter_path}")
    else:
        logger.info("  Starting MLX server (base model, no adapter)...")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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


def stop_mlx_server(proc):
    """Stop an MLX server process."""
    logger.info("  Stopping MLX server...")
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    # Wait for port to free
    time.sleep(3)


# ── Phase 1: Collection ───────────────────────────────────────────────


def is_server_alive():
    """Check if MLX server is responding."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{MLX_URL}/v1/models", timeout=3)
        return True
    except Exception:
        return False


def collect_trajectories_phase(tasks, traj_dir: Path, server_proc=None,
                                adapter_path: str | None = None,
                                chat_fn=None):
    """Run model on tasks, save trajectories. Restarts server on crash.

    Returns (list of AgentTrajectory, server_proc).
    """
    if chat_fn is None:
        chat_fn = chat_with_collection_temp

    traj_dir.mkdir(parents=True, exist_ok=True)
    trajectories = []

    for i, task in enumerate(tasks):
        filepath = traj_dir / f"{task.instance_id}.json"

        if filepath.exists():
            traj = load_trajectory(filepath)
            has_patch = bool(traj.generated_patch.strip())
            logger.info(
                f"  [{i+1}/{len(tasks)}] {task.instance_id}: "
                f"LOADED ({traj.outcome}, {len(traj.steps)} steps, "
                f"patch={'yes' if has_patch else 'no'})"
            )
            trajectories.append(traj)
            continue

        # Ensure server is alive before each task
        if not is_server_alive():
            logger.warning("  MLX server not responding, restarting...")
            if server_proc:
                try:
                    server_proc.terminate()
                    server_proc.wait(timeout=5)
                except Exception:
                    pass
            server_proc = start_mlx_server(adapter_path=adapter_path)

        logger.info(f"  [{i+1}/{len(tasks)}] {task.instance_id}...")
        start = time.monotonic()

        try:
            traj = run_agent(
                task=task,
                workspace_root=WORKSPACE_ROOT,
                model=MLX_MODEL,
                base_url=MLX_URL,
                max_steps=MAX_STEPS,
                max_tokens=MAX_TOKENS,
                chat_fn=chat_fn,
            )
        except Exception as e:
            logger.error(f"    FAILED: {e}")
            # Server may have crashed — will be restarted on next iteration
            continue

        duration = time.monotonic() - start
        filepath.write_text(traj.model_dump_json(indent=2))
        trajectories.append(traj)

        has_patch = bool(traj.generated_patch.strip())
        logger.info(
            f"    -> {traj.outcome} ({len(traj.steps)} steps, "
            f"patch={'yes' if has_patch else 'no'}, {duration:.0f}s)"
        )

    return trajectories, server_proc


# ── Phase 2: Verify and Filter ────────────────────────────────────────


def verify_trajectories(trajectories, tasks):
    """Verify trajectories with FAIL_TO_PASS tests.

    Returns (resolved_trajectories, all_results).
    resolved_trajectories: list of (task_id, trajectory) for resolved.
    all_results: list of TaskResult for reporting.
    """
    task_map = {t.instance_id: t for t in tasks}
    resolved = []
    results = []

    for traj in trajectories:
        task_id = traj.meta.task_id
        task = task_map.get(task_id)
        is_resolved = False

        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
            is_resolved = vr["resolved"]
            logger.info(
                f"  {task_id}: {'RESOLVED' if is_resolved else 'FAIL'} "
                f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
                + (f" error={vr['error']}" if vr.get("error") else "")
            )
            if is_resolved:
                resolved.append((task_id, traj))
        else:
            logger.info(f"  {task_id}: SKIP (no patch)")

        results.append(
            TaskResult(
                task_id=task_id,
                condition="base_collection",
                resolved=is_resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            )
        )

    return resolved, results


# ── Phase 3: Training Data Preparation + LoRA Training ─────────────────


def prepare_training_data(resolved_trajectories, tasks):
    """Convert resolved trajectories to MLX training format.

    Uses both full-trajectory and per-step examples.
    """
    task_map = {t.instance_id: t for t in tasks}

    full_examples = []
    step_examples = []

    for task_id, traj in resolved_trajectories:
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

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # 80/20 split: hold out ~20% of full examples for validation
    n_val = max(1, len(full_examples) // 5)
    val_examples = full_examples[-n_val:]
    train_full = full_examples[:-n_val]
    train_examples = step_examples + train_full

    logger.info(f"\n  Train: {len(train_examples)} examples "
                f"({len(step_examples)} step + {len(train_full)} full)")
    logger.info(f"  Valid: {len(val_examples)} examples (full trajectories)")

    with open(DATASET_DIR / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")

    with open(DATASET_DIR / "valid.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")

    with open(DATASET_DIR / "metadata.json", "w") as f:
        json.dump({
            "unique_tasks": [tid for tid, _ in resolved_trajectories],
            "num_train": len(train_examples),
            "num_val": len(val_examples),
            "num_step_examples": len(step_examples),
            "num_full_examples": len(full_examples),
        }, f, indent=2)

    return train_examples, val_examples


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

    logger.info(f"  Training output (last 2000 chars):\n{result.stdout[-2000:]}")
    logger.info(f"  Final train loss: {metrics.get('final_train_loss', '?')}")
    logger.info(f"  Final val loss: {metrics.get('final_val_loss', '?')}")

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


# ── Phase 4: Evaluation ───────────────────────────────────────────────


def evaluate_condition(tasks, traj_dir: Path, condition_name: str,
                       server_proc=None,
                       adapter_path: str | None = None):
    """Run evaluation on tasks with server crash resilience, then verify.

    Uses openai_chat (temp=0.0) for deterministic evaluation.
    Returns (trajectories, results_list, n_resolved, server_proc).
    """
    # Use collect_trajectories_phase for server resilience, with temp=0.0
    trajectories, server_proc = collect_trajectories_phase(
        tasks, traj_dir, server_proc=server_proc, adapter_path=adapter_path,
        chat_fn=openai_chat,  # temp=0.0 default
    )

    # Verify
    task_map = {t.instance_id: t for t in tasks}
    results = []
    n_resolved = 0
    for traj in trajectories:
        task = task_map.get(traj.meta.task_id)
        resolved = False
        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
            resolved = vr["resolved"]
            if resolved:
                n_resolved += 1
            logger.info(
                f"  VERIFY {traj.meta.task_id}: "
                f"{'RESOLVED' if resolved else 'FAIL'} "
                f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
            )
        results.append(
            TaskResult(
                task_id=traj.meta.task_id,
                condition=condition_name,
                resolved=resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            )
        )

    return trajectories, results, n_resolved, server_proc


# ── Main ──────────────────────────────────────────────────────────────


def main():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 70)
    logger.info("SELF-PLAY TRAJECTORY COLLECTION + LoRA TRAINING (14B)")
    logger.info("=" * 70)
    logger.info(f"Model: {MLX_MODEL}")
    logger.info(f"Collection temp: {COLLECTION_TEMP}, Eval temp: {EVAL_TEMP}")
    logger.info(f"LoRA: rank={LORA_RANK}, lr={LORA_LR}, iters={LORA_ITERS}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("")

    # Load all 50 Django tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = list(get_repo_tasks(curriculum, "django/django"))
    train_tasks = all_tasks[:25]
    test_tasks = all_tasks[25:]
    logger.info(f"Loaded {len(all_tasks)} tasks (25 train, 25 test)")

    # ── Phase 1: Pilot Collection (25 tasks) ──────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: Pilot Collection (25 tasks, temp=0.3)")
    logger.info("=" * 70)

    mlx_proc = start_mlx_server()

    collection_dir = OUTPUT_DIR / "trajectories" / "collection"
    pilot_trajectories, mlx_proc = collect_trajectories_phase(
        train_tasks, collection_dir, server_proc=mlx_proc,
    )

    # Quick stats
    n_with_patch = sum(1 for t in pilot_trajectories if t.generated_patch.strip())
    logger.info(f"\nPilot collection: {len(pilot_trajectories)} runs, "
                f"{n_with_patch} with patches")

    # ── Phase 2: Verify Collected Trajectories ────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: Verify collected trajectories (FAIL_TO_PASS)")
    logger.info("=" * 70)

    resolved, collection_results = verify_trajectories(pilot_trajectories, train_tasks)
    n_resolved = len(resolved)
    resolved_ids = [tid for tid, _ in resolved]

    logger.info(f"\nResolved: {n_resolved}/25 ({n_resolved/25*100:.1f}%)")
    logger.info(f"Resolved tasks: {', '.join(sorted(resolved_ids))}")

    # Save collection summary
    with open(OUTPUT_DIR / "collection_summary.json", "w") as f:
        json.dump({
            "total_tasks": len(train_tasks),
            "with_patch": n_with_patch,
            "resolved": n_resolved,
            "resolved_tasks": sorted(resolved_ids),
            "temperature": COLLECTION_TEMP,
        }, f, indent=2)

    # ── Go/No-Go Decision ─────────────────────────────────────────────
    if n_resolved < MIN_RESOLVED:
        logger.warning(
            f"\nONLY {n_resolved} resolved (need >{MIN_RESOLVED}). "
            "Insufficient data for LoRA training. STOPPING."
        )
        if mlx_proc:
            stop_mlx_server(mlx_proc)
        write_report_negative(n_resolved, n_with_patch, collection_results)
        return

    logger.info(f"\n{n_resolved} resolved > {MIN_RESOLVED} minimum. Proceeding to training.")

    # ── Phase 3a: Run base eval on test tasks ─────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3a: Base evaluation on test tasks (25 tasks)")
    logger.info("=" * 70)

    base_traj_dir = OUTPUT_DIR / "trajectories" / "base_eval"
    base_trajectories, base_results, base_resolved, mlx_proc = evaluate_condition(
        test_tasks, base_traj_dir, "base",
        server_proc=mlx_proc,
    )

    logger.info(f"\nBase eval: {base_resolved}/25 ({base_resolved/25*100:.1f}%)")

    # Stop server for training
    if mlx_proc:
        stop_mlx_server(mlx_proc)

    # ── Phase 3b: Prepare training data + Train LoRA ──────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3b: Prepare training data + Train LoRA")
    logger.info("=" * 70)

    train_examples, val_examples = prepare_training_data(resolved, train_tasks)
    logger.info(f"\nTraining: {len(train_examples)} train, {len(val_examples)} val")

    metrics = train_lora_adapter()

    # ── Phase 4: Adapted Evaluation ───────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: Adapted evaluation on test tasks (25 tasks)")
    logger.info("=" * 70)

    mlx_proc = start_mlx_server(adapter_path=str(ADAPTER_DIR))

    adapted_traj_dir = OUTPUT_DIR / "trajectories" / "adapted_eval"
    adapted_trajectories, adapted_results, adapted_resolved, mlx_proc = evaluate_condition(
        test_tasks, adapted_traj_dir, "adapted",
        server_proc=mlx_proc, adapter_path=str(ADAPTER_DIR),
    )

    if mlx_proc:
        stop_mlx_server(mlx_proc)

    logger.info(f"\nAdapted eval: {adapted_resolved}/25 ({adapted_resolved/25*100:.1f}%)")

    # ── Final Report ──────────────────────────────────────────────────
    ft = (adapted_resolved - base_resolved) / 25
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Collection: {n_resolved}/25 resolved ({n_resolved/25*100:.1f}%)")
    logger.info(f"  Training: {len(train_examples)} examples, "
                f"loss={metrics.get('final_train_loss', '?')}")
    logger.info(f"  Base eval: {base_resolved}/25 ({base_resolved/25*100:.1f}%)")
    logger.info(f"  Adapted eval: {adapted_resolved}/25 ({adapted_resolved/25*100:.1f}%)")
    logger.info(f"  Forward Transfer: {ft:+.4f}")

    # Save experiment result
    config = ExperimentConfig(
        name="experiment_selfplay_14b",
        experiment_id="selfplay_14b_v1",
        repo="django/django",
        model_name=f"{MLX_MODEL} + LoRA (rank={LORA_RANK})",
        train_task_ids=sorted(resolved_ids),
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="selfplay_14b_v1",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=collection_results + base_results + adapted_results,
        forward_transfer=ft,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "experiment_result.json")

    write_report(
        resolved_ids, n_with_patch, n_resolved,
        train_examples, val_examples, metrics,
        base_resolved, adapted_resolved, ft,
        base_results, adapted_results,
    )

    logger.info(f"\nResults saved to {OUTPUT_DIR}")
    return result


def write_report(
    resolved_ids, n_with_patch, n_resolved,
    train_ex, val_ex, metrics,
    base_resolved, adapted_resolved, ft,
    base_results, adapted_results,
):
    """Write full experiment report."""
    # Build per-task comparison table
    base_map = {r.task_id: r for r in base_results}
    adapted_map = {r.task_id: r for r in adapted_results}

    task_rows = ""
    for task_id in sorted(set(list(base_map.keys()) + list(adapted_map.keys()))):
        br = base_map.get(task_id)
        ar = adapted_map.get(task_id)
        task_rows += (
            f"| {task_id} "
            f"| {'YES' if br and br.resolved else 'no'} "
            f"| {br.steps if br else '-'} "
            f"| {'YES' if ar and ar.resolved else 'no'} "
            f"| {ar.steps if ar else '-'} |\n"
        )

    report = f"""# Self-Play Trajectory Collection + LoRA Training Report (14B)

**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Model:** {MLX_MODEL}

## Hypothesis

The base 14B model already generates XML tool-call formatted trajectories.
If we collect resolved trajectories from the base model itself (self-play),
train LoRA on those, the format is already correct (no mismatch).
Does the adapted model improve on held-out tasks?

## Collection (Phase 1-2)

- Tasks: 25 (django/django tasks 1-25)
- Temperature: {COLLECTION_TEMP}
- Patches generated: {n_with_patch}/25 ({n_with_patch/25*100:.1f}%)
- **Resolved (FAIL_TO_PASS verified): {n_resolved}/25 ({n_resolved/25*100:.1f}%)**
- Resolved tasks: {', '.join(sorted(resolved_ids))}

## Training (Phase 3)

- LoRA: rank={LORA_RANK}, layers={LORA_NUM_LAYERS}, lr={LORA_LR}, iters={LORA_ITERS}
- Training examples: {len(train_ex)} ({len(train_ex) - len(val_ex)} step + full)
- Validation examples: {len(val_ex)}
- Final train loss: {metrics.get('final_train_loss', '?')}
- Final val loss: {metrics.get('final_val_loss', '?')}

## Evaluation (Phase 4)

| Condition | Resolved | Rate |
|-----------|----------|------|
| Base (temp=0.0) | {base_resolved}/25 | {base_resolved/25*100:.1f}% |
| Adapted (temp=0.0) | {adapted_resolved}/25 | {adapted_resolved/25*100:.1f}% |

**Forward Transfer: {ft:+.4f}**

## Per-Task Comparison

| Task | Base Resolved | Base Steps | Adapted Resolved | Adapted Steps |
|------|---------------|------------|------------------|---------------|
{task_rows}

## Analysis

### Key Question: Did self-play LoRA help?

{'YES — adapted model resolved more tasks than base. Forward transfer is positive.' if ft > 0 else 'NO — adapted model did not improve over base.' if ft == 0 else 'NEGATIVE — adapted model performed WORSE than base. LoRA caused regression.'}

### Format Mismatch Test

Unlike prior experiments (oracle SFT, Phase 1 Q&A pairs), this experiment trains
on the exact same format used at inference time: multi-turn XML tool calls with
<bash>, <read>, <write>, <edit>, <done> tags. The training data IS the agent's
own successful trajectories.

If FT > 0: Self-play with format-aligned training data works. The skill layer
thesis is supported.

If FT <= 0: Even with correct format, the model cannot improve from its own
successful trajectories. Possible causes:
1. Too few resolved tasks ({n_resolved}) for generalization
2. Surface pattern learning (finishing fast) rather than skill encoding
3. LoRA capacity insufficient for behavioral skill at this scale

### Comparison with trajectory_collection experiment

The trajectory_collection experiment (temp=0.7, 3 attempts, rank=8) showed:
- 12 unique resolved tasks, 65 training examples
- Base: 10% → Adapted: 2% (FT = -0.08, REGRESSION)
- Root cause: model learned "finish fast in 3 steps" surface pattern

This experiment differs:
- temp=0.3 (more deterministic, cleaner trajectories)
- rank=16 (more capacity)
- Single attempt (no selection bias from multiple attempts)

## Raw Data

- Collection trajectories: {OUTPUT_DIR}/trajectories/collection/
- Base eval trajectories: {OUTPUT_DIR}/trajectories/base_eval/
- Adapted eval trajectories: {OUTPUT_DIR}/trajectories/adapted_eval/
- Training data: {DATASET_DIR}/
- Adapter weights: {ADAPTER_DIR}/

Generated at {datetime.now(timezone.utc).isoformat()}
"""
    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write(report)
    logger.info(f"Report written to {OUTPUT_DIR / 'REPORT.md'}")


def write_report_negative(n_resolved, n_with_patch, collection_results):
    """Write report for when pilot fails (too few resolved)."""
    report = f"""# Self-Play Trajectory Collection Report (14B) — PILOT FAILED

**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Model:** {MLX_MODEL}

## Result: INSUFFICIENT DATA

- Tasks: 25 (django/django tasks 1-25)
- Temperature: {COLLECTION_TEMP}
- Patches generated: {n_with_patch}/25
- **Resolved: {n_resolved}/25 (need >{MIN_RESOLVED})**

The pilot did not produce enough resolved trajectories for meaningful
LoRA training. The experiment was stopped before training.

## Implications

With only {n_resolved} resolved tasks at temp={COLLECTION_TEMP}, the base 14B model
does not produce enough successful trajectories for self-play training on Django tasks.

## Per-Task Results

| Task | Resolved | Steps | Tokens |
|------|----------|-------|--------|
"""
    for r in sorted(collection_results, key=lambda x: x.task_id):
        report += f"| {r.task_id} | {'YES' if r.resolved else 'no'} | {r.steps} | {r.tokens} |\n"

    report += f"\nGenerated at {datetime.now(timezone.utc).isoformat()}\n"
    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
