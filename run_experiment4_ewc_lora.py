#!/usr/bin/env python3
"""Experiment 4: EWC-LoRA (Elastic Weight Consolidation for LoRA).

Tests whether EWC regularization on LoRA parameters during sequential updates
reduces catastrophic forgetting, or whether LoRA's low-rank constraint already
provides sufficient implicit regularization.

From docs/experiments.md:
- After training on batch t, compute diagonal Fisher Information Matrix
- On batch t+1, add EWC penalty: lambda * sum(F_i * (theta_i - theta_i*)^2)
- Sweep lambda values: [0, 10, 100, 1000]

Uses XML tool-call trajectories (format-matched from experiment 2).

Conditions on test tasks:
1. base       — Ollama, no adapter (reuse experiment9 base trajectories)
2. naive_seq  — Sequential LoRA updates without EWC (lambda=0)
3. ewc_10     — EWC with lambda=10
4. ewc_100    — EWC with lambda=100
5. ewc_1000   — EWC with lambda=1000

Sequential batches: 5 batches of ~5 trajectories each.
Training at each batch: new examples only + EWC penalty (no replay).
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

from looper.agent.ollama_client import ChatMessage, chat, openai_chat
from looper.collectors.trajectory_store import (
    collect_trajectories,
    load_all_trajectories,
)
from looper.evaluators.metrics import compare_conditions, resolve_rate
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results
from looper.models import (
    AgentTrajectory,
    ExperimentConfig,
    ExperimentResult,
    TaskInfo,
    TaskResult,
    TrainingExample,
)
from looper.synthesizers.trajectory_synthesizer import (
    trajectories_to_training_examples,
)
from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks
from looper.trainers.lora_trainer import LoRAConfig

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434"
HF_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MLX_PORT = 8080

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
PHASE1_TRAJ_DIR = Path("/Volumes/1TB_SSD/looper/results/phase1/trajectories/base")

# Reuse experiment9 base trajectories
EXP9_BASE_DIR = Path(
    "/Volumes/1TB_SSD/looper/results/experiment9_ablation/trajectories/base"
)

OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment4_ewc_lora")

MAX_STEPS = 10
OLLAMA_MAX_TOKENS = 4096
MLX_MAX_TOKENS = 512

LORA_RANK = 16
LORA_ITERS = 50  # Fewer iters per batch to prevent overfitting → NaN on next batch
LORA_LR = 5e-5  # Lower LR for sequential training stability
LORA_BATCH = 1
LORA_MAX_SEQ = 1024
FISHER_SAMPLES = 50  # Faster Fisher computation

NUM_BATCHES = 5
LAMBDA_VALUES = [0, 10, 100, 1000]
PILOT_LAMBDAS = [0, 100, 1000]
PILOT_TEST_TASKS = 10


# ── Logging ───────────────────────────────────────────────────────────


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


# ── MLX Server ────────────────────────────────────────────────────────


def start_mlx_server(model: str, port: int, adapter_path: str | None = None):
    """Start mlx_lm.server and wait for it to be ready."""
    import httpx

    venv_bin = Path(sys.executable).parent
    cmd = [str(venv_bin / "mlx_lm.server"), "--model", model, "--port", str(port)]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    logger.info(f"  Starting MLX server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(60):
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


# ── Training ──────────────────────────────────────────────────────────


def save_batch_data(
    examples: list[TrainingExample], path: Path
) -> None:
    """Save training examples as JSONL for EWC trainer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")


def train_ewc_batch(
    data_path: Path,
    adapter_dir: Path,
    label: str,
    ewc_lambda: float,
    prev_adapter_path: Path | None = None,
    fisher_path: Path | None = None,
    old_params_path: Path | None = None,
) -> dict:
    """Train one EWC batch in a subprocess to free GPU memory afterwards."""
    adapter_file = adapter_dir / "adapters.safetensors"
    metrics_file = adapter_dir.parent / f"training_metrics_{label}.json"

    if adapter_file.exists():
        logger.info(f"  [{label}] Adapter already trained, skipping...")
        if metrics_file.exists():
            return json.loads(metrics_file.read_text())
        return {"cached": True}

    # Stop Ollama to free GPU memory
    logger.info(f"  [{label}] Stopping Ollama to free GPU memory...")
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(3)

    prev_str = repr(str(prev_adapter_path)) if prev_adapter_path else "None"
    fisher_str = repr(str(fisher_path)) if fisher_path else "None"
    params_str = repr(str(old_params_path)) if old_params_path else "None"

    train_script = f"""
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
import json
from looper.trainers.ewc_trainer import train_lora_ewc

metrics = train_lora_ewc(
    model_name='{HF_MODEL}',
    data_path='{data_path}',
    adapter_dir='{adapter_dir}',
    rank={LORA_RANK},
    num_layers=16,
    learning_rate={LORA_LR},
    iters={LORA_ITERS},
    batch_size={LORA_BATCH},
    max_seq_length={LORA_MAX_SEQ},
    ewc_lambda={ewc_lambda},
    prev_adapter_path={prev_str},
    fisher_path={fisher_str},
    old_params_path={params_str},
    fisher_samples={FISHER_SAMPLES},
)
print(json.dumps(metrics))
"""
    logger.info(
        f"  [{label}] Training EWC batch (lambda={ewc_lambda})..."
    )
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
        timeout=3600,
    )

    if result.returncode != 0:
        logger.error(f"[{label}] Training failed:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"EWC training failed for {label}")

    metrics = {}
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info(
        f"  [{label}] Done: train_loss={metrics.get('final_train_loss', '?')}"
    )
    return metrics


def train_ewc_sequential(
    batches: list[list[TrainingExample]],
    ewc_lambda: float,
    adapter_base_dir: Path,
    data_base_dir: Path,
) -> dict:
    """Train LoRA sequentially through batches with EWC regularization.

    At each batch:
    1. Train on new data only (+ EWC penalty if lambda > 0)
    2. Compute Fisher on current data
    3. Accumulate Fisher for next batch
    4. Save params snapshot for next batch

    Returns metrics from the final batch.
    """
    label_prefix = f"ewc_{int(ewc_lambda)}" if ewc_lambda > 0 else "naive_seq"
    prev_adapter_path = None
    fisher_path = None
    old_params_path = None

    batch_metrics = []
    for batch_idx, batch_examples in enumerate(batches):
        batch_label = f"{label_prefix}_b{batch_idx}"
        logger.info(
            f"  [{label_prefix}] Batch {batch_idx + 1}/{len(batches)}: "
            f"{len(batch_examples)} examples, lambda={ewc_lambda}"
        )

        # Save batch data
        data_path = data_base_dir / f"{batch_label}.jsonl"
        save_batch_data(batch_examples, data_path)

        # Adapter directory for this batch
        adapter_dir = adapter_base_dir / batch_label
        adapter_dir.mkdir(parents=True, exist_ok=True)

        metrics = train_ewc_batch(
            data_path=data_path,
            adapter_dir=adapter_dir,
            label=batch_label,
            ewc_lambda=ewc_lambda,
            prev_adapter_path=prev_adapter_path,
            fisher_path=fisher_path,
            old_params_path=old_params_path,
        )
        batch_metrics.append(metrics)

        # Update paths for next batch
        prev_adapter_path = adapter_dir / "adapters.safetensors"
        fisher_path = adapter_dir / "fisher.safetensors"
        old_params_path = adapter_dir / "params_snapshot.safetensors"

    # Copy final adapter to a clean location for evaluation
    final_label = label_prefix
    final_adapter_dir = adapter_base_dir / final_label
    if not (final_adapter_dir / "adapters.safetensors").exists():
        final_adapter_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        last_batch_dir = adapter_base_dir / f"{label_prefix}_b{len(batches) - 1}"
        for f in ["adapters.safetensors", "adapter_config.json"]:
            src = last_batch_dir / f
            if src.exists():
                shutil.copy2(src, final_adapter_dir / f)

    return {
        "label": final_label,
        "ewc_lambda": ewc_lambda,
        "num_batches": len(batches),
        "batch_metrics": batch_metrics,
        "final_train_loss": batch_metrics[-1].get("final_train_loss", 0.0)
        if batch_metrics
        else 0.0,
    }


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_trajectories(
    trajectories: list[AgentTrajectory],
    tasks: list[TaskInfo],
    workspace_root: Path,
    condition: str,
) -> list[TaskResult]:
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


def sanity_check_adapter(adapter_dir: Path) -> bool:
    """Quick check: does the adapted model produce XML tool calls?"""
    logger.info("  Sanity check: verifying adapter produces XML tool calls...")

    mlx_proc = start_mlx_server(HF_MODEL, MLX_PORT, str(adapter_dir))
    try:
        from looper.agent.runner import SYSTEM_PROMPT

        test_prompt = SYSTEM_PROMPT.format(
            workspace_dir="/workspace",
            problem_statement="Fix a bug where URLValidator accepts invalid chars.",
        )
        messages = [
            ChatMessage(role="system", content=test_prompt),
            ChatMessage(
                role="user",
                content="Fix a bug where URLValidator accepts invalid chars.",
            ),
        ]

        response = openai_chat(
            messages,
            model=HF_MODEL,
            base_url=f"http://127.0.0.1:{MLX_PORT}",
            max_tokens=MLX_MAX_TOKENS,
        )

        has_tools = any(
            tag in response.content
            for tag in ["<bash>", "<read>", "<write>", "<done>"]
        )

        logger.info(f"  Sanity check response ({len(response.content)} chars):")
        logger.info(f"  {response.content[:300]}")
        logger.info(f"  Contains tool calls: {has_tools}")

        return has_tools
    except Exception as e:
        logger.error(f"  Sanity check failed with error: {e}")
        return False
    finally:
        stop_mlx_server(mlx_proc)


def patch_rate(trajectories: list[AgentTrajectory]) -> float:
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t.generated_patch.strip()) / len(
        trajectories
    )


# ── Report ────────────────────────────────────────────────────────────


def write_report(
    all_results: dict[str, list[TaskResult]],
    all_trajectories: dict[str, list[AgentTrajectory]],
    training_metrics: dict[str, dict],
    total_examples: int,
    output_dir: Path,
):
    """Write experiment report."""
    lines = [
        "# Experiment 4: EWC-LoRA (Elastic Weight Consolidation for LoRA)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {OLLAMA_MODEL} (base) / {HF_MODEL} (adapted)",
        f"**Training data:** XML tool-call trajectories (trajectory_synthesizer.py)",
        f"**Total training examples:** {total_examples}",
        f"**Batches:** {NUM_BATCHES} sequential batches",
        f"**Lambda values:** {[m.get('ewc_lambda', '?') for m in training_metrics.values()]}",
        "",
        "## Hypothesis",
        "",
        "EWC regularization on LoRA parameters during sequential updates will reduce",
        "catastrophic forgetting. However, LoRA's low-rank constraint may already",
        "provide sufficient implicit regularization, making EWC redundant.",
        "",
        "## Training Metrics",
        "",
        "| Condition | Lambda | Final Loss | Batches |",
        "|-----------|--------|------------|---------|",
    ]

    for label, metrics in sorted(training_metrics.items()):
        lam = metrics.get("ewc_lambda", "?")
        tl = metrics.get("final_train_loss", "?")
        nb = metrics.get("num_batches", "?")
        if isinstance(tl, float):
            tl = f"{tl:.4f}"
        lines.append(f"| {label:15s} | {lam} | {tl} | {nb} |")

    # Per-batch loss progression
    lines += ["", "## Per-Batch Loss Progression", ""]
    for label, metrics in sorted(training_metrics.items()):
        batch_metrics = metrics.get("batch_metrics", [])
        if batch_metrics:
            lines.append(f"### {label} (lambda={metrics.get('ewc_lambda', '?')})")
            for i, bm in enumerate(batch_metrics):
                lines.append(
                    f"  Batch {i + 1}: loss={bm.get('final_train_loss', '?')}"
                )
            lines.append("")

    lines += [
        "## Evaluation Results",
        "",
        "| Condition | Resolved | Patch Rate | Avg Steps | FT vs Base |",
        "|-----------|----------|------------|-----------|------------|",
    ]

    base_rr = resolve_rate(all_results.get("base", []))
    for cond in sorted(all_results.keys()):
        results = all_results[cond]
        trajs = all_trajectories.get(cond, [])
        rr = resolve_rate(results)
        pr = patch_rate(trajs)
        resolved_n = sum(1 for r in results if r.resolved)
        patched_n = sum(1 for t in trajs if t.generated_patch.strip())
        avg_steps = (
            sum(r.steps for r in results) / len(results) if results else 0
        )
        ft = rr - base_rr

        lines.append(
            f"| {cond:15s} | {resolved_n}/{len(results)} ({rr:.0%}) "
            f"| {patched_n}/{len(trajs)} ({pr:.0%}) "
            f"| {avg_steps:.1f} | {ft:+.4f} |"
        )

    # Key findings
    lines += ["", "## Key Findings", ""]

    adapted_conditions = [c for c in all_results if c != "base"]

    # Check if EWC conditions beat naive sequential
    naive_rr = resolve_rate(all_results.get("naive_seq", []))
    ewc_rrs = {
        c: resolve_rate(all_results[c])
        for c in adapted_conditions
        if c.startswith("ewc_")
    }

    if ewc_rrs:
        best_ewc_label = max(ewc_rrs, key=ewc_rrs.get)
        best_ewc_rr = ewc_rrs[best_ewc_label]

        if best_ewc_rr > naive_rr:
            lines.append(
                f"- **EWC HELPS**: Best EWC ({best_ewc_label}, {best_ewc_rr:.0%}) "
                f"> naive sequential ({naive_rr:.0%})"
            )
        elif best_ewc_rr == naive_rr:
            lines.append(
                f"- **EWC NEUTRAL**: Best EWC = naive sequential ({naive_rr:.0%}). "
                "LoRA's low-rank constraint may provide sufficient regularization."
            )
        else:
            lines.append(
                f"- **EWC HURTS**: Best EWC ({best_ewc_rr:.0%}) "
                f"< naive sequential ({naive_rr:.0%}). "
                "EWC penalty may be too restrictive in low-rank space."
            )

    best_adapted_rr = max(
        (resolve_rate(all_results[c]) for c in adapted_conditions),
        default=0.0,
    )
    if best_adapted_rr > base_rr:
        lines.append(
            f"- **POSITIVE FORWARD TRANSFER**: Best adapted ({best_adapted_rr:.0%}) "
            f"> base ({base_rr:.0%})"
        )
    elif best_adapted_rr == base_rr:
        lines.append(
            f"- **ZERO FORWARD TRANSFER**: Adapted = base ({base_rr:.0%})"
        )
    else:
        lines.append(
            f"- **NEGATIVE FORWARD TRANSFER**: Best adapted ({best_adapted_rr:.0%}) "
            f"< base ({base_rr:.0%})"
        )

    report = "\n".join(lines) + "\n"
    report_path = output_dir / "EXPERIMENT4_REPORT.md"
    report_path.write_text(report)
    logger.info(f"  Report written to {report_path}")
    return report


# ── Main ──────────────────────────────────────────────────────────────


def run_experiment4(
    output_dir: Path = OUTPUT_DIR,
    pilot: bool = False,
):
    setup_logging(output_dir)
    started_at = datetime.now(timezone.utc).isoformat()

    lambdas = PILOT_LAMBDAS if pilot else LAMBDA_VALUES
    mode = "PILOT" if pilot else "FULL"

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 4: EWC-LoRA ({mode})")
    logger.info(f"Lambda values: {lambdas}")
    logger.info("Training data: XML tool-call trajectories (FORMAT-MATCHED)")
    logger.info("=" * 60)

    # ── Step 0: Load data ─────────────────────────────────────────────

    logger.info("\nStep 0: Loading data...")
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, test_tasks = split_tasks(all_tasks, train_size=25, seed=None)

    if pilot:
        test_tasks = test_tasks[:PILOT_TEST_TASKS]

    logger.info(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Load phase1 train trajectories
    all_phase1 = load_all_trajectories(PHASE1_TRAJ_DIR)
    train_ids = {t.instance_id for t in train_tasks}
    train_trajectories = [t for t in all_phase1 if t.meta.task_id in train_ids]
    logger.info(f"  Train trajectories: {len(train_trajectories)}")

    # ── Step 1: Convert to XML training examples ─────────────────────

    logger.info("\nStep 1: Converting trajectories to XML training data...")
    all_examples = trajectories_to_training_examples(
        train_trajectories, train_tasks, per_step=True
    )
    logger.info(f"  Total per-step XML training examples: {len(all_examples)}")

    if not all_examples:
        logger.error("No training examples generated! Aborting.")
        return

    # Save all training data for reference
    synthesis_dir = output_dir / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    all_training_file = synthesis_dir / "all_xml_training.jsonl"
    save_batch_data(all_examples, all_training_file)
    logger.info(f"  Saved to {all_training_file}")

    # ── Step 2: Split into sequential batches ────────────────────────

    logger.info("\nStep 2: Splitting into sequential batches...")

    # Group examples by source trajectory (task)
    traj_order = [t.meta.task_id for t in train_trajectories]
    examples_by_task: dict[str, list[TrainingExample]] = {}
    for ex in all_examples:
        task_id = ex.source_pair_id.rsplit("_step", 1)[0]
        examples_by_task.setdefault(task_id, []).append(ex)

    # Split trajectories into NUM_BATCHES groups
    batch_size = len(traj_order) // NUM_BATCHES
    batches: list[list[TrainingExample]] = []
    for i in range(NUM_BATCHES):
        start = i * batch_size
        end = start + batch_size if i < NUM_BATCHES - 1 else len(traj_order)
        batch_task_ids = traj_order[start:end]
        batch_examples = []
        for tid in batch_task_ids:
            batch_examples.extend(examples_by_task.get(tid, []))
        batches.append(batch_examples)
        logger.info(
            f"  Batch {i + 1}: {len(batch_task_ids)} trajectories, "
            f"{len(batch_examples)} examples"
        )

    # ── Step 3: Train EWC conditions ─────────────────────────────────

    logger.info("\nStep 3: Training EWC conditions...")
    adapters_dir = output_dir / "adapters"
    data_dir = output_dir / "batch_data"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_training_metrics: dict[str, dict] = {}

    for ewc_lambda in lambdas:
        label = f"ewc_{int(ewc_lambda)}" if ewc_lambda > 0 else "naive_seq"
        logger.info(f"\n  Training {label} (lambda={ewc_lambda})...")

        metrics = train_ewc_sequential(
            batches=batches,
            ewc_lambda=ewc_lambda,
            adapter_base_dir=adapters_dir,
            data_base_dir=data_dir,
        )
        all_training_metrics[label] = metrics

    # Log training comparison
    logger.info("\n  Training comparison:")
    logger.info(f"  {'Condition':15s} {'Lambda':>7s} {'Final Loss':>11s}")
    for label, m in sorted(all_training_metrics.items()):
        lam = m.get("ewc_lambda", "?")
        tl = m.get("final_train_loss", "?")
        tl_s = f"{tl:.4f}" if isinstance(tl, float) else str(tl)
        logger.info(f"  {label:15s} {str(lam):>7s} {tl_s:>11s}")

    # Save training metrics
    (output_dir / "all_training_metrics.json").write_text(
        json.dumps(all_training_metrics, indent=2, default=str)
    )

    # ── Step 4: Sanity check best adapter ─────────────────────────────

    logger.info("\nStep 4: Sanity check — verifying adapter produces tool calls...")
    # Check the naive_seq adapter (lambda=0, should be most different from base)
    naive_adapter = adapters_dir / "naive_seq"
    sanity_ok = False
    if (naive_adapter / "adapters.safetensors").exists():
        sanity_ok = sanity_check_adapter(naive_adapter)

    if not sanity_ok:
        logger.warning(
            "  SANITY CHECK FAILED: adapter does not produce tool calls. "
            "Continuing anyway to collect data."
        )
    else:
        logger.info("  SANITY CHECK PASSED: adapter produces XML tool calls!")

    # ── Step 5: Evaluate base condition (reuse experiment9) ──────────

    logger.info("\nStep 5: Loading base condition trajectories...")
    base_trajs = load_all_trajectories(EXP9_BASE_DIR)
    test_ids = {t.instance_id for t in test_tasks}
    base_trajs = [t for t in base_trajs if t.meta.task_id in test_ids]

    if len(base_trajs) < len(test_tasks):
        logger.warning(
            f"  Only {len(base_trajs)} base trajectories found "
            f"(expected {len(test_tasks)}). Running missing tasks..."
        )
        existing_ids = {t.meta.task_id for t in base_trajs}
        missing_tasks = [t for t in test_tasks if t.instance_id not in existing_ids]
        if missing_tasks:
            new_dir = output_dir / "trajectories" / "base"
            new_trajs = collect_trajectories(
                tasks=missing_tasks,
                output_dir=new_dir,
                workspace_root=WORKSPACE_ROOT,
                model=OLLAMA_MODEL,
                base_url=OLLAMA_URL,
                max_steps=MAX_STEPS,
                max_tokens=OLLAMA_MAX_TOKENS,
            )
            base_trajs.extend(new_trajs)

    logger.info(f"  Base trajectories: {len(base_trajs)}")
    base_results = evaluate_trajectories(
        base_trajs, test_tasks, WORKSPACE_ROOT, "base"
    )
    base_resolved = sum(1 for r in base_results if r.resolved)
    logger.info(f"  Base: {base_resolved}/{len(base_results)} resolved")

    # ── Step 6: Evaluate adapted conditions ──────────────────────────

    logger.info("\nStep 6: Evaluating adapted conditions...")

    # Stop Ollama before MLX
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(3)

    all_trajectories: dict[str, list[AgentTrajectory]] = {"base": base_trajs}
    all_results: dict[str, list[TaskResult]] = {"base": base_results}

    eval_conditions = list(all_training_metrics.keys())
    logger.info(f"  Evaluation conditions: base + {eval_conditions}")

    for cond_label in eval_conditions:
        adapter_dir = adapters_dir / cond_label
        if not (adapter_dir / "adapters.safetensors").exists():
            logger.warning(f"  [{cond_label}] No adapter found, skipping")
            continue

        logger.info(f"\n  Evaluating {cond_label}...")
        mlx_proc = start_mlx_server(HF_MODEL, MLX_PORT, str(adapter_dir))

        try:

            def make_chat_fn():
                def adapted_chat_fn(messages, model="", base_url="", **kwargs):
                    return openai_chat(
                        messages,
                        model=HF_MODEL,
                        base_url=f"http://127.0.0.1:{MLX_PORT}",
                        max_tokens=MLX_MAX_TOKENS,
                    )

                return adapted_chat_fn

            cond_dir = output_dir / "trajectories" / cond_label
            cond_trajs = collect_trajectories(
                tasks=test_tasks,
                output_dir=cond_dir,
                workspace_root=WORKSPACE_ROOT,
                model=HF_MODEL,
                base_url=f"http://127.0.0.1:{MLX_PORT}",
                max_steps=MAX_STEPS,
                max_tokens=MLX_MAX_TOKENS,
                chat_fn=make_chat_fn(),
                on_complete=lambda tid, traj: logger.info(
                    f"  [{cond_label}] {tid} -> {traj.outcome} "
                    f"({len(traj.steps)} steps, "
                    f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
                ),
            )

            all_trajectories[cond_label] = cond_trajs

            # Evaluate
            cond_results = evaluate_trajectories(
                cond_trajs, test_tasks, WORKSPACE_ROOT, cond_label
            )
            all_results[cond_label] = cond_results
            resolved = sum(1 for r in cond_results if r.resolved)
            patched = sum(1 for t in cond_trajs if t.generated_patch.strip())
            logger.info(
                f"  [{cond_label}] {resolved}/{len(cond_results)} resolved, "
                f"{patched}/{len(cond_trajs)} patched"
            )
        finally:
            stop_mlx_server(mlx_proc)

    # Restart Ollama
    logger.info("  Restarting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        env={
            **os.environ,
            "OLLAMA_MODELS": "/Volumes/1TB_SSD/looper/ollama_models",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # ── Step 7: Compute metrics and save ─────────────────────────────

    logger.info("\nStep 7: Computing metrics and saving results...")

    comparison = compare_conditions(all_results)

    base_rr = resolve_rate(all_results["base"])
    ft = {}
    for cond in all_results:
        if cond != "base":
            ft[f"{cond}_vs_base"] = resolve_rate(all_results[cond]) - base_rr

    logger.info("\n" + "=" * 60)
    logger.info(f"EXPERIMENT 4 RESULTS ({mode})")
    logger.info("=" * 60)
    for cond, metrics in sorted(comparison.items()):
        logger.info(
            f"  {cond:15s}: resolve={metrics['resolve_rate']:.2%}, "
            f"steps={metrics['avg_steps']:.1f}, "
            f"tokens={metrics['avg_tokens']:.0f}"
        )
    for label, val in sorted(ft.items()):
        logger.info(f"  FT ({label}): {val:+.4f}")

    # Save ExperimentResult
    all_task_results = []
    for results in all_results.values():
        all_task_results.extend(results)

    config = ExperimentConfig(
        name="experiment4_ewc_lora",
        experiment_id=f"exp4_ewc_lora_{mode.lower()}",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="ewc_lora_sequential",
        lora_rank=LORA_RANK,
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=all_task_results,
        forward_transfer=max(ft.values()) if ft else 0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, output_dir / "experiment_result.json")

    # Save detailed results
    detailed = {
        "experiment": "experiment4_ewc_lora",
        "mode": mode.lower(),
        "model": OLLAMA_MODEL,
        "hf_model": HF_MODEL,
        "training_data_format": "xml_tool_call_trajectories",
        "total_training_examples": len(all_examples),
        "num_batches": NUM_BATCHES,
        "lambda_values": lambdas,
        "training_metrics": all_training_metrics,
        "sanity_check_passed": sanity_ok,
        "conditions_evaluated": list(all_results.keys()),
        "comparison": comparison,
        "forward_transfer": ft,
        "patch_rates": {
            cond: patch_rate(trajs) for cond, trajs in all_trajectories.items()
        },
        "per_task": {
            cond: [
                {"task_id": r.task_id, "resolved": r.resolved, "steps": r.steps}
                for r in results
            ]
            for cond, results in all_results.items()
        },
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "detailed_results.json").write_text(
        json.dumps(detailed, indent=2, default=str)
    )

    # ── Step 8: Write report ──────────────────────────────────────────

    write_report(
        all_results,
        all_trajectories,
        all_training_metrics,
        len(all_examples),
        output_dir,
    )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("=" * 60)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 4: EWC-LoRA")
    parser.add_argument(
        "--pilot", action="store_true", help="Run pilot (3 lambdas, 10 test tasks)"
    )
    args = parser.parse_args()

    run_experiment4(pilot=args.pilot)
