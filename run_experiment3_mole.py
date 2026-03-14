#!/usr/bin/env python3
"""Experiment 3: Mixture of LoRA Experts (MoLE).

Tests whether separating skills into specialized adapters (by tool-call type)
and merging them reduces inter-skill interference compared to a single adapter.

Also tests the critical untested hypothesis from DEEP_AUDIT: training on
ONLY successful trajectories (patch_generated) instead of all trajectories.

Expert Categories (by XML tool-call type in training data):
  1. "search"  — steps using <bash> (find, grep, ls, etc.)
  2. "read"    — steps using <read> (file reading)
  3. "modify"  — steps using <write> or <done> (code changes, completion)

Configurations evaluated on 10 test tasks (pilot):
  1. base            — Ollama, no adapter (reuse experiment9 base trajectories)
  2. single_all      — Single rank-16 adapter, ALL training examples
  3. single_success  — Single rank-16 adapter, ONLY successful trajectories (KEY TEST)
  4. mole_3_all      — 3 experts (rank 5 each), ALL examples, uniform merge
  5. mole_3_success  — 3 experts (rank 5 each), successful-only, uniform merge

Uses XML tool-call trajectories (trajectory_synthesizer.py) per DEEP_AUDIT.
"""

import json
import logging
import os
import shutil
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

OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment3_mole")

MAX_STEPS = 10
OLLAMA_MAX_TOKENS = 4096
MLX_MAX_TOKENS = 512

LORA_ITERS = 100
LORA_BATCH = 1
LORA_MAX_SEQ = 1024

PILOT_TEST_TASKS = 10


# ── Skill Categories ─────────────────────────────────────────────────

SKILL_CATEGORIES = {
    "search": lambda msg: "<bash>" in msg,
    "read": lambda msg: "<read>" in msg,
    "modify": lambda msg: "<write" in msg or "<done>" in msg,
}


def categorize_examples(
    examples: list[TrainingExample],
) -> dict[str, list[TrainingExample]]:
    """Split training examples by tool-call type in the assistant message.

    Each example is assigned to exactly one category based on the last
    assistant message. If an example matches multiple categories, the first
    match in priority order (search, read, modify) wins.
    """
    categorized: dict[str, list[TrainingExample]] = {k: [] for k in SKILL_CATEGORIES}
    uncategorized = []

    for ex in examples:
        # Get the assistant message (last message in per-step format)
        assistant_msg = ""
        for msg in reversed(ex.messages):
            if msg["role"] == "assistant":
                assistant_msg = msg["content"]
                break

        matched = False
        for cat_name, matcher in SKILL_CATEGORIES.items():
            if matcher(assistant_msg):
                categorized[cat_name].append(ex)
                matched = True
                break

        if not matched:
            uncategorized.append(ex)

    logger.info("  Categorized examples:")
    for cat, exs in categorized.items():
        logger.info(f"    {cat}: {len(exs)} examples")
    if uncategorized:
        logger.info(f"    uncategorized: {len(uncategorized)}")
        # Add uncategorized to the largest category
        largest = max(categorized, key=lambda k: len(categorized[k]))
        categorized[largest].extend(uncategorized)
        logger.info(f"    → added to {largest}")

    return categorized


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


def train_adapter(
    examples: list[TrainingExample],
    adapter_dir: Path,
    label: str,
    rank: int = 16,
) -> dict:
    """Train LoRA adapter in a subprocess to free GPU memory afterwards."""
    from looper.synthesizers.synthesizer import save_training_data

    adapter_file = adapter_dir / "adapters.safetensors"
    metrics_file = adapter_dir.parent / f"training_metrics_{label}.json"

    if adapter_file.exists():
        logger.info(f"  [{label}] Adapter already trained, skipping...")
        if metrics_file.exists():
            return json.loads(metrics_file.read_text())
        return {"cached": True}

    if len(examples) < 2:
        logger.warning(
            f"  [{label}] Only {len(examples)} examples — too few to train. Skipping."
        )
        return {"skipped": True, "reason": "too_few_examples", "num_examples": len(examples)}

    # Save training data to temp file
    training_file = adapter_dir.parent / f"training_{label}.jsonl"
    save_training_data(examples, training_file)

    # Stop Ollama to free GPU memory
    logger.info(f"  [{label}] Stopping Ollama to free GPU memory...")
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

examples = load_training_data(Path('{training_file}'))
config = LoRAConfig(
    rank={rank},
    iters={LORA_ITERS},
    batch_size={LORA_BATCH},
    max_seq_length={LORA_MAX_SEQ},
)
metrics = full_replay_train(examples, '{HF_MODEL}', Path('{adapter_dir}'), config)
print(json.dumps(metrics))
"""
    logger.info(f"  [{label}] Training LoRA adapter ({len(examples)} examples, rank={rank})...")
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
        timeout=3600,
    )

    if result.returncode != 0:
        logger.error(f"[{label}] Training failed:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"LoRA training failed for {label}")

    metrics = {}
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    metrics["num_examples"] = len(examples)
    metrics["rank"] = rank
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info(
        f"  [{label}] Done: train_loss={metrics.get('final_train_loss', '?')}, "
        f"val_loss={metrics.get('final_val_loss', '?')}"
    )
    return metrics


# ── MoLE Merge ───────────────────────────────────────────────────────


def merge_expert_adapters(
    expert_dirs: list[Path],
    output_dir: Path,
    weights: list[float] | None = None,
) -> bool:
    """Merge multiple LoRA expert adapters by weight-averaging.

    This is the standard MoLE merge: for each tensor in the adapter,
    compute weighted average across experts. All experts must have
    the same rank and architecture.

    Returns True if merge succeeded.
    """
    import mlx.core as mx

    if weights is None:
        weights = [1.0 / len(expert_dirs)] * len(expert_dirs)

    # Filter to experts that actually exist
    valid_dirs = [d for d in expert_dirs if (d / "adapters.safetensors").exists()]
    if len(valid_dirs) < 2:
        logger.warning(
            f"  Only {len(valid_dirs)} valid expert adapters found "
            f"(need at least 2 for merge). Skipping merge."
        )
        if len(valid_dirs) == 1:
            # Just copy the single expert
            output_dir.mkdir(parents=True, exist_ok=True)
            for f in ["adapters.safetensors", "adapter_config.json"]:
                src = valid_dirs[0] / f
                if src.exists():
                    shutil.copy2(src, output_dir / f)
            return True
        return False

    # Recompute uniform weights for valid experts
    w = [1.0 / len(valid_dirs)] * len(valid_dirs)

    # Load all expert weights
    expert_weights = []
    for d in valid_dirs:
        ew = mx.load(str(d / "adapters.safetensors"))
        expert_weights.append(ew)
        logger.info(f"  Loaded expert: {d.name} ({len(ew)} tensors)")

    # Verify all experts have the same keys
    keys = set(expert_weights[0].keys())
    for i, ew in enumerate(expert_weights[1:], 1):
        if set(ew.keys()) != keys:
            logger.error(f"  Expert {i} has different tensor keys!")
            return False

    # Weight-average each tensor
    merged = {}
    for key in sorted(keys):
        tensors = [ew[key] for ew in expert_weights]
        # Verify shapes match
        shapes = [t.shape for t in tensors]
        if len(set(str(s) for s in shapes)) > 1:
            logger.error(f"  Shape mismatch for {key}: {shapes}")
            return False
        merged[key] = sum(wi * t for wi, t in zip(w, tensors))

    # Save merged adapter
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), merged)

    # Copy adapter_config from first expert (all share same rank)
    config_src = valid_dirs[0] / "adapter_config.json"
    if config_src.exists():
        shutil.copy2(config_src, output_dir / "adapter_config.json")

    logger.info(
        f"  Merged {len(valid_dirs)} experts → {output_dir.name} "
        f"({len(merged)} tensors, weights={[f'{wi:.2f}' for wi in w]})"
    )
    return True


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
    category_counts: dict[str, dict[str, int]],
    total_examples: int,
    success_examples: int,
    output_dir: Path,
):
    """Write experiment report."""
    lines = [
        "# Experiment 3: Mixture of LoRA Experts (MoLE)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {OLLAMA_MODEL} (base) / {HF_MODEL} (adapted)",
        f"**Training data:** XML tool-call trajectories (trajectory_synthesizer.py)",
        f"**Total training examples (all):** {total_examples}",
        f"**Successful-only examples:** {success_examples}",
        "",
        "## Hypotheses",
        "",
        "1. **MoLE hypothesis**: Separating skills into specialized adapters (search,",
        "   read, modify) reduces inter-skill interference vs. single monolithic adapter",
        "2. **Successful-only hypothesis** (from DEEP_AUDIT): Training ONLY on successful",
        "   trajectories (patch_generated) avoids self-distillation from failures",
        "",
        "## Skill Category Distribution",
        "",
        "| Category | All Data | Successful Only |",
        "|----------|----------|-----------------|",
    ]

    for cat in SKILL_CATEGORIES:
        all_n = category_counts.get("all", {}).get(cat, 0)
        suc_n = category_counts.get("success", {}).get(cat, 0)
        lines.append(f"| {cat:8s} | {all_n:>8d} | {suc_n:>15d} |")

    lines += [
        "",
        "## Training Metrics",
        "",
        "| Condition | Rank | Examples | Train Loss | Val Loss |",
        "|-----------|------|----------|------------|----------|",
    ]

    for label, metrics in sorted(training_metrics.items()):
        n = metrics.get("num_examples", "?")
        r = metrics.get("rank", "?")
        tl = metrics.get("final_train_loss", "?")
        vl = metrics.get("final_val_loss", "?")
        if isinstance(tl, float):
            tl = f"{tl:.4f}"
        if isinstance(vl, float):
            vl = f"{vl:.4f}"
        if metrics.get("skipped"):
            tl = "SKIPPED"
            vl = metrics.get("reason", "")
        lines.append(f"| {label:20s} | {r} | {n} | {tl} | {vl} |")

    lines += [
        "",
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
            f"| {cond:20s} | {resolved_n}/{len(results)} ({rr:.0%}) "
            f"| {patched_n}/{len(trajs)} ({pr:.0%}) "
            f"| {avg_steps:.1f} | {ft:+.4f} |"
        )

    # Key findings
    lines += ["", "## Key Findings", ""]

    adapted_conditions = [c for c in all_results if c != "base"]

    # Check successful-only vs all
    single_all_rr = resolve_rate(all_results.get("single_all", []))
    single_success_rr = resolve_rate(all_results.get("single_success", []))
    if "single_success" in all_results:
        if single_success_rr > single_all_rr:
            lines.append(
                f"- **SUCCESSFUL-ONLY HELPS**: single_success ({single_success_rr:.0%}) "
                f"> single_all ({single_all_rr:.0%}). "
                "Training on successful trajectories only is superior."
            )
        elif single_success_rr == single_all_rr:
            lines.append(
                f"- **SUCCESSFUL-ONLY NEUTRAL**: single_success = single_all ({single_all_rr:.0%})"
            )
        else:
            lines.append(
                f"- **SUCCESSFUL-ONLY HURTS**: single_success ({single_success_rr:.0%}) "
                f"< single_all ({single_all_rr:.0%}). "
                "More data (even from failures) is better than less data."
            )

    # Check MoLE vs single
    mole_all_rr = resolve_rate(all_results.get("mole_3_all", []))
    mole_success_rr = resolve_rate(all_results.get("mole_3_success", []))
    if "mole_3_all" in all_results:
        if mole_all_rr > single_all_rr:
            lines.append(
                f"- **MOLE HELPS**: mole_3_all ({mole_all_rr:.0%}) "
                f"> single_all ({single_all_rr:.0%})"
            )
        else:
            lines.append(
                f"- **MOLE DOES NOT HELP**: mole_3_all ({mole_all_rr:.0%}) "
                f"vs single_all ({single_all_rr:.0%})"
            )

    # Overall FT assessment
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

    # Context from prior experiments
    lines += [
        "",
        "## Context (Prior Experiments)",
        "",
        "| Experiment | FT | Notes |",
        "|------------|-----|-------|",
        "| Phase 1 (full replay) | -0.08 | Q&A format, format mismatch |",
        "| Exp 2 (partial replay) | -0.08 to -0.10 | XML format, all data |",
        "| Exp 4 (EWC-LoRA) | -0.10 | XML format, sequential batches |",
        "| Exp 9 (ablation) | 0.0 | All conditions equal |",
        "",
        "All prior experiments trained on ALL trajectories (92% failures).",
        "This experiment is the first to test successful-only training.",
    ]

    report = "\n".join(lines) + "\n"
    report_path = output_dir / "EXPERIMENT3_REPORT.md"
    report_path.write_text(report)
    logger.info(f"  Report written to {report_path}")
    return report


# ── Main ──────────────────────────────────────────────────────────────


def run_experiment3(
    output_dir: Path = OUTPUT_DIR,
    pilot: bool = True,
):
    setup_logging(output_dir)
    started_at = datetime.now(timezone.utc).isoformat()

    mode = "PILOT" if pilot else "FULL"

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 3: Mixture of LoRA Experts ({mode})")
    logger.info("Training data: XML tool-call trajectories (FORMAT-MATCHED)")
    logger.info("Key test: successful-only training hypothesis")
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

    patched_train = [t for t in train_trajectories if t.generated_patch.strip()]
    logger.info(
        f"  Patch-generating train trajectories: {len(patched_train)}"
    )
    for t in patched_train:
        logger.info(
            f"    {t.meta.task_id}: {len(t.steps)} steps, "
            f"{sum(len(s.tool_calls) for s in t.steps)} tool_calls"
        )

    # ── Step 1: Generate training examples ────────────────────────────

    logger.info("\nStep 1: Converting trajectories to XML training data...")

    # All trajectories
    all_examples = trajectories_to_training_examples(
        train_trajectories, train_tasks, per_step=True
    )
    logger.info(f"  All per-step XML training examples: {len(all_examples)}")

    # Successful-only trajectories
    success_examples = trajectories_to_training_examples(
        train_trajectories, train_tasks, only_successful=True, per_step=True
    )
    logger.info(f"  Successful-only per-step XML training examples: {len(success_examples)}")

    if not all_examples:
        logger.error("No training examples generated! Aborting.")
        return

    # Save all training data
    synthesis_dir = output_dir / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    from looper.synthesizers.synthesizer import save_training_data

    save_training_data(all_examples, synthesis_dir / "all_xml_training.jsonl")
    save_training_data(success_examples, synthesis_dir / "success_xml_training.jsonl")

    # ── Step 2: Categorize by skill type ──────────────────────────────

    logger.info("\nStep 2: Categorizing examples by skill type...")

    logger.info("  All data categories:")
    all_categorized = categorize_examples(all_examples)

    logger.info("  Successful-only categories:")
    success_categorized = categorize_examples(success_examples)

    category_counts = {
        "all": {cat: len(exs) for cat, exs in all_categorized.items()},
        "success": {cat: len(exs) for cat, exs in success_categorized.items()},
    }

    # ── Step 3: Train all adapter configurations ──────────────────────

    logger.info("\nStep 3: Training adapter configurations...")
    adapters_dir = output_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    all_training_metrics: dict[str, dict] = {}

    # Config 1: single_all (rank 16, all examples)
    logger.info("\n  --- single_all (rank 16, all examples) ---")
    single_all_dir = adapters_dir / "single_all"
    all_training_metrics["single_all"] = train_adapter(
        all_examples, single_all_dir, "single_all", rank=16
    )

    # Config 2: single_success (rank 16, successful-only)
    logger.info("\n  --- single_success (rank 16, successful-only) ---")
    single_success_dir = adapters_dir / "single_success"
    all_training_metrics["single_success"] = train_adapter(
        success_examples, single_success_dir, "single_success", rank=16
    )

    # Config 3: mole_3_all (3 experts at rank 5, all examples)
    logger.info("\n  --- mole_3_all (3 experts at rank 5, all examples) ---")
    mole_3_all_experts_dir = adapters_dir / "mole_3_all_experts"
    expert_dirs_all = []
    for cat_name, cat_examples in all_categorized.items():
        expert_label = f"mole_3_all_{cat_name}"
        expert_dir = mole_3_all_experts_dir / cat_name
        metrics = train_adapter(cat_examples, expert_dir, expert_label, rank=5)
        all_training_metrics[expert_label] = metrics
        expert_dirs_all.append(expert_dir)

    # Merge all experts
    mole_3_all_dir = adapters_dir / "mole_3_all"
    merge_ok_all = merge_expert_adapters(expert_dirs_all, mole_3_all_dir)
    if not merge_ok_all:
        logger.warning("  mole_3_all merge failed!")
    all_training_metrics["mole_3_all"] = {
        "merged": True,
        "merge_ok": merge_ok_all,
        "num_experts": len(expert_dirs_all),
        "rank": 5,
        "num_examples": len(all_examples),
    }

    # Config 4: mole_3_success (3 experts at rank 5, successful-only)
    logger.info("\n  --- mole_3_success (3 experts at rank 5, successful-only) ---")
    mole_3_success_experts_dir = adapters_dir / "mole_3_success_experts"
    expert_dirs_success = []
    for cat_name, cat_examples in success_categorized.items():
        expert_label = f"mole_3_success_{cat_name}"
        expert_dir = mole_3_success_experts_dir / cat_name
        metrics = train_adapter(cat_examples, expert_dir, expert_label, rank=5)
        all_training_metrics[expert_label] = metrics
        expert_dirs_success.append(expert_dir)

    # Merge success experts
    mole_3_success_dir = adapters_dir / "mole_3_success"
    merge_ok_success = merge_expert_adapters(expert_dirs_success, mole_3_success_dir)
    if not merge_ok_success:
        logger.warning("  mole_3_success merge failed!")
    all_training_metrics["mole_3_success"] = {
        "merged": True,
        "merge_ok": merge_ok_success,
        "num_experts": len(expert_dirs_success),
        "rank": 5,
        "num_examples": len(success_examples),
    }

    # Log training comparison
    logger.info("\n  Training comparison:")
    logger.info(f"  {'Condition':25s} {'Rank':>5s} {'#Ex':>5s} {'Train Loss':>11s}")
    for label, m in sorted(all_training_metrics.items()):
        r = m.get("rank", "?")
        n = m.get("num_examples", "?")
        tl = m.get("final_train_loss", "?")
        if m.get("skipped"):
            tl = "SKIPPED"
        elif m.get("merged"):
            tl = "MERGED"
        elif isinstance(tl, float):
            tl = f"{tl:.4f}"
        logger.info(f"  {label:25s} {str(r):>5s} {str(n):>5s} {str(tl):>11s}")

    # Save training metrics
    (output_dir / "all_training_metrics.json").write_text(
        json.dumps(all_training_metrics, indent=2, default=str)
    )

    # ── Step 4: Sanity check adapters ─────────────────────────────────

    logger.info("\nStep 4: Sanity check — verifying adapters produce tool calls...")

    sanity_results = {}
    for label in ["single_all", "single_success", "mole_3_all", "mole_3_success"]:
        adapter_dir = adapters_dir / label
        if (adapter_dir / "adapters.safetensors").exists():
            sanity_ok = sanity_check_adapter(adapter_dir)
            sanity_results[label] = sanity_ok
            logger.info(f"  {label}: {'PASS' if sanity_ok else 'FAIL'}")
        else:
            sanity_results[label] = False
            logger.info(f"  {label}: NO ADAPTER")

    # ── Step 5: Evaluate base condition ───────────────────────────────

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

    # ── Step 6: Evaluate adapted conditions ───────────────────────────

    logger.info("\nStep 6: Evaluating adapted conditions...")

    # Stop Ollama before MLX
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(3)

    all_trajectories: dict[str, list[AgentTrajectory]] = {"base": base_trajs}
    all_results: dict[str, list[TaskResult]] = {"base": base_results}

    eval_conditions = ["single_all", "single_success", "mole_3_all", "mole_3_success"]
    # Only evaluate conditions with valid adapters
    eval_conditions = [
        c for c in eval_conditions
        if (adapters_dir / c / "adapters.safetensors").exists()
    ]
    logger.info(f"  Evaluation conditions: base + {eval_conditions}")

    for cond_label in eval_conditions:
        adapter_dir = adapters_dir / cond_label

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

    # ── Step 7: Compute metrics and save ──────────────────────────────

    logger.info("\nStep 7: Computing metrics and saving results...")

    comparison = compare_conditions(all_results)

    base_rr = resolve_rate(all_results["base"])
    ft = {}
    for cond in all_results:
        if cond != "base":
            ft[f"{cond}_vs_base"] = resolve_rate(all_results[cond]) - base_rr

    logger.info("\n" + "=" * 60)
    logger.info(f"EXPERIMENT 3 RESULTS ({mode})")
    logger.info("=" * 60)
    for cond, metrics in sorted(comparison.items()):
        logger.info(
            f"  {cond:20s}: resolve={metrics['resolve_rate']:.2%}, "
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
        name="experiment3_mole",
        experiment_id=f"exp3_mole_{mode.lower()}",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="mole_xml_trajectory",
        lora_rank=16,
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
        "experiment": "experiment3_mole",
        "mode": mode.lower(),
        "model": OLLAMA_MODEL,
        "hf_model": HF_MODEL,
        "training_data_format": "xml_tool_call_trajectories",
        "total_training_examples_all": len(all_examples),
        "total_training_examples_success": len(success_examples),
        "category_counts": category_counts,
        "training_metrics": all_training_metrics,
        "sanity_checks": sanity_results,
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
        category_counts,
        len(all_examples),
        len(success_examples),
        output_dir,
    )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("=" * 60)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: MoLE")
    parser.add_argument(
        "--full", action="store_true", help="Run full experiment (25 test tasks)"
    )
    args = parser.parse_args()

    run_experiment3(pilot=not args.full)
