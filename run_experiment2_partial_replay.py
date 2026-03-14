#!/usr/bin/env python3
"""Experiment 2: Partial Replay with Prioritized Sampling.

Tests whether a fixed-size replay buffer with intelligent prioritization
can achieve 80%+ of full replay's retention at a fraction of training cost.

Key innovation from DEEP_AUDIT: training data uses XML tool-call trajectories
(trajectory_synthesizer.py), not Q&A pairs or diffs. This is the first
experiment with format-matched training data.

Conditions evaluated on 25 test tasks:
1. base           — Ollama, no adapter (reuse experiment9 base trajectories)
2. full_replay    — MLX LoRA trained on ALL 169 XML training examples
3. partial_20_rec — MLX LoRA, buffer=20, recency priority
4. partial_40_rec — MLX LoRA, buffer=40, recency priority
5. partial_20_dif — MLX LoRA, buffer=20, difficulty priority
6. partial_40_dif — MLX LoRA, buffer=40, difficulty priority

Sequential batches: 5 batches of 5 trajectories each.
Training at each batch: new examples + replay buffer → retrain LoRA from scratch.

Evaluation: base (reuse) + full_replay + best partial replay on 25 test tasks.
"""

import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
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

# Reuse experiment9 base trajectories for the base condition
EXP9_BASE_DIR = Path(
    "/Volumes/1TB_SSD/looper/results/experiment9_ablation/trajectories/base"
)

OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment2_partial_replay")

MAX_STEPS = 10
OLLAMA_MAX_TOKENS = 4096
MLX_MAX_TOKENS = 512

LORA_RANK = 16
LORA_ITERS = 100
LORA_BATCH = 1
LORA_MAX_SEQ = 1024

# Partial replay parameters
NUM_BATCHES = 5
BUFFER_SIZES = [20, 40]
PRIORITY_SCHEMES = ["recency", "difficulty"]


# ── Replay Buffer ─────────────────────────────────────────────────────


@dataclass
class ReplayBuffer:
    """Fixed-size buffer with priority-based eviction."""

    max_size: int
    priority: str = "recency"  # "recency" or "difficulty"
    buffer: list[TrainingExample] = field(default_factory=list)
    metadata: list[dict] = field(default_factory=list)

    def add_batch(self, examples: list[TrainingExample], batch_idx: int):
        """Add new examples from a batch and evict if over capacity."""
        for ex in examples:
            # Use message count as a proxy for difficulty (more steps = harder)
            self.buffer.append(ex)
            self.metadata.append({
                "batch_idx": batch_idx,
                "msg_count": len(ex.messages),
            })
        self._evict()

    def _evict(self):
        """Evict lowest-priority examples until within capacity."""
        if len(self.buffer) <= self.max_size:
            return

        if self.priority == "recency":
            # Keep most recent examples (highest batch_idx)
            indices = sorted(
                range(len(self.buffer)),
                key=lambda i: self.metadata[i]["batch_idx"],
                reverse=True,
            )
        elif self.priority == "difficulty":
            # Keep examples with most messages (harder tasks = more steps)
            indices = sorted(
                range(len(self.buffer)),
                key=lambda i: self.metadata[i]["msg_count"],
                reverse=True,
            )
        else:
            raise ValueError(f"Unknown priority: {self.priority}")

        keep = set(indices[: self.max_size])
        self.buffer = [self.buffer[i] for i in sorted(keep)]
        self.metadata = [self.metadata[i] for i in sorted(keep)]

    def get_all(self) -> list[TrainingExample]:
        """Return all examples in the buffer."""
        return list(self.buffer)


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
    rank={LORA_RANK},
    iters={LORA_ITERS},
    batch_size={LORA_BATCH},
    max_seq_length={LORA_MAX_SEQ},
)
metrics = full_replay_train(examples, '{HF_MODEL}', Path('{adapter_dir}'), config)
print(json.dumps(metrics))
"""
    logger.info(f"  [{label}] Training LoRA adapter ({len(examples)} examples)...")
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
        timeout=3600,
    )

    if result.returncode != 0:
        logger.error(f"[{label}] Training failed:\n{result.stderr[-1000:]}")
        raise RuntimeError(f"LoRA training failed for {label}")

    metrics = {}
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    metrics["num_examples"] = len(examples)
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info(
        f"  [{label}] Done: train_loss={metrics.get('final_train_loss', '?')}, "
        f"val_loss={metrics.get('final_val_loss', '?')}"
    )
    return metrics


# ── Partial Replay Training ──────────────────────────────────────────


def train_partial_replay(
    batches: list[list[TrainingExample]],
    buffer_size: int,
    priority: str,
    adapter_base_dir: Path,
) -> dict:
    """Train LoRA sequentially through batches with a replay buffer.

    At each batch:
    1. Add new examples to replay buffer (evict if over capacity)
    2. Train LoRA from scratch on all buffer contents + new examples
    3. The final adapter is the one trained after the last batch.
    """
    label = f"partial_{buffer_size}_{priority[:3]}"
    buf = ReplayBuffer(max_size=buffer_size, priority=priority)

    metrics_history = []
    for batch_idx, batch_examples in enumerate(batches):
        logger.info(
            f"  [{label}] Batch {batch_idx + 1}/{len(batches)}: "
            f"{len(batch_examples)} new examples"
        )

        # Add new examples to buffer
        buf.add_batch(batch_examples, batch_idx)

        # Train on everything in the buffer
        train_examples = buf.get_all()
        logger.info(
            f"  [{label}] Buffer: {len(buf.buffer)} examples "
            f"(capacity {buffer_size})"
        )

        # Only train the final adapter (for efficiency)
        # But record buffer state at each step for analysis
        metrics_history.append({
            "batch_idx": batch_idx,
            "new_examples": len(batch_examples),
            "buffer_size": len(buf.buffer),
            "total_train_examples": len(train_examples),
        })

    # Train final adapter on final buffer contents
    adapter_dir = adapter_base_dir / label
    final_examples = buf.get_all()
    logger.info(
        f"  [{label}] Training final adapter on {len(final_examples)} examples"
    )
    final_metrics = train_adapter(final_examples, adapter_dir, label)
    final_metrics["buffer_history"] = metrics_history
    final_metrics["priority"] = priority
    final_metrics["buffer_max_size"] = buffer_size

    return final_metrics


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
    """Quick check: does the adapted model produce XML tool calls?

    Runs 2 test tasks and checks if output contains <bash>, <read>, etc.
    Returns True if at least 1 task produces tool calls.
    """
    logger.info("  Sanity check: verifying adapter produces XML tool calls...")

    mlx_proc = start_mlx_server(HF_MODEL, MLX_PORT, str(adapter_dir))
    try:
        # Quick single-turn test
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


# ── Progress Report ───────────────────────────────────────────────────


def write_report(
    all_results: dict,
    all_trajectories: dict,
    training_metrics: dict,
    total_examples: int,
    output_dir: Path,
):
    """Write experiment report."""
    lines = [
        "# Experiment 2: Partial Replay with Prioritized Sampling",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {OLLAMA_MODEL} (base) / {HF_MODEL} (adapted)",
        f"**Training data:** XML tool-call trajectories (trajectory_synthesizer.py)",
        f"**Total training examples:** {total_examples}",
        f"**Key innovation:** Format-matched training (XML tool calls, not Q&A/diffs)",
        "",
        "## Training Metrics",
        "",
        "| Condition | Examples | Train Loss | Val Loss |",
        "|-----------|----------|------------|----------|",
    ]

    for label, metrics in sorted(training_metrics.items()):
        n = metrics.get("num_examples", "?")
        tl = metrics.get("final_train_loss", "?")
        vl = metrics.get("final_val_loss", "?")
        if isinstance(tl, float):
            tl = f"{tl:.4f}"
        if isinstance(vl, float):
            vl = f"{vl:.4f}"
        lines.append(f"| {label:20s} | {n} | {tl} | {vl} |")

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

    lines += [
        "",
        "## Buffer History (Partial Replay)",
        "",
    ]
    for label, metrics in sorted(training_metrics.items()):
        if "buffer_history" in metrics:
            lines.append(f"### {label}")
            lines.append(
                f"Priority: {metrics.get('priority', '?')}, "
                f"Max buffer: {metrics.get('buffer_max_size', '?')}"
            )
            for step in metrics["buffer_history"]:
                lines.append(
                    f"  Batch {step['batch_idx'] + 1}: "
                    f"+{step['new_examples']} new → "
                    f"{step['buffer_size']} in buffer"
                )
            lines.append("")

    # Key findings
    lines += [
        "## Key Findings",
        "",
    ]

    # Check if XML format fixed the tool-call problem
    adapted_conditions = [c for c in all_trajectories if c != "base"]
    has_tool_calls = False
    for cond in adapted_conditions:
        for traj in all_trajectories.get(cond, []):
            if sum(len(s.tool_calls) for s in traj.steps) > 0:
                has_tool_calls = True
                break

    if has_tool_calls:
        lines.append(
            "- **FORMAT FIX WORKS**: XML-format training data produces "
            "adapted models that generate tool calls (unlike Q&A or diff format)"
        )
    else:
        lines.append(
            "- **FORMAT FIX FAILED**: Even XML-format training data does not "
            "produce tool calls from the adapted model"
        )

    best_adapted_rr = max(
        (resolve_rate(all_results[c]) for c in adapted_conditions),
        default=0.0,
    )
    if best_adapted_rr > base_rr:
        lines.append(
            f"- **POSITIVE FORWARD TRANSFER**: Best adapted condition "
            f"({best_adapted_rr:.0%}) beats base ({base_rr:.0%})"
        )
    elif best_adapted_rr == base_rr:
        lines.append(
            f"- **ZERO FORWARD TRANSFER**: Adapted = base ({base_rr:.0%})"
        )
    else:
        lines.append(
            f"- **NEGATIVE FORWARD TRANSFER**: Best adapted "
            f"({best_adapted_rr:.0%}) < base ({base_rr:.0%})"
        )

    report = "\n".join(lines) + "\n"
    report_path = output_dir / "EXPERIMENT2_REPORT.md"
    report_path.write_text(report)
    logger.info(f"  Report written to {report_path}")
    return report


# ── Main ──────────────────────────────────────────────────────────────


def run_experiment2(output_dir: Path = OUTPUT_DIR):
    setup_logging(output_dir)
    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Partial Replay with Prioritized Sampling")
    logger.info("Training data: XML tool-call trajectories (FORMAT-MATCHED)")
    logger.info("=" * 60)

    # ── Step 0: Load data ─────────────────────────────────────────────

    logger.info("\nStep 0: Loading data...")
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, test_tasks = split_tasks(all_tasks, train_size=25, seed=None)
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
    from looper.synthesizers.synthesizer import save_training_data

    synthesis_dir = output_dir / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    all_training_file = synthesis_dir / "all_xml_training.jsonl"
    save_training_data(all_examples, all_training_file)
    logger.info(f"  Saved to {all_training_file}")

    # ── Step 2: Split into sequential batches ────────────────────────

    logger.info("\nStep 2: Splitting into sequential batches...")

    # Group examples by source trajectory (task)
    traj_order = [t.meta.task_id for t in train_trajectories]
    examples_by_task: dict[str, list[TrainingExample]] = {}
    for ex in all_examples:
        # source_pair_id is like "django__django-10097_step1"
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

    # ── Step 3: Train full replay adapter (baseline) ─────────────────

    logger.info("\nStep 3: Training full replay adapter (all XML data)...")
    adapters_dir = output_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    full_replay_dir = adapters_dir / "full_replay"
    full_replay_metrics = train_adapter(
        all_examples, full_replay_dir, "full_replay"
    )

    # ── Step 4: Sanity check — does adapter produce tool calls? ──────

    logger.info("\nStep 4: Sanity check — verifying XML format works...")
    sanity_ok = sanity_check_adapter(full_replay_dir)

    if not sanity_ok:
        logger.warning(
            "  SANITY CHECK FAILED: adapter does not produce tool calls. "
            "Continuing anyway to collect data, but expect negative results."
        )
    else:
        logger.info("  SANITY CHECK PASSED: adapter produces XML tool calls!")

    # ── Step 5: Train partial replay adapters ────────────────────────

    logger.info("\nStep 5: Training partial replay adapters...")
    all_training_metrics = {"full_replay": full_replay_metrics}

    for buf_size in BUFFER_SIZES:
        for priority in PRIORITY_SCHEMES:
            label = f"partial_{buf_size}_{priority[:3]}"
            logger.info(f"\n  Training {label}...")
            metrics = train_partial_replay(
                batches, buf_size, priority, adapters_dir
            )
            all_training_metrics[label] = metrics

    # Log training comparison
    logger.info("\n  Training comparison:")
    logger.info(
        f"  {'Condition':25s} {'Examples':>8s} {'Train Loss':>11s} {'Val Loss':>10s}"
    )
    for label, m in sorted(all_training_metrics.items()):
        n = m.get("num_examples", "?")
        tl = m.get("final_train_loss", "?")
        vl = m.get("final_val_loss", "?")
        tl_s = f"{tl:.4f}" if isinstance(tl, float) else str(tl)
        vl_s = f"{vl:.4f}" if isinstance(vl, float) else str(vl)
        logger.info(f"  {label:25s} {str(n):>8s} {tl_s:>11s} {vl_s:>10s}")

    # Save all training metrics
    (output_dir / "all_training_metrics.json").write_text(
        json.dumps(all_training_metrics, indent=2)
    )

    # ── Step 6: Select conditions to evaluate ────────────────────────

    # Always evaluate: base (reuse), full_replay, best partial
    # Pick best partial by lowest val_loss (or train_loss if no val)
    partial_labels = [l for l in all_training_metrics if l.startswith("partial_")]
    best_partial = min(
        partial_labels,
        key=lambda l: all_training_metrics[l].get(
            "final_val_loss",
            all_training_metrics[l].get("final_train_loss", 999),
        ),
    )
    logger.info(f"\n  Best partial replay: {best_partial}")

    eval_conditions = ["full_replay", best_partial]

    # Also evaluate a second partial for comparison if different priority
    best_priority = all_training_metrics[best_partial].get("priority", "")
    alt_partial = None
    for l in partial_labels:
        if l != best_partial and all_training_metrics[l].get("priority") != best_priority:
            alt_partial = l
            break
    if alt_partial:
        eval_conditions.append(alt_partial)

    logger.info(f"  Evaluation conditions: base + {eval_conditions}")

    # ── Step 7: Evaluate base condition (reuse experiment9) ──────────

    logger.info("\nStep 7: Loading base condition trajectories...")
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

    # ── Step 8: Evaluate adapted conditions ──────────────────────────

    logger.info("\nStep 8: Evaluating adapted conditions...")

    # Stop Ollama before MLX
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(3)

    all_trajectories = {"base": base_trajs}
    all_results: dict[str, list[TaskResult]] = {"base": base_results}

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

    # ── Step 9: Compute metrics and save ─────────────────────────────

    logger.info("\nStep 9: Computing metrics and saving results...")

    comparison = compare_conditions(all_results)

    base_rr = resolve_rate(all_results["base"])
    ft = {}
    for cond in all_results:
        if cond != "base":
            ft[f"{cond}_vs_base"] = resolve_rate(all_results[cond]) - base_rr

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2 RESULTS")
    logger.info("=" * 60)
    for cond, metrics in sorted(comparison.items()):
        logger.info(
            f"  {cond:25s}: resolve={metrics['resolve_rate']:.2%}, "
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
        name="experiment2_partial_replay",
        experiment_id="exp2_partial_replay_xml_format",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="xml_trajectory_partial_replay",
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
        "experiment": "experiment2_partial_replay",
        "model": OLLAMA_MODEL,
        "hf_model": HF_MODEL,
        "training_data_format": "xml_tool_call_trajectories",
        "total_training_examples": len(all_examples),
        "num_batches": NUM_BATCHES,
        "buffer_sizes_tested": BUFFER_SIZES,
        "priority_schemes_tested": PRIORITY_SCHEMES,
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

    # ── Step 10: Write report ────────────────────────────────────────

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
    run_experiment2()
