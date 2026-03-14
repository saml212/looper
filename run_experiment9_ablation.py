#!/usr/bin/env python3
"""Experiment 9: Skill Layer + Knowledge Layer Ablation.

Tests whether a hybrid system (LoRA skill layer + RAG knowledge layer)
outperforms either in isolation. Applies Exp6+7 findings:
D_reflexion synthesis format, 3 pairs per trajectory.

4 conditions on 25 test tasks:
1. base       — no adapter, no RAG
2. knowledge  — base model + RAG over past trajectories
3. skill      — LoRA adapter, no RAG
4. both       — LoRA adapter + RAG

Uses Phase 1 train trajectories as source for both layers.

Note: Ollama conditions (1, 2) use max_tokens=4096; MLX conditions (3, 4)
use max_tokens=512 due to Apple Silicon memory constraints. This disadvantages
conditions 3 & 4, making any improvement more impressive.
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
from looper.evaluators.results_io import save_results, results_summary
from looper.models import (
    AgentTrajectory,
    ExperimentConfig,
    ExperimentResult,
    SynthesizedPair,
    TaskInfo,
    TaskResult,
    TrainingExample,
)
from looper.synthesizers.synthesizer import (
    _extract_json_array,
    pairs_to_training_examples,
    save_training_data,
    load_training_data,
)
from looper.synthesizers.trajectory_to_text import trajectory_to_text
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

OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment9_ablation")

# Consistent across all conditions
MAX_STEPS = 10
# Ollama can handle larger context; MLX 7B 4-bit OOMs at 4096 over 10 steps
OLLAMA_MAX_TOKENS = 4096
MLX_MAX_TOKENS = 512

LORA_RANK = 16
LORA_ITERS = 100
LORA_BATCH = 1
LORA_MAX_SEQ = 1024

# Exp6+7 optimal findings
SYNTHESIS_BUDGET = 3  # pairs per trajectory
RAG_TOP_K = 3  # similar trajectories per test task
RAG_MAX_CHARS = 1500  # cap RAG context length

D_REFLEXION_PROMPT = """You are analyzing an AI agent's work session to extract skills through reflexion.

Given the following trajectory, identify {num_pairs} situations where the agent could have done something differently. For each, create a pair showing what went wrong and what the correct approach should be.

Format: "I tried X and it [worked/failed] because [reason]. The [better/correct] approach is Y because [explanation]."

Output your pairs as a JSON array:
[
  {{
    "instruction": "I tried [wrong approach] and ...",
    "response": "That approach [fails/is suboptimal] in this project because ... The correct approach is ... because ...",
    "pair_type": "error_recovery",
    "confidence": 0.8
  }},
  ...
]

IMPORTANT: Output exactly {num_pairs} pairs. Only output the JSON array, nothing else.

Trajectory:
{trajectory_text}
"""


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


# ── RAG Context Builder ───────────────────────────────────────────────


def build_rag_contexts(
    train_trajectories: list[AgentTrajectory],
    test_tasks: list[TaskInfo],
    train_tasks: list[TaskInfo],
    top_k: int = RAG_TOP_K,
) -> dict[str, str]:
    """Build RAG context for each test task from similar train trajectories.

    For each test task, finds the top-k most similar train tasks by word
    overlap in problem statements, then builds a context string summarizing
    the agent's approach and outcome on those past tasks.
    """
    train_task_map = {t.instance_id: t for t in train_tasks}
    traj_map = {t.meta.task_id: t for t in train_trajectories}

    rag_contexts = {}
    for test_task in test_tasks:
        test_words = set(test_task.problem_statement.lower().split())

        scored = []
        for train_task in train_tasks:
            train_words = set(train_task.problem_statement.lower().split())
            overlap = len(test_words & train_words) / max(
                len(test_words | train_words), 1
            )
            scored.append((overlap, train_task.instance_id))

        scored.sort(reverse=True)
        top_ids = [tid for _, tid in scored[:top_k]]

        parts = [
            "## Relevant past experience fixing bugs in this Django codebase:\n"
        ]
        total_chars = 0
        for tid in top_ids:
            traj = traj_map.get(tid)
            task = train_task_map.get(tid)
            if not traj or not task:
                continue

            # Extract files from patch
            patch_files = []
            if traj.generated_patch:
                for line in traj.generated_patch.split("\n"):
                    if line.startswith("diff --git"):
                        line_parts = line.split()
                        if len(line_parts) >= 4:
                            patch_files.append(line_parts[3].lstrip("b/"))

            summary = f"### Past fix: {tid}\n"
            summary += f"Problem: {task.problem_statement[:200]}\n"
            summary += f"Outcome: {traj.outcome} ({len(traj.steps)} steps)\n"
            if patch_files:
                summary += f"Files: {', '.join(patch_files[:5])}\n"
            if traj.generated_patch:
                patch_excerpt = traj.generated_patch[:300]
                summary += f"Patch:\n```\n{patch_excerpt}\n```\n"

            if total_chars + len(summary) > RAG_MAX_CHARS:
                break
            parts.append(summary)
            total_chars += len(summary)

        rag_contexts[test_task.instance_id] = "\n".join(parts)

    return rag_contexts


# ── Synthesis ──────────────────────────────────────────────────────────


def synthesize_d_reflexion(
    train_trajectories: list[AgentTrajectory],
    output_dir: Path,
) -> list[TrainingExample]:
    """Synthesize D_reflexion training data with budget=3 pairs/trajectory."""
    pairs_file = output_dir / "pairs.json"
    training_file = output_dir / "training.jsonl"

    if training_file.exists():
        examples = load_training_data(training_file)
        logger.info(f"  Loaded cached synthesis: {len(examples)} examples")
        return examples

    all_pairs: list[SynthesizedPair] = []
    for i, traj in enumerate(train_trajectories):
        traj_text = trajectory_to_text(traj)
        prompt = D_REFLEXION_PROMPT.format(
            trajectory_text=traj_text, num_pairs=SYNTHESIS_BUDGET
        )
        messages = [ChatMessage(role="user", content=prompt)]

        try:
            response = chat(messages, model=OLLAMA_MODEL, base_url=OLLAMA_URL)
            raw_pairs = _extract_json_array(response.content)
            if raw_pairs is None:
                logger.warning(f"  Invalid JSON for {traj.meta.task_id}")
                continue

            count = 0
            for raw in raw_pairs:
                try:
                    pair = SynthesizedPair(
                        instruction=raw["instruction"],
                        response=raw["response"],
                        pair_type=raw.get("pair_type", "error_recovery"),
                        confidence=float(raw.get("confidence", 0.5)),
                        source_session_id=traj.meta.session_id,
                        source_task_id=traj.meta.task_id,
                    )
                    if pair.confidence >= 0.3:
                        all_pairs.append(pair)
                        count += 1
                except (KeyError, ValueError, TypeError):
                    continue

            logger.info(
                f"  [{i + 1}/{len(train_trajectories)}] {traj.meta.task_id}: "
                f"{count} pairs"
            )
        except Exception as e:
            logger.warning(
                f"  [{i + 1}/{len(train_trajectories)}] {traj.meta.task_id}: "
                f"error: {e}"
            )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_data = [p.model_dump() for p in all_pairs]
    pairs_file.write_text(json.dumps(pairs_data, indent=2))

    examples = pairs_to_training_examples(all_pairs)
    save_training_data(examples, training_file)

    logger.info(f"  Synthesized {len(all_pairs)} pairs -> {len(examples)} examples")
    return examples


# ── Training ───────────────────────────────────────────────────────────


def train_adapter(training_file: Path, adapter_dir: Path) -> dict:
    """Train LoRA adapter in a subprocess to free GPU memory afterwards."""
    adapter_file = adapter_dir / "adapters.safetensors"
    metrics_file = adapter_dir.parent / "training_metrics.json"

    if adapter_file.exists():
        logger.info("  Adapter already trained, skipping...")
        if metrics_file.exists():
            return json.loads(metrics_file.read_text())
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

    metrics = {}
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    metrics_file.write_text(json.dumps(metrics, indent=2))
    return metrics


# ── Evaluation ─────────────────────────────────────────────────────────


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


def patch_rate(trajectories: list[AgentTrajectory]) -> float:
    """Fraction of trajectories that generated a non-empty patch."""
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t.generated_patch.strip()) / len(
        trajectories
    )


# ── Progress Report ────────────────────────────────────────────────────


def write_progress_report(
    comparison: dict,
    ft: dict,
    all_results: dict,
    all_trajectories: dict,
    train_metrics: dict,
    num_training_examples: int,
):
    """Write progress report to /tmp/looper_exp9_progress.md."""
    # Compute patch rates
    patch_rates = {
        cond: patch_rate(trajs) for cond, trajs in all_trajectories.items()
    }

    # Find winners
    best_resolve_cond = max(comparison, key=lambda c: comparison[c]["resolve_rate"])
    best_patch_cond = max(patch_rates, key=lambda c: patch_rates[c])

    lines = [
        "# Experiment 9: Skill + Knowledge Layer Ablation",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {OLLAMA_MODEL} (base) / {HF_MODEL} (adapted)",
        f"**Synthesis:** D_reflexion format, {SYNTHESIS_BUDGET} pairs/trajectory",
        f"**Training examples:** {num_training_examples}",
        f"**Training loss:** train={train_metrics.get('final_train_loss', 'N/A')}, "
        f"val={train_metrics.get('final_val_loss', 'N/A')}",
        "",
        "## Results",
        "",
        "| Condition | Resolve Rate | Patch Rate | Avg Steps | Avg Tokens |",
        "|-----------|-------------|------------|-----------|------------|",
    ]

    for cond in ["base", "knowledge", "skill", "both"]:
        m = comparison[cond]
        pr = patch_rates[cond]
        resolved_count = sum(1 for r in all_results[cond] if r.resolved)
        total = len(all_results[cond])
        patched_count = sum(
            1 for t in all_trajectories[cond] if t.generated_patch.strip()
        )
        lines.append(
            f"| {cond:12s} | {m['resolve_rate']:.0%} ({resolved_count}/{total}) "
            f"| {pr:.0%} ({patched_count}/{total}) "
            f"| {m['avg_steps']:.1f} | {m['avg_tokens']:.0f} |"
        )

    lines += [
        "",
        "## Forward Transfer (vs base)",
        "",
        f"- Knowledge: {ft['knowledge_vs_base']:+.4f}",
        f"- Skill:     {ft['skill_vs_base']:+.4f}",
        f"- Both:      {ft['both_vs_base']:+.4f}",
        "",
        "## Winners",
        "",
        f"- **Best patch rate:** {best_patch_cond} ({patch_rates[best_patch_cond]:.0%})",
        f"- **Best resolve rate:** {best_resolve_cond} "
        f"({comparison[best_resolve_cond]['resolve_rate']:.0%})",
        "",
        "## Key Takeaway",
        "",
    ]

    # Generate takeaway based on results
    base_rr = comparison["base"]["resolve_rate"]
    knowledge_rr = comparison["knowledge"]["resolve_rate"]
    skill_rr = comparison["skill"]["resolve_rate"]
    both_rr = comparison["both"]["resolve_rate"]

    if both_rr > base_rr and both_rr >= max(knowledge_rr, skill_rr):
        lines.append(
            "The combined skill+knowledge system outperforms both individual "
            "layers and the base model, supporting the hypothesis that LoRA "
            "skills and RAG knowledge are complementary."
        )
    elif knowledge_rr >= skill_rr and knowledge_rr > base_rr:
        lines.append(
            "The knowledge layer (RAG) outperforms the skill layer (LoRA). "
            "This suggests that for this model size and task difficulty, "
            "retrieved context provides more value than learned behavioral "
            "patterns. The skill layer may need a higher base resolve rate "
            "to learn useful signal."
        )
    elif skill_rr > knowledge_rr and skill_rr > base_rr:
        lines.append(
            "The skill layer (LoRA) outperforms the knowledge layer (RAG). "
            "Learned behavioral patterns provide more value than retrieved "
            "trajectory context for this task set."
        )
    elif base_rr >= max(knowledge_rr, skill_rr, both_rr) or all(
        rr == base_rr for rr in [knowledge_rr, skill_rr, both_rr]
    ):
        lines.append(
            "Neither layer improves over the base model. At 7B scale with "
            f"{base_rr:.0%} base resolve rate, there is insufficient signal "
            "in the training trajectories for either mechanism to help. "
            "The 8% base rate means most training data comes from failed "
            "attempts, limiting what both LoRA and RAG can learn."
        )
    else:
        lines.append(
            f"Mixed results: base={base_rr:.0%}, knowledge={knowledge_rr:.0%}, "
            f"skill={skill_rr:.0%}, both={both_rr:.0%}. "
            "No clear winner emerges from this configuration."
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Ollama conditions (base, knowledge) use max_tokens=4096",
        "- MLX conditions (skill, both) use max_tokens=512 (memory constraint)",
        "- All conditions use max_steps=10",
        f"- RAG: top-{RAG_TOP_K} similar trajectories by word overlap",
        f"- LoRA: rank={LORA_RANK}, iters={LORA_ITERS}",
    ]

    report = "\n".join(lines) + "\n"
    Path("/tmp/looper_exp9_progress.md").write_text(report)
    logger.info(f"  Progress report written to /tmp/looper_exp9_progress.md")
    return report


# ── Main ───────────────────────────────────────────────────────────────


def run_experiment9(output_dir: Path = OUTPUT_DIR):
    setup_logging(output_dir)
    started_at = datetime.now(timezone.utc).isoformat()

    synthesis_dir = output_dir / "synthesis"
    adapter_dir = output_dir / "adapter"
    for d in [synthesis_dir, adapter_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXPERIMENT 9: Skill + Knowledge Layer Ablation")
    logger.info("D_reflexion format, 3 pairs/trajectory (Exp6+7 findings)")
    logger.info("=" * 60)

    # ── Step 0: Load data ──────────────────────────────────────────────

    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, test_tasks = split_tasks(all_tasks, train_size=25, seed=None)

    logger.info(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Load Phase 1 train trajectories
    all_phase1 = load_all_trajectories(PHASE1_TRAJ_DIR)
    train_ids = {t.instance_id for t in train_tasks}
    train_trajectories = [t for t in all_phase1 if t.meta.task_id in train_ids]
    logger.info(f"Train trajectories loaded: {len(train_trajectories)}")

    # ── Step 1: Synthesize D_reflexion training data ──────────────────

    logger.info("\nStep 1: Synthesizing D_reflexion training data (budget=3)...")
    training_examples = synthesize_d_reflexion(train_trajectories, synthesis_dir)
    if not training_examples:
        logger.error("No training examples generated! Aborting.")
        return
    logger.info(f"  Training examples: {len(training_examples)}")

    # ── Step 2: Build RAG contexts ────────────────────────────────────

    logger.info("\nStep 2: Building RAG contexts for test tasks...")
    rag_contexts = build_rag_contexts(
        train_trajectories, test_tasks, train_tasks
    )
    logger.info(f"  RAG contexts built for {len(rag_contexts)} test tasks")
    sample_id = list(rag_contexts.keys())[0]
    logger.info(f"  Sample ({sample_id}): {len(rag_contexts[sample_id])} chars")

    # ── Step 3: Run conditions 1 & 2 (Ollama) ────────────────────────

    logger.info("\nStep 3a: Running Condition 1 -- Base model...")
    cond1_dir = output_dir / "trajectories" / "base"
    cond1_trajectories = collect_trajectories(
        tasks=test_tasks,
        output_dir=cond1_dir,
        workspace_root=WORKSPACE_ROOT,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        max_steps=MAX_STEPS,
        max_tokens=OLLAMA_MAX_TOKENS,
        on_complete=lambda tid, traj: logger.info(
            f"  [base] {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, "
            f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
        ),
    )

    logger.info("\nStep 3b: Running Condition 2 -- Knowledge only (base + RAG)...")
    cond2_dir = output_dir / "trajectories" / "knowledge"
    cond2_trajectories = collect_trajectories(
        tasks=test_tasks,
        output_dir=cond2_dir,
        workspace_root=WORKSPACE_ROOT,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        max_steps=MAX_STEPS,
        max_tokens=OLLAMA_MAX_TOKENS,
        rag_contexts=rag_contexts,
        on_complete=lambda tid, traj: logger.info(
            f"  [knowledge] {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, "
            f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
        ),
    )

    # ── Step 4: Train LoRA adapter ────────────────────────────────────

    logger.info("\nStep 4: Training LoRA adapter...")
    training_file = synthesis_dir / "training.jsonl"
    train_metrics = train_adapter(training_file, adapter_dir)
    logger.info(f"  Training metrics: {train_metrics}")

    # ── Step 5: Run conditions 3 & 4 (MLX with adapter) ──────────────

    # Make sure Ollama is stopped (resource contention)
    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)
    time.sleep(3)

    mlx_proc = start_mlx_server(HF_MODEL, MLX_PORT, str(adapter_dir))

    try:
        def adapted_chat_fn(messages, model="", base_url="", **kwargs):
            return openai_chat(
                messages,
                model=HF_MODEL,
                base_url=f"http://127.0.0.1:{MLX_PORT}",
                max_tokens=MLX_MAX_TOKENS,
            )

        logger.info("\nStep 5a: Running Condition 3 -- Skill only (LoRA adapter)...")
        cond3_dir = output_dir / "trajectories" / "skill"
        cond3_trajectories = collect_trajectories(
            tasks=test_tasks,
            output_dir=cond3_dir,
            workspace_root=WORKSPACE_ROOT,
            model=HF_MODEL,
            base_url=f"http://127.0.0.1:{MLX_PORT}",
            max_steps=MAX_STEPS,
            max_tokens=MLX_MAX_TOKENS,
            chat_fn=adapted_chat_fn,
            on_complete=lambda tid, traj: logger.info(
                f"  [skill] {tid} -> {traj.outcome} "
                f"({len(traj.steps)} steps, "
                f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
            ),
        )

        logger.info("\nStep 5b: Running Condition 4 -- Both layers (LoRA + RAG)...")
        cond4_dir = output_dir / "trajectories" / "both"
        cond4_trajectories = collect_trajectories(
            tasks=test_tasks,
            output_dir=cond4_dir,
            workspace_root=WORKSPACE_ROOT,
            model=HF_MODEL,
            base_url=f"http://127.0.0.1:{MLX_PORT}",
            max_steps=MAX_STEPS,
            max_tokens=MLX_MAX_TOKENS,
            chat_fn=adapted_chat_fn,
            rag_contexts=rag_contexts,
            on_complete=lambda tid, traj: logger.info(
                f"  [both] {tid} -> {traj.outcome} "
                f"({len(traj.steps)} steps, "
                f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
            ),
        )
    finally:
        stop_mlx_server(mlx_proc)

    # Restart Ollama for future use
    logger.info("  Restarting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        env={**os.environ, "OLLAMA_MODELS": "/Volumes/1TB_SSD/looper/ollama_models"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # ── Step 6: Evaluate all conditions ──────────────────────────────

    logger.info("\nStep 6: Evaluating all conditions...")

    all_trajectories_map = {
        "base": cond1_trajectories,
        "knowledge": cond2_trajectories,
        "skill": cond3_trajectories,
        "both": cond4_trajectories,
    }

    all_results: dict[str, list[TaskResult]] = {}
    for condition, trajectories in all_trajectories_map.items():
        logger.info(f"\n  Evaluating {condition}...")
        results = evaluate_trajectories(
            trajectories, test_tasks, WORKSPACE_ROOT, condition
        )
        all_results[condition] = results
        resolved = sum(1 for r in results if r.resolved)
        patched = sum(1 for t in trajectories if t.generated_patch.strip())
        logger.info(
            f"  {condition}: {resolved}/{len(results)} resolved, "
            f"{patched}/{len(trajectories)} patched"
        )

    # ── Step 7: Compute metrics and save ─────────────────────────────

    comparison = compare_conditions(all_results)

    ft = {
        "knowledge_vs_base": resolve_rate(all_results["knowledge"])
        - resolve_rate(all_results["base"]),
        "skill_vs_base": resolve_rate(all_results["skill"])
        - resolve_rate(all_results["base"]),
        "both_vs_base": resolve_rate(all_results["both"])
        - resolve_rate(all_results["base"]),
    }

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 9 RESULTS")
    logger.info("=" * 60)
    for cond, metrics in comparison.items():
        logger.info(
            f"  {cond:12s}: resolve={metrics['resolve_rate']:.2%}, "
            f"steps={metrics['avg_steps']:.1f}, "
            f"tokens={metrics['avg_tokens']:.0f}"
        )
    logger.info(f"\n  FT (knowledge vs base): {ft['knowledge_vs_base']:+.4f}")
    logger.info(f"  FT (skill vs base):     {ft['skill_vs_base']:+.4f}")
    logger.info(f"  FT (both vs base):      {ft['both_vs_base']:+.4f}")

    # Save experiment result (ExperimentResult format)
    all_task_results = []
    for results in all_results.values():
        all_task_results.extend(results)

    config = ExperimentConfig(
        name="experiment9_ablation",
        experiment_id="exp9_skill_knowledge_ablation",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="d_reflexion_budget3",
        lora_rank=LORA_RANK,
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=all_task_results,
        forward_transfer=ft["both_vs_base"],
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, output_dir / "experiment_result.json")

    # Save detailed comparison
    detailed = {
        "experiment": "experiment9_skill_knowledge_ablation",
        "model": OLLAMA_MODEL,
        "hf_model": HF_MODEL,
        "max_steps": MAX_STEPS,
        "ollama_max_tokens": OLLAMA_MAX_TOKENS,
        "mlx_max_tokens": MLX_MAX_TOKENS,
        "synthesis": {
            "format": "D_reflexion",
            "budget": SYNTHESIS_BUDGET,
            "num_examples": len(training_examples),
        },
        "training_metrics": train_metrics,
        "rag": {"top_k": RAG_TOP_K, "method": "word_overlap"},
        "conditions": comparison,
        "forward_transfer": ft,
        "patch_rates": {
            cond: patch_rate(trajs)
            for cond, trajs in all_trajectories_map.items()
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

    # ── Step 8: Write progress report ────────────────────────────────

    write_progress_report(
        comparison, ft, all_results, all_trajectories_map,
        train_metrics, len(training_examples),
    )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("=" * 60)
    return result


if __name__ == "__main__":
    run_experiment9()
