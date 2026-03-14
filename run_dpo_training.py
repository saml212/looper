#!/usr/bin/env python3
"""DPO-style training on multi-repo resolved vs failed trajectories.

Uses trajectories aggregated by run_multirepo_expansion.py to train a LoRA
adapter with preference-based signal. Resolved trajectories are "chosen"
(positive), failed-but-patched trajectories are "rejected" (negative).

Since MLX LoRA does not support native DPO, this implements two approaches:

1. **Reward-weighted SFT** (default): Train on all trajectories but weight
   the loss by trajectory quality. Resolved trajectories get weight=1.0,
   failed-with-patch trajectories get a configurable negative/lower weight.
   In practice: we train on resolved trajectories with standard SFT loss
   and additionally train on "contrast" examples from failed trajectories
   where the assistant response is replaced with a brief rejection signal.

2. **Positive-only SFT** (fallback): If there are too few resolved
   trajectories for meaningful contrast, just do SFT on resolved-only.

The key insight: prior experiments used SFT on ALL trajectories (including
failures), which taught the model failure patterns. Here we explicitly
separate signal from noise using the resolved/failed classification.

Trajectory pairing for DPO:
- For each resolved trajectory, find a failed-with-patch trajectory from
  the SAME repo (closest match). This gives (prompt, chosen, rejected)
  triples sharing the same problem domain.
- Unpaired resolved trajectories are still used as positive SFT examples.
- Unpaired failed trajectories are discarded.
"""

import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.collectors.trajectory_store import load_all_trajectories, load_trajectory
from looper.models import AgentTrajectory, TaskInfo, TrainingExample
from looper.synthesizers.trajectory_synthesizer import (
    trajectory_to_step_examples,
    trajectory_to_training_example,
)
from looper.tasks.loader import get_repo_tasks, load_curriculum
from looper.trainers.lora_trainer import LoRAConfig

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

HF_MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
RESULTS_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_multirepo_expansion")
AGGREGATION_SUMMARY = RESULTS_DIR / "aggregation_summary.json"
ADAPTER_OUTPUT = RESULTS_DIR / "dpo_adapter"

# Training hyperparameters
LORA_RANK = 16
LORA_ITERS = 100
LORA_BATCH = 1
LORA_MAX_SEQ = 1024
LORA_LR = 1e-4

# DPO configuration
USE_PER_STEP = True  # Per-step examples (shorter sequences, better for 14B)
POSITIVE_WEIGHT = 1.0  # Weight for resolved trajectory examples
NEGATIVE_WEIGHT = 0.2  # Weight for failed trajectory "contrast" examples
MIN_RESOLVED_FOR_DPO = 3  # Minimum resolved trajectories to attempt DPO

# All SWE-Bench-CL repos
ALL_REPOS = [
    "django/django",
    "sympy/sympy",
    "sphinx-doc/sphinx",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "astropy/astropy",
    "pydata/xarray",
    "pytest-dev/pytest",
]

# Prior experiment directories with Django trajectories
PRIOR_DJANGO_DIRS = [
    Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_full/trajectories/base"),
    Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_remaining/trajectories/base"),
    Path("/Volumes/1TB_SSD/looper/results/experiment_selfplay_14b/trajectories/collection"),
]


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "dpo_training.log"),
        ],
    )


# ── Trajectory Loading ────────────────────────────────────────────────


def load_aggregation_summary() -> dict:
    """Load the aggregation summary from the multirepo expansion experiment."""
    if not AGGREGATION_SUMMARY.exists():
        logger.error(f"Aggregation summary not found: {AGGREGATION_SUMMARY}")
        logger.error("Run run_multirepo_expansion.py first to generate trajectories.")
        sys.exit(1)

    summary = json.loads(AGGREGATION_SUMMARY.read_text())
    logger.info(f"Aggregation summary:")
    logger.info(f"  Total resolved:          {summary['total_resolved']}")
    logger.info(f"  Total failed-with-patch: {summary['total_failed_with_patch']}")
    logger.info(f"  DPO viable:              {summary['dpo_viable']}")
    return summary


def find_trajectory_file(task_id: str, repo: str) -> Path | None:
    """Find the trajectory JSON file for a given task across all sources.

    Checks:
    1. Multirepo expansion trajectory directory (per-repo subdirs)
    2. Prior Django experiment directories
    """
    # Check multirepo expansion trajectories (organized by repo)
    repo_dir_name = repo.replace("/", "_")
    multirepo_path = RESULTS_DIR / "trajectories" / repo_dir_name / f"{task_id}.json"
    if multirepo_path.exists():
        return multirepo_path

    # Check prior Django experiment directories
    if "django" in repo:
        for prior_dir in PRIOR_DJANGO_DIRS:
            path = prior_dir / f"{task_id}.json"
            if path.exists():
                return path

    return None


def load_resolved_and_failed_trajectories(
    summary: dict, curriculum: dict
) -> tuple[
    list[tuple[str, AgentTrajectory, TaskInfo]],
    list[tuple[str, AgentTrajectory, TaskInfo]],
]:
    """Load resolved and failed-with-patch trajectories based on aggregation summary.

    Returns:
        (resolved, failed_with_patch) where each is a list of
        (repo, trajectory, task_info) tuples.
    """
    # Build task info lookup for all repos
    task_map: dict[str, tuple[str, TaskInfo]] = {}
    for repo in ALL_REPOS:
        for task in get_repo_tasks(curriculum, repo):
            task_map[task.instance_id] = (repo, task)

    # Load resolved trajectories from the summary
    resolved_task_ids = set()
    resolved: list[tuple[str, AgentTrajectory, TaskInfo]] = []
    for entry in summary.get("resolved_tasks", []):
        task_id = entry["task_id"]
        repo = entry["repo"]
        resolved_task_ids.add(task_id)

        traj_path = find_trajectory_file(task_id, repo)
        if traj_path is None:
            logger.warning(f"  Resolved trajectory not found: {task_id} ({repo})")
            continue

        traj = load_trajectory(traj_path)
        task_info = task_map.get(task_id)
        if task_info is None:
            logger.warning(f"  Task info not found for: {task_id}")
            continue

        _, task = task_info
        resolved.append((repo, traj, task))
        logger.info(f"  Loaded resolved: {task_id} ({repo}, {traj.meta.total_steps} steps)")

    # Load failed-with-patch trajectories by scanning all trajectory directories
    failed_with_patch: list[tuple[str, AgentTrajectory, TaskInfo]] = []

    # Scan multirepo expansion trajectories
    traj_base = RESULTS_DIR / "trajectories"
    if traj_base.exists():
        for repo_dir in sorted(traj_base.iterdir()):
            if not repo_dir.is_dir():
                continue
            # Convert dir name back to repo format
            repo_name = repo_dir.name.replace("_", "/", 1)
            for traj_file in sorted(repo_dir.glob("*.json")):
                task_id = traj_file.stem
                if task_id in resolved_task_ids:
                    continue  # Skip resolved ones
                try:
                    traj = load_trajectory(traj_file)
                except Exception as e:
                    logger.warning(f"  Failed to load {traj_file}: {e}")
                    continue

                if not traj.generated_patch.strip():
                    continue  # Skip no-patch failures

                task_entry = task_map.get(task_id)
                if task_entry is None:
                    continue
                _, task = task_entry
                failed_with_patch.append((repo_name, traj, task))

    # Scan prior Django directories
    for prior_dir in PRIOR_DJANGO_DIRS:
        if not prior_dir.exists():
            continue
        for traj_file in sorted(prior_dir.glob("*.json")):
            task_id = traj_file.stem
            if task_id in resolved_task_ids:
                continue
            # Avoid duplicates (might already be loaded from multirepo)
            if any(t.meta.task_id == task_id for _, t, _ in failed_with_patch):
                continue
            try:
                traj = load_trajectory(traj_file)
            except Exception as e:
                logger.warning(f"  Failed to load {traj_file}: {e}")
                continue
            if not traj.generated_patch.strip():
                continue
            task_entry = task_map.get(task_id)
            if task_entry is None:
                continue
            _, task = task_entry
            failed_with_patch.append(("django/django", traj, task))

    logger.info(f"  Loaded {len(resolved)} resolved trajectories")
    logger.info(f"  Loaded {len(failed_with_patch)} failed-with-patch trajectories")

    return resolved, failed_with_patch


# ── DPO Pair Construction ─────────────────────────────────────────────


def pair_trajectories(
    resolved: list[tuple[str, AgentTrajectory, TaskInfo]],
    failed: list[tuple[str, AgentTrajectory, TaskInfo]],
) -> list[tuple[tuple[str, AgentTrajectory, TaskInfo], tuple[str, AgentTrajectory, TaskInfo]]]:
    """Pair resolved trajectories with failed-with-patch trajectories.

    Pairing strategy:
    1. Same repo preferred (closest domain match)
    2. Within same repo, pair by closest sequence_position (temporal proximity)
    3. If no same-repo failed trajectory, pair with any repo's failed trajectory
    4. Failed trajectories can be reused across pairs

    Returns list of (resolved_tuple, failed_tuple) pairs.
    """
    # Group failed trajectories by repo
    failed_by_repo: dict[str, list[tuple[str, AgentTrajectory, TaskInfo]]] = defaultdict(list)
    for entry in failed:
        failed_by_repo[entry[0]].append(entry)

    all_failed_flat = list(failed)
    pairs = []

    for res_entry in resolved:
        res_repo, res_traj, res_task = res_entry

        # Try same-repo first
        candidates = failed_by_repo.get(res_repo, [])
        if not candidates:
            # Fall back to all failed trajectories
            candidates = all_failed_flat

        if not candidates:
            logger.warning(
                f"  No failed trajectory to pair with {res_traj.meta.task_id} — "
                f"will use as positive-only SFT"
            )
            continue

        # Pick closest by sequence_position
        best = min(
            candidates,
            key=lambda c: abs(c[2].sequence_position - res_task.sequence_position),
        )
        pairs.append((res_entry, best))
        logger.info(
            f"  Paired: {res_traj.meta.task_id} (resolved) <-> "
            f"{best[1].meta.task_id} (failed, {best[0]})"
        )

    return pairs


# ── Training Data Generation ──────────────────────────────────────────


def create_dpo_training_data(
    resolved: list[tuple[str, AgentTrajectory, TaskInfo]],
    failed: list[tuple[str, AgentTrajectory, TaskInfo]],
    pairs: list[tuple[tuple, tuple]],
    per_step: bool = True,
) -> tuple[list[TrainingExample], dict]:
    """Create training examples with DPO-style reward weighting.

    Approach: Reward-weighted SFT
    - Resolved trajectories -> standard training examples (weight=1.0)
    - Failed trajectories -> training examples with truncated/early-exit
      responses to teach the model NOT to follow those patterns

    For each DPO pair:
    - "chosen": Full multi-turn from resolved trajectory (teaches correct behavior)
    - "rejected": We create contrast examples from the failed trajectory where
      the assistant's final action is replaced with <done> (teaches early exit
      over wrong-fix persistence)

    Returns (examples, stats_dict).
    """
    positive_examples: list[TrainingExample] = []
    contrast_examples: list[TrainingExample] = []
    stats = {
        "num_resolved": len(resolved),
        "num_failed": len(failed),
        "num_pairs": len(pairs),
        "positive_examples": 0,
        "contrast_examples": 0,
    }

    # 1. Generate positive examples from ALL resolved trajectories
    resolved_task_ids = set()
    for repo, traj, task in resolved:
        resolved_task_ids.add(traj.meta.task_id)
        if per_step:
            examples = trajectory_to_step_examples(traj, task)
            positive_examples.extend(examples)
            logger.info(
                f"  [+] {traj.meta.task_id}: {len(examples)} step examples (resolved)"
            )
        else:
            example = trajectory_to_training_example(traj, task)
            if example is not None:
                positive_examples.append(example)
                logger.info(
                    f"  [+] {traj.meta.task_id}: {len(example.messages)} msgs (resolved)"
                )

    stats["positive_examples"] = len(positive_examples)
    logger.info(f"  Total positive examples: {len(positive_examples)}")

    # 2. Generate contrast examples from paired failed trajectories
    #    Strategy: take the first N steps of the failed trajectory (the
    #    exploration/diagnosis phase that is often correct) but replace
    #    the final response with <done> — teaching the model that once
    #    it has explored, it should produce a targeted fix rather than
    #    persisting with wrong approaches.
    seen_failed = set()
    for (res_repo, res_traj, res_task), (fail_repo, fail_traj, fail_task) in pairs:
        if fail_traj.meta.task_id in seen_failed:
            continue
        seen_failed.add(fail_traj.meta.task_id)

        if per_step:
            # For per-step: take only the first 2-3 steps of the failed
            # trajectory (usually the correct exploration phase).
            # This teaches the model HOW to explore without the wrong-fix steps.
            max_contrast_steps = min(3, len(fail_traj.steps))
            from looper.models import AgentStep, AgentTrajectory as AT

            truncated = fail_traj.model_copy(deep=True)
            truncated.steps = truncated.steps[:max_contrast_steps]

            examples = trajectory_to_step_examples(truncated, fail_task)
            # Mark these as contrast examples (lower weight during training)
            for ex in examples:
                ex.source_pair_id = f"contrast_{ex.source_pair_id}"
            contrast_examples.extend(examples)
            logger.info(
                f"  [-] {fail_traj.meta.task_id}: {len(examples)} contrast steps "
                f"(first {max_contrast_steps} of {len(fail_traj.steps)} steps)"
            )
        else:
            # For full-trajectory: use the complete failed trajectory as-is
            # but with reduced weight (handled in training data formatting)
            example = trajectory_to_training_example(fail_traj, fail_task)
            if example is not None:
                example.source_pair_id = f"contrast_{fail_traj.meta.task_id}"
                contrast_examples.append(example)
                logger.info(
                    f"  [-] {fail_traj.meta.task_id}: contrast example "
                    f"({len(example.messages)} msgs)"
                )

    stats["contrast_examples"] = len(contrast_examples)
    logger.info(f"  Total contrast examples: {len(contrast_examples)}")

    # 3. Combine: positive examples first, then contrast examples
    #    The training script will handle weighting via data duplication:
    #    positive examples are included POSITIVE_WEIGHT times (round up),
    #    contrast examples NEGATIVE_WEIGHT times.
    all_examples = []

    # Duplicate positive examples for weighting (if weight > 1)
    pos_repeats = max(1, round(POSITIVE_WEIGHT))
    for _ in range(pos_repeats):
        all_examples.extend(positive_examples)

    # Include contrast examples (with reduced representation)
    # NEGATIVE_WEIGHT < 1 means we include a fraction of contrast examples
    if NEGATIVE_WEIGHT > 0 and contrast_examples:
        import random
        rng = random.Random(42)
        n_contrast = max(1, round(len(contrast_examples) * NEGATIVE_WEIGHT))
        sampled = rng.sample(contrast_examples, min(n_contrast, len(contrast_examples)))
        all_examples.extend(sampled)
        logger.info(
            f"  Sampled {len(sampled)}/{len(contrast_examples)} contrast examples "
            f"(weight={NEGATIVE_WEIGHT})"
        )

    logger.info(f"  Final training set: {len(all_examples)} examples")
    stats["total_examples"] = len(all_examples)

    return all_examples, stats


# ── Training ──────────────────────────────────────────────────────────


def train_adapter(
    examples: list[TrainingExample],
    adapter_dir: Path,
    label: str = "dpo",
) -> dict:
    """Train LoRA adapter in a subprocess to free GPU memory afterwards.

    Uses the same subprocess pattern as other experiment scripts to ensure
    MLX GPU memory is fully released after training.
    """
    from looper.synthesizers.synthesizer import save_training_data

    adapter_file = adapter_dir / "adapters.safetensors"
    metrics_file = adapter_dir.parent / f"training_metrics_{label}.json"

    if adapter_file.exists():
        logger.info(f"  [{label}] Adapter already trained at {adapter_dir}")
        if metrics_file.exists():
            return json.loads(metrics_file.read_text())
        return {"cached": True}

    if len(examples) < 2:
        logger.error(
            f"  [{label}] Only {len(examples)} examples — too few to train."
        )
        return {"skipped": True, "reason": "too_few_examples", "num_examples": len(examples)}

    # Save training data to temp file
    training_file = adapter_dir.parent / f"training_{label}.jsonl"
    save_training_data(examples, training_file)
    logger.info(f"  [{label}] Saved {len(examples)} training examples to {training_file}")

    # Training in subprocess (releases GPU memory when done)
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
    learning_rate={LORA_LR},
)
metrics = full_replay_train(examples, '{HF_MODEL}', Path('{adapter_dir}'), config)
print(json.dumps(metrics))
"""
    logger.info(
        f"  [{label}] Training LoRA adapter "
        f"({len(examples)} examples, rank={LORA_RANK}, iters={LORA_ITERS})..."
    )
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
        timeout=3600,
    )

    if result.returncode != 0:
        logger.error(f"  [{label}] Training failed:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"LoRA training failed for {label}")

    # Parse metrics from subprocess output (last JSON line)
    metrics = {}
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    metrics["num_examples"] = len(examples)
    metrics["rank"] = LORA_RANK
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info(
        f"  [{label}] Done: train_loss={metrics.get('final_train_loss', '?')}, "
        f"val_loss={metrics.get('final_val_loss', '?')}"
    )
    return metrics


# ── Main ──────────────────────────────────────────────────────────────


def main():
    setup_logging(RESULTS_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("DPO-STYLE LORA TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model:           {HF_MODEL}")
    logger.info(f"LoRA rank:       {LORA_RANK}")
    logger.info(f"LoRA iters:      {LORA_ITERS}")
    logger.info(f"Batch size:      {LORA_BATCH}")
    logger.info(f"Max seq length:  {LORA_MAX_SEQ}")
    logger.info(f"Learning rate:   {LORA_LR}")
    logger.info(f"Per-step:        {USE_PER_STEP}")
    logger.info(f"Positive weight: {POSITIVE_WEIGHT}")
    logger.info(f"Negative weight: {NEGATIVE_WEIGHT}")
    logger.info(f"Adapter output:  {ADAPTER_OUTPUT}")
    logger.info("=" * 60)

    # Step 1: Load aggregation summary
    logger.info("\n[Step 1] Loading aggregation summary...")
    summary = load_aggregation_summary()

    # Step 2: Load curriculum and trajectories
    logger.info("\n[Step 2] Loading trajectories...")
    curriculum = load_curriculum(CURRICULUM)
    resolved, failed_with_patch = load_resolved_and_failed_trajectories(
        summary, curriculum
    )

    if not resolved:
        logger.error("No resolved trajectories found. Cannot train.")
        sys.exit(1)

    # Step 3: Decide training strategy
    logger.info("\n[Step 3] Selecting training strategy...")
    if len(resolved) >= MIN_RESOLVED_FOR_DPO and len(failed_with_patch) >= 1:
        strategy = "reward_weighted_sft"
        logger.info(
            f"  Strategy: Reward-weighted SFT "
            f"({len(resolved)} resolved, {len(failed_with_patch)} failed-with-patch)"
        )
    else:
        strategy = "positive_only_sft"
        logger.info(
            f"  Strategy: Positive-only SFT "
            f"({len(resolved)} resolved, insufficient failed for DPO)"
        )

    # Step 4: Create DPO pairs
    logger.info("\n[Step 4] Creating trajectory pairs...")
    if strategy == "reward_weighted_sft":
        pairs = pair_trajectories(resolved, failed_with_patch)
        logger.info(f"  Created {len(pairs)} resolved-failed pairs")
    else:
        pairs = []

    # Step 5: Generate training data
    logger.info("\n[Step 5] Generating training examples...")
    if strategy == "reward_weighted_sft":
        examples, data_stats = create_dpo_training_data(
            resolved, failed_with_patch, pairs, per_step=USE_PER_STEP,
        )
    else:
        # Positive-only: just convert resolved trajectories
        examples = []
        for repo, traj, task in resolved:
            if USE_PER_STEP:
                step_exs = trajectory_to_step_examples(traj, task)
                examples.extend(step_exs)
            else:
                ex = trajectory_to_training_example(traj, task)
                if ex is not None:
                    examples.append(ex)
        data_stats = {
            "num_resolved": len(resolved),
            "num_failed": 0,
            "num_pairs": 0,
            "positive_examples": len(examples),
            "contrast_examples": 0,
            "total_examples": len(examples),
        }

    if not examples:
        logger.error("No training examples generated. Cannot train.")
        sys.exit(1)

    logger.info(f"  Generated {len(examples)} total training examples")

    # Step 6: Train LoRA adapter
    logger.info("\n[Step 6] Training LoRA adapter...")
    metrics = train_adapter(examples, ADAPTER_OUTPUT, label="dpo")

    # Step 7: Save experiment summary
    logger.info("\n[Step 7] Saving experiment summary...")
    experiment_summary = {
        "strategy": strategy,
        "model": HF_MODEL,
        "lora_config": {
            "rank": LORA_RANK,
            "iters": LORA_ITERS,
            "batch_size": LORA_BATCH,
            "max_seq_length": LORA_MAX_SEQ,
            "learning_rate": LORA_LR,
        },
        "data_stats": data_stats,
        "training_metrics": metrics,
        "positive_weight": POSITIVE_WEIGHT,
        "negative_weight": NEGATIVE_WEIGHT,
        "per_step": USE_PER_STEP,
        "adapter_path": str(ADAPTER_OUTPUT),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = RESULTS_DIR / "dpo_training_summary.json"
    summary_path.write_text(json.dumps(experiment_summary, indent=2))
    logger.info(f"  Summary saved to {summary_path}")

    # Final report
    logger.info("")
    logger.info("=" * 60)
    logger.info("DPO TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Strategy:          {strategy}")
    logger.info(f"  Resolved trajs:    {data_stats['num_resolved']}")
    logger.info(f"  Failed trajs:      {data_stats.get('num_failed', 0)}")
    logger.info(f"  DPO pairs:         {data_stats['num_pairs']}")
    logger.info(f"  Positive examples: {data_stats['positive_examples']}")
    logger.info(f"  Contrast examples: {data_stats['contrast_examples']}")
    logger.info(f"  Total examples:    {data_stats['total_examples']}")
    logger.info(f"  Train loss:        {metrics.get('final_train_loss', '?')}")
    logger.info(f"  Val loss:          {metrics.get('final_val_loss', '?')}")
    logger.info(f"  Adapter saved to:  {ADAPTER_OUTPUT}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start MLX server with adapter:")
    logger.info(f"     .venv/bin/mlx_lm.server --model {HF_MODEL} \\")
    logger.info(f"       --adapter-path {ADAPTER_OUTPUT} --port 8080")
    logger.info("  2. Run evaluation on held-out Django tasks")
    logger.info("  3. Compare resolve rate: base vs DPO-adapted")

    return experiment_summary


if __name__ == "__main__":
    main()
