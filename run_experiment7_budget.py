#!/usr/bin/env python3
"""Experiment 7: Synthesis Budget Sweep.

Tests whether more synthesis pairs per trajectory leads to better adapters
or introduces noise. Uses existing Phase 1 trajectories (no new inference).

Budget levels: 3, 5, 10, 20 pairs per trajectory
For each: synthesize with budget-aware prompt -> train LoRA -> record loss
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment7_budget")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "experiment.log"),
    ],
)
logger = logging.getLogger(__name__)

BUDGET_LEVELS = [3, 5, 10, 20]
HF_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434"
LORA_RANK = 16
LORA_ITERS = 100

# Budget-aware prompt template — {num_pairs} is interpolated per budget level
BUDGET_PROMPT = """You are analyzing an AI agent's work session to extract reusable skills.

Given the following trajectory of an AI agent working on a coding task, extract exactly {num_pairs} instruction/response pairs that capture the skills demonstrated.

Focus on:
- tool_usage: How to use tools effectively in this codebase
- error_recovery: How to diagnose and fix errors
- convention: Project-specific conventions and patterns
- workflow: Effective workflow patterns

Output your pairs as a JSON array:
[
  {{
    "instruction": "How should you ...",
    "response": "You should ...",
    "pair_type": "tool_usage",
    "confidence": 0.8
  }},
  ...
]

IMPORTANT: Output exactly {num_pairs} pairs. Only output the JSON array, nothing else.

Trajectory:
{trajectory_text}
"""


def synthesize_with_budget(trajectories, num_pairs, output_dir):
    """Synthesize training data with a specific pair budget per trajectory."""
    from looper.agent.ollama_client import ChatMessage, chat
    from looper.models import SynthesizedPair
    from looper.synthesizers.synthesizer import (
        _extract_json_array,
        pairs_to_training_examples,
        save_training_data,
    )
    from looper.synthesizers.trajectory_to_text import trajectory_to_text

    pairs_file = output_dir / "pairs.json"
    training_file = output_dir / "training.jsonl"

    if training_file.exists():
        from looper.synthesizers.synthesizer import load_training_data
        examples = load_training_data(training_file)
        pairs_data = json.loads(pairs_file.read_text()) if pairs_file.exists() else []
        logger.info(f"  Loaded cached: {len(examples)} examples from {len(pairs_data)} pairs")
        return pairs_data, examples

    all_pairs = []
    for i, traj in enumerate(trajectories):
        traj_text = trajectory_to_text(traj)
        prompt = BUDGET_PROMPT.format(
            trajectory_text=traj_text, num_pairs=num_pairs
        )
        messages = [ChatMessage(role="user", content=prompt)]

        try:
            response = chat(messages, model=OLLAMA_MODEL, base_url=OLLAMA_URL)
            raw_pairs = _extract_json_array(response.content)
            if raw_pairs is None:
                logger.warning(f"  Invalid JSON for {traj.meta.task_id}: {response.content[:100]}")
                continue

            count = 0
            for raw in raw_pairs:
                try:
                    pair = SynthesizedPair(
                        instruction=raw["instruction"],
                        response=raw["response"],
                        pair_type=raw.get("pair_type", "tool_usage"),
                        confidence=float(raw.get("confidence", 0.5)),
                        source_session_id=traj.meta.session_id,
                        source_task_id=traj.meta.task_id,
                    )
                    if pair.confidence >= 0.3:
                        all_pairs.append(pair)
                        count += 1
                except (KeyError, ValueError, TypeError):
                    continue

            logger.info(f"  [{i+1}/{len(trajectories)}] {traj.meta.task_id}: {count} pairs")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(trajectories)}] {traj.meta.task_id}: error: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_data = [p.model_dump() for p in all_pairs]
    pairs_file.write_text(json.dumps(pairs_data, indent=2))

    examples = pairs_to_training_examples(all_pairs)
    save_training_data(examples, training_file)

    return pairs_data, examples


def train_adapter(training_file, adapter_dir, metrics_dir):
    """Train a LoRA adapter in a subprocess."""
    adapter_file = adapter_dir / "adapters.safetensors"
    metrics_file = metrics_dir / "training_metrics.json"

    if adapter_file.exists() and metrics_file.exists():
        return json.loads(metrics_file.read_text())

    train_script = f"""
import sys
sys.path.insert(0, '.')
from looper.trainers.full_replay import full_replay_train
from looper.trainers.lora_trainer import LoRAConfig
from looper.synthesizers.synthesizer import load_training_data
from pathlib import Path
import json

examples = load_training_data(Path('{training_file}'))
config = LoRAConfig(rank={LORA_RANK}, iters={LORA_ITERS}, batch_size=1, max_seq_length=1024)
metrics = full_replay_train(examples, '{HF_MODEL}', Path('{adapter_dir}'), config)
print(json.dumps(metrics))
"""
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )

    if result.returncode != 0:
        logger.error(f"  Training failed: {result.stderr[-300:]}")
        return {"error": result.stderr[-200:]}

    metrics = {}
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    metrics_file.write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    from looper.collectors.trajectory_store import load_all_trajectories
    from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks

    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("EXPERIMENT 7: Synthesis Budget Sweep")
    logger.info(f"Budget levels: {BUDGET_LEVELS}")
    logger.info("=" * 60)

    # Load train trajectories from Phase 1
    traj_dir = Path("/Volumes/1TB_SSD/looper/results/phase1/trajectories/base")
    all_trajectories = load_all_trajectories(traj_dir)
    logger.info(f"Loaded {len(all_trajectories)} trajectories")

    curriculum = load_curriculum(
        Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
    )
    all_tasks = get_repo_tasks(curriculum, "django/django")
    train_tasks, _ = split_tasks(all_tasks, train_size=25)
    train_ids = {t.instance_id for t in train_tasks}
    train_trajectories = [
        t for t in all_trajectories if t.meta.task_id in train_ids
    ]
    logger.info(f"Train trajectories: {len(train_trajectories)}")

    results = {}

    for budget in BUDGET_LEVELS:
        logger.info("")
        logger.info(f"{'='*40}")
        logger.info(f"BUDGET LEVEL: {budget} pairs per trajectory")
        logger.info(f"{'='*40}")

        budget_dir = OUTPUT_DIR / f"budget_{budget}"
        budget_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Synthesize with budget-aware prompt
        t0 = time.time()
        pairs_data, examples = synthesize_with_budget(
            train_trajectories, budget, budget_dir
        )
        synthesis_time = time.time() - t0
        logger.info(
            f"  Total: {len(pairs_data)} pairs, "
            f"{len(examples)} examples, {synthesis_time:.0f}s"
        )

        # Analyze synthesis quality
        if pairs_data:
            types = {}
            confidences = []
            for p in pairs_data:
                t = p.get("pair_type", "unknown")
                types[t] = types.get(t, 0) + 1
                confidences.append(p.get("confidence", 0.5))
            avg_conf = sum(confidences) / max(len(confidences), 1)
            avg_resp_len = sum(len(p.get("response", "")) for p in pairs_data) / len(pairs_data)
            logger.info(f"  Types: {types}")
            logger.info(f"  Avg confidence: {avg_conf:.2f}")
            logger.info(f"  Avg response length: {avg_resp_len:.0f} chars")
            logger.info(
                f"  Actual pairs/trajectory: {len(pairs_data)/max(len(train_trajectories),1):.1f}"
            )

        # Step 2: Train LoRA adapter
        adapter_dir = budget_dir / "adapter"
        if len(examples) > 0:
            logger.info(f"  Training LoRA adapter ({len(examples)} examples)...")
            t0 = time.time()
            train_metrics = train_adapter(
                budget_dir / "training.jsonl", adapter_dir, budget_dir
            )
            training_time = time.time() - t0
            logger.info(f"  Training metrics: {train_metrics}")
        else:
            train_metrics = {"error": "no_training_data"}
            training_time = 0

        results[str(budget)] = {
            "budget": budget,
            "num_pairs": len(pairs_data),
            "num_training_examples": len(examples),
            "synthesis_time_s": synthesis_time,
            "training_time_s": training_time,
            "training_metrics": train_metrics,
            "actual_pairs_per_trajectory": (
                len(pairs_data) / max(len(train_trajectories), 1)
            ),
            "avg_response_length": (
                sum(len(p.get("response", "")) for p in pairs_data) / max(len(pairs_data), 1)
                if pairs_data else 0
            ),
        }

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENT 7 RESULTS SUMMARY")
    logger.info("=" * 60)

    for budget_str, r in results.items():
        m = r["training_metrics"]
        logger.info(
            f"  Budget {r['budget']:>2}: {r['num_pairs']:>3} pairs "
            f"({r['actual_pairs_per_trajectory']:.1f}/traj), "
            f"train_loss={m.get('final_train_loss', 'N/A')}, "
            f"val_loss={m.get('final_val_loss', 'N/A')}"
        )

    # Save
    overall = {
        "experiment": "experiment7_synthesis_budget",
        "budget_levels": BUDGET_LEVELS,
        "model": HF_MODEL,
        "synthesis_model": OLLAMA_MODEL,
        "train_trajectories": len(train_trajectories),
        "lora_config": {"rank": LORA_RANK, "iters": LORA_ITERS},
        "results": results,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "experiment_result.json").write_text(
        json.dumps(overall, indent=2, default=str)
    )
    logger.info(f"\nResults saved to {OUTPUT_DIR / 'experiment_result.json'}")


if __name__ == "__main__":
    main()
