#!/usr/bin/env python3
"""Experiment 6: Synthesis Format Comparison.

Tests whether different synthesis formats produce better LoRA adapters.
Uses existing Phase 1 trajectories. Compares 5 formats:

Format A — Simple QA (current default)
Format B — Chain-of-thought
Format C — DPO preference pairs (requires both good and bad trajectories)
Format D — Reflexion-style self-critique
Format E — Contextual memory (includes project metadata)

Since Format C requires failed trajectories (which we have), all 5 can be tested.
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment6_format")
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

HF_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434"
LORA_RANK = 16
LORA_ITERS = 100

# Format-specific synthesis prompts
FORMAT_PROMPTS = {
    "A_simple_qa": """You are analyzing an AI agent's work session to extract reusable skills.

Given the following trajectory of an AI agent working on a coding task, extract {num_pairs} instruction/response pairs that capture the skills demonstrated.

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

Only output the JSON array, nothing else.

Trajectory:
{trajectory_text}
""",

    "B_chain_of_thought": """You are analyzing an AI agent's work session to extract reusable skills with detailed reasoning.

Given the following trajectory, extract {num_pairs} instruction/response pairs. Each response must include a chain-of-thought reasoning process that explains WHY the approach works.

Format each response as: "Let me think through this step by step. First, [reason]. Then, [reason]. Therefore, [action]. This works because [explanation]."

Output your pairs as a JSON array:
[
  {{
    "instruction": "How should you ...",
    "response": "Let me think through this step by step. First, ...",
    "pair_type": "tool_usage",
    "confidence": 0.8
  }},
  ...
]

Only output the JSON array, nothing else.

Trajectory:
{trajectory_text}
""",

    "D_reflexion": """You are analyzing an AI agent's work session to extract skills through reflexion.

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

Only output the JSON array, nothing else.

Trajectory:
{trajectory_text}
""",

    "E_contextual": """You are analyzing an AI agent's work session to extract context-rich skills.

Given the following trajectory working on a Django project, extract {num_pairs} instruction/response pairs that include specific project context (file paths, module names, framework conventions).

Each instruction should mention the project/framework by name. Each response should reference specific files, directories, or configuration details from the trajectory.

Output your pairs as a JSON array:
[
  {{
    "instruction": "In the Django project, how should you ...",
    "response": "In this Django codebase, the [specific pattern] is at [specific path]. You should ... because Django's [specific feature] ...",
    "pair_type": "convention",
    "confidence": 0.8
  }},
  ...
]

Only output the JSON array, nothing else.

Trajectory:
{trajectory_text}
""",
}


def synthesize_with_format(
    trajectories, format_name, prompt_template, output_dir, num_pairs=5
):
    """Synthesize training data using a specific format prompt."""
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
        logger.info(f"  [{format_name}] Loaded cached: {len(examples)} examples")
        return pairs_data, examples

    all_pairs = []
    for traj in trajectories:
        traj_text = trajectory_to_text(traj)
        prompt = prompt_template.format(
            trajectory_text=traj_text, num_pairs=num_pairs
        )
        messages = [ChatMessage(role="user", content=prompt)]

        try:
            response = chat(messages, model=OLLAMA_MODEL, base_url=OLLAMA_URL)
            raw_pairs = _extract_json_array(response.content)
            if raw_pairs is None:
                logger.warning(
                    f"  [{format_name}] Invalid JSON for {traj.meta.task_id}"
                )
                continue

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
                except (KeyError, ValueError, TypeError):
                    continue
        except Exception as e:
            logger.warning(f"  [{format_name}] LLM error for {traj.meta.task_id}: {e}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_data = [p.model_dump() for p in all_pairs]
    pairs_file.write_text(json.dumps(pairs_data, indent=2))

    examples = pairs_to_training_examples(all_pairs)
    save_training_data(examples, training_file)

    return pairs_data, examples


def train_adapter(training_file, adapter_dir, budget_dir):
    """Train a LoRA adapter in a subprocess."""
    adapter_file = adapter_dir / "adapters.safetensors"
    if adapter_file.exists():
        metrics_file = budget_dir / "training_metrics.json"
        if metrics_file.exists():
            return json.loads(metrics_file.read_text())
        return {"cached": True}

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

    (budget_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    from looper.collectors.trajectory_store import load_all_trajectories
    from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks

    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: Synthesis Format Comparison")
    logger.info(f"Formats: {list(FORMAT_PROMPTS.keys())}")
    logger.info("=" * 60)

    # Load train trajectories
    traj_dir = Path("/Volumes/1TB_SSD/looper/results/phase1/trajectories/base")
    all_trajectories = load_all_trajectories(traj_dir)

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

    for format_name, prompt_template in FORMAT_PROMPTS.items():
        logger.info("")
        logger.info(f"{'='*40}")
        logger.info(f"FORMAT: {format_name}")
        logger.info(f"{'='*40}")

        format_dir = OUTPUT_DIR / format_name
        format_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Synthesize
        t0 = time.time()
        pairs_data, examples = synthesize_with_format(
            train_trajectories, format_name, prompt_template, format_dir
        )
        synthesis_time = time.time() - t0
        logger.info(
            f"  [{format_name}] {len(pairs_data)} pairs, "
            f"{len(examples)} examples, {synthesis_time:.0f}s"
        )

        # Analyze
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

        # Step 2: Train
        if examples:
            t0 = time.time()
            train_metrics = train_adapter(
                format_dir / "training.jsonl",
                format_dir / "adapter",
                format_dir,
            )
            training_time = time.time() - t0
            logger.info(f"  [{format_name}] Training: {train_metrics}")
        else:
            train_metrics = {"error": "no_training_data"}
            training_time = 0

        results[format_name] = {
            "format": format_name,
            "num_pairs": len(pairs_data),
            "num_examples": len(examples),
            "synthesis_time_s": synthesis_time,
            "training_time_s": training_time,
            "training_metrics": train_metrics,
            "avg_response_length": (
                sum(len(p.get("response", "")) for p in pairs_data) / max(len(pairs_data), 1)
                if pairs_data else 0
            ),
        }

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6 RESULTS SUMMARY")
    logger.info("=" * 60)

    for name, r in results.items():
        m = r["training_metrics"]
        logger.info(
            f"  {name}: {r['num_pairs']} pairs, "
            f"train_loss={m.get('final_train_loss', 'N/A')}, "
            f"val_loss={m.get('final_val_loss', 'N/A')}, "
            f"avg_resp_len={r['avg_response_length']:.0f}"
        )

    # Save
    overall = {
        "experiment": "experiment6_synthesis_format",
        "formats": list(FORMAT_PROMPTS.keys()),
        "model": HF_MODEL,
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
