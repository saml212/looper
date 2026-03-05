"""Synthesize training pairs from agent trajectories using an LLM."""

import json
import logging
from pathlib import Path

from looper.agent.ollama_client import ChatMessage, chat
from looper.models import AgentTrajectory, SynthesizedPair, TrainingExample
from looper.synthesizers.trajectory_to_text import trajectory_to_text

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are analyzing an AI agent's work session to extract reusable skills.

Given the following trajectory of an AI agent working on a coding task, extract 3-5 instruction/response pairs that capture the skills demonstrated.

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
"""

CONFIDENCE_THRESHOLD = 0.3


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common 7B model output issue)."""
    import re

    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_json_array(text: str) -> list[dict] | None:
    """Try to extract a JSON array from text that may contain prose around it.

    Handles cases where the 7B model wraps JSON in markdown fences, adds
    explanatory text, or uses trailing commas.
    """
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array within the text
    start = text.find("[")
    if start == -1:
        return None

    # Find matching closing bracket
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                # Try parsing, with trailing comma fix as fallback
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    pass
                try:
                    result = json.loads(_fix_trailing_commas(candidate))
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    return None

    return None


def synthesize_trajectory(
    trajectory: AgentTrajectory,
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434",
    num_pairs: int = 5,
    chat_fn=None,
) -> list[SynthesizedPair]:
    """Synthesize training pairs from a trajectory using an LLM.

    1. Convert trajectory to text
    2. Send to LLM with synthesis prompt
    3. Parse JSON response into SynthesizedPairs
    4. Filter out low-confidence pairs (< 0.3)

    If chat_fn is provided, it is used instead of the default Ollama chat function.
    """
    if chat_fn is None:
        chat_fn = chat
    trajectory_text = trajectory_to_text(trajectory)
    prompt = SYNTHESIS_PROMPT.format(trajectory_text=trajectory_text)

    messages = [ChatMessage(role="user", content=prompt)]

    try:
        response = chat_fn(messages, model=model, base_url=base_url)
    except Exception:
        logger.exception("LLM call failed during synthesis")
        return []

    raw_pairs = _extract_json_array(response.content)
    if raw_pairs is None:
        logger.warning("LLM returned invalid JSON: %s", response.content[:200])
        return []

    if not isinstance(raw_pairs, list):
        logger.warning("LLM response is not a JSON array")
        return []

    pairs: list[SynthesizedPair] = []
    for raw in raw_pairs:
        try:
            pair = SynthesizedPair(
                instruction=raw["instruction"],
                response=raw["response"],
                pair_type=raw.get("pair_type", "tool_usage"),
                confidence=float(raw.get("confidence", 0.5)),
                source_session_id=trajectory.meta.session_id,
                source_task_id=trajectory.meta.task_id,
            )
            if pair.confidence >= CONFIDENCE_THRESHOLD:
                pairs.append(pair)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed pair: %s (%s)", raw, exc)
            continue

    return pairs


def synthesize_batch(
    trajectories: list[AgentTrajectory],
    output_path: Path,
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434",
    num_pairs: int = 5,
    chat_fn=None,
) -> list[SynthesizedPair]:
    """Synthesize pairs from multiple trajectories and save to JSON.

    Returns all synthesized pairs.
    """
    all_pairs: list[SynthesizedPair] = []

    for traj in trajectories:
        pairs = synthesize_trajectory(
            traj, model=model, base_url=base_url, num_pairs=num_pairs,
            chat_fn=chat_fn,
        )
        all_pairs.extend(pairs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([p.model_dump() for p in all_pairs], f, indent=2)

    return all_pairs


def pairs_to_training_examples(
    pairs: list[SynthesizedPair],
) -> list[TrainingExample]:
    """Convert SynthesizedPairs to TrainingExamples in chat format.

    Each pair becomes:
    messages = [
        {"role": "user", "content": pair.instruction},
        {"role": "assistant", "content": pair.response}
    ]
    """
    return [
        TrainingExample(
            messages=[
                {"role": "user", "content": pair.instruction},
                {"role": "assistant", "content": pair.response},
            ],
        )
        for pair in pairs
    ]


def save_training_data(examples: list[TrainingExample], path: Path) -> None:
    """Save training examples as a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example.model_dump()) + "\n")


def load_training_data(path: Path) -> list[TrainingExample]:
    """Load training examples from a JSONL file."""
    examples: list[TrainingExample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(TrainingExample(**json.loads(line)))
    return examples
