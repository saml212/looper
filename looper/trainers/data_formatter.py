"""Prepare training data directories for MLX LoRA training."""

import json
import random
from pathlib import Path

from looper.models import TrainingExample


def prepare_training_dir(
    examples: list[TrainingExample],
    output_dir: Path,
    val_split: float = 0.1,
    seed: int = 42,
) -> Path:
    """Prepare a directory with train.jsonl and valid.jsonl for MLX training.

    Each line is {"messages": [{"role": "...", "content": "..."}, ...]}.

    Args:
        examples: Training examples to write.
        output_dir: Directory to create train.jsonl and valid.jsonl in.
        val_split: Fraction of examples for validation (default 0.1).
        seed: Random seed for reproducible split.

    Returns:
        The output_dir path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(len(examples) * val_split)
    val_indices = set(indices[:n_val])

    with open(output_dir / "train.jsonl", "w") as train_f, \
         open(output_dir / "valid.jsonl", "w") as valid_f:
        for i, ex in enumerate(examples):
            line = json.dumps({"messages": ex.messages}) + "\n"
            if i in val_indices:
                valid_f.write(line)
            else:
                train_f.write(line)

    return output_dir
