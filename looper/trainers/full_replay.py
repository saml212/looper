"""Full replay training strategy -- retrain from scratch on all accumulated data."""

import tempfile
from pathlib import Path

from looper.models import TrainingExample
from looper.trainers.data_formatter import prepare_training_dir
from looper.trainers.lora_trainer import LoRAConfig, train_lora


def full_replay_train(
    all_examples: list[TrainingExample],
    model_name: str,
    adapter_dir: Path,
    config: LoRAConfig | None = None,
) -> dict:
    """Full replay training strategy.

    1. Take ALL accumulated training examples.
    2. Prepare training directory.
    3. Train LoRA from scratch (no previous adapter).
    4. Save adapter to adapter_dir.

    Returns training metrics.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        prepare_training_dir(all_examples, data_dir)
        return train_lora(model_name, data_dir, adapter_dir, config=config)
