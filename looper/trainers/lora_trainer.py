"""LoRA training wrapper using MLX."""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.tuner import train as mlx_train
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.tuner.trainer import CacheDataset, TrainingArgs


def save_adapters(model, path: Path, config: "LoRAConfig | None" = None) -> None:
    """Save adapter weights and config to a directory."""
    import json as _json

    import mlx.core as mx
    from mlx.utils import tree_flatten

    path.mkdir(parents=True, exist_ok=True)
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path / "adapters.safetensors"), adapter_weights)

    # Save adapter_config.json required by mlx_lm fuse
    if config is None:
        config = LoRAConfig()
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": config.num_layers,
        "lora_parameters": {
            "rank": config.rank,
            "scale": config.rank * 2.0,
            "dropout": 0.05,
        },
    }
    with open(path / "adapter_config.json", "w") as f:
        _json.dump(adapter_config, f, indent=2)


# Alias for monkeypatching in tests
linear_to_loro_layers = linear_to_lora_layers


@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""

    rank: int = 16
    num_layers: int = 16
    learning_rate: float = 1e-4
    iters: int = 100
    batch_size: int = 4
    max_seq_length: int = 2048
    seed: int = 42


class _MetricsCallback(TrainingCallback):
    """Captures training and validation loss from MLX train loop."""

    def __init__(self):
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_loss_report(self, train_info: dict):
        if "train_loss" in train_info:
            self.train_losses.append(train_info["train_loss"])

    def on_val_loss_report(self, val_info: dict):
        if "val_loss" in val_info:
            self.val_losses.append(val_info["val_loss"])


def train_lora(
    model_name: str,
    data_dir: Path,
    adapter_output_dir: Path,
    config: LoRAConfig | None = None,
) -> dict:
    """Train a LoRA adapter using MLX.

    Args:
        model_name: HuggingFace model ID.
        data_dir: Directory containing train.jsonl and valid.jsonl.
        adapter_output_dir: Where to save the adapter weights.
        config: LoRA training configuration.

    Returns:
        dict with training metrics.
    """
    if config is None:
        config = LoRAConfig()

    adapter_output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load(model_name)
    model.freeze()

    # Apply LoRA layers
    lora_config = {
        "rank": config.rank,
        "scale": config.rank * 2.0,
        "dropout": 0.05,
    }
    linear_to_loro_layers(model, num_layers=config.num_layers, config=lora_config)

    # Prepare optimizer
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Load datasets
    train_data = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_data.append(json.loads(line))

    val_data = []
    val_path = data_dir / "valid.jsonl"
    if val_path.exists():
        with open(val_path) as f:
            for line in f:
                val_data.append(json.loads(line))

    train_set = CacheDataset(ChatDataset(train_data, tokenizer, chat_key="messages"))
    val_set = CacheDataset(ChatDataset(val_data, tokenizer, chat_key="messages")) if val_data else None

    # Training args
    adapter_file = str(adapter_output_dir / "adapters.safetensors")
    training_args = TrainingArgs(
        batch_size=config.batch_size,
        iters=config.iters,
        max_seq_length=config.max_seq_length,
        adapter_file=adapter_file,
        steps_per_eval=config.iters,  # Evaluate at end of training
    )

    # Train with metrics callback
    callback = _MetricsCallback()
    mlx_train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        args=training_args,
        training_callback=callback,
    )

    # Save adapter weights and config for later loading/fusing
    save_adapters(model, adapter_output_dir, config)

    return {
        "final_train_loss": callback.train_losses[-1] if callback.train_losses else 0.0,
        "final_val_loss": callback.val_losses[-1] if callback.val_losses else 0.0,
        "iters": config.iters,
    }


def load_model_with_adapter(
    model_name: str,
    adapter_path: Path | None = None,
) -> tuple:
    """Load a model, optionally with a LoRA adapter applied.

    Returns (model, tokenizer) tuple.
    """
    kwargs = {}
    if adapter_path is not None:
        kwargs["adapter_path"] = str(adapter_path)
    return load(model_name, **kwargs)
