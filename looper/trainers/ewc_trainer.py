"""EWC-LoRA trainer: Elastic Weight Consolidation for LoRA parameters.

Implements sequential LoRA training with EWC regularization to prevent
catastrophic forgetting when training on sequential batches of data.

After each batch, computes the diagonal Fisher Information Matrix for LoRA
parameters. On subsequent batches, adds the EWC penalty:

    lambda * sum(F_i * (theta_i - theta_i*)^2)

Uses online EWC: Fisher is accumulated across all previous batches.
"""

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.tuner.trainer import CacheDataset

from looper.trainers.lora_trainer import LoRAConfig, save_adapters

logger = logging.getLogger(__name__)


def _prepare_sequences(
    data_path: Path, tokenizer, max_seq_length: int
) -> list[list[int]]:
    """Load training data and tokenize into sequences."""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    dataset = CacheDataset(ChatDataset(data, tokenizer, chat_key="messages"))

    sequences = []
    for i in range(len(dataset)):
        item = dataset[i]
        # CacheDataset returns (tokens, loss_offset) tuple
        tokens = item[0] if isinstance(item, tuple) else item
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            tokens = list(tokens)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        if len(tokens) > 1:
            sequences.append(tokens)

    return sequences


def _make_batch(
    sequences: list[list[int]], indices: list[int]
) -> tuple[mx.array, mx.array, mx.array]:
    """Create a padded batch from sequence indices."""
    batch_seqs = [sequences[i] for i in indices]
    max_len = max(len(s) for s in batch_seqs)

    padded = []
    lengths = []
    for s in batch_seqs:
        lengths.append(len(s))
        padded.append(s + [0] * (max_len - len(s)))

    return mx.array(padded), mx.array(padded), mx.array(lengths)


def compute_fisher(
    model,
    sequences: list[list[int]],
    max_seq_length: int,
    num_samples: int = 100,
) -> dict[str, mx.array]:
    """Compute diagonal Fisher Information Matrix for LoRA parameters.

    Uses empirical Fisher: F_i = (1/N) * sum_n (dL/d_theta_i)^2
    Computed sample-by-sample for correctness.
    """
    fisher: dict[str, mx.array] = {}
    for name, param in tree_flatten(model.trainable_parameters()):
        fisher[name] = mx.zeros_like(param)

    def base_loss_fn(model, inputs, targets, lengths):
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < lengths[:, None]
        ce = nn.losses.cross_entropy(logits[:, :-1, :], targets[:, 1:]) * length_mask
        ntoks = length_mask.sum()
        return ce.sum() / ntoks

    grad_fn = nn.value_and_grad(model, base_loss_fn)

    n = min(num_samples, len(sequences))
    for i in range(n):
        inputs, targets, lengths = _make_batch(sequences, [i])
        _, grads = grad_fn(model, inputs, targets, lengths)

        for name, grad in tree_flatten(grads):
            if name in fisher:
                fisher[name] = fisher[name] + grad ** 2
        mx.eval(*fisher.values())

    for name in fisher:
        fisher[name] = fisher[name] / max(n, 1)
    mx.eval(*fisher.values())

    logger.info(f"Fisher computed over {n} samples")
    return fisher


def train_lora_ewc(
    model_name: str,
    data_path: str,
    adapter_dir: str,
    rank: int = 16,
    num_layers: int = 16,
    learning_rate: float = 1e-4,
    iters: int = 100,
    batch_size: int = 1,
    max_seq_length: int = 1024,
    ewc_lambda: float = 0.0,
    prev_adapter_path: str | None = None,
    fisher_path: str | None = None,
    old_params_path: str | None = None,
    fisher_samples: int = 100,
) -> dict:
    """Train LoRA with optional EWC regularization.

    Args:
        model_name: HuggingFace model ID.
        data_path: JSONL file with training examples ({"messages": [...]}).
        adapter_dir: Directory to save adapter weights.
        ewc_lambda: EWC penalty strength. 0 = no penalty (naive sequential).
        prev_adapter_path: Path to previous adapter's adapters.safetensors.
        fisher_path: Path to accumulated Fisher (fisher.safetensors).
        old_params_path: Path to old params snapshot (params_snapshot.safetensors).
        fisher_samples: Number of samples for Fisher computation.

    Returns:
        dict with training metrics + paths to saved Fisher and params snapshot.
    """
    data_path = Path(data_path)
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = LoRAConfig(
        rank=rank,
        num_layers=num_layers,
        learning_rate=learning_rate,
        iters=iters,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    # Load model with LoRA layers
    model, tokenizer = load(model_name)
    model.freeze()

    lora_config = {"rank": rank, "scale": rank * 2.0, "dropout": 0.05}
    linear_to_lora_layers(model, num_layers=num_layers, config=lora_config)

    # Load previous adapter weights if continuing sequential training
    if prev_adapter_path and Path(prev_adapter_path).exists():
        prev_weights = mx.load(str(prev_adapter_path))
        model.load_weights(list(prev_weights.items()), strict=False)
        logger.info(f"Loaded previous adapter from {prev_adapter_path}")

    # Load EWC components
    fisher = None
    old_params = None
    if ewc_lambda > 0 and fisher_path and old_params_path:
        fp, pp = Path(fisher_path), Path(old_params_path)
        if fp.exists() and pp.exists():
            fisher = dict(mx.load(str(fp)).items())
            old_params = dict(mx.load(str(pp)).items())
            logger.info(
                f"Loaded EWC: {len(fisher)} Fisher params, lambda={ewc_lambda}"
            )

    # Prepare data
    sequences = _prepare_sequences(data_path, tokenizer, max_seq_length)
    if not sequences:
        raise ValueError(f"No valid sequences in {data_path}")
    logger.info(f"Loaded {len(sequences)} training sequences")

    # Optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Loss function with optional EWC penalty
    def loss_fn(model, inputs, targets, lengths):
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < lengths[:, None]
        ce = nn.losses.cross_entropy(logits[:, :-1, :], targets[:, 1:]) * length_mask
        ntoks = length_mask.sum()
        base_loss = ce.sum() / ntoks

        ewc_loss = mx.array(0.0)
        if fisher is not None and old_params is not None and ewc_lambda > 0:
            trainable = dict(tree_flatten(model.trainable_parameters()))
            for name in fisher:
                if name in trainable and name in old_params:
                    diff = trainable[name] - old_params[name]
                    ewc_loss = ewc_loss + (fisher[name] * diff * diff).sum()
            ewc_loss = ewc_lambda * ewc_loss

        total_loss = base_loss + ewc_loss
        return total_loss, (ntoks, base_loss, ewc_loss)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop with NaN detection
    train_losses = []
    nan_count = 0
    n_seqs = len(sequences)

    for step in range(iters):
        batch_indices = [
            (step * batch_size + i) % n_seqs
            for i in range(min(batch_size, n_seqs))
        ]
        inputs, targets, lengths = _make_batch(sequences, batch_indices)

        (total_loss, (ntoks, base_loss, ewc_loss)), grads = loss_grad_fn(
            model, inputs, targets, lengths
        )

        # Gradient clipping to prevent explosion in sequential training
        grads = tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_val = total_loss.item()

        import math

        if math.isnan(total_val) or math.isinf(total_val):
            nan_count += 1
            if nan_count >= 5:
                logger.warning(
                    f"  Step {step + 1}: NaN/Inf detected {nan_count} times, "
                    "stopping early"
                )
                break
            continue

        train_losses.append(total_val)

        if (step + 1) % 10 == 0 or step == 0:
            base_val = base_loss.item()
            ewc_val = ewc_loss.item()
            logger.info(
                f"  Step {step + 1}/{iters}: "
                f"total={total_val:.4f} base={base_val:.4f} ewc={ewc_val:.4f}"
            )

    final_loss = train_losses[-1] if train_losses else float("nan")
    logger.info(
        f"Training complete: {len(train_losses)} valid steps, "
        f"{nan_count} NaN steps, final_loss={final_loss}"
    )

    # Save adapter
    save_adapters(model, adapter_dir, config)
    logger.info(f"Saved adapter to {adapter_dir}")

    # Compute Fisher on this batch's data for next batch
    logger.info("Computing Fisher Information Matrix...")
    new_fisher = compute_fisher(model, sequences, max_seq_length, fisher_samples)

    # Accumulate Fisher (online EWC: sum over all previous batches)
    if fisher is not None:
        for name in new_fisher:
            if name in fisher:
                new_fisher[name] = fisher[name] + new_fisher[name]
        mx.eval(*new_fisher.values())

    fisher_out = str(adapter_dir / "fisher.safetensors")
    mx.save_safetensors(fisher_out, new_fisher)

    params_out = str(adapter_dir / "params_snapshot.safetensors")
    current_params = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(params_out, current_params)

    logger.info(f"Saved Fisher to {fisher_out}")
    logger.info(f"Saved params snapshot to {params_out}")

    return {
        "final_train_loss": final_loss,
        "ewc_lambda": ewc_lambda,
        "iters": iters,
        "valid_steps": len(train_losses),
        "nan_steps": nan_count,
        "num_sequences": len(sequences),
        "fisher_path": fisher_out,
        "params_path": params_out,
    }
