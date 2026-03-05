#!/usr/bin/env python3
"""Run Phase 1 experiment: django/django with Qwen 2.5 Coder 7B."""

import logging
import sys
from pathlib import Path

from looper.pipeline import PipelineConfig, run_phase1

# Ensure log directory exists before setting up file handler
Path("/Volumes/1TB_SSD/looper/results/phase1").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/Volumes/1TB_SSD/looper/results/phase1/experiment.log"),
    ],
)

logger = logging.getLogger(__name__)


def main():

    config = PipelineConfig(
        curriculum_path=Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json"),
        repo="django/django",
        train_size=25,
        split_seed=None,  # Chronological split: first 25 train, last 25 test
        model_name="qwen2.5-coder:7b",
        ollama_url="http://localhost:11434",
        hf_model_name="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        max_steps=15,  # 15 steps per task (enough for most fixes, saves time)
        lora_rank=16,
        lora_iters=100,
        output_dir=Path("/Volumes/1TB_SSD/looper/results/phase1"),
        workspace_root=Path("/Volumes/1TB_SSD/looper/cache/workspaces"),
        num_pairs_per_trajectory=5,
        adapted_test_size=5,  # Subset for pilot (MLX in-process is slow)
    )

    logger.info("=" * 60)
    logger.info("LOOPER Phase 1 Experiment")
    logger.info("=" * 60)
    logger.info(f"Repo: {config.repo}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Train/Test split: {config.train_size} / (50 - {config.train_size})")
    logger.info(f"Max steps per task: {config.max_steps}")
    logger.info(f"LoRA rank: {config.lora_rank}, iters: {config.lora_iters}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)

    result = run_phase1(config)

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Forward Transfer: {result.forward_transfer:.4f}")
    logger.info(f"Total task results: {len(result.task_results)}")

    base_results = [r for r in result.task_results if r.condition == "base"]
    adapted_results = [r for r in result.task_results if r.condition == "adapted"]

    base_resolved = sum(1 for r in base_results if r.resolved)
    adapted_resolved = sum(1 for r in adapted_results if r.resolved)

    logger.info(f"Base resolve rate: {base_resolved}/{len(base_results)}")
    logger.info(f"Adapted resolve rate: {adapted_resolved}/{len(adapted_results)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
