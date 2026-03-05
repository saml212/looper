#!/usr/bin/env python3
"""Run Phase 1 experiment using OpenClaw as the agent framework."""

import logging
import sys
from pathlib import Path

from looper.integrations.run_openclaw_experiment import (
    OpenClawExperimentConfig,
    run_openclaw_experiment,
)
from looper.tasks.loader import load_curriculum, get_repo_tasks

# Ensure log directory exists
output_dir = Path("/Volumes/1TB_SSD/looper/results/phase1_openclaw")
output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "experiment.log"),
    ],
)

logger = logging.getLogger(__name__)


def main():
    config = OpenClawExperimentConfig(
        model_name="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        hf_model_name="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        provider_name="looper-mlx",
        provider_port=8080,
        max_steps=15,
        agent_timeout=900,
        lora_rank=16,
        lora_iters=100,
        output_dir=output_dir,
        workspace_root=Path("/Volumes/1TB_SSD/looper/cache/workspaces"),
        train_size=3,  # Small pilot to verify OpenClaw integration works
        split_seed=None,
        adapted_test_size=3,
        num_pairs_per_trajectory=5,
    )

    logger.info("=" * 60)
    logger.info("LOOPER Phase 1 — OpenClaw Integration")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Provider: {config.provider_name} on port {config.provider_port}")
    logger.info(f"Train: {config.train_size}, Adapted test: {config.adapted_test_size}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)

    # Load tasks
    curriculum_path = Path(
        "/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json"
    )
    curriculum = load_curriculum(curriculum_path)
    tasks = get_repo_tasks(curriculum, "django/django")
    logger.info(f"Loaded {len(tasks)} django tasks")

    result = run_openclaw_experiment(config, tasks)

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Forward Transfer: {result.forward_transfer:.4f}")

    base_results = [r for r in result.task_results if r.condition == "base"]
    adapted_results = [r for r in result.task_results if r.condition == "adapted"]
    base_resolved = sum(1 for r in base_results if r.resolved)
    adapted_resolved = sum(1 for r in adapted_results if r.resolved)

    logger.info(f"Base resolve rate: {base_resolved}/{len(base_results)}")
    logger.info(f"Adapted resolve rate: {adapted_resolved}/{len(adapted_results)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
