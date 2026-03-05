"""Phase 1 pipeline orchestrator.

Wires all components together to run the complete Phase 1 experiment:
load tasks, run base model, synthesize training data, train LoRA,
run adapted model, evaluate, and save results.
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from looper.collectors.trajectory_store import collect_trajectories
from looper.evaluators.metrics import forward_transfer
from looper.evaluators.patch_verifier import verify_patch_simple, verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import (
    ExperimentConfig,
    ExperimentResult,
    TaskResult,
)
from looper.synthesizers.synthesizer import (
    pairs_to_training_examples,
    save_training_data,
    synthesize_batch,
)
from looper.tasks.loader import get_repo_tasks, load_curriculum, split_tasks
from looper.trainers.full_replay import full_replay_train
from looper.trainers.lora_trainer import LoRAConfig
# serve_adapter/cleanup_ollama_model unused — using MLX direct inference instead

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a Phase 1 pipeline run."""

    # Data
    curriculum_path: Path = field(default_factory=lambda: Path("curriculum.json"))
    repo: str = "django/django"
    train_size: int = 25
    split_seed: int | None = None  # None = chronological

    # Model
    model_name: str = "qwen2.5-coder:7b"
    ollama_url: str = "http://localhost:11434"
    hf_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Agent
    max_steps: int = 25

    # Training
    lora_rank: int = 16
    lora_iters: int = 100

    # Paths (on SSD)
    output_dir: Path = field(
        default_factory=lambda: Path("/Volumes/1TB_SSD/looper/results/phase1")
    )
    workspace_root: Path = field(
        default_factory=lambda: Path("/Volumes/1TB_SSD/looper/cache/workspaces")
    )

    # Synthesis
    num_pairs_per_trajectory: int = 5

    # Adapted model inference
    adapted_test_size: int | None = None  # None = all test tasks, int = subset


def run_phase1(config: PipelineConfig) -> ExperimentResult:
    """Run the complete Phase 1 experiment.

    Steps:
    1. Load tasks and split into train/test
    2. Run base model on all tasks, collect trajectories
    3. Evaluate base model on test tasks
    4. Synthesize training data from train trajectories
    5. Train LoRA adapter
    6. Run adapted model on test tasks
    7. Evaluate adapted model and compute forward transfer
    8. Save results
    """
    started_at = datetime.now(timezone.utc).isoformat()

    # Create output directories
    base_traj_dir = config.output_dir / "trajectories" / "base"
    adapted_traj_dir = config.output_dir / "trajectories" / "adapted"
    synthesis_dir = config.output_dir / "synthesis"
    adapter_dir = config.output_dir / "adapter"

    for d in [base_traj_dir, adapted_traj_dir, synthesis_dir, adapter_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and split tasks
    logger.info("Step 1: Loading tasks...")
    curriculum = load_curriculum(config.curriculum_path)
    all_tasks = get_repo_tasks(curriculum, config.repo)
    train_tasks, test_tasks = split_tasks(
        all_tasks, config.train_size, config.split_seed
    )

    logger.info(f"Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")

    # Step 2: Run base model on ALL tasks
    logger.info("Step 2: Running base model on all tasks...")
    all_base_trajectories = collect_trajectories(
        tasks=all_tasks,
        output_dir=base_traj_dir,
        workspace_root=config.workspace_root,
        model=config.model_name,
        base_url=config.ollama_url,
        max_steps=config.max_steps,
        on_complete=lambda tid, traj: logger.info(f"  Base: {tid} -> {traj.outcome}"),
    )

    # Separate train and test trajectories
    train_task_ids = {t.instance_id for t in train_tasks}
    test_task_ids = {t.instance_id for t in test_tasks}
    train_trajectories = [
        t for t in all_base_trajectories if t.meta.task_id in train_task_ids
    ]
    test_base_trajectories = [
        t for t in all_base_trajectories if t.meta.task_id in test_task_ids
    ]

    # Step 3: Evaluate base model on test tasks using FAIL_TO_PASS tests
    logger.info("Step 3: Evaluating base model on test tasks (running FAIL_TO_PASS tests)...")
    base_results: list[TaskResult] = []
    for traj in test_base_trajectories:
        task = next(
            (t for t in test_tasks if t.instance_id == traj.meta.task_id), None
        )
        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(
                task, traj.generated_patch, config.workspace_root
            )
            resolved = vr["resolved"]
            logger.info(
                f"  Verify {traj.meta.task_id}: "
                f"{'PASS' if resolved else 'FAIL'} "
                f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
                + (f" error={vr['error']}" if vr["error"] else "")
            )
        else:
            resolved = False
        base_results.append(
            TaskResult(
                task_id=traj.meta.task_id,
                condition="base",
                resolved=resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            )
        )

    # Step 4: Synthesize training data from train trajectories
    training_jsonl = synthesis_dir / "training.jsonl"
    if training_jsonl.exists():
        logger.info("Step 4: Loading cached training data...")
        from looper.synthesizers.synthesizer import load_training_data

        training_examples = load_training_data(training_jsonl)
        logger.info(f"  Loaded {len(training_examples)} cached training examples")
    else:
        logger.info("Step 4: Synthesizing training data...")
        pairs = synthesize_batch(
            trajectories=train_trajectories,
            output_path=synthesis_dir / "pairs.json",
            model=config.model_name,
            base_url=config.ollama_url,
            num_pairs=config.num_pairs_per_trajectory,
        )

        training_examples = pairs_to_training_examples(pairs)
        save_training_data(training_examples, synthesis_dir / "training.jsonl")
        logger.info(
            f"  Synthesized {len(pairs)} pairs -> {len(training_examples)} training examples"
        )

    # Step 5: Train LoRA adapter (skip if no training data)
    adapted_model_name = config.model_name  # Default: use base model
    if training_examples:
        # Stop Ollama to free GPU memory for training
        logger.info("Step 5: Stopping Ollama to free GPU memory for training...")
        import subprocess as _sp

        _sp.run(["ollama", "stop", config.model_name], capture_output=True)

        # Check if adapter already exists (skip re-training)
        adapter_file = adapter_dir / "adapters.safetensors"
        if adapter_file.exists():
            logger.info("Step 5: Adapter already trained, skipping...")
            train_metrics = {"cached": True}
        else:
            logger.info("Step 5: Training LoRA adapter in subprocess...")
            # Run training in a subprocess so GPU memory is fully freed afterwards
            import json as _json

            train_script = f"""
import sys
sys.path.insert(0, '.')
from looper.trainers.full_replay import full_replay_train
from looper.trainers.lora_trainer import LoRAConfig
from looper.synthesizers.synthesizer import load_training_data
from pathlib import Path
import json

examples = load_training_data(Path('{synthesis_dir / "training.jsonl"}'))
config = LoRAConfig(rank={config.lora_rank}, iters={config.lora_iters}, batch_size=1, max_seq_length=1024)
metrics = full_replay_train(examples, '{config.hf_model_name}', Path('{adapter_dir}'), config)
print(json.dumps(metrics))
"""
            train_result = _sp.run(
                [sys.executable, "-c", train_script],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent),
            )
            if train_result.returncode != 0:
                logger.error(f"Training failed: {train_result.stderr[-500:]}")
                raise RuntimeError("LoRA training failed")
            # Parse metrics from last line of stdout
            for line in reversed(train_result.stdout.strip().split("\n")):
                try:
                    train_metrics = _json.loads(line)
                    break
                except _json.JSONDecodeError:
                    continue
            else:
                train_metrics = {}
        logger.info(f"  Training metrics: {train_metrics}")

        # Step 5b: Load adapted model in-process via MLX
        adapted_model_name = config.hf_model_name
        logger.info("Step 5b: Loading adapted model in-process via MLX...")
        from looper.agent.ollama_client import load_mlx_model, mlx_chat

        load_mlx_model(config.hf_model_name, adapter_path=str(adapter_dir))
        logger.info("  Adapted model loaded in-process")
        adapted_chat_fn_to_use = mlx_chat
    else:
        logger.warning(
            "Step 5: SKIPPED — no training examples from synthesis. "
            "Adapted model will use the base model for comparison."
        )
        adapted_chat_fn_to_use = None  # Use default Ollama chat

    # Step 6: Run adapted model on test tasks
    adapted_test_tasks = test_tasks
    if config.adapted_test_size is not None:
        adapted_test_tasks = test_tasks[: config.adapted_test_size]
    logger.info(
        f"Step 6: Running adapted model on {len(adapted_test_tasks)} test tasks..."
    )
    adapted_trajectories = collect_trajectories(
        tasks=adapted_test_tasks,
        output_dir=adapted_traj_dir,
        workspace_root=config.workspace_root,
        model=adapted_model_name,
        base_url=config.ollama_url,
        max_steps=config.max_steps,
        max_tokens=512,  # Agent tool calls are short
        on_complete=lambda tid, traj: logger.info(
            f"  Adapted: {tid} -> {traj.outcome}"
        ),
        chat_fn=adapted_chat_fn_to_use,
    )

    # Step 7: Evaluate adapted model on test tasks using FAIL_TO_PASS tests
    logger.info("Step 7: Evaluating adapted model (running FAIL_TO_PASS tests)...")
    adapted_results: list[TaskResult] = []
    for traj in adapted_trajectories:
        task = next(
            (t for t in test_tasks if t.instance_id == traj.meta.task_id), None
        )
        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(
                task, traj.generated_patch, config.workspace_root
            )
            resolved = vr["resolved"]
            logger.info(
                f"  Verify {traj.meta.task_id}: "
                f"{'PASS' if resolved else 'FAIL'} "
                f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
                + (f" error={vr['error']}" if vr["error"] else "")
            )
        else:
            resolved = False
        adapted_results.append(
            TaskResult(
                task_id=traj.meta.task_id,
                condition="adapted",
                resolved=resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            )
        )

    # Filter base results to match adapted test subset for fair comparison
    adapted_task_ids = {r.task_id for r in adapted_results}
    base_results_matched = [r for r in base_results if r.task_id in adapted_task_ids]
    ft = forward_transfer(base_results_matched, adapted_results)

    # Step 8: Build and save results
    logger.info("Step 8: Saving results...")
    experiment_config = ExperimentConfig(
        name="phase1_pilot",
        experiment_id="phase1_full_replay",
        repo=config.repo,
        model_name=config.model_name,
        train_task_ids=[t.instance_id for t in train_tasks],
        test_task_ids=[t.instance_id for t in test_tasks],
        strategy="full_replay",
        lora_rank=config.lora_rank,
        seed=config.split_seed or 0,
    )

    result = ExperimentResult(
        config=experiment_config,
        task_results=base_results + adapted_results,
        forward_transfer=ft,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    save_results(result, config.output_dir / "experiment_result.json")
    summary = results_summary(result)
    logger.info(f"\n{summary}")

    return result
