"""Tests for the pipeline orchestrator and MLX runner."""

from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from looper.models import (
    TaskInfo,
    AgentTrajectory,
    AgentStep,
    ToolCall,
    SessionMeta,
    SynthesizedPair,
    TrainingExample,
    ExperimentConfig,
    ExperimentResult,
    TaskResult,
)
from looper.pipeline import PipelineConfig, run_phase1
from looper.agent.mlx_runner import run_agent_mlx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(instance_id: str, pos: int = 0) -> TaskInfo:
    return TaskInfo(
        instance_id=instance_id,
        repo="django/django",
        base_commit="a" * 40,
        problem_statement=f"Fix {instance_id}",
        patch=f"diff --git a/django/foo.py b/django/foo.py\n--- a/django/foo.py\n+++ b/django/foo.py",
        test_patch="",
        difficulty="easy",
        created_at="2025-01-01",
        sequence_position=pos,
    )


def _make_trajectory(task_id: str, model: str = "qwen2.5-coder:7b") -> AgentTrajectory:
    return AgentTrajectory(
        meta=SessionMeta(
            session_id=f"sess-{task_id}",
            task_id=task_id,
            model_name=model,
            started_at="2025-01-01T00:00:00Z",
            ended_at="2025-01-01T00:01:00Z",
            total_tokens=500,
            total_steps=3,
        ),
        steps=[
            AgentStep(
                step_number=1,
                reasoning="Looking at code",
                tool_calls=[
                    ToolCall(
                        tool_name="bash",
                        tool_input={"tool": "bash", "input": "ls"},
                        tool_result="foo.py",
                        success=True,
                    )
                ],
            ),
        ],
        outcome="patch_generated",
        generated_patch=f"diff --git a/django/foo.py b/django/foo.py\n--- a/django/foo.py\n+++ b/django/foo.py\n@@ -1 +1 @@\n-old\n+new",
    )


def _make_pair(task_id: str) -> SynthesizedPair:
    return SynthesizedPair(
        instruction=f"How to fix {task_id}?",
        response=f"Fix by editing the file for {task_id}.",
        pair_type="tool_usage",
        confidence=0.8,
        source_session_id=f"sess-{task_id}",
        source_task_id=task_id,
    )


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_values(self):
        cfg = PipelineConfig(curriculum_path=Path("/tmp/curriculum.json"))
        assert cfg.repo == "django/django"
        assert cfg.train_size == 25
        assert cfg.split_seed is None
        assert cfg.model_name == "qwen2.5-coder:7b"
        assert cfg.ollama_url == "http://localhost:11434"
        assert cfg.hf_model_name == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert cfg.max_steps == 25
        assert cfg.lora_rank == 16
        assert cfg.lora_iters == 100
        assert cfg.num_pairs_per_trajectory == 5

    def test_custom_values(self):
        cfg = PipelineConfig(
            curriculum_path=Path("/data/tasks.json"),
            repo="astropy/astropy",
            train_size=10,
            split_seed=42,
            model_name="llama3:8b",
            ollama_url="http://localhost:9999",
            hf_model_name="meta-llama/Llama-3-8B-Instruct",
            max_steps=50,
            lora_rank=32,
            lora_iters=200,
            output_dir=Path("/tmp/results"),
            workspace_root=Path("/tmp/workspaces"),
            num_pairs_per_trajectory=10,
        )
        assert cfg.repo == "astropy/astropy"
        assert cfg.train_size == 10
        assert cfg.split_seed == 42
        assert cfg.model_name == "llama3:8b"
        assert cfg.lora_rank == 32
        assert cfg.lora_iters == 200


# ---------------------------------------------------------------------------
# run_phase1 integration test (fully mocked)
# ---------------------------------------------------------------------------


class TestRunPhase1:
    def test_full_pipeline_mocked(self, tmp_path):
        """Full mock integration test: all external calls are mocked.

        Verifies that the pipeline:
        - Loads tasks correctly
        - Calls collect_trajectories for base model
        - Calls synthesize_batch with train trajectories
        - Calls full_replay_train with training examples
        - Calls collect_trajectories for adapted model on test tasks
        - Produces an ExperimentResult with correct structure
        - Saves results to disk
        """
        # Create 4 tasks (2 train, 2 test for simplicity)
        all_tasks = [_make_task(f"django__django-{i}", pos=i) for i in range(1, 5)]
        train_tasks = all_tasks[:2]
        test_tasks = all_tasks[2:]

        # Trajectories for base model (all 4 tasks)
        base_trajectories = [_make_trajectory(t.instance_id) for t in all_tasks]

        # Trajectories for adapted model (test tasks only)
        adapted_trajectories = [
            _make_trajectory(t.instance_id, model="adapted") for t in test_tasks
        ]

        # Synthesized pairs from train trajectories
        pairs = [_make_pair(t.instance_id) for t in train_tasks]

        # Training examples
        training_examples = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": p.instruction},
                    {"role": "assistant", "content": p.response},
                ]
            )
            for p in pairs
        ]

        # Build curriculum JSON fixture
        curriculum = {
            "sequences": [
                {
                    "repo": "django/django",
                    "tasks": [
                        {
                            "metadata": {
                                "instance_id": t.instance_id,
                                "repo": t.repo,
                                "base_commit": t.base_commit,
                                "difficulty": t.difficulty,
                                "created_at": t.created_at,
                            },
                            "task": {
                                "problem_statement": t.problem_statement,
                                "hints_text": "",
                            },
                            "evaluation": {
                                "patch": t.patch,
                                "test_patch": t.test_patch,
                            },
                            "continual_learning": {
                                "sequence_position": t.sequence_position,
                            },
                        }
                        for t in all_tasks
                    ],
                }
            ]
        }

        import json
        curriculum_path = tmp_path / "curriculum.json"
        curriculum_path.write_text(json.dumps(curriculum))

        config = PipelineConfig(
            curriculum_path=curriculum_path,
            train_size=2,
            output_dir=tmp_path / "results",
            workspace_root=tmp_path / "workspaces",
        )

        # Mock all external dependencies
        # Pre-create the adapter file so training is skipped
        adapter_dir = config.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapters.safetensors").write_text("fake")

        with (
            patch("looper.pipeline.collect_trajectories") as mock_collect,
            patch("looper.pipeline.synthesize_batch") as mock_synthesize,
            patch("looper.pipeline.pairs_to_training_examples") as mock_pairs_to_examples,
            patch("looper.pipeline.save_training_data") as mock_save_training,
            patch("looper.pipeline.verify_patch_simple") as mock_verify,
            patch("looper.pipeline.save_results") as mock_save_results,
            patch("looper.agent.ollama_client.load_mlx_model"),
        ):
            # Configure mocks
            # First call: base model on all tasks; second call: adapted on test tasks
            mock_collect.side_effect = [base_trajectories, adapted_trajectories]
            mock_synthesize.return_value = pairs
            mock_pairs_to_examples.return_value = training_examples
            # verify_patch_simple: let's say first test task resolves, second doesn't
            mock_verify.side_effect = [True, False, True, False]  # base test1, base test2, adapted test1, adapted test2

            result = run_phase1(config)

        # --- Assertions ---

        # 1. collect_trajectories called twice (base + adapted)
        assert mock_collect.call_count == 2

        # First call: base model on all tasks
        base_call = mock_collect.call_args_list[0]
        assert len(base_call.kwargs["tasks"]) == 4
        assert base_call.kwargs["model"] == "qwen2.5-coder:7b"

        # Second call: adapted model on test tasks only
        adapted_call = mock_collect.call_args_list[1]
        assert len(adapted_call.kwargs["tasks"]) == 2
        # Adapted model name is the HF model name (served via MLX server)
        assert adapted_call.kwargs["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"

        # 2. synthesize_batch called with train trajectories
        mock_synthesize.assert_called_once()
        synth_call = mock_synthesize.call_args
        synth_trajectories = synth_call.kwargs["trajectories"]
        assert len(synth_trajectories) == 2
        synth_task_ids = {t.meta.task_id for t in synth_trajectories}
        assert synth_task_ids == {t.instance_id for t in train_tasks}

        # 3. Training was skipped (adapter pre-exists)
        # The pipeline detects existing adapter and skips training

        # 4. Result structure is correct
        assert isinstance(result, ExperimentResult)
        assert result.config.name == "phase1_pilot"
        assert result.config.experiment_id == "phase1_full_replay"
        assert result.config.repo == "django/django"
        assert len(result.config.train_task_ids) == 2
        assert len(result.config.test_task_ids) == 2

        # 5. Task results: 2 base + 2 adapted = 4
        assert len(result.task_results) == 4
        base_results = [r for r in result.task_results if r.condition == "base"]
        adapted_results = [r for r in result.task_results if r.condition == "adapted"]
        assert len(base_results) == 2
        assert len(adapted_results) == 2

        # 6. Forward transfer computed
        # base: 1/2 resolved = 0.5, adapted: 1/2 = 0.5 -> FT = 0.0
        assert result.forward_transfer == pytest.approx(0.0)

        # 7. save_results called
        mock_save_results.assert_called_once()

    def test_output_dirs_created(self, tmp_path):
        """Pipeline creates output directories."""
        import json

        curriculum = {
            "sequences": [
                {
                    "repo": "django/django",
                    "tasks": [
                        {
                            "metadata": {
                                "instance_id": f"django__django-{i}",
                                "repo": "django/django",
                                "base_commit": "a" * 40,
                                "difficulty": "easy",
                                "created_at": "2025-01-01",
                            },
                            "task": {"problem_statement": "fix", "hints_text": ""},
                            "evaluation": {"patch": "diff --git a/f.py b/f.py", "test_patch": ""},
                            "continual_learning": {"sequence_position": i},
                        }
                        for i in range(1, 5)
                    ],
                }
            ]
        }

        curriculum_path = tmp_path / "curriculum.json"
        curriculum_path.write_text(json.dumps(curriculum))

        config = PipelineConfig(
            curriculum_path=curriculum_path,
            train_size=2,
            output_dir=tmp_path / "results",
            workspace_root=tmp_path / "workspaces",
        )

        with (
            patch("looper.pipeline.collect_trajectories") as mock_collect,
            patch("looper.pipeline.synthesize_batch") as mock_synthesize,
            patch("looper.pipeline.pairs_to_training_examples") as mock_p2e,
            patch("looper.pipeline.save_training_data"),
            patch("looper.pipeline.full_replay_train") as mock_train,
            patch("looper.pipeline.verify_patch_simple", return_value=False),
            patch("looper.pipeline.save_results"),
            patch("looper.agent.ollama_client.load_mlx_model"),
        ):
            mock_collect.side_effect = [
                [_make_trajectory(f"django__django-{i}") for i in range(1, 5)],
                [_make_trajectory(f"django__django-{i}") for i in range(3, 5)],
            ]
            mock_synthesize.return_value = []
            mock_p2e.return_value = []
            mock_train.return_value = {"final_train_loss": 0.5, "final_val_loss": 0.6, "iters": 100}

            run_phase1(config)

        # Verify directories were created
        assert (tmp_path / "results" / "trajectories" / "base").is_dir()
        assert (tmp_path / "results" / "trajectories" / "adapted").is_dir()
        assert (tmp_path / "results" / "synthesis").is_dir()
        assert (tmp_path / "results" / "adapter").is_dir()


# ---------------------------------------------------------------------------
# run_agent_mlx tests
# ---------------------------------------------------------------------------


class TestRunAgentMlx:
    def test_integration_mocked(self, tmp_path, monkeypatch):
        """Mock test verifying MLX model loading with optional adapter."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        (workspace_dir / "main.py").write_text("print('hello')")

        # Mock workspace creation
        monkeypatch.setattr(
            "looper.agent.mlx_runner.create_workspace",
            lambda repo, base_commit, workspace_root: workspace_dir,
        )

        # Mock get_patch
        monkeypatch.setattr(
            "looper.agent.mlx_runner.get_patch",
            lambda ws: "diff --git a/main.py b/main.py\n--- a/main.py\n+++ b/main.py",
        )

        # Mock MLX model loading
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        monkeypatch.setattr(
            "looper.agent.mlx_runner.load_model_with_adapter",
            lambda model_name, adapter_path=None: (mock_model, mock_tokenizer),
        )

        # Mock mlx_lm.generate to produce tool calls then <done>
        call_count = 0

        def mock_generate(model, tokenizer, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "<bash>echo hello</bash>"
            else:
                return "<done>"

        monkeypatch.setattr("looper.agent.mlx_runner.mlx_generate", mock_generate)

        task = _make_task("django__django-99")
        trajectory = run_agent_mlx(
            task=task,
            workspace_root=tmp_path,
            hf_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=None,
            max_steps=5,
        )

        assert isinstance(trajectory, AgentTrajectory)
        assert trajectory.meta.task_id == "django__django-99"
        assert trajectory.meta.total_steps == 2
        assert trajectory.outcome in ("completed", "patch_generated")

    def test_with_adapter_path(self, tmp_path, monkeypatch):
        """Verify adapter_path is passed through to load_model_with_adapter."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()

        monkeypatch.setattr(
            "looper.agent.mlx_runner.create_workspace",
            lambda repo, base_commit, workspace_root: workspace_dir,
        )
        monkeypatch.setattr(
            "looper.agent.mlx_runner.get_patch",
            lambda ws: "",
        )

        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()

        load_calls = []

        def mock_load(model_name, adapter_path=None):
            load_calls.append({"model_name": model_name, "adapter_path": adapter_path})
            return MagicMock(), MagicMock()

        monkeypatch.setattr("looper.agent.mlx_runner.load_model_with_adapter", mock_load)
        monkeypatch.setattr(
            "looper.agent.mlx_runner.mlx_generate",
            lambda model, tokenizer, prompt, **kw: "<done>",
        )

        task = _make_task("django__django-100")
        run_agent_mlx(
            task=task,
            workspace_root=tmp_path,
            hf_model_name="test-model",
            adapter_path=adapter_path,
            max_steps=3,
        )

        assert len(load_calls) == 1
        assert load_calls[0]["adapter_path"] == adapter_path

    def test_max_steps_enforced(self, tmp_path, monkeypatch):
        """Agent stops after max_steps without <done>."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()

        monkeypatch.setattr(
            "looper.agent.mlx_runner.create_workspace",
            lambda repo, base_commit, workspace_root: workspace_dir,
        )
        monkeypatch.setattr(
            "looper.agent.mlx_runner.get_patch",
            lambda ws: "",
        )
        monkeypatch.setattr(
            "looper.agent.mlx_runner.load_model_with_adapter",
            lambda model_name, adapter_path=None: (MagicMock(), MagicMock()),
        )
        monkeypatch.setattr(
            "looper.agent.mlx_runner.mlx_generate",
            lambda model, tokenizer, prompt, **kw: "<bash>echo looping</bash>",
        )

        task = _make_task("django__django-101")
        trajectory = run_agent_mlx(
            task=task,
            workspace_root=tmp_path,
            hf_model_name="test-model",
            max_steps=3,
        )

        assert trajectory.meta.total_steps == 3
        assert trajectory.outcome == "max_steps"
