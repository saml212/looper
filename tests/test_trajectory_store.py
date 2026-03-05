"""Tests for the trajectory store (save/load/collect)."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from looper.models import (
    AgentTrajectory,
    AgentStep,
    SessionMeta,
    TaskInfo,
    ToolCall,
)
from looper.collectors.trajectory_store import (
    save_trajectory,
    load_trajectory,
    load_all_trajectories,
    collect_trajectories,
)


def make_trajectory(task_id: str = "repo__issue__1") -> AgentTrajectory:
    """Build a minimal valid AgentTrajectory for testing."""
    return AgentTrajectory(
        meta=SessionMeta(
            session_id="sess-001",
            agent_id="default",
            task_id=task_id,
            model_name="test-model",
            started_at="2025-01-01T00:00:00Z",
            ended_at="2025-01-01T00:01:00Z",
            total_tokens=100,
            total_steps=1,
        ),
        steps=[
            AgentStep(
                step_number=1,
                reasoning="Looking at the code",
                tool_calls=[
                    ToolCall(
                        tool_name="bash",
                        tool_input={"tool": "bash", "input": "ls"},
                        tool_result="file.py",
                        success=True,
                        duration_ms=50,
                    )
                ],
                timestamp="2025-01-01T00:00:30Z",
            )
        ],
        outcome="patch_generated",
        generated_patch="diff --git a/f.py b/f.py\n",
        resolve_rate=1.0,
    )


def make_task(instance_id: str = "repo__issue__1") -> TaskInfo:
    """Build a minimal valid TaskInfo for testing."""
    return TaskInfo(
        instance_id=instance_id,
        repo="test/repo",
        base_commit="abc123",
        problem_statement="Fix the bug",
        patch="diff",
        test_patch="diff",
        difficulty="easy",
        created_at="2025-01-01",
        sequence_position=0,
    )


class TestSaveTrajectory:
    def test_creates_json_file_with_correct_name(self, tmp_path: Path):
        traj = make_trajectory("myrepo__issue__42")
        result_path = save_trajectory(traj, tmp_path)

        assert result_path == tmp_path / "myrepo__issue__42.json"
        assert result_path.exists()

    def test_file_contains_valid_json(self, tmp_path: Path):
        traj = make_trajectory()
        path = save_trajectory(traj, tmp_path)

        data = json.loads(path.read_text())
        assert data["meta"]["task_id"] == "repo__issue__1"
        assert data["outcome"] == "patch_generated"

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        traj = make_trajectory()
        path = save_trajectory(traj, nested)

        assert path.exists()


class TestLoadTrajectory:
    def test_loads_from_json(self, tmp_path: Path):
        traj = make_trajectory()
        path = save_trajectory(traj, tmp_path)

        loaded = load_trajectory(path)
        assert isinstance(loaded, AgentTrajectory)
        assert loaded.meta.task_id == "repo__issue__1"

    def test_round_trip_equivalence(self, tmp_path: Path):
        original = make_trajectory()
        path = save_trajectory(original, tmp_path)
        loaded = load_trajectory(path)

        assert loaded == original

    def test_preserves_all_fields(self, tmp_path: Path):
        original = make_trajectory()
        path = save_trajectory(original, tmp_path)
        loaded = load_trajectory(path)

        assert loaded.meta.session_id == original.meta.session_id
        assert loaded.steps[0].reasoning == original.steps[0].reasoning
        assert loaded.steps[0].tool_calls[0].tool_name == "bash"
        assert loaded.generated_patch == original.generated_patch
        assert loaded.resolve_rate == original.resolve_rate


class TestLoadAllTrajectories:
    def test_loads_multiple_sorted_by_task_id(self, tmp_path: Path):
        save_trajectory(make_trajectory("z_task"), tmp_path)
        save_trajectory(make_trajectory("a_task"), tmp_path)
        save_trajectory(make_trajectory("m_task"), tmp_path)

        results = load_all_trajectories(tmp_path)

        assert len(results) == 3
        assert results[0].meta.task_id == "a_task"
        assert results[1].meta.task_id == "m_task"
        assert results[2].meta.task_id == "z_task"

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        results = load_all_trajectories(tmp_path)
        assert results == []

    def test_ignores_non_json_files(self, tmp_path: Path):
        save_trajectory(make_trajectory("task_1"), tmp_path)
        (tmp_path / "notes.txt").write_text("not a trajectory")

        results = load_all_trajectories(tmp_path)
        assert len(results) == 1


class TestCollectTrajectories:
    @patch("looper.collectors.trajectory_store.run_agent")
    def test_runs_agent_and_saves_trajectories(self, mock_run, tmp_path: Path):
        tasks = [make_task("task_a"), make_task("task_b")]
        mock_run.side_effect = [make_trajectory("task_a"), make_trajectory("task_b")]

        results = collect_trajectories(
            tasks=tasks,
            output_dir=tmp_path / "out",
            workspace_root=tmp_path / "ws",
        )

        assert len(results) == 2
        assert mock_run.call_count == 2
        assert (tmp_path / "out" / "task_a.json").exists()
        assert (tmp_path / "out" / "task_b.json").exists()

    @patch("looper.collectors.trajectory_store.run_agent")
    def test_calls_on_complete_callback(self, mock_run, tmp_path: Path):
        tasks = [make_task("task_x")]
        mock_run.return_value = make_trajectory("task_x")

        callback = MagicMock()
        collect_trajectories(
            tasks=tasks,
            output_dir=tmp_path / "out",
            workspace_root=tmp_path / "ws",
            on_complete=callback,
        )

        callback.assert_called_once()
        call_args = callback.call_args
        assert call_args[0][0] == "task_x"
        assert isinstance(call_args[0][1], AgentTrajectory)

    @patch("looper.collectors.trajectory_store.run_agent")
    def test_skips_existing_trajectory_files(self, mock_run, tmp_path: Path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Pre-save one trajectory so it already exists on disk
        existing = make_trajectory("task_a")
        save_trajectory(existing, out_dir)

        tasks = [make_task("task_a"), make_task("task_b")]
        mock_run.return_value = make_trajectory("task_b")

        results = collect_trajectories(
            tasks=tasks,
            output_dir=out_dir,
            workspace_root=tmp_path / "ws",
        )

        # Only task_b should have triggered the agent
        mock_run.assert_called_once()
        # But both trajectories should be returned
        assert len(results) == 2

    @patch("looper.collectors.trajectory_store.run_agent")
    def test_passes_model_and_config_to_agent(self, mock_run, tmp_path: Path):
        tasks = [make_task("task_1")]
        mock_run.return_value = make_trajectory("task_1")

        collect_trajectories(
            tasks=tasks,
            output_dir=tmp_path / "out",
            workspace_root=tmp_path / "ws",
            model="custom-model",
            base_url="http://custom:1234",
            max_steps=10,
        )

        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["model"] == "custom-model"
        assert call_kwargs[1]["base_url"] == "http://custom:1234"
        assert call_kwargs[1]["max_steps"] == 10
