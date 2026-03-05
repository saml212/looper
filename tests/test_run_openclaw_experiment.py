"""Tests for the OpenClaw experiment orchestrator."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from looper.integrations.run_openclaw_experiment import (
    render_skill,
    run_openclaw_on_task,
    collect_task_trajectory,
    find_session_file,
    OpenClawExperimentConfig,
)
from looper.models import TaskInfo, AgentTrajectory, SessionMeta, AgentStep, ToolCall


def _make_task(instance_id="django__django-12304") -> TaskInfo:
    return TaskInfo(
        instance_id=instance_id,
        repo="django/django",
        base_commit="abc123" + "0" * 34,
        problem_statement="Fix the aggregation bug.",
        patch="diff --git a/test.py",
        test_patch="diff --git a/test.py",
        difficulty="easy",
        created_at="2026-01-01",
        sequence_position=1,
    )


def _make_trajectory(task_id="django__django-12304", outcome="completed") -> AgentTrajectory:
    return AgentTrajectory(
        meta=SessionMeta(
            session_id="sess-001",
            task_id=task_id,
            model_name="qwen2.5-coder:7b",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:01:00Z",
            total_tokens=5000,
            total_steps=2,
        ),
        steps=[
            AgentStep(
                step_number=1,
                reasoning="exploring",
                tool_calls=[
                    ToolCall(
                        tool_name="bash",
                        tool_input={"command": "ls"},
                        tool_result="file.py",
                        success=True,
                        duration_ms=50,
                    )
                ],
            ),
        ],
        outcome=outcome,
        generated_patch="diff --git a/fix.py",
    )


class TestRenderSkill:
    def test_renders_placeholders(self):
        result = render_skill(
            problem_statement="Fix the bug",
            workspace_dir=Path("/tmp/workspace"),
        )
        assert "Fix the bug" in result
        assert "/tmp/workspace" in result
        assert "{{problem_statement}}" not in result
        assert "{{workspace_dir}}" not in result

    def test_preserves_structure(self):
        result = render_skill(
            problem_statement="test",
            workspace_dir=Path("/test"),
        )
        assert "# SWE-Bench Task Solver" in result
        assert "ONE tool per response" in result


class TestRunOpenClawOnTask:
    @patch("looper.integrations.run_openclaw_experiment._openclaw_agent_turn")
    @patch("looper.integrations.run_openclaw_experiment.create_workspace")
    def test_multi_turn_loop(self, mock_workspace, mock_turn, tmp_path):
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        mock_workspace.return_value = workspace_dir

        # Simulate: first turn does bash, second turn does <done>
        mock_turn.side_effect = [
            '<bash>echo hello</bash>',
            '<done>',
        ]

        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        task = _make_task()
        session_id, ws = run_openclaw_on_task(
            task=task,
            workspace_root=tmp_path,
            provider_name="looper-mlx",
            model_name="test-model",
            skill_dir=skill_dir,
            max_steps=5,
        )

        assert len(session_id) > 0
        assert ws == workspace_dir
        assert mock_turn.call_count == 2

    @patch("looper.integrations.run_openclaw_experiment._openclaw_agent_turn")
    @patch("looper.integrations.run_openclaw_experiment.create_workspace")
    def test_writes_rendered_skill(self, mock_workspace, mock_turn, tmp_path):
        mock_workspace.return_value = tmp_path / "workspace"
        (tmp_path / "workspace").mkdir()
        mock_turn.return_value = '<done>'

        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        task = _make_task()
        run_openclaw_on_task(
            task=task,
            workspace_root=tmp_path,
            provider_name="looper-mlx",
            model_name="test-model",
            skill_dir=skill_dir,
        )

        rendered = (skill_dir / "SKILL.md").read_text()
        assert "Fix the aggregation bug" in rendered

    @patch("looper.integrations.run_openclaw_experiment._openclaw_agent_turn")
    @patch("looper.integrations.run_openclaw_experiment.create_workspace")
    def test_max_steps_enforced(self, mock_workspace, mock_turn, tmp_path):
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        mock_workspace.return_value = workspace_dir

        # Never sends <done> — always does bash
        mock_turn.return_value = '<bash>echo looping</bash>'

        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        task = _make_task()
        run_openclaw_on_task(
            task=task,
            workspace_root=tmp_path,
            provider_name="looper-mlx",
            model_name="test-model",
            skill_dir=skill_dir,
            max_steps=3,
        )
        assert mock_turn.call_count == 3


class TestFindSessionFile:
    def test_finds_by_exact_name(self, tmp_path):
        session_file = tmp_path / "abc123.jsonl"
        session_file.write_text('{"type":"session","id":"abc123"}')
        result = find_session_file(tmp_path, "abc123")
        assert result == session_file

    def test_returns_none_when_missing(self, tmp_path):
        result = find_session_file(tmp_path, "nonexistent")
        assert result is None


class TestCollectTaskTrajectory:
    @patch("looper.integrations.run_openclaw_experiment.parse_session")
    def test_collects_trajectory(self, mock_parse, tmp_path):
        session_file = tmp_path / "sess-001.jsonl"
        session_file.write_text('{"type":"session"}')

        traj = _make_trajectory()
        mock_parse.return_value = traj

        result = collect_task_trajectory(
            tmp_path, "sess-001", tmp_path / "workspace", "task-1"
        )
        assert result is not None
        assert result.meta.task_id == "task-1"

    def test_returns_none_when_session_missing(self, tmp_path):
        result = collect_task_trajectory(
            tmp_path, "nonexistent", tmp_path / "workspace", "task-1"
        )
        assert result is None


class TestOpenClawExperimentConfig:
    def test_default_values(self):
        config = OpenClawExperimentConfig()
        assert config.provider_port == 8080
        assert config.max_steps == 15
        assert config.adapted_test_size == 5

    def test_custom_values(self):
        config = OpenClawExperimentConfig(
            provider_port=9090,
            model_name="test-model",
        )
        assert config.provider_port == 9090
        assert config.model_name == "test-model"
