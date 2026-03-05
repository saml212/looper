"""Tests for the agent runner: parse_tool_calls, execute_tool, run_agent."""

import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from looper.agent.runner import parse_tool_calls, execute_tool, run_agent
from looper.agent.ollama_client import ChatMessage, ChatResponse
from looper.models import TaskInfo, AgentTrajectory


# ---------------------------------------------------------------------------
# parse_tool_calls tests
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_parse_single_bash(self):
        text = "<bash>ls -la</bash>"
        result = parse_tool_calls(text)
        assert result == [{"tool": "bash", "input": "ls -la"}]

    def test_parse_single_read(self):
        text = "<read>src/main.py</read>"
        result = parse_tool_calls(text)
        assert result == [{"tool": "read", "input": "src/main.py"}]

    def test_parse_write(self):
        text = '<write path="src/main.py">content here</write>'
        result = parse_tool_calls(text)
        assert result == [{"tool": "write", "input": "content here", "path": "src/main.py"}]

    def test_parse_done(self):
        text = "<done>"
        result = parse_tool_calls(text)
        assert result == [{"tool": "done", "input": ""}]

    def test_parse_multiple_tool_calls(self):
        text = "<bash>ls</bash>\n<read>foo.py</read>\n<done>"
        result = parse_tool_calls(text)
        assert len(result) == 3
        assert result[0]["tool"] == "bash"
        assert result[1]["tool"] == "read"
        assert result[2]["tool"] == "done"

    def test_no_tool_calls(self):
        text = "I'm thinking about what to do next."
        result = parse_tool_calls(text)
        assert result == []

    def test_tool_calls_mixed_with_reasoning(self):
        text = (
            "Let me look at the file structure first.\n"
            "<bash>find . -name '*.py'</bash>\n"
            "Now let me read the main file.\n"
            "<read>main.py</read>"
        )
        result = parse_tool_calls(text)
        assert len(result) == 2
        assert result[0] == {"tool": "bash", "input": "find . -name '*.py'"}
        assert result[1] == {"tool": "read", "input": "main.py"}


# ---------------------------------------------------------------------------
# execute_tool tests
# ---------------------------------------------------------------------------


class TestExecuteTool:
    def test_execute_bash(self, tmp_path):
        result_text, success = execute_tool({"tool": "bash", "input": "echo hello"}, tmp_path)
        assert "hello" in result_text
        assert success is True

    def test_execute_read(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("file contents here")
        result_text, success = execute_tool({"tool": "read", "input": "test.txt"}, tmp_path)
        assert "file contents here" in result_text
        assert success is True

    def test_execute_write(self, tmp_path):
        result_text, success = execute_tool(
            {"tool": "write", "input": "new content", "path": "out.txt"}, tmp_path
        )
        assert success is True
        assert (tmp_path / "out.txt").read_text() == "new content"

    def test_execute_bash_failure(self, tmp_path):
        result_text, success = execute_tool(
            {"tool": "bash", "input": "exit 1"}, tmp_path
        )
        assert success is False


# ---------------------------------------------------------------------------
# run_agent tests
# ---------------------------------------------------------------------------


def _make_task() -> TaskInfo:
    return TaskInfo(
        instance_id="test__test-1",
        repo="test/test-repo",
        base_commit="abc12345" * 5,  # 40-char hash
        problem_statement="Fix the bug in main.py",
        patch="",
        test_patch="",
        difficulty="easy",
        created_at="2025-01-01",
        sequence_position=0,
    )


class TestRunAgent:
    def test_integration_mocked(self, tmp_path, monkeypatch):
        """Integration test: mock Ollama + workspace, verify trajectory structure."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        # Create the file the agent will try to read
        src_dir = workspace_dir / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("original content")

        # Mock create_workspace to return our tmp dir
        monkeypatch.setattr(
            "looper.agent.runner.create_workspace",
            lambda repo, base_commit, workspace_root: workspace_dir,
        )

        # Mock get_patch
        monkeypatch.setattr(
            "looper.agent.runner.get_patch",
            lambda ws: "diff --git a/src/main.py ...",
        )

        # Build the sequence of chat responses
        responses = [
            ChatResponse(content="<bash>cat src/main.py</bash>", total_tokens=100, model="test"),
            ChatResponse(
                content='<write path="src/main.py">fixed content</write>',
                total_tokens=100,
                model="test",
            ),
            ChatResponse(content="<done>", total_tokens=50, model="test"),
        ]
        call_count = 0

        def mock_chat(messages, **kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        monkeypatch.setattr("looper.agent.runner.chat", mock_chat)

        task = _make_task()
        trajectory = run_agent(task, workspace_root=tmp_path, max_steps=10)

        assert isinstance(trajectory, AgentTrajectory)
        assert trajectory.meta.task_id == task.instance_id
        assert trajectory.meta.total_steps == 3
        assert len(trajectory.steps) == 3
        assert trajectory.generated_patch == "diff --git a/src/main.py ..."
        assert trajectory.outcome in ("completed", "patch_generated")

    def test_max_steps_enforced(self, tmp_path, monkeypatch):
        """Agent loop stops after max_steps even without <done>."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()

        monkeypatch.setattr(
            "looper.agent.runner.create_workspace",
            lambda repo, base_commit, workspace_root: workspace_dir,
        )
        monkeypatch.setattr(
            "looper.agent.runner.get_patch",
            lambda ws: "",
        )

        def mock_chat(messages, **kwargs):
            return ChatResponse(
                content="<bash>echo still going</bash>", total_tokens=10, model="test"
            )

        monkeypatch.setattr("looper.agent.runner.chat", mock_chat)

        task = _make_task()
        trajectory = run_agent(task, workspace_root=tmp_path, max_steps=3)

        assert trajectory.meta.total_steps == 3
        assert len(trajectory.steps) == 3
        assert trajectory.outcome == "max_steps"
