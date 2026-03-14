"""Tests for the agent runner: parse_tool_calls, execute_tool, run_agent."""

import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from looper.agent.runner import (
    parse_tool_calls,
    execute_tool,
    run_agent,
    prune_messages,
)
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

    def test_parse_edit(self):
        text = '<edit path="src/main.py">\nold code\n=======\nnew code\n</edit>'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["tool"] == "edit"
        assert result[0]["path"] == "src/main.py"
        assert "old code" in result[0]["input"]
        assert "new code" in result[0]["input"]

    def test_parse_edit_with_reasoning(self):
        text = (
            "I need to fix the return value.\n"
            '<edit path="utils.py">\n'
            "return False\n"
            "=======\n"
            "return True\n"
            "</edit>"
        )
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["tool"] == "edit"
        assert result[0]["path"] == "utils.py"


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

    def test_execute_edit_basic(self, tmp_path):
        target = tmp_path / "test.py"
        target.write_text("def foo():\n    return False\n")
        result_text, success = execute_tool(
            {"tool": "edit", "input": "\nreturn False\n=======\nreturn True\n", "path": "test.py"},
            tmp_path,
        )
        assert success is True
        assert "Edited" in result_text
        assert target.read_text() == "def foo():\n    return True\n"

    def test_execute_edit_multiline(self, tmp_path):
        target = tmp_path / "test.py"
        target.write_text("line1\nline2\nline3\nline4\n")
        result_text, success = execute_tool(
            {"tool": "edit", "input": "\nline2\nline3\n=======\nreplaced2\nreplaced3\n", "path": "test.py"},
            tmp_path,
        )
        assert success is True
        assert target.read_text() == "line1\nreplaced2\nreplaced3\nline4\n"

    def test_execute_edit_not_found(self, tmp_path):
        target = tmp_path / "test.py"
        target.write_text("actual content\n")
        result_text, success = execute_tool(
            {"tool": "edit", "input": "\nwrong content\n=======\nnew content\n", "path": "test.py"},
            tmp_path,
        )
        assert success is False
        assert "not found" in result_text
        # File should be unchanged
        assert target.read_text() == "actual content\n"

    def test_execute_edit_file_not_found(self, tmp_path):
        result_text, success = execute_tool(
            {"tool": "edit", "input": "\nold\n=======\nnew\n", "path": "nonexistent.py"},
            tmp_path,
        )
        assert success is False
        assert "does not exist" in result_text

    def test_execute_edit_preserves_rest(self, tmp_path):
        target = tmp_path / "test.py"
        original = "# header\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
        target.write_text(original)
        result_text, success = execute_tool(
            {"tool": "edit", "input": "\n    return 1\n=======\n    return 42\n", "path": "test.py"},
            tmp_path,
        )
        assert success is True
        expected = "# header\ndef foo():\n    return 42\n\ndef bar():\n    return 2\n"
        assert target.read_text() == expected

    def test_execute_edit_bad_format(self, tmp_path):
        target = tmp_path / "test.py"
        target.write_text("content\n")
        result_text, success = execute_tool(
            {"tool": "edit", "input": "no separator here", "path": "test.py"},
            tmp_path,
        )
        assert success is False
        assert "could not parse" in result_text.lower()


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


# ---------------------------------------------------------------------------
# Fix 1: Line-range read tests
# ---------------------------------------------------------------------------


class TestLineRangeRead:
    def test_read_with_line_range(self, tmp_path):
        """<read>file.py:3-5</read> returns only lines 3-5."""
        target = tmp_path / "big.py"
        target.write_text("\n".join(f"line {i}" for i in range(1, 21)))
        result, success = execute_tool({"tool": "read", "input": "big.py:3-5"}, tmp_path)
        assert success is True
        assert "line 3" in result
        assert "line 5" in result
        assert "line 2" not in result
        assert "line 6" not in result

    def test_read_with_line_range_single_line(self, tmp_path):
        """<read>file.py:5-5</read> returns just line 5."""
        target = tmp_path / "big.py"
        target.write_text("\n".join(f"line {i}" for i in range(1, 21)))
        result, success = execute_tool({"tool": "read", "input": "big.py:5-5"}, tmp_path)
        assert success is True
        assert "line 5" in result
        assert "line 4" not in result
        assert "line 6" not in result

    def test_read_without_line_range_unchanged(self, tmp_path):
        """Regular read still works as before."""
        target = tmp_path / "test.txt"
        target.write_text("hello world")
        result, success = execute_tool({"tool": "read", "input": "test.txt"}, tmp_path)
        assert success is True
        assert "hello world" in result

    def test_read_line_range_out_of_bounds(self, tmp_path):
        """Line range beyond file length returns what's available."""
        target = tmp_path / "short.py"
        target.write_text("line 1\nline 2\nline 3\n")
        result, success = execute_tool({"tool": "read", "input": "short.py:2-100"}, tmp_path)
        assert success is True
        assert "line 2" in result
        assert "line 3" in result

    def test_parse_read_with_colon_range(self):
        """Parser captures the full path:start-end as the read input."""
        text = "<read>django/db/models.py:100-200</read>"
        result = parse_tool_calls(text)
        assert result == [{"tool": "read", "input": "django/db/models.py:100-200"}]


# ---------------------------------------------------------------------------
# Fix 2: Context pruning tests
# ---------------------------------------------------------------------------


class TestContextPruning:
    def test_prune_shortens_long_tool_results(self):
        """Messages exceeding token threshold get old file reads pruned."""
        messages = [
            ChatMessage(role="system", content="System prompt"),
            ChatMessage(role="user", content="Problem statement"),
            # Old assistant + tool result pair
            ChatMessage(role="assistant", content="<read>big_file.py</read>"),
            ChatMessage(role="user", content="[read] " + "x" * 10000),
            # Recent assistant + tool result pair
            ChatMessage(role="assistant", content="<read>small.py</read>"),
            ChatMessage(role="user", content="[read] small content"),
        ]
        pruned = prune_messages(messages, max_tokens=2000)
        # The old long read result should be summarized
        assert len(pruned[3].content) < 500
        # The recent message should be kept intact
        assert "small content" in pruned[5].content

    def test_prune_no_op_when_under_threshold(self):
        """Messages under threshold are returned unchanged."""
        messages = [
            ChatMessage(role="system", content="Short"),
            ChatMessage(role="user", content="Problem"),
        ]
        pruned = prune_messages(messages, max_tokens=2000)
        assert len(pruned) == 2
        assert pruned[0].content == "Short"
        assert pruned[1].content == "Problem"


# ---------------------------------------------------------------------------
# Fix 4: Loop detection tests
# ---------------------------------------------------------------------------


class TestLoopDetection:
    def test_loop_detection_injects_nudge(self, tmp_path, monkeypatch):
        """After reading same file 3 times without writing, agent gets a nudge."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        (workspace_dir / "target.py").write_text("some code")

        monkeypatch.setattr(
            "looper.agent.runner.create_workspace",
            lambda repo, base_commit, workspace_root: workspace_dir,
        )
        monkeypatch.setattr(
            "looper.agent.runner.get_patch",
            lambda ws: "",
        )

        call_count = 0
        received_messages = []

        def mock_chat(messages, **kwargs):
            nonlocal call_count
            received_messages.append([m.content for m in messages])
            call_count += 1
            if call_count <= 4:
                return ChatResponse(
                    content="<read>target.py</read>", total_tokens=10, model="test"
                )
            return ChatResponse(content="<done>", total_tokens=10, model="test")

        monkeypatch.setattr("looper.agent.runner.chat", mock_chat)

        task = _make_task()
        trajectory = run_agent(task, workspace_root=tmp_path, max_steps=6)

        # After 3rd read of same file, the 4th call's messages should contain the nudge
        # Check that some message contains the loop detection nudge
        all_contents = []
        for msg_list in received_messages:
            all_contents.extend(msg_list)
        nudge_found = any("already read" in c.lower() for c in all_contents)
        assert nudge_found, "Loop detection nudge was not injected"
