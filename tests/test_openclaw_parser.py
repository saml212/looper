"""Tests for OpenClaw session JSONL parser."""

import json
from pathlib import Path

import pytest

from looper.integrations.openclaw_parser import parse_session, _parse_events
from looper.models import AgentTrajectory, AgentStep, ToolCall, SessionMeta


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "openclaw_session.jsonl"


class TestParseSession:
    def test_returns_agent_trajectory(self):
        traj = parse_session(FIXTURE_PATH)
        assert isinstance(traj, AgentTrajectory)

    def test_session_meta_extracted(self):
        traj = parse_session(FIXTURE_PATH)
        assert traj.meta.session_id == "sess-abc123"
        assert traj.meta.agent_id == "agent-01"
        assert traj.meta.model_name == "qwen2.5-coder:7b"
        assert traj.meta.started_at == "2026-03-04T10:00:00Z"

    def test_task_id_from_custom_event(self):
        traj = parse_session(FIXTURE_PATH)
        assert traj.meta.task_id == "django__django-12304"

    def test_total_tokens_summed(self):
        traj = parse_session(FIXTURE_PATH)
        # 1500 + 2800 + 4200 + 5000 = 13500
        assert traj.meta.total_tokens == 13500

    def test_steps_count(self):
        traj = parse_session(FIXTURE_PATH)
        # 3 assistant messages with tool calls = 3 steps
        assert len(traj.steps) == 3

    def test_step_reasoning_extracted(self):
        traj = parse_session(FIXTURE_PATH)
        assert "explore the repository" in traj.steps[0].reasoning

    def test_step_tool_calls(self):
        traj = parse_session(FIXTURE_PATH)
        step = traj.steps[0]
        assert len(step.tool_calls) == 1
        tc = step.tool_calls[0]
        assert tc.tool_name == "bash"
        assert tc.tool_input == {"command": "find . -name '*.py' | head -20"}
        assert tc.success is True
        assert tc.duration_ms == 150

    def test_tool_result_content(self):
        traj = parse_session(FIXTURE_PATH)
        tc = traj.steps[0].tool_calls[0]
        assert "django/db/models" in tc.tool_result

    def test_write_tool_call(self):
        traj = parse_session(FIXTURE_PATH)
        tc = traj.steps[2].tool_calls[0]
        assert tc.tool_name == "write"
        assert tc.tool_result == "File written successfully."

    def test_step_numbering(self):
        traj = parse_session(FIXTURE_PATH)
        for i, step in enumerate(traj.steps):
            assert step.step_number == i + 1

    def test_total_steps_in_meta(self):
        traj = parse_session(FIXTURE_PATH)
        assert traj.meta.total_steps == 3

    def test_outcome_completed_when_no_error(self):
        traj = parse_session(FIXTURE_PATH)
        # All tool calls succeeded, so outcome should be "completed"
        assert traj.outcome == "completed"

    def test_ended_at_from_last_message(self):
        traj = parse_session(FIXTURE_PATH)
        assert traj.meta.ended_at == "2026-03-04T10:00:30Z"


class TestParseEvents:
    def test_returns_meta_and_steps(self):
        lines = [
            {"type": "session", "session_id": "s1", "agent_id": "a1", "created_at": "2026-01-01T00:00:00Z"},
            {"type": "model_change", "model": "test-model"},
            {"type": "message", "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thinking"},
                    {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "ls"}},
                ],
                "usage": {"totalTokens": 100},
                "timestamp": "2026-01-01T00:00:05Z",
            }},
            {"type": "message", "message": {
                "role": "toolResult",
                "toolCallId": "t1",
                "toolName": "bash",
                "content": "file.py",
                "details": {"exitCode": 0, "durationMs": 50},
                "timestamp": "2026-01-01T00:00:06Z",
            }},
        ]
        meta, steps = _parse_events(lines)
        assert meta.session_id == "s1"
        assert meta.model_name == "test-model"
        assert len(steps) == 1
        assert steps[0].tool_calls[0].tool_name == "bash"

    def test_failed_tool_call(self):
        lines = [
            {"type": "session", "session_id": "s1", "agent_id": "a1", "created_at": "2026-01-01T00:00:00Z"},
            {"type": "message", "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "bad"}},
                ],
                "usage": {"totalTokens": 50},
                "timestamp": "2026-01-01T00:00:05Z",
            }},
            {"type": "message", "message": {
                "role": "toolResult",
                "toolCallId": "t1",
                "toolName": "bash",
                "content": "command not found",
                "details": {"exitCode": 1, "durationMs": 10},
                "timestamp": "2026-01-01T00:00:06Z",
            }},
        ]
        _, steps = _parse_events(lines)
        assert steps[0].tool_calls[0].success is False

    def test_assistant_text_only_no_step(self):
        """Assistant messages without tool calls don't create steps."""
        lines = [
            {"type": "session", "session_id": "s1", "agent_id": "a1", "created_at": "2026-01-01T00:00:00Z"},
            {"type": "message", "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Just thinking..."}],
                "usage": {"totalTokens": 50},
                "timestamp": "2026-01-01T00:00:05Z",
            }},
        ]
        _, steps = _parse_events(lines)
        assert len(steps) == 0

    def test_missing_task_id_defaults(self):
        lines = [
            {"type": "session", "session_id": "s1", "agent_id": "a1", "created_at": "2026-01-01T00:00:00Z"},
        ]
        meta, _ = _parse_events(lines)
        assert meta.task_id == "unknown"


class TestParseSessionEdgeCases:
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_session(Path("/nonexistent/session.jsonl"))

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            parse_session(empty)
