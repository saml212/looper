"""Tests for looper.synthesizers — trajectory synthesis to training data."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from looper.agent.ollama_client import ChatResponse
from looper.models import (
    AgentStep,
    AgentTrajectory,
    SessionMeta,
    SynthesizedPair,
    ToolCall,
    TrainingExample,
)
from looper.synthesizers.synthesizer import (
    load_training_data,
    pairs_to_training_examples,
    save_training_data,
    synthesize_batch,
    synthesize_trajectory,
)
from looper.synthesizers.trajectory_to_text import trajectory_to_text


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

def _make_trajectory(**overrides) -> AgentTrajectory:
    """Build a minimal valid AgentTrajectory for testing."""
    defaults = dict(
        meta=SessionMeta(
            session_id="sess-test-001",
            agent_id="default",
            task_id="django__django-9296",
            model_name="qwen2.5-coder:7b",
            started_at="2024-01-15T10:30:00Z",
            ended_at="2024-01-15T10:35:00Z",
            total_tokens=5000,
            total_steps=2,
        ),
        steps=[
            AgentStep(
                step_number=1,
                reasoning="I need to find the failing test first.",
                tool_calls=[
                    ToolCall(
                        tool_name="bash",
                        tool_input={"command": "python -m pytest tests/ -x"},
                        tool_result="FAILED tests/test_migrations.py::test_rename - AssertionError",
                        success=False,
                        duration_ms=3200,
                    )
                ],
                timestamp="2024-01-15T10:31:00Z",
            ),
            AgentStep(
                step_number=2,
                reasoning="The test failed. Let me look at the migration autodetector.",
                tool_calls=[
                    ToolCall(
                        tool_name="read_file",
                        tool_input={"path": "django/db/migrations/autodetector.py"},
                        tool_result="class MigrationAutodetector:\n    def detect_changes(self):\n        pass",
                        success=True,
                        duration_ms=100,
                    )
                ],
                timestamp="2024-01-15T10:32:00Z",
            ),
        ],
        outcome="completed",
        generated_patch="diff --git a/django/db/migrations/autodetector.py ...",
    )
    defaults.update(overrides)
    return AgentTrajectory(**defaults)


# ---------------------------------------------------------------------------
# trajectory_to_text
# ---------------------------------------------------------------------------

class TestTrajectoryToText:
    def test_produces_nonempty_with_task_id_and_outcome(self):
        traj = _make_trajectory()
        text = trajectory_to_text(traj)
        assert len(text) > 0
        assert "django__django-9296" in text
        assert "completed" in text

    def test_includes_tool_call_info(self):
        traj = _make_trajectory()
        text = trajectory_to_text(traj)
        assert "bash" in text
        assert "read_file" in text
        assert "python -m pytest" in text

    def test_truncates_long_tool_results(self):
        long_result = "x" * 1000
        traj = _make_trajectory(
            steps=[
                AgentStep(
                    step_number=1,
                    reasoning="Testing",
                    tool_calls=[
                        ToolCall(
                            tool_name="bash",
                            tool_input={"command": "cat big_file.py"},
                            tool_result=long_result,
                            success=True,
                        )
                    ],
                )
            ]
        )
        text = trajectory_to_text(traj)
        # The full 1000-char result should not appear; it should be truncated
        assert long_result not in text
        assert "..." in text


# ---------------------------------------------------------------------------
# synthesize_trajectory
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE = json.dumps([
    {
        "instruction": "How should you run Django tests for the migrations module?",
        "response": "Use: python -m pytest tests/test_migrations.py -x",
        "pair_type": "tool_usage",
        "confidence": 0.85,
    },
    {
        "instruction": "How do you read source files to understand Django internals?",
        "response": "Use the read_file tool on the relevant module.",
        "pair_type": "workflow",
        "confidence": 0.7,
    },
])


class TestSynthesizeTrajectory:
    @patch("looper.synthesizers.synthesizer.chat")
    def test_returns_valid_pairs_from_llm(self, mock_chat):
        mock_chat.return_value = ChatResponse(
            content=MOCK_LLM_RESPONSE,
            total_tokens=500,
            model="qwen2.5-coder:7b",
        )
        traj = _make_trajectory()
        pairs = synthesize_trajectory(traj)

        assert len(pairs) == 2
        assert all(isinstance(p, SynthesizedPair) for p in pairs)
        assert pairs[0].pair_type == "tool_usage"
        assert pairs[0].confidence == 0.85
        assert pairs[0].source_session_id == "sess-test-001"
        assert pairs[0].source_task_id == "django__django-9296"
        mock_chat.assert_called_once()

    @patch("looper.synthesizers.synthesizer.chat")
    def test_filters_low_confidence_pairs(self, mock_chat):
        low_conf_response = json.dumps([
            {
                "instruction": "Low quality pair",
                "response": "Not useful",
                "pair_type": "tool_usage",
                "confidence": 0.2,
            },
            {
                "instruction": "Good pair",
                "response": "Useful advice",
                "pair_type": "workflow",
                "confidence": 0.8,
            },
        ])
        mock_chat.return_value = ChatResponse(
            content=low_conf_response,
            total_tokens=300,
            model="qwen2.5-coder:7b",
        )
        traj = _make_trajectory()
        pairs = synthesize_trajectory(traj)

        assert len(pairs) == 1
        assert pairs[0].confidence == 0.8

    @patch("looper.synthesizers.synthesizer.chat")
    def test_returns_empty_on_invalid_json(self, mock_chat):
        mock_chat.return_value = ChatResponse(
            content="This is not valid JSON at all {{{",
            total_tokens=100,
            model="qwen2.5-coder:7b",
        )
        traj = _make_trajectory()
        pairs = synthesize_trajectory(traj)

        assert pairs == []


# ---------------------------------------------------------------------------
# synthesize_batch
# ---------------------------------------------------------------------------

class TestSynthesizeBatch:
    @patch("looper.synthesizers.synthesizer.synthesize_trajectory")
    def test_processes_all_trajectories_and_saves(self, mock_synth, tmp_path):
        pair1 = SynthesizedPair(
            instruction="How to run tests?",
            response="python -m pytest",
            pair_type="tool_usage",
            confidence=0.9,
            source_session_id="sess-1",
            source_task_id="task-1",
        )
        pair2 = SynthesizedPair(
            instruction="How to read files?",
            response="Use read_file tool",
            pair_type="workflow",
            confidence=0.8,
            source_session_id="sess-2",
            source_task_id="task-2",
        )
        mock_synth.side_effect = [[pair1], [pair2]]

        traj1 = _make_trajectory(meta=SessionMeta(
            session_id="sess-1", task_id="task-1",
            model_name="m", started_at="2024-01-01",
        ))
        traj2 = _make_trajectory(meta=SessionMeta(
            session_id="sess-2", task_id="task-2",
            model_name="m", started_at="2024-01-01",
        ))

        output_path = tmp_path / "pairs.json"
        all_pairs = synthesize_batch([traj1, traj2], output_path)

        assert len(all_pairs) == 2
        assert mock_synth.call_count == 2
        assert output_path.exists()

        saved = json.loads(output_path.read_text())
        assert len(saved) == 2


# ---------------------------------------------------------------------------
# pairs_to_training_examples
# ---------------------------------------------------------------------------

class TestPairsToTrainingExamples:
    def test_converts_to_chat_format(self):
        pairs = [
            SynthesizedPair(
                instruction="How to run tests?",
                response="python -m pytest",
                pair_type="tool_usage",
                confidence=0.9,
                source_session_id="sess-1",
                source_task_id="task-1",
            ),
        ]
        examples = pairs_to_training_examples(pairs)

        assert len(examples) == 1
        assert len(examples[0].messages) == 2
        assert examples[0].messages[0] == {"role": "user", "content": "How to run tests?"}
        assert examples[0].messages[1] == {"role": "assistant", "content": "python -m pytest"}

    def test_handles_empty_list(self):
        examples = pairs_to_training_examples([])
        assert examples == []


# ---------------------------------------------------------------------------
# save_training_data / load_training_data
# ---------------------------------------------------------------------------

class TestTrainingDataIO:
    def test_roundtrip_save_load(self, tmp_path):
        examples = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                ],
                source_pair_id="p1",
            ),
            TrainingExample(
                messages=[
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Answer 2"},
                ],
                source_pair_id="p2",
            ),
        ]
        path = tmp_path / "train.jsonl"
        save_training_data(examples, path)
        loaded = load_training_data(path)

        assert len(loaded) == len(examples)
        for orig, reloaded in zip(examples, loaded):
            assert orig.messages == reloaded.messages
            assert orig.source_pair_id == reloaded.source_pair_id

    def test_file_format_is_jsonl(self, tmp_path):
        examples = [
            TrainingExample(
                messages=[{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}],
            ),
            TrainingExample(
                messages=[{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}],
            ),
        ]
        path = tmp_path / "train.jsonl"
        save_training_data(examples, path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        # Each line must be valid JSON
        for line in lines:
            obj = json.loads(line)
            assert "messages" in obj
