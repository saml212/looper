"""Tests for looper.models — core Pydantic data models."""

import pytest
from pydantic import ValidationError

from looper.models import (
    AgentStep,
    AgentTrajectory,
    ExperimentConfig,
    ExperimentResult,
    SessionMeta,
    SynthesizedPair,
    TaskInfo,
    TaskResult,
    ToolCall,
    TrainingExample,
)


# ---------------------------------------------------------------------------
# Fixtures with realistic data
# ---------------------------------------------------------------------------

def make_task_info(**overrides):
    defaults = dict(
        instance_id="django__django-9296",
        repo="django/django",
        base_commit="a1b2c3d4e5f6",
        problem_statement="Fix migration crash when renaming model.",
        hints_text="Check MigrationAutodetector.",
        patch="diff --git a/django/db/migrations ...",
        test_patch="diff --git a/tests/migrations ...",
        difficulty="<15 min fix",
        created_at="2024-01-15T10:30:00Z",
        sequence_position=1,
    )
    defaults.update(overrides)
    return TaskInfo(**defaults)


def make_tool_call(**overrides):
    defaults = dict(
        tool_name="bash",
        tool_input={"command": "python manage.py test"},
        tool_result="OK\n3 tests passed.",
        success=True,
        duration_ms=1200,
    )
    defaults.update(overrides)
    return ToolCall(**defaults)


def make_agent_step(**overrides):
    defaults = dict(
        step_number=1,
        reasoning="I should run the tests first to see the failure.",
        tool_calls=[make_tool_call()],
        timestamp="2024-01-15T10:31:00Z",
    )
    defaults.update(overrides)
    return AgentStep(**defaults)


def make_session_meta(**overrides):
    defaults = dict(
        session_id="sess-abc-123",
        agent_id="default",
        task_id="django__django-9296",
        model_name="claude-sonnet-4-20250514",
        started_at="2024-01-15T10:30:00Z",
        ended_at="2024-01-15T10:35:00Z",
        total_tokens=15000,
        total_steps=5,
    )
    defaults.update(overrides)
    return SessionMeta(**defaults)


def make_trajectory(**overrides):
    defaults = dict(
        meta=make_session_meta(),
        steps=[make_agent_step(step_number=i) for i in range(1, 4)],
        outcome="success",
        generated_patch="diff --git a/fix ...",
        resolve_rate=1.0,
    )
    defaults.update(overrides)
    return AgentTrajectory(**defaults)


def make_synthesized_pair(**overrides):
    defaults = dict(
        instruction="Run the Django test suite for the migrations module.",
        response="bash: python manage.py test migrations",
        pair_type="tool_usage",
        confidence=0.85,
        source_session_id="sess-abc-123",
        source_task_id="django__django-9296",
    )
    defaults.update(overrides)
    return SynthesizedPair(**defaults)


def make_training_example(**overrides):
    defaults = dict(
        messages=[
            {"role": "user", "content": "Run the test suite."},
            {"role": "assistant", "content": "bash: python -m pytest tests/"},
        ],
        source_pair_id="pair-001",
    )
    defaults.update(overrides)
    return TrainingExample(**defaults)


def make_experiment_config(**overrides):
    defaults = dict(
        name="Within-Repo Learning",
        experiment_id="exp-001",
        repo="django/django",
        model_name="codellama-7b",
        train_task_ids=["django__django-9296", "django__django-9297"],
        test_task_ids=["django__django-9300"],
        strategy="full_replay",
        lora_rank=16,
        lora_alpha=32,
        seed=42,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def make_task_result(**overrides):
    defaults = dict(
        task_id="django__django-9300",
        condition="base_lora",
        resolved=True,
        steps=8,
        tokens=12000,
        duration_seconds=45.2,
        error="",
    )
    defaults.update(overrides)
    return TaskResult(**defaults)


def make_experiment_result(**overrides):
    defaults = dict(
        config=make_experiment_config(),
        task_results=[make_task_result()],
        forward_transfer=0.15,
        forgetting=0.02,
        started_at="2024-01-15T10:00:00Z",
        completed_at="2024-01-15T12:00:00Z",
    )
    defaults.update(overrides)
    return ExperimentResult(**defaults)


# ---------------------------------------------------------------------------
# TaskInfo
# ---------------------------------------------------------------------------

class TestTaskInfo:
    def test_create_with_valid_data(self):
        t = make_task_info()
        assert t.instance_id == "django__django-9296"
        assert t.repo == "django/django"
        assert t.sequence_position == 1

    def test_hints_text_defaults_empty(self):
        t = make_task_info(hints_text="")
        assert t.hints_text == ""

    def test_roundtrip_dict(self):
        t = make_task_info()
        d = t.model_dump()
        t2 = TaskInfo(**d)
        assert t == t2

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            TaskInfo(instance_id="x", repo="r")  # missing many fields


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------

class TestToolCall:
    def test_create_with_valid_data(self):
        tc = make_tool_call()
        assert tc.tool_name == "bash"
        assert tc.success is True
        assert tc.tool_input == {"command": "python manage.py test"}

    def test_duration_defaults_zero(self):
        tc = make_tool_call(duration_ms=0)
        assert tc.duration_ms == 0

    def test_roundtrip_dict(self):
        tc = make_tool_call()
        d = tc.model_dump()
        t2 = ToolCall(**d)
        assert tc == t2


# ---------------------------------------------------------------------------
# AgentStep
# ---------------------------------------------------------------------------

class TestAgentStep:
    def test_create_with_valid_data(self):
        s = make_agent_step()
        assert s.step_number == 1
        assert len(s.tool_calls) == 1

    def test_empty_tool_calls(self):
        s = make_agent_step(tool_calls=[])
        assert s.tool_calls == []

    def test_timestamp_defaults_empty(self):
        s = AgentStep(step_number=1, reasoning="think", tool_calls=[])
        assert s.timestamp == ""

    def test_roundtrip_dict(self):
        s = make_agent_step()
        d = s.model_dump()
        s2 = AgentStep(**d)
        assert s == s2


# ---------------------------------------------------------------------------
# SessionMeta
# ---------------------------------------------------------------------------

class TestSessionMeta:
    def test_create_with_valid_data(self):
        m = make_session_meta()
        assert m.session_id == "sess-abc-123"
        assert m.model_name == "claude-sonnet-4-20250514"

    def test_defaults(self):
        m = SessionMeta(
            session_id="s1",
            task_id="t1",
            model_name="m1",
            started_at="2024-01-01",
        )
        assert m.agent_id == "default"
        assert m.ended_at == ""
        assert m.total_tokens == 0
        assert m.total_steps == 0

    def test_roundtrip_dict(self):
        m = make_session_meta()
        d = m.model_dump()
        m2 = SessionMeta(**d)
        assert m == m2


# ---------------------------------------------------------------------------
# AgentTrajectory
# ---------------------------------------------------------------------------

class TestAgentTrajectory:
    def test_create_with_valid_data(self):
        traj = make_trajectory()
        assert traj.outcome == "success"
        assert len(traj.steps) == 3
        assert traj.resolve_rate == 1.0

    def test_defaults(self):
        traj = AgentTrajectory(
            meta=make_session_meta(),
            steps=[],
            outcome="failure",
        )
        assert traj.generated_patch == ""
        assert traj.resolve_rate == 0.0

    def test_roundtrip_dict(self):
        traj = make_trajectory()
        d = traj.model_dump()
        traj2 = AgentTrajectory(**d)
        assert traj == traj2


# ---------------------------------------------------------------------------
# SynthesizedPair
# ---------------------------------------------------------------------------

class TestSynthesizedPair:
    def test_create_with_valid_data(self):
        sp = make_synthesized_pair()
        assert sp.pair_type == "tool_usage"
        assert sp.confidence == 0.85

    def test_confidence_must_be_between_0_and_1(self):
        with pytest.raises(ValidationError):
            make_synthesized_pair(confidence=1.5)
        with pytest.raises(ValidationError):
            make_synthesized_pair(confidence=-0.1)

    def test_confidence_boundaries(self):
        sp0 = make_synthesized_pair(confidence=0.0)
        assert sp0.confidence == 0.0
        sp1 = make_synthesized_pair(confidence=1.0)
        assert sp1.confidence == 1.0

    def test_roundtrip_dict(self):
        sp = make_synthesized_pair()
        d = sp.model_dump()
        sp2 = SynthesizedPair(**d)
        assert sp == sp2


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------

class TestTrainingExample:
    def test_create_with_valid_data(self):
        te = make_training_example()
        assert len(te.messages) == 2
        assert te.messages[0]["role"] == "user"

    def test_source_pair_id_defaults_empty(self):
        te = TrainingExample(messages=[{"role": "user", "content": "hi"}])
        assert te.source_pair_id == ""

    def test_roundtrip_dict(self):
        te = make_training_example()
        d = te.model_dump()
        te2 = TrainingExample(**d)
        assert te == te2


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_create_with_valid_data(self):
        ec = make_experiment_config()
        assert ec.name == "Within-Repo Learning"
        assert ec.lora_rank == 16
        assert ec.seed == 42

    def test_defaults(self):
        ec = ExperimentConfig(
            name="test",
            experiment_id="e1",
            repo="r",
            model_name="m",
            train_task_ids=[],
            test_task_ids=[],
            strategy="full_replay",
        )
        assert ec.lora_rank == 16
        assert ec.lora_alpha == 32
        assert ec.seed == 42

    def test_roundtrip_dict(self):
        ec = make_experiment_config()
        d = ec.model_dump()
        ec2 = ExperimentConfig(**d)
        assert ec == ec2


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_create_with_valid_data(self):
        tr = make_task_result()
        assert tr.condition == "base_lora"
        assert tr.resolved is True
        assert tr.duration_seconds == 45.2

    def test_error_defaults_empty(self):
        tr = TaskResult(
            task_id="t1",
            condition="base",
            resolved=False,
            steps=1,
            tokens=100,
            duration_seconds=1.0,
        )
        assert tr.error == ""

    def test_roundtrip_dict(self):
        tr = make_task_result()
        d = tr.model_dump()
        tr2 = TaskResult(**d)
        assert tr == tr2


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------

class TestExperimentResult:
    def test_create_with_valid_data(self):
        er = make_experiment_result()
        assert er.forward_transfer == 0.15
        assert er.forgetting == 0.02
        assert len(er.task_results) == 1

    def test_defaults(self):
        er = ExperimentResult(
            config=make_experiment_config(),
            task_results=[],
            started_at="2024-01-01",
        )
        assert er.forward_transfer == 0.0
        assert er.forgetting == 0.0
        assert er.completed_at == ""

    def test_roundtrip_dict(self):
        er = make_experiment_result()
        d = er.model_dump()
        er2 = ExperimentResult(**d)
        assert er == er2

    def test_nested_serialization(self):
        """Verify deeply nested models serialize and deserialize correctly."""
        er = make_experiment_result()
        d = er.model_dump()
        assert isinstance(d["config"], dict)
        assert isinstance(d["task_results"], list)
        assert isinstance(d["task_results"][0], dict)
        er2 = ExperimentResult(**d)
        assert er2.config.name == "Within-Repo Learning"
        assert er2.task_results[0].task_id == "django__django-9300"
