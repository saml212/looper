"""Core data models for the Looper framework.

Pure Pydantic v2 models representing the data flowing through the pipeline:
trajectories, training data, experiment configs, and results.
"""

from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    """A SWE-Bench-CL task definition."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    patch: str
    test_patch: str
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    difficulty: str
    created_at: str
    sequence_position: int


class ToolCall(BaseModel):
    """A single tool invocation within an agent step."""

    tool_name: str
    tool_input: dict
    tool_result: str
    success: bool
    duration_ms: int = 0


class AgentStep(BaseModel):
    """One reasoning + action cycle."""

    step_number: int
    reasoning: str
    tool_calls: list[ToolCall]
    timestamp: str = ""


class SessionMeta(BaseModel):
    """Metadata about an agent session."""

    session_id: str
    agent_id: str = "default"
    task_id: str
    model_name: str
    started_at: str
    ended_at: str = ""
    total_tokens: int = 0
    total_steps: int = 0


class AgentTrajectory(BaseModel):
    """Complete session trajectory: metadata, steps, and outcome."""

    meta: SessionMeta
    steps: list[AgentStep]
    outcome: str
    generated_patch: str = ""
    resolve_rate: float = 0.0


class SynthesizedPair(BaseModel):
    """A training pair synthesized from an agent trajectory."""

    instruction: str
    response: str
    pair_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_session_id: str
    source_task_id: str


class TrainingExample(BaseModel):
    """A training example formatted for LoRA fine-tuning (chat format)."""

    messages: list[dict]
    source_pair_id: str = ""


class ExperimentConfig(BaseModel):
    """Configuration for a Looper experiment."""

    name: str
    experiment_id: str
    repo: str
    model_name: str
    train_task_ids: list[str]
    test_task_ids: list[str]
    strategy: str
    lora_rank: int = 16
    lora_alpha: int = 32
    seed: int = 42


class TaskResult(BaseModel):
    """Result of running a single task under one condition."""

    task_id: str
    condition: str
    resolved: bool
    steps: int
    tokens: int
    duration_seconds: float
    error: str = ""


class ExperimentResult(BaseModel):
    """Full results from an experiment run."""

    config: ExperimentConfig
    task_results: list[TaskResult]
    forward_transfer: float = 0.0
    forgetting: float = 0.0
    started_at: str
    completed_at: str = ""
