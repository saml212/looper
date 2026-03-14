"""Convert agent trajectories directly into training examples in XML tool-call format.

Instead of using an LLM to synthesize Q&A pairs (which causes format mismatch),
this converts trajectories into multi-turn chat examples that preserve the exact
XML tool-calling format used at inference time (<bash>, <read>, <write>, <done>).

See DEEP_AUDIT.md — format mismatch is the dominant failure mode for LoRA.
"""

import logging
from pathlib import Path

from looper.agent.runner import SYSTEM_PROMPT
from looper.models import AgentStep, AgentTrajectory, TaskInfo, TrainingExample

logger = logging.getLogger(__name__)


def _reconstruct_assistant_message(step: AgentStep) -> str:
    """Reconstruct the assistant message from a step: reasoning + XML tool call."""
    parts = []
    if step.reasoning:
        parts.append(step.reasoning)

    for tc in step.tool_calls:
        name = tc.tool_name
        inp = tc.tool_input

        if name == "bash":
            parts.append(f"<bash>{inp.get('input', '')}</bash>")
        elif name == "read":
            parts.append(f"<read>{inp.get('input', '')}</read>")
        elif name == "write":
            path = inp.get("path", "")
            content = inp.get("input", "")
            parts.append(f'<write path="{path}">{content}</write>')
        elif name == "done":
            parts.append("<done>")

    return "\n".join(parts) if parts else step.reasoning


def _reconstruct_tool_result(step: AgentStep) -> str:
    """Reconstruct the user message (tool results) from a step."""
    parts = []
    for tc in step.tool_calls:
        if tc.tool_name == "done":
            continue
        parts.append(f"[{tc.tool_name}] {tc.tool_result}")
    return "\n".join(parts)


def trajectory_to_training_example(
    trajectory: AgentTrajectory,
    task: TaskInfo,
    workspace_dir: str = "/workspace",
) -> TrainingExample | None:
    """Convert one trajectory into a single multi-turn training example.

    Returns None if the trajectory has no tool calls.
    """
    if not trajectory.steps:
        return None

    total_tool_calls = sum(len(s.tool_calls) for s in trajectory.steps)
    if total_tool_calls == 0:
        return None

    system_prompt = SYSTEM_PROMPT.format(
        workspace_dir=workspace_dir,
        problem_statement=task.problem_statement,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.problem_statement},
    ]

    for step in trajectory.steps:
        assistant_msg = _reconstruct_assistant_message(step)
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

        has_done = any(tc.tool_name == "done" for tc in step.tool_calls)
        if not has_done:
            tool_result = _reconstruct_tool_result(step)
            if tool_result:
                messages.append({"role": "user", "content": tool_result})

    return TrainingExample(
        messages=messages,
        source_pair_id=trajectory.meta.task_id,
    )


# Truncation limit for tool results in per-step examples
_MAX_TOOL_RESULT_CHARS = 2000


def trajectory_to_step_examples(
    trajectory: AgentTrajectory,
    task: TaskInfo,
    workspace_dir: str = "/workspace",
) -> list[TrainingExample]:
    """Split a trajectory into per-step training examples.

    Each example contains:
    - System prompt (tool descriptions + problem)
    - User message: problem statement (+ prior context summary for later steps)
    - Assistant message: reasoning + XML tool call

    This produces shorter sequences that fit in memory for larger models (14B+).
    Each example still teaches the XML tool-calling format.
    """
    if not trajectory.steps:
        return []

    system_prompt = SYSTEM_PROMPT.format(
        workspace_dir=workspace_dir,
        problem_statement=task.problem_statement,
    )

    examples = []
    context_parts = []  # accumulate prior tool interactions as context

    for step in trajectory.steps:
        assistant_msg = _reconstruct_assistant_message(step)
        if not assistant_msg:
            continue

        # Skip steps with no tool calls (pure reasoning)
        if not step.tool_calls:
            continue

        # Build user message: problem + prior context
        if context_parts:
            user_msg = (
                task.problem_statement
                + "\n\nPrior steps:\n"
                + "\n".join(context_parts[-3:])  # last 3 steps for context
            )
        else:
            user_msg = task.problem_statement

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        examples.append(TrainingExample(
            messages=messages,
            source_pair_id=f"{trajectory.meta.task_id}_step{step.step_number}",
        ))

        # Update context for subsequent examples
        tool_result = _reconstruct_tool_result(step)
        if tool_result and len(tool_result) > _MAX_TOOL_RESULT_CHARS:
            tool_result = tool_result[:_MAX_TOOL_RESULT_CHARS] + "..."
        context_parts.append(f"Step {step.step_number}: {assistant_msg[:200]}")
        if tool_result:
            context_parts.append(f"Result: {tool_result[:200]}")

    return examples


def trajectories_to_training_examples(
    trajectories: list[AgentTrajectory],
    tasks: list[TaskInfo],
    only_successful: bool = False,
    per_step: bool = False,
) -> list[TrainingExample]:
    """Convert trajectories to XML-format training examples.

    Args:
        trajectories: Agent trajectories from base model runs.
        tasks: Corresponding task definitions (needed for problem_statement).
        only_successful: If True, only include trajectories that generated patches.
        per_step: If True, split each trajectory into per-step examples (shorter,
                  fits in memory for 14B+ models).
    """
    task_map = {t.instance_id: t for t in tasks}
    examples = []

    for traj in trajectories:
        task = task_map.get(traj.meta.task_id)
        if task is None:
            logger.warning(f"No task found for trajectory {traj.meta.task_id}")
            continue

        if only_successful and not traj.generated_patch.strip():
            continue

        if per_step:
            step_examples = trajectory_to_step_examples(traj, task)
            examples.extend(step_examples)
            logger.info(
                f"  Trajectory {traj.meta.task_id}: "
                f"{len(step_examples)} step examples, "
                f"{traj.outcome}"
            )
        else:
            example = trajectory_to_training_example(traj, task)
            if example is not None:
                examples.append(example)
                logger.info(
                    f"  Trajectory {traj.meta.task_id}: "
                    f"{len(example.messages)} messages, "
                    f"{traj.outcome}"
                )

    logger.info(
        f"Converted {len(trajectories)} trajectories "
        f"to {len(examples)} training examples"
    )
    return examples
