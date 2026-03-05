"""Convert an AgentTrajectory to a human-readable text summary for LLM synthesis."""

from looper.models import AgentTrajectory

MAX_TOOL_RESULT_LENGTH = 500


def trajectory_to_text(trajectory: AgentTrajectory) -> str:
    """Convert a trajectory to a human-readable text summary.

    Includes: task_id, outcome, each step's reasoning and tool calls with results.
    Truncates very long tool results to keep the summary manageable.
    """
    lines: list[str] = []
    lines.append(f"Task: {trajectory.meta.task_id}")
    lines.append(f"Outcome: {trajectory.outcome}")
    lines.append(f"Model: {trajectory.meta.model_name}")
    lines.append(f"Steps: {len(trajectory.steps)}")
    lines.append("")

    for step in trajectory.steps:
        lines.append(f"--- Step {step.step_number} ---")
        lines.append(f"Reasoning: {step.reasoning}")

        for tc in step.tool_calls:
            lines.append(f"  Tool: {tc.tool_name}")
            lines.append(f"  Input: {tc.tool_input}")
            result = tc.tool_result
            if len(result) > MAX_TOOL_RESULT_LENGTH:
                result = result[:MAX_TOOL_RESULT_LENGTH] + "..."
            lines.append(f"  Result: {result}")
            lines.append(f"  Success: {tc.success}")
        lines.append("")

    if trajectory.generated_patch:
        lines.append(f"Generated patch: {trajectory.generated_patch[:500]}")

    return "\n".join(lines)
