"""Parse OpenClaw session JSONL files into looper AgentTrajectory models.

OpenClaw stores session data as newline-delimited JSON with these event types:
- session: session metadata (session_id, agent_id, created_at)
- model_change: model switch events
- thinking_level_change: thinking level updates
- custom: arbitrary key-value metadata (used for task_id)
- message: conversation messages (user, assistant, toolResult)

Assistant messages contain content items: text, thinking, toolCall.
Tool results contain toolCallId, toolName, content, and details (exitCode, durationMs).
"""

import json
from pathlib import Path

from looper.models import AgentTrajectory, AgentStep, ToolCall, SessionMeta


def parse_session(path: Path) -> AgentTrajectory:
    """Parse an OpenClaw session JSONL file into an AgentTrajectory."""
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Session file is empty: {path}")

    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    meta, steps = _parse_events(lines)

    # Determine outcome
    all_success = all(tc.success for step in steps for tc in step.tool_calls)
    outcome = "completed" if all_success else "error"

    return AgentTrajectory(
        meta=meta,
        steps=steps,
        outcome=outcome,
    )


def _parse_events(lines: list[dict]) -> tuple[SessionMeta, list[AgentStep]]:
    """Process raw JSONL events into SessionMeta and AgentSteps."""
    session_id = ""
    agent_id = "default"
    task_id = "unknown"
    model_name = ""
    started_at = ""
    ended_at = ""
    total_tokens = 0

    # Collect assistant messages with their tool calls pending results
    pending_steps: list[dict] = []  # {reasoning, tool_calls_pending: [{id, name, arguments}]}
    steps: list[AgentStep] = []
    tool_results: dict[str, dict] = {}  # toolCallId -> {content, success, duration_ms}

    for event in lines:
        etype = event.get("type")

        if etype == "session":
            session_id = event.get("session_id", "")
            agent_id = event.get("agent_id", "default")
            started_at = event.get("created_at", "")

        elif etype == "model_change":
            model_name = event.get("model", "")

        elif etype == "custom":
            if event.get("key") == "task_id":
                task_id = event.get("value", "unknown")

        elif etype == "message":
            msg = event.get("message", {})
            role = msg.get("role", "")
            timestamp = msg.get("timestamp", "")

            if role == "assistant":
                content_items = msg.get("content", [])
                usage = msg.get("usage", {})
                total_tokens += usage.get("totalTokens", 0)
                ended_at = timestamp

                # Extract reasoning text and tool calls
                reasoning_parts = []
                tool_calls_pending = []

                for item in content_items:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            reasoning_parts.append(item.get("text", ""))
                        elif item.get("type") == "toolCall":
                            tool_calls_pending.append({
                                "id": item.get("id", ""),
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", {}),
                            })

                if tool_calls_pending:
                    pending_steps.append({
                        "reasoning": " ".join(reasoning_parts).strip(),
                        "tool_calls_pending": tool_calls_pending,
                        "timestamp": timestamp,
                    })

            elif role == "toolResult":
                tool_call_id = msg.get("toolCallId", "")
                details = msg.get("details", {})
                exit_code = details.get("exitCode", 0)
                duration_ms = details.get("durationMs", 0)
                ended_at = timestamp

                # Content can be a string or list of {type: "text", text: "..."}
                raw_content = msg.get("content", "")
                if isinstance(raw_content, list):
                    raw_content = "\n".join(
                        item.get("text", "") for item in raw_content
                        if isinstance(item, dict)
                    )

                tool_results[tool_call_id] = {
                    "content": raw_content,
                    "success": exit_code == 0,
                    "duration_ms": duration_ms,
                }

    # Build steps from pending assistant messages + their tool results
    step_number = 0
    for pending in pending_steps:
        resolved_tools = []
        for tc in pending["tool_calls_pending"]:
            result = tool_results.get(tc["id"], {})
            resolved_tools.append(ToolCall(
                tool_name=tc["name"],
                tool_input=tc["arguments"],
                tool_result=result.get("content", ""),
                success=result.get("success", True),
                duration_ms=result.get("duration_ms", 0),
            ))

        step_number += 1
        step_ts = pending["timestamp"]
        if isinstance(step_ts, (int, float)):
            from datetime import datetime, timezone
            step_ts = datetime.fromtimestamp(step_ts / 1000, tz=timezone.utc).isoformat()
        steps.append(AgentStep(
            step_number=step_number,
            reasoning=pending["reasoning"],
            tool_calls=resolved_tools,
            timestamp=str(step_ts) if step_ts else "",
        ))

    # Convert timestamps if they are integers (ms since epoch)
    if isinstance(started_at, (int, float)):
        from datetime import datetime, timezone
        started_at = datetime.fromtimestamp(started_at / 1000, tz=timezone.utc).isoformat()
    if isinstance(ended_at, (int, float)):
        from datetime import datetime, timezone
        ended_at = datetime.fromtimestamp(ended_at / 1000, tz=timezone.utc).isoformat()

    meta = SessionMeta(
        session_id=session_id,
        agent_id=agent_id,
        task_id=task_id,
        model_name=model_name,
        started_at=str(started_at) if started_at else "",
        ended_at=str(ended_at) if ended_at else "",
        total_tokens=total_tokens,
        total_steps=len(steps),
    )

    return meta, steps
