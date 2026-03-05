"""Agent runner for SWE-Bench tasks.

Implements a simple text-based tool protocol using XML-style tags,
suitable for small (7B) models that struggle with JSON tool calling.
"""

import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from looper.models import TaskInfo, AgentTrajectory, AgentStep, ToolCall, SessionMeta
from looper.agent.ollama_client import chat, ChatMessage, ChatResponse, mlx_chat
from looper.agent.workspace import create_workspace, get_patch, run_in_workspace

SYSTEM_PROMPT = """You are a software engineer fixing a bug in a Python project.

The repository is checked out at: {workspace_dir}

Available tools (use ONE tool per response, using XML tags):
- <bash>command</bash> — Run a shell command in the repo directory
- <read>path/to/file</read> — Read a file (relative to repo root)
- <write path="path/to/file">new file content</write> — Write/overwrite a file
- <done> — Signal you are finished (ONLY use this ALONE, after you have made and verified your fix)

IMPORTANT RULES:
1. Use exactly ONE tool per response. Wait for the result before using the next tool.
2. Start by exploring: use <bash>find . -type f -name "*.py" | head -20</bash> or <bash>grep -r "keyword" --include="*.py" -l</bash>
3. Read the relevant file BEFORE writing any changes.
4. Make minimal changes — edit only what's needed to fix the issue.
5. Only use <done> AFTER you have written your fix. Never combine <done> with other tools.

Problem to fix:
{problem_statement}
"""

# Regex patterns for parsing tool calls
_BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
_READ_RE = re.compile(r"<read>(.*?)</read>", re.DOTALL)
_WRITE_RE = re.compile(r'<write\s+path="([^"]+)">(.*?)</write>', re.DOTALL)
_DONE_RE = re.compile(r"<done>")

# Combined pattern to find all tool calls in order of appearance
_ALL_RE = re.compile(
    r"<bash>(.*?)</bash>"
    r"|<read>(.*?)</read>"
    r'|<write\s+path="([^"]+)">(.*?)</write>'
    r"|<done>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from model output.

    Returns list of dicts like:
        {"tool": "bash", "input": "ls -la"}
        {"tool": "read", "input": "path/to/file"}
        {"tool": "write", "input": "content", "path": "path/to/file"}
        {"tool": "done", "input": ""}
    """
    results = []
    for m in _ALL_RE.finditer(text):
        if m.group(1) is not None:
            results.append({"tool": "bash", "input": m.group(1)})
        elif m.group(2) is not None:
            results.append({"tool": "read", "input": m.group(2)})
        elif m.group(3) is not None:
            results.append({"tool": "write", "input": m.group(4), "path": m.group(3)})
        else:
            results.append({"tool": "done", "input": ""})
    return results


def execute_tool(tool: dict, workspace_dir: Path) -> tuple[str, bool]:
    """Execute a parsed tool call in the workspace.

    Returns (result_text, success).
    """
    name = tool["tool"]

    if name == "bash":
        output, returncode = run_in_workspace(workspace_dir, tool["input"])
        return output, returncode == 0

    elif name == "read":
        raw_path = tool["input"].strip()
        # Handle absolute paths: resolve relative to workspace
        if raw_path.startswith("/"):
            # Try to make it relative to workspace_dir
            try:
                rel = Path(raw_path).relative_to(workspace_dir)
                filepath = workspace_dir / rel
            except ValueError:
                # Absolute path outside workspace — strip leading /
                filepath = workspace_dir / raw_path.lstrip("/")
        else:
            filepath = workspace_dir / raw_path
        try:
            content = filepath.read_text()
            # Truncate large files to avoid context saturation
            max_chars = 12000
            if len(content) > max_chars:
                content = (
                    content[:max_chars]
                    + f"\n\n... [TRUNCATED: {len(content)} total chars, showing first {max_chars}]"
                )
            return content, True
        except Exception as e:
            return f"Error reading file: {e}", False

    elif name == "write":
        filepath = workspace_dir / tool["path"]
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(tool["input"])
            return f"Wrote {tool['path']}", True
        except Exception as e:
            return f"Error writing file: {e}", False

    elif name == "done":
        return "Done.", True

    else:
        return f"Unknown tool: {name}", False


def run_agent(
    task: TaskInfo,
    workspace_root: Path,
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434",
    max_steps: int = 25,
    max_tokens: int = 4096,
    chat_fn=None,
    rag_context: str = "",
) -> AgentTrajectory:
    """Run the agent on a SWE-Bench task and return the trajectory.

    1. Create workspace (clone repo at base_commit)
    2. Build system prompt (with optional RAG context)
    3. Agent loop: chat -> parse tools -> execute -> feed results back
    4. Extract patch from git diff
    5. Return AgentTrajectory
    """
    session_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # 1. Create workspace
    workspace_dir = create_workspace(task.repo, task.base_commit, workspace_root)

    # 2. Build system prompt
    system_prompt = SYSTEM_PROMPT.format(
        workspace_dir=workspace_dir,
        problem_statement=task.problem_statement,
    )
    if rag_context:
        system_prompt += f"\n\n{rag_context}"

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=task.problem_statement),
    ]

    steps: list[AgentStep] = []
    total_tokens = 0
    outcome = "max_steps"

    # Use custom chat function if provided (e.g., mlx_chat for adapted model)
    _chat_fn = chat_fn or chat

    # 3. Agent loop
    for step_num in range(1, max_steps + 1):
        response = _chat_fn(
            messages,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
        )
        total_tokens += response.total_tokens

        # Parse tool calls from model output
        parsed_tools = parse_tool_calls(response.content)

        # Build tool call records and execute
        tool_calls: list[ToolCall] = []
        results_text_parts: list[str] = []
        done = False

        for tc in parsed_tools:
            if tc["tool"] == "done":
                done = True
                tool_calls.append(
                    ToolCall(
                        tool_name="done",
                        tool_input=tc,
                        tool_result="Done.",
                        success=True,
                    )
                )
                break

            t_start = time.monotonic()
            result_text, success = execute_tool(tc, workspace_dir)
            duration_ms = int((time.monotonic() - t_start) * 1000)

            tool_calls.append(
                ToolCall(
                    tool_name=tc["tool"],
                    tool_input=tc,
                    tool_result=result_text,
                    success=success,
                    duration_ms=duration_ms,
                )
            )
            results_text_parts.append(f"[{tc['tool']}] {result_text}")

        # Strip tool calls from the response to get reasoning text
        reasoning = _ALL_RE.sub("", response.content).strip()

        steps.append(
            AgentStep(
                step_number=step_num,
                reasoning=reasoning,
                tool_calls=tool_calls,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

        if done:
            outcome = "completed"
            break

        # Append results as user message for next iteration
        messages.append(ChatMessage(role="assistant", content=response.content))
        if results_text_parts:
            messages.append(
                ChatMessage(role="user", content="\n".join(results_text_parts))
            )
        else:
            messages.append(
                ChatMessage(
                    role="user",
                    content="No tool calls detected. Please use one of the available tools.",
                )
            )

    # 4. Extract patch
    generated_patch = get_patch(workspace_dir)
    if generated_patch and outcome == "completed":
        outcome = "patch_generated"

    # 5. Return trajectory
    return AgentTrajectory(
        meta=SessionMeta(
            session_id=session_id,
            task_id=task.instance_id,
            model_name=model,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            total_tokens=total_tokens,
            total_steps=len(steps),
        ),
        steps=steps,
        outcome=outcome,
        generated_patch=generated_patch,
    )
