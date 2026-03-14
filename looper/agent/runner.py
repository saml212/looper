"""Agent runner for SWE-Bench tasks.

Implements a simple text-based tool protocol using XML-style tags,
suitable for small (7B) models that struggle with JSON tool calling.
"""

import difflib
import re
import time
import uuid
from collections import defaultdict
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
- <read>path/to/file:100-200</read> — Read lines 100-200 of a file
- <edit path="path/to/file">
exact old text
=======
new text
</edit> — Replace text in a file (PREFERRED for targeted fixes in files with 50+ lines)
- <write path="path/to/file">new file content</write> — Write/overwrite entire file (use for new files or files under 50 lines)
- <done> — Signal you are finished (ONLY use this ALONE, after you have made and verified your fix)

IMPORTANT RULES:
1. Use exactly ONE tool per response. Wait for the result before using the next tool.
2. Start by exploring: use <bash>find . -type f -name "*.py" | head -20</bash> or <bash>grep -r "keyword" --include="*.py" -l</bash>
3. Read the relevant file BEFORE writing any changes.
4. Make minimal changes — use <edit> for targeted fixes instead of rewriting entire files with <write>.
5. Only use <done> AFTER you have written your fix. Never combine <done> with other tools.
6. For large files, use line-range reads: <read>path/to/file:100-200</read>
7. Do NOT re-read the same file repeatedly. After reading a file, make your changes.
8. Do not attempt to install packages or create virtual environments. The workspace has all necessary dependencies pre-installed. To verify your fix is complete, use <done> immediately after writing — the evaluation framework handles test execution.
9. BEFORE making any edit, you MUST output a <think> block analyzing the fix:
   <think>
   1. What is the failing behavior described in the bug report?
   2. What is the current code doing wrong?
   3. What specific change will fix it, and why?
   </think>
   Then use ONE tool. The <think> block helps you avoid wrong fixes.

COMPLETE EXAMPLE — django-11066 (RenameContentType._rename() saves to wrong database):

Problem: content_type.save(update_fields={{'model'}}) doesn't pass `using=db`, so it saves to the default database instead of the one specified by the schema_editor.

Step 1 — Find relevant files:
<bash>grep -r "RenameContentType" --include="*.py" -l</bash>

Result: ./django/contrib/contenttypes/management/__init__.py

Step 2 — Read the relevant code:
<read>django/contrib/contenttypes/management/__init__.py:20-40</read>

Result:
20:             content_type = ContentType.objects.db_manager(db).get_by_natural_key(self.app_label, old_model)
21:         except ContentType.DoesNotExist:
22:             pass
23:         else:
24:             content_type.model = new_model
25:             try:
26:                 with transaction.atomic(using=db):
27:                     content_type.save(update_fields={{'model'}})
28:             except IntegrityError:
...

Step 3 — Analyze and fix:
<think>
1. The bug says content_type.save() writes to the default database instead of the one passed via schema_editor.connection.alias (variable `db`).
2. The code wraps save() in transaction.atomic(using=db), but save() itself doesn't receive `using=db` — so Django's default router picks the default database.
3. Fix: pass using=db to content_type.save() so it writes to the correct database.
</think>
<edit path="django/contrib/contenttypes/management/__init__.py">
    content_type.save(update_fields={{'model'}})
=======
    content_type.save(using=db, update_fields={{'model'}})
</edit>

Result: Edited django/contrib/contenttypes/management/__init__.py

Step 4 — Done:
<done>

Problem to fix:
{problem_statement}
"""

# Regex for <think> blocks (parsed but not counted as tool calls)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Regex patterns for parsing tool calls
_BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
_READ_RE = re.compile(r"<read>(.*?)</read>", re.DOTALL)
_WRITE_RE = re.compile(r'<write\s+path="([^"]+)">(.*?)</write>', re.DOTALL)
_DONE_RE = re.compile(r"<done>")
_EDIT_RE = re.compile(r'<edit\s+path="([^"]+)">(.*?)</edit>', re.DOTALL)

# Combined pattern to find all tool calls in order of appearance
_ALL_RE = re.compile(
    r"<bash>(.*?)</bash>"
    r"|<read>(.*?)</read>"
    r'|<write\s+path="([^"]+)">(.*?)</write>'
    r'|<edit\s+path="([^"]+)">(.*?)</edit>'
    r"|<done>",
    re.DOTALL,
)

# Separator for edit tool find/replace sections (3+ equals signs on own line)
_EDIT_SEPARATOR_RE = re.compile(r"\n={3,}\n")

# Pattern for line-range in read paths: file.py:100-200
_LINE_RANGE_RE = re.compile(r"^(.+):(\d+)-(\d+)$")


def parse_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from model output.

    Returns list of dicts like:
        {"tool": "bash", "input": "ls -la"}
        {"tool": "read", "input": "path/to/file"}
        {"tool": "read", "input": "path/to/file:100-200"}
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
        elif m.group(5) is not None:
            results.append({"tool": "edit", "input": m.group(6), "path": m.group(5)})
        else:
            results.append({"tool": "done", "input": ""})
    return results


def _resolve_read_path(
    raw_path: str, workspace_dir: Path
) -> tuple[Path, int | None, int | None]:
    """Parse a read path, extracting optional line range.

    Returns (filepath, start_line, end_line).
    start_line and end_line are 1-indexed, or None if no range specified.
    """
    # Check for line-range suffix (e.g., "file.py:100-200")
    range_match = _LINE_RANGE_RE.match(raw_path)
    if range_match:
        path_part = range_match.group(1).strip()
        start_line = int(range_match.group(2))
        end_line = int(range_match.group(3))
    else:
        path_part = raw_path
        start_line = None
        end_line = None

    # Handle absolute paths: resolve relative to workspace
    if path_part.startswith("/"):
        try:
            rel = Path(path_part).relative_to(workspace_dir)
            filepath = workspace_dir / rel
        except ValueError:
            filepath = workspace_dir / path_part.lstrip("/")
    else:
        filepath = workspace_dir / path_part

    return filepath, start_line, end_line


_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:\w+)?\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)


def _strip_code_fences(content: str) -> str:
    """Strip markdown code fences wrapping file content.

    Larger models (14B+) often wrap <write> content in ```python ... ```
    which corrupts source files. This strips the outermost fence if the
    entire content is wrapped in one.
    """
    m = _CODE_FENCE_RE.match(content)
    if m:
        return m.group(1)
    return content


def _parse_edit_content(raw: str) -> tuple[str, str] | None:
    """Parse edit content into (find_text, replace_text).

    Expected format (separated by a line of 3+ equals signs):
        old text to find
        =======
        new replacement text
    """
    content = raw.strip("\n")
    parts = _EDIT_SEPARATOR_RE.split(content, maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _fuzzy_find(
    content: str, find_text: str, threshold: float = 0.7
) -> tuple[int, int, float] | None:
    """Find the best fuzzy match for find_text in content.

    Uses a sliding window of lines from content, comparing against find_text.
    Returns (start_char_index, end_char_index, similarity_ratio) or None.
    """
    find_lines = find_text.splitlines(keepends=True)
    content_lines = content.splitlines(keepends=True)
    n_find = len(find_lines)

    if n_find == 0 or len(content_lines) == 0:
        return None

    best_ratio = 0.0
    best_start_line = 0

    # Slide a window of size n_find (±2 lines) across content
    for window_size in [n_find, n_find - 1, n_find + 1, n_find - 2, n_find + 2]:
        if window_size < 1 or window_size > len(content_lines):
            continue
        for i in range(len(content_lines) - window_size + 1):
            candidate = "".join(content_lines[i : i + window_size])
            ratio = difflib.SequenceMatcher(None, find_text, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start_line = i
                best_window_size = window_size

    if best_ratio < threshold:
        return None

    # Convert line indices to char indices
    start = sum(len(content_lines[i]) for i in range(best_start_line))
    end = sum(len(content_lines[i]) for i in range(best_start_line + best_window_size))
    return start, end, best_ratio


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
        filepath, start_line, end_line = _resolve_read_path(raw_path, workspace_dir)
        try:
            content = filepath.read_text()

            # Apply line range if specified
            if start_line is not None and end_line is not None:
                lines = content.splitlines()
                # Convert to 0-indexed, clamp to file length
                s = max(0, start_line - 1)
                e = min(len(lines), end_line)
                selected = lines[s:e]
                # Add line numbers for context
                content = "\n".join(
                    f"{s + i + 1}: {line}" for i, line in enumerate(selected)
                )
                total_lines = len(lines)
                content += f"\n\n[Showing lines {start_line}-{min(end_line, total_lines)} of {total_lines} total]"
                return content, True

            # Truncate large files to avoid context saturation
            max_chars = 12000
            if len(content) > max_chars:
                total = len(content)
                total_lines = len(content.splitlines())
                content = (
                    content[:max_chars]
                    + f"\n\n... [TRUNCATED: {total} total chars, {total_lines} lines."
                    f" Use <read>{raw_path}:START-END</read> to read specific line ranges.]"
                )
            return content, True
        except Exception as e:
            return f"Error reading file: {e}", False

    elif name == "write":
        filepath = workspace_dir / tool["path"]
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            content = _strip_code_fences(tool["input"])
            filepath.write_text(content)
            return f"Wrote {tool['path']}", True
        except Exception as e:
            return f"Error writing file: {e}", False

    elif name == "edit":
        filepath = workspace_dir / tool["path"]
        try:
            if not filepath.exists():
                return f"Error: file '{tool['path']}' does not exist", False
            parsed = _parse_edit_content(tool["input"])
            if parsed is None:
                return (
                    "Error: could not parse edit content. Use format:\n"
                    "old text\n=======\nnew text"
                ), False
            find_text, replace_text = parsed
            content = filepath.read_text()
            if find_text in content:
                new_content = content.replace(find_text, replace_text, 1)
                filepath.write_text(new_content)
                return f"Edited {tool['path']}", True
            # Fuzzy matching fallback
            match_result = _fuzzy_find(content, find_text, threshold=0.7)
            if match_result is not None:
                start, end, ratio = match_result
                new_content = content[:start] + replace_text + content[end:]
                filepath.write_text(new_content)
                return (
                    f"Edited {tool['path']} (fuzzy match, {ratio:.0%} similarity)"
                ), True
            return (
                f"Error: text not found in '{tool['path']}' (even with fuzzy matching). "
                "Re-read the file and copy the exact text, including whitespace and indentation."
            ), False
        except Exception as e:
            return f"Error editing file: {e}", False

    elif name == "done":
        return "Done.", True

    else:
        return f"Unknown tool: {name}", False


def prune_messages(
    messages: list[ChatMessage], max_tokens: int = 8000
) -> list[ChatMessage]:
    """Prune conversation history to stay under token budget.

    Replaces old, long tool-result messages with short summaries.
    Keeps the system prompt, initial user message, and recent messages intact.
    """
    # Rough token estimate: 1 token ~ 4 chars
    total_chars = sum(len(m.content) for m in messages)
    estimated_tokens = total_chars // 4

    if estimated_tokens <= max_tokens:
        return messages

    # Keep first 2 messages (system + initial user) and last 2 messages intact
    protected_start = 2
    protected_end = 2
    if len(messages) <= protected_start + protected_end:
        return messages

    pruned = list(messages)
    middle_end = len(pruned) - protected_end

    for i in range(protected_start, middle_end):
        msg = pruned[i]
        # Only prune long user messages (these contain tool results)
        if msg.role == "user" and len(msg.content) > 500:
            # Keep first 200 chars as summary
            summary = msg.content[:200]
            pruned[i] = ChatMessage(
                role="user",
                content=f"{summary}\n... [content pruned to save context — re-read the file if needed]",
            )

    return pruned


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

    # Loop detection: track consecutive reads of the same file without writes
    read_counts: dict[str, int] = defaultdict(int)
    # Edit loop detection: track consecutive failed edits per file
    edit_fail_counts: dict[str, int] = defaultdict(int)

    # Use custom chat function if provided (e.g., mlx_chat for adapted model)
    _chat_fn = chat_fn or chat

    # 3. Agent loop
    for step_num in range(1, max_steps + 1):
        # Context pruning: keep messages under budget
        pruned_messages = prune_messages(messages)

        response = _chat_fn(
            pruned_messages,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
        )
        total_tokens += response.total_tokens

        # Extract <think> blocks (logged but not counted as tool calls)
        think_blocks = _THINK_RE.findall(response.content)

        # Parse tool calls from model output
        parsed_tools = parse_tool_calls(response.content)

        # Build tool call records and execute
        # Enforce one-tool-per-response: only process the first tool call.
        # With higher temperatures, models often emit multiple tools + <done>
        # in a single response, leading to premature termination.
        tool_calls: list[ToolCall] = []
        results_text_parts: list[str] = []
        done = False

        for tc in parsed_tools[:1]:  # Only first tool call
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

            # Loop detection: track reads and reset on writes
            if tc["tool"] == "read":
                # Normalize path (strip line range for counting)
                read_path = tc["input"].strip()
                range_m = _LINE_RANGE_RE.match(read_path)
                if range_m:
                    read_path = range_m.group(1)
                read_counts[read_path] += 1
            elif tc["tool"] in ("write", "edit"):
                read_counts.clear()

            # Edit loop detection: track consecutive failed edits per file
            if tc["tool"] == "edit" and not success:
                edit_path = tc.get("path", "")
                edit_fail_counts[edit_path] += 1
            elif tc["tool"] == "edit" and success:
                edit_path = tc.get("path", "")
                edit_fail_counts.pop(edit_path, None)

        # Strip tool calls from the response to get reasoning text
        # Keep <think> content in reasoning for analysis
        reasoning = _ALL_RE.sub("", response.content).strip()
        if think_blocks:
            reasoning = "\n".join(f"<think>{tb}</think>" for tb in think_blocks) + "\n" + reasoning

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
            result_content = "\n".join(results_text_parts)

            # Loop detection: inject nudge if model keeps re-reading
            for path, count in read_counts.items():
                if count >= 3:
                    result_content += (
                        f"\n\n**WARNING: You have already read '{path}' {count} times."
                        " Please make your changes now using <write> or move on.**"
                    )

            # Edit loop detection: nudge after 3 consecutive failures on same file
            for path, count in edit_fail_counts.items():
                if count >= 3:
                    result_content += (
                        f"\n\n**Edit failed {count} times on '{path}'."
                        " Try re-reading it with a line-range read, or use <write> for small files.**"
                    )

            messages.append(ChatMessage(role="user", content=result_content))
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
