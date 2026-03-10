"""Oracle SFT synthesizer: training data from ground truth patches.

Instead of extracting Q&A pairs from (mostly failed) agent trajectories,
this synthesizer uses the GOLD PATCHES from SWE-Bench as the training signal.

For each task, it creates training examples:
  User: {problem_statement} + {relevant code context}
  Assistant: {gold patch as diff}

This teaches the model what correct Django fixes look like, rather than
teaching it how to narrate tool usage from failed attempts.
"""

import json
import logging
from pathlib import Path

from looper.models import TaskInfo, TrainingExample

logger = logging.getLogger(__name__)


def _extract_modified_files(patch: str) -> list[str]:
    """Extract file paths modified in a diff patch."""
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                # 'diff --git a/path b/path' -> 'path'
                files.append(parts[3].lstrip("b/"))
    return files


def oracle_sft_examples(
    tasks: list[TaskInfo],
    format: str = "patch",
) -> list[TrainingExample]:
    """Generate training examples from ground truth patches.

    Args:
        tasks: Tasks with gold patches in task.patch
        format: One of:
            - "patch": Problem -> diff patch (teaches patch writing)
            - "agentic": Problem -> XML tool calls with the fix (matches agent format)
            - "code_context": Problem + file context -> corrected code

    Returns:
        List of TrainingExamples in chat format.
    """
    examples = []
    for task in tasks:
        if not task.patch.strip():
            logger.warning(f"Skipping {task.instance_id}: no gold patch")
            continue

        if format == "patch":
            examples.extend(_format_patch(task))
        elif format == "agentic":
            examples.extend(_format_agentic(task))
        elif format == "code_context":
            examples.extend(_format_code_context(task))
        else:
            raise ValueError(f"Unknown format: {format}")

    logger.info(f"Generated {len(examples)} oracle SFT examples from {len(tasks)} tasks")
    return examples


def _format_patch(task: TaskInfo) -> list[TrainingExample]:
    """Format as: problem statement -> git diff patch."""
    modified_files = _extract_modified_files(task.patch)
    files_str = ", ".join(modified_files) if modified_files else "unknown"

    user_msg = (
        f"Fix the following Django bug. The relevant file(s): {files_str}\n\n"
        f"Bug report:\n{task.problem_statement}\n\n"
        f"Provide your fix as a unified diff patch."
    )

    assistant_msg = task.patch

    return [TrainingExample(
        messages=[
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        source_pair_id=f"oracle_{task.instance_id}",
    )]


def _format_agentic(task: TaskInfo) -> list[TrainingExample]:
    """Format as XML tool calls matching the agent's action format.

    Creates a multi-turn example that teaches the agent to:
    1. Read the relevant file
    2. Write the fix
    3. Signal done
    """
    modified_files = _extract_modified_files(task.patch)
    if not modified_files:
        return _format_patch(task)

    # Build a plausible agentic trajectory from the gold patch
    target_file = modified_files[0]

    user_msg = (
        f"{task.problem_statement}\n\n"
        f"The repository is cloned at /workspace. Use XML tool tags:\n"
        f"- <bash>command</bash> - Run a shell command\n"
        f"- <read>path/to/file</read> - Read a file\n"
        f"- <write path=\"path/to/file\">content</write> - Write a file\n"
        f"- <done> - Signal you are finished\n\n"
        f"IMPORTANT: Use exactly ONE tool per response."
    )

    # Step 1: Read the relevant file
    assistant_step1 = (
        f"Let me examine the relevant code.\n"
        f"<read>{target_file}</read>"
    )

    # Step 2: Apply the fix (we use the diff as the response since
    # we don't have the full file content for a write)
    assistant_step2 = (
        f"I can see the issue. Here's the fix as a patch:\n\n"
        f"```diff\n{task.patch}\n```\n\n"
        f"<done>"
    )

    return [TrainingExample(
        messages=[
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_step1},
            {"role": "user", "content": f"[read] Contents of {target_file}:\n(file contents)"},
            {"role": "assistant", "content": assistant_step2},
        ],
        source_pair_id=f"oracle_agentic_{task.instance_id}",
    )]


def _format_code_context(task: TaskInfo) -> list[TrainingExample]:
    """Format as: problem + diff context -> corrected code.

    Extracts the 'before' code from the diff and teaches the model
    to produce the 'after' code.
    """
    # Parse the diff to extract before/after code
    chunks = _parse_diff_chunks(task.patch)
    if not chunks:
        return _format_patch(task)

    examples = []
    for file_path, before_code, after_code in chunks:
        user_msg = (
            f"Fix the following bug in {file_path}:\n\n"
            f"Bug: {task.problem_statement[:500]}\n\n"
            f"Current code:\n```python\n{before_code}\n```\n\n"
            f"Provide the corrected code."
        )
        assistant_msg = f"```python\n{after_code}\n```"

        examples.append(TrainingExample(
            messages=[
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            source_pair_id=f"oracle_ctx_{task.instance_id}_{file_path}",
        ))

    return examples


def _parse_diff_chunks(patch: str) -> list[tuple[str, str, str]]:
    """Parse a unified diff into (file_path, before_code, after_code) tuples."""
    chunks = []
    current_file = None
    before_lines = []
    after_lines = []
    in_chunk = False

    for line in patch.splitlines():
        if line.startswith("diff --git"):
            # Save previous chunk
            if current_file and (before_lines or after_lines):
                chunks.append((
                    current_file,
                    "\n".join(before_lines),
                    "\n".join(after_lines),
                ))
            parts = line.split()
            current_file = parts[3].lstrip("b/") if len(parts) >= 4 else None
            before_lines = []
            after_lines = []
            in_chunk = False
        elif line.startswith("@@"):
            in_chunk = True
        elif in_chunk:
            if line.startswith("-") and not line.startswith("---"):
                before_lines.append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                after_lines.append(line[1:])
            elif not line.startswith("\\"):
                # Context line - appears in both
                before_lines.append(line[1:] if line.startswith(" ") else line)
                after_lines.append(line[1:] if line.startswith(" ") else line)

    # Save last chunk
    if current_file and (before_lines or after_lines):
        chunks.append((
            current_file,
            "\n".join(before_lines),
            "\n".join(after_lines),
        ))

    return chunks


def save_oracle_training_data(
    tasks: list[TaskInfo],
    output_path: Path,
    format: str = "patch",
) -> list[TrainingExample]:
    """Generate oracle SFT examples and save as JSONL."""
    examples = oracle_sft_examples(tasks, format=format)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump()) + "\n")

    logger.info(f"Saved {len(examples)} oracle examples to {output_path}")
    return examples
