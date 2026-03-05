"""Agent runner using MLX for direct inference with optional LoRA adapter.

Similar to runner.py but uses mlx_lm.generate instead of Ollama,
allowing direct use of LoRA adapters without a serving layer.
"""

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from looper.models import TaskInfo, AgentTrajectory, AgentStep, ToolCall, SessionMeta
from looper.agent.runner import parse_tool_calls, execute_tool, SYSTEM_PROMPT
from looper.agent.workspace import create_workspace, get_patch
from looper.trainers.lora_trainer import load_model_with_adapter

try:
    from mlx_lm import generate as mlx_generate
except ImportError:
    mlx_generate = None  # type: ignore[assignment]


def _build_prompt(messages: list[dict]) -> str:
    """Build a simple chat prompt string from message dicts.

    Uses a straightforward format compatible with instruction-tuned models.
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def run_agent_mlx(
    task: TaskInfo,
    workspace_root: Path,
    hf_model_name: str,
    adapter_path: Path | None = None,
    max_steps: int = 25,
    max_tokens: int = 4096,
) -> AgentTrajectory:
    """Run the agent using MLX directly (with optional LoRA adapter).

    Similar to run_agent but uses mlx_lm.generate instead of Ollama.
    This allows us to use the LoRA adapter directly without serving it.

    Args:
        task: The SWE-Bench task to work on.
        workspace_root: Root directory for creating workspaces.
        hf_model_name: HuggingFace model ID for MLX.
        adapter_path: Path to LoRA adapter directory (None for base model).
        max_steps: Maximum agent loop iterations.
        max_tokens: Maximum tokens per generation.

    Returns:
        AgentTrajectory with full session record.
    """
    session_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # 1. Create workspace
    workspace_dir = create_workspace(task.repo, task.base_commit, workspace_root)

    # 2. Load model (with optional adapter)
    model, tokenizer = load_model_with_adapter(hf_model_name, adapter_path=adapter_path)

    # 3. Build system prompt
    system_prompt = SYSTEM_PROMPT.format(
        workspace_dir=workspace_dir,
        problem_statement=task.problem_statement,
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.problem_statement},
    ]

    steps: list[AgentStep] = []
    outcome = "max_steps"

    # 4. Agent loop
    for step_num in range(1, max_steps + 1):
        prompt = _build_prompt(messages)
        response_text = mlx_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens
        )

        # Parse tool calls from model output
        parsed_tools = parse_tool_calls(response_text)

        # Build tool call records and execute
        tool_calls: list[ToolCall] = []
        results_text_parts: list[str] = []
        done = False

        for tc in parsed_tools:
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

            if tc["tool"] == "done":
                done = True

        # Strip tool calls from response for reasoning text
        from looper.agent.runner import _ALL_RE

        reasoning = _ALL_RE.sub("", response_text).strip()

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

        # Append results as next turn
        messages.append({"role": "assistant", "content": response_text})
        if results_text_parts:
            messages.append({"role": "user", "content": "\n".join(results_text_parts)})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "No tool calls detected. Please use one of the available tools.",
                }
            )

    # 5. Extract patch
    generated_patch = get_patch(workspace_dir)
    if generated_patch and outcome == "completed":
        outcome = "patch_generated"

    # 6. Return trajectory
    return AgentTrajectory(
        meta=SessionMeta(
            session_id=session_id,
            task_id=task.instance_id,
            model_name=f"mlx:{hf_model_name}",
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            total_tokens=0,  # MLX generate doesn't easily expose token counts
            total_steps=len(steps),
        ),
        steps=steps,
        outcome=outcome,
        generated_patch=generated_patch,
    )
