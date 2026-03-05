"""Orchestrate a Looper experiment using OpenClaw as the agent framework.

Uses `openclaw agent --local` to run agent turns, with an MLX server providing
the local model. Collects session JSONL files from ~/.openclaw/agents/ and
feeds them through the standard looper synthesis → training → evaluation pipeline.
"""

import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from looper.agent.workspace import create_workspace, get_patch
from looper.evaluators.metrics import forward_transfer
from looper.evaluators.patch_verifier import verify_patch_simple, verify_patch_tests
from looper.evaluators.results_io import save_results
from looper.integrations.openclaw_parser import parse_session
from looper.integrations.openclaw_provider import (
    generate_provider_config,
    write_provider_config,
    set_default_model,
    restore_default_model,
)
from looper.models import (
    AgentTrajectory,
    ExperimentConfig,
    ExperimentResult,
    TaskInfo,
    TaskResult,
)
from looper.synthesizers.synthesizer import (
    pairs_to_training_examples,
    save_training_data,
    synthesize_batch,
)
from looper.tasks.loader import split_tasks

logger = logging.getLogger(__name__)

SKILL_TEMPLATE_PATH = Path(__file__).parent / "openclaw_skill" / "SKILL.md"


@dataclass
class OpenClawExperimentConfig:
    """Configuration for an OpenClaw-based experiment run."""

    # Model
    model_name: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    hf_model_name: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    provider_name: str = "looper-mlx"
    provider_port: int = 8080

    # Agent
    max_steps: int = 15
    agent_timeout: int = 900  # seconds per task

    # Training
    lora_rank: int = 16
    lora_iters: int = 100
    num_pairs_per_trajectory: int = 5

    # Paths
    output_dir: Path = field(
        default_factory=lambda: Path("/Volumes/1TB_SSD/looper/results/phase1_openclaw")
    )
    workspace_root: Path = field(
        default_factory=lambda: Path("/Volumes/1TB_SSD/looper/cache/workspaces")
    )
    openclaw_config_path: Path = field(
        default_factory=lambda: Path.home() / ".openclaw" / "openclaw.json"
    )
    openclaw_sessions_dir: Path = field(
        default_factory=lambda: Path.home() / ".openclaw" / "agents" / "main" / "sessions"
    )

    # Split
    train_size: int = 25
    split_seed: int | None = None

    # Pilot mode
    adapted_test_size: int | None = 5  # None = all test tasks


def render_skill(problem_statement: str, workspace_dir: Path) -> str:
    """Render the SKILL.md template with task-specific values."""
    template = SKILL_TEMPLATE_PATH.read_text()
    return (
        template
        .replace("{{problem_statement}}", problem_statement)
        .replace("{{workspace_dir}}", str(workspace_dir))
    )


def start_mlx_server(
    model_name: str,
    port: int,
    adapter_path: str | None = None,
) -> subprocess.Popen:
    """Start mlx_lm.server and wait until it's ready."""
    import sys
    venv_bin = Path(sys.executable).parent
    mlx_server = str(venv_bin / "mlx_lm.server")
    cmd = [mlx_server, "--model", model_name, "--port", str(port)]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for server to become ready
    import httpx
    for _ in range(30):
        time.sleep(2)
        try:
            httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
            logger.info(f"  MLX server ready on port {port}")
            return proc
        except Exception:
            continue

    proc.terminate()
    raise RuntimeError(f"MLX server failed to start on port {port}")


def stop_mlx_server(proc: subprocess.Popen | None) -> None:
    """Stop an MLX server process."""
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=10)
        logger.info("  MLX server stopped")


def _openclaw_agent_turn(
    session_id: str,
    message: str,
    timeout: int = 300,
) -> str:
    """Run one openclaw agent turn and return the assistant's text response."""
    cmd = [
        "openclaw", "agent",
        "--local",
        "--session-id", session_id,
        "--message", message,
        "--timeout", str(timeout),
        "--json",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout + 60
    )
    if result.returncode != 0:
        logger.warning(f"  OpenClaw error: {result.stderr[:200]}")
        return ""

    # Parse JSON response to get the text
    try:
        import json as _json
        data = _json.loads(result.stdout)
        payloads = data.get("payloads", [])
        if payloads:
            return payloads[0].get("text", "")
    except (ValueError, KeyError):
        pass
    return result.stdout


def run_openclaw_on_task(
    task: TaskInfo,
    workspace_root: Path,
    provider_name: str,
    model_name: str,
    skill_dir: Path,
    max_steps: int = 15,
    timeout: int = 300,
) -> tuple[str, Path | None]:
    """Run OpenClaw agent loop on a single SWE-Bench task.

    Since the 7B model can't do native tool calling, we drive a multi-turn
    loop: send task → parse XML tool calls from text → execute locally →
    send results back as next message.

    Returns (session_id, workspace_dir).
    """
    from looper.agent.runner import parse_tool_calls, execute_tool

    workspace_dir = create_workspace(task.repo, task.base_commit, workspace_root)

    # Render skill with task-specific content
    rendered = render_skill(task.problem_statement, workspace_dir)
    (skill_dir / "SKILL.md").write_text(rendered)

    session_id = str(uuid.uuid4())

    # Initial message with the task
    initial_message = (
        f"{rendered}\n\n"
        f"The repository is cloned at {workspace_dir}. Use XML tool tags:\n"
        f"- <bash>command</bash> — Run a shell command in the repo\n"
        f"- <read>path/to/file</read> — Read a file (paths relative to repo root)\n"
        f'- <write path="path/to/file">content</write> — Write a file\n'
        f"- <done> — Signal you are finished\n\n"
        f"IMPORTANT: Use exactly ONE tool per response.\n"
        f"FIRST STEP: Explore the repo to find relevant files. For example:\n"
        f"<bash>find {workspace_dir} -type f -name '*.py' | grep -i 'relevant_keyword' | head -20</bash>\n\n"
        f"Start by exploring. Do NOT guess file paths."
    )

    logger.info(
        f"  Running OpenClaw on {task.instance_id} (session={session_id[:8]}...)..."
    )

    # Multi-turn agent loop
    next_message = initial_message
    for step in range(1, max_steps + 1):
        response_text = _openclaw_agent_turn(session_id, next_message, timeout)
        if not response_text:
            logger.warning(f"    Step {step}: empty response, stopping")
            break

        # Parse tool calls from the model's text response
        tool_calls = parse_tool_calls(response_text)
        logger.debug(f"    Raw response ({len(response_text)} chars): {response_text[:300]}")
        if not tool_calls:
            # No tool calls detected — nudge the model
            next_message = (
                "No tool calls detected. Please use one of the XML tool tags: "
                "<bash>command</bash>, <read>path</read>, "
                '<write path="path">content</write>, or <done>.'
            )
            continue

        # Execute the first tool call
        tc = tool_calls[0]
        if tc["tool"] == "done":
            logger.info(f"    Step {step}: <done>")
            break

        result_text, success = execute_tool(tc, workspace_dir)
        logger.info(
            f"    Step {step}: <{tc['tool']}> -> "
            f"{'ok' if success else 'error'} ({len(result_text)} chars)"
        )
        if not success:
            logger.info(f"    Error detail: {result_text[:200]}")
            logger.info(f"    Tool input: {tc.get('input', '')[:200]}")

        # Send tool result as next message (truncate to avoid context saturation)
        max_result_chars = 8000
        if len(result_text) > max_result_chars:
            result_text = (
                result_text[:max_result_chars]
                + f"\n... [TRUNCATED: {len(result_text)} total chars]"
            )
        next_message = f"[{tc['tool']}] {result_text}"

    return session_id, workspace_dir


def find_session_file(sessions_dir: Path, session_id: str) -> Path | None:
    """Find the session JSONL file for a given session ID."""
    target = sessions_dir / f"{session_id}.jsonl"
    if target.exists():
        return target
    # Fallback: search all files
    for path in sessions_dir.glob("*.jsonl"):
        try:
            first_line = path.open().readline()
            import json
            event = json.loads(first_line)
            if event.get("id") == session_id:
                return path
        except Exception:
            continue
    return None


def collect_task_trajectory(
    sessions_dir: Path,
    session_id: str,
    workspace_dir: Path | None,
    task_id: str,
) -> AgentTrajectory | None:
    """Collect a single task's trajectory from its OpenClaw session."""
    session_file = find_session_file(sessions_dir, session_id)
    if session_file is None:
        logger.warning(f"  Session file not found for {task_id} ({session_id[:8]})")
        return None

    try:
        traj = parse_session(session_file)
        # Override task_id since OpenClaw doesn't know our task ID scheme
        traj.meta.task_id = task_id
        # Extract patch from workspace if available
        if workspace_dir and workspace_dir.exists():
            traj.generated_patch = get_patch(workspace_dir)
        return traj
    except (ValueError, Exception) as e:
        logger.warning(f"  Failed to parse session for {task_id}: {e}")
        return None


def run_openclaw_experiment(
    config: OpenClawExperimentConfig,
    tasks: list[TaskInfo],
) -> ExperimentResult:
    """Run the full OpenClaw experiment pipeline.

    Steps:
    1. Start MLX server with base model
    2. Configure OpenClaw to use it
    3. Split tasks, run on train tasks
    4. Synthesize training data + train LoRA
    5. Restart MLX server with adapter, reconfigure OpenClaw
    6. Run on test tasks
    7. Evaluate, restore config, save results
    """
    started_at = datetime.now(timezone.utc).isoformat()
    original_model = "anthropic/claude-sonnet-4-6"  # Will restore after experiment

    # Create output directories
    synthesis_dir = config.output_dir / "synthesis"
    adapter_dir = config.output_dir / "adapter"
    skill_dir = config.output_dir / "skill"
    for d in [synthesis_dir, adapter_dir, skill_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mlx_proc = None
    try:
        # Step 1: Start MLX server with base model
        logger.info("Step 1: Starting MLX server (base model)...")
        mlx_proc = start_mlx_server(config.model_name, config.provider_port)

        # Step 2: Configure OpenClaw
        logger.info("Step 2: Configuring OpenClaw provider...")
        provider_config = generate_provider_config(
            port=config.provider_port,
            model_name=config.model_name,
        )
        write_provider_config(
            config.openclaw_config_path, provider_config, config.provider_name
        )
        set_default_model(
            config.openclaw_config_path, config.provider_name, config.model_name
        )

        # Step 3: Split and run on train tasks
        train_tasks, test_tasks = split_tasks(
            tasks, config.train_size, config.split_seed
        )
        logger.info(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")

        logger.info("Step 3: Running OpenClaw on train tasks...")
        train_sessions: list[tuple[str, str, Path | None]] = []  # (session_id, task_id, workspace)
        for task in train_tasks:
            session_id, workspace_dir = run_openclaw_on_task(
                task=task,
                workspace_root=config.workspace_root,
                provider_name=config.provider_name,
                model_name=config.model_name,
                skill_dir=skill_dir,
                max_steps=config.max_steps,
                timeout=config.agent_timeout,
            )
            train_sessions.append((session_id, task.instance_id, workspace_dir))

        # Collect train trajectories
        train_trajectories: list[AgentTrajectory] = []
        for session_id, task_id, workspace_dir in train_sessions:
            traj = collect_task_trajectory(
                config.openclaw_sessions_dir, session_id, workspace_dir, task_id
            )
            if traj:
                train_trajectories.append(traj)
        logger.info(f"  Collected {len(train_trajectories)} train trajectories")

        # Step 3b: Evaluate base model on train tasks
        base_results: list[TaskResult] = []
        for traj in train_trajectories:
            task = next(
                (t for t in train_tasks if t.instance_id == traj.meta.task_id), None
            )
            resolved = verify_patch_simple(task, traj.generated_patch) if task else False
            base_results.append(TaskResult(
                task_id=traj.meta.task_id,
                condition="base",
                resolved=resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            ))

        # Step 4: Synthesize + train
        adapter_path = None
        training_jsonl = synthesis_dir / "training.jsonl"

        if train_trajectories:
            if training_jsonl.exists():
                logger.info("Step 4: Loading cached training data...")
                from looper.synthesizers.synthesizer import load_training_data
                training_examples = load_training_data(training_jsonl)
            else:
                logger.info("Step 4: Synthesizing training data...")
                from looper.agent.ollama_client import openai_chat
                pairs = synthesize_batch(
                    trajectories=train_trajectories,
                    output_path=synthesis_dir / "pairs.json",
                    model=config.model_name,
                    base_url=f"http://127.0.0.1:{config.provider_port}",
                    num_pairs=config.num_pairs_per_trajectory,
                    chat_fn=openai_chat,
                )

                if pairs:
                    training_examples = pairs_to_training_examples(pairs)
                    save_training_data(training_examples, training_jsonl)
                    logger.info(
                        f"  {len(pairs)} pairs -> {len(training_examples)} examples"
                    )
                else:
                    training_examples = []

            if training_examples:
                adapter_file = adapter_dir / "adapters.safetensors"
                if adapter_file.exists():
                    logger.info("Step 4b: Adapter already trained, skipping...")
                else:
                    logger.info("Step 4b: Training LoRA adapter...")
                    # Stop MLX server to free GPU memory
                    stop_mlx_server(mlx_proc)
                    mlx_proc = None

                    import sys
                    import json as _json
                    train_script = f"""
import sys
sys.path.insert(0, '.')
from looper.trainers.full_replay import full_replay_train
from looper.trainers.lora_trainer import LoRAConfig
from looper.synthesizers.synthesizer import load_training_data
from pathlib import Path
import json

examples = load_training_data(Path('{training_jsonl}'))
config = LoRAConfig(rank={config.lora_rank}, iters={config.lora_iters}, batch_size=1, max_seq_length=1024)
metrics = full_replay_train(examples, '{config.hf_model_name}', Path('{adapter_dir}'), config)
print(json.dumps(metrics))
"""
                    train_result = subprocess.run(
                        [sys.executable, "-c", train_script],
                        capture_output=True, text=True,
                        cwd=str(Path(__file__).parent.parent.parent),
                    )
                    if train_result.returncode != 0:
                        logger.error(f"Training failed: {train_result.stderr[-500:]}")
                        raise RuntimeError("LoRA training failed")
                    logger.info("  Training complete")

                adapter_path = adapter_dir

        # Step 5: Restart MLX server with adapter (if trained)
        if adapter_path:
            logger.info("Step 5: Restarting MLX server with adapter...")
            stop_mlx_server(mlx_proc)
            mlx_proc = start_mlx_server(
                config.model_name, config.provider_port,
                adapter_path=str(adapter_path),
            )
            # Update provider config with adapter metadata
            provider_config = generate_provider_config(
                port=config.provider_port,
                model_name=config.model_name,
                adapter_path=adapter_path,
            )
            write_provider_config(
                config.openclaw_config_path, provider_config, config.provider_name
            )
        elif mlx_proc is None:
            # Restart base server if we stopped it for training
            mlx_proc = start_mlx_server(config.model_name, config.provider_port)

        # Step 6: Run on test tasks
        adapted_test_tasks = test_tasks
        if config.adapted_test_size is not None:
            adapted_test_tasks = test_tasks[:config.adapted_test_size]
        logger.info(
            f"Step 6: Running OpenClaw on {len(adapted_test_tasks)} test tasks..."
        )

        test_sessions: list[tuple[str, str, Path | None]] = []
        for task in adapted_test_tasks:
            session_id, workspace_dir = run_openclaw_on_task(
                task=task,
                workspace_root=config.workspace_root,
                provider_name=config.provider_name,
                model_name=config.model_name,
                skill_dir=skill_dir,
                max_steps=config.max_steps,
                timeout=config.agent_timeout,
            )
            test_sessions.append((session_id, task.instance_id, workspace_dir))

        # Collect test trajectories
        adapted_trajectories: list[AgentTrajectory] = []
        for session_id, task_id, workspace_dir in test_sessions:
            traj = collect_task_trajectory(
                config.openclaw_sessions_dir, session_id, workspace_dir, task_id
            )
            if traj:
                adapted_trajectories.append(traj)

        # Step 7: Evaluate
        logger.info("Step 7: Evaluating...")
        adapted_results: list[TaskResult] = []
        for traj in adapted_trajectories:
            task = next(
                (t for t in adapted_test_tasks if t.instance_id == traj.meta.task_id),
                None,
            )
            resolved = (
                verify_patch_simple(task, traj.generated_patch) if task else False
            )
            adapted_results.append(TaskResult(
                task_id=traj.meta.task_id,
                condition="adapted",
                resolved=resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            ))

        ft = forward_transfer(base_results, adapted_results)

        # Build and save results
        experiment_config = ExperimentConfig(
            name="phase1_openclaw",
            experiment_id="phase1_openclaw_full_replay",
            repo=tasks[0].repo if tasks else "unknown",
            model_name=config.model_name,
            train_task_ids=[t.instance_id for t in train_tasks],
            test_task_ids=[t.instance_id for t in adapted_test_tasks],
            strategy="full_replay",
            lora_rank=config.lora_rank,
            seed=config.split_seed or 0,
        )

        result = ExperimentResult(
            config=experiment_config,
            task_results=base_results + adapted_results,
            forward_transfer=ft,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        save_results(result, config.output_dir / "experiment_result.json")
        logger.info(f"Results saved to {config.output_dir / 'experiment_result.json'}")

        return result

    finally:
        # Cleanup: stop MLX server and restore OpenClaw config
        stop_mlx_server(mlx_proc)
        restore_default_model(config.openclaw_config_path, original_model)
        logger.info("Restored OpenClaw default model")
