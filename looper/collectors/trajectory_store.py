"""Save, load, and batch-collect agent trajectories."""

from pathlib import Path
from typing import Callable, Optional

from looper.models import AgentTrajectory, TaskInfo
from looper.agent.runner import run_agent


def save_trajectory(trajectory: AgentTrajectory, output_dir: Path) -> Path:
    """Save a trajectory to a JSON file.

    File is saved as: output_dir/<task_id>.json
    Returns the file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{trajectory.meta.task_id}.json"
    path.write_text(trajectory.model_dump_json(indent=2))
    return path


def load_trajectory(path: Path) -> AgentTrajectory:
    """Load a trajectory from a JSON file."""
    return AgentTrajectory.model_validate_json(path.read_text())


def load_all_trajectories(directory: Path) -> list[AgentTrajectory]:
    """Load all trajectory JSON files from a directory.

    Returns them sorted by task_id.
    """
    if not directory.exists():
        return []

    trajectories = []
    for path in directory.glob("*.json"):
        trajectories.append(load_trajectory(path))

    trajectories.sort(key=lambda t: t.meta.task_id)
    return trajectories


def collect_trajectories(
    tasks: list[TaskInfo],
    output_dir: Path,
    workspace_root: Path,
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434",
    max_steps: int = 25,
    max_tokens: int = 4096,
    on_complete: Optional[Callable] = None,
    chat_fn=None,
    rag_contexts: dict[str, str] | None = None,
) -> list[AgentTrajectory]:
    """Run the agent on each task and save trajectories.

    Saves each trajectory to disk immediately after completion.
    If a trajectory file already exists for a task, skip it (resume support).
    Returns list of all trajectories (including previously saved ones).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories: list[AgentTrajectory] = []

    for task in tasks:
        existing_path = output_dir / f"{task.instance_id}.json"

        if existing_path.exists():
            # Resume support: load existing trajectory instead of re-running
            traj = load_trajectory(existing_path)
        else:
            rag_ctx = (rag_contexts or {}).get(task.instance_id, "")
            traj = run_agent(
                task=task,
                workspace_root=workspace_root,
                model=model,
                base_url=base_url,
                max_steps=max_steps,
                max_tokens=max_tokens,
                chat_fn=chat_fn,
                rag_context=rag_ctx,
            )
            save_trajectory(traj, output_dir)

            if on_complete is not None:
                on_complete(task.instance_id, traj)

        trajectories.append(traj)

    return trajectories
