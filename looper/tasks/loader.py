"""SWE-Bench-CL curriculum loader.

Loads the curriculum JSON, extracts tasks by repo, and provides
simple utilities for splitting and lookup.
"""

import json
import random
from pathlib import Path

from looper.models import TaskInfo


def load_curriculum(path: Path) -> dict:
    """Load the SWE-Bench-CL curriculum JSON file. Returns raw dict."""
    with open(path) as f:
        return json.load(f)


def get_repo_tasks(curriculum: dict, repo: str) -> list[TaskInfo]:
    """Extract all tasks for a given repo, returned as TaskInfo models.

    Maps from the nested JSON structure to flat TaskInfo objects.
    """
    for sequence in curriculum["sequences"]:
        if sequence["repo"] == repo:
            return [_task_from_json(t) for t in sequence["tasks"]]
    return []


def split_tasks(
    tasks: list[TaskInfo],
    train_size: int = 25,
    seed: int | None = None,
) -> tuple[list[TaskInfo], list[TaskInfo]]:
    """Split tasks into train/test sets.

    If seed is None: first train_size tasks go to train, rest to test
    (chronological split).
    If seed is provided: random split with that seed.
    """
    if seed is None:
        return tasks[:train_size], tasks[train_size:]

    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:train_size], shuffled[train_size:]


def get_task_by_id(tasks: list[TaskInfo], instance_id: str) -> TaskInfo | None:
    """Find a task by its instance_id."""
    for task in tasks:
        if task.instance_id == instance_id:
            return task
    return None


def _task_from_json(raw: dict) -> TaskInfo:
    """Convert a raw task dict from the curriculum JSON to a TaskInfo."""
    meta = raw["metadata"]
    task = raw["task"]
    evl = raw["evaluation"]
    cl = raw["continual_learning"]

    return TaskInfo(
        instance_id=meta["instance_id"],
        repo=meta["repo"],
        base_commit=meta["base_commit"],
        problem_statement=task["problem_statement"],
        hints_text=task.get("hints_text", ""),
        patch=evl["patch"],
        test_patch=evl["test_patch"],
        fail_to_pass=evl.get("FAIL_TO_PASS", []),
        pass_to_pass=evl.get("PASS_TO_PASS", []),
        difficulty=meta["difficulty"],
        created_at=meta["created_at"],
        sequence_position=cl["sequence_position"],
    )
