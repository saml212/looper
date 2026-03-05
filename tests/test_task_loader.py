"""Tests for the SWE-Bench-CL task loader."""

import os
from pathlib import Path

import pytest

from looper.models import TaskInfo
from looper.tasks.loader import (
    get_repo_tasks,
    get_task_by_id,
    load_curriculum,
    split_tasks,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_CURRICULUM = FIXTURES_DIR / "mini_curriculum.json"

REAL_DATASET = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")


# --- load_curriculum ---


def test_load_curriculum_reads_fixture():
    data = load_curriculum(MINI_CURRICULUM)
    assert isinstance(data, dict)
    assert "metadata" in data
    assert "sequences" in data
    assert data["metadata"]["total_tasks"] == 4


# --- get_repo_tasks ---


def test_get_repo_tasks_returns_task_info_list():
    data = load_curriculum(MINI_CURRICULUM)
    tasks = get_repo_tasks(data, "fake/repo")
    assert len(tasks) == 4
    assert all(isinstance(t, TaskInfo) for t in tasks)


def test_get_repo_tasks_maps_fields_correctly():
    data = load_curriculum(MINI_CURRICULUM)
    tasks = get_repo_tasks(data, "fake/repo")
    first = tasks[0]
    assert first.instance_id == "fake__repo-100"
    assert first.repo == "fake/repo"
    assert first.base_commit == "aaa111"
    assert first.problem_statement == "Add __iter__ to Paginator"
    assert first.hints_text == "Some hint"
    assert first.patch.startswith("diff --git")
    assert first.test_patch.startswith("diff --git")
    assert first.fail_to_pass == ["tests.test::Test::test_iter"]
    assert first.pass_to_pass == ["tests.test::Test::test_other"]
    assert first.difficulty == "<15 min fix"
    assert first.created_at == "2020-01-01T00:00:00+00:00"
    assert first.sequence_position == 1


def test_get_repo_tasks_unknown_repo_returns_empty():
    data = load_curriculum(MINI_CURRICULUM)
    tasks = get_repo_tasks(data, "nonexistent/repo")
    assert tasks == []


# --- split_tasks ---


def test_split_tasks_chronological_no_seed():
    data = load_curriculum(MINI_CURRICULUM)
    tasks = get_repo_tasks(data, "fake/repo")
    train, test = split_tasks(tasks, train_size=2)
    assert len(train) == 2
    assert len(test) == 2
    # Chronological: first 2 in train, last 2 in test
    assert train[0].instance_id == "fake__repo-100"
    assert train[1].instance_id == "fake__repo-200"
    assert test[0].instance_id == "fake__repo-300"
    assert test[1].instance_id == "fake__repo-400"


def test_split_tasks_with_seed_is_random_and_deterministic():
    data = load_curriculum(MINI_CURRICULUM)
    tasks = get_repo_tasks(data, "fake/repo")

    train1, test1 = split_tasks(tasks, train_size=2, seed=42)
    train2, test2 = split_tasks(tasks, train_size=2, seed=42)

    # Deterministic: same seed produces same split
    assert [t.instance_id for t in train1] == [t.instance_id for t in train2]
    assert [t.instance_id for t in test1] == [t.instance_id for t in test2]

    # Correct sizes
    assert len(train1) == 2
    assert len(test1) == 2

    # All tasks accounted for
    all_ids = {t.instance_id for t in train1} | {t.instance_id for t in test1}
    assert len(all_ids) == 4


# --- get_task_by_id ---


def test_get_task_by_id_finds_existing():
    data = load_curriculum(MINI_CURRICULUM)
    tasks = get_repo_tasks(data, "fake/repo")
    task = get_task_by_id(tasks, "fake__repo-300")
    assert task is not None
    assert task.instance_id == "fake__repo-300"
    assert task.problem_statement == "Refactor cache layer"


def test_get_task_by_id_returns_none_for_missing():
    data = load_curriculum(MINI_CURRICULUM)
    tasks = get_repo_tasks(data, "fake/repo")
    task = get_task_by_id(tasks, "fake__repo-999")
    assert task is None


# --- Integration test with real dataset ---


@pytest.mark.skipif(
    not REAL_DATASET.exists(),
    reason=f"Real dataset not found at {REAL_DATASET}",
)
def test_real_dataset_django_has_50_tasks():
    data = load_curriculum(REAL_DATASET)
    tasks = get_repo_tasks(data, "django/django")
    assert len(tasks) == 50
    # Verify all are valid TaskInfo instances
    assert all(isinstance(t, TaskInfo) for t in tasks)
    # Verify sequence positions are 1-50
    positions = sorted(t.sequence_position for t in tasks)
    assert positions == list(range(1, 51))
