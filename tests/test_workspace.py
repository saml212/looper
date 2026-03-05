"""Tests for looper.agent.workspace — workspace management for SWE-Bench tasks."""

import subprocess
from pathlib import Path

import pytest

from looper.agent.workspace import (
    apply_patch,
    cleanup_workspace,
    create_workspace,
    get_patch,
    run_in_workspace,
)


@pytest.fixture
def local_bare_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a local bare git repo with two commits.

    Returns:
        (bare_repo_path, first_commit_hash, second_commit_hash)
    """
    # Create a working repo first
    work = tmp_path / "work_repo"
    work.mkdir()
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=work, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=work, check=True, capture_output=True,
    )

    # First commit
    (work / "hello.txt").write_text("hello world\n")
    subprocess.run(["git", "add", "hello.txt"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "first commit"],
        cwd=work, check=True, capture_output=True,
    )
    first = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work, check=True, capture_output=True, text=True,
    ).stdout.strip()

    # Second commit
    (work / "hello.txt").write_text("hello world\nupdated\n")
    (work / "readme.md").write_text("# readme\n")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "second commit"],
        cwd=work, check=True, capture_output=True,
    )
    second = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work, check=True, capture_output=True, text=True,
    ).stdout.strip()

    # Clone to a bare repo (used as the "remote" for tests)
    bare = tmp_path / "bare_repo.git"
    subprocess.run(
        ["git", "clone", "--bare", str(work), str(bare)],
        check=True, capture_output=True,
    )

    return bare, first, second


class TestCreateWorkspace:
    """Tests for create_workspace."""

    def test_clones_repo_and_checks_out_commit(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, first_commit, _second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()

        # Use the bare repo path as the "repo" argument
        ws = create_workspace(str(bare), first_commit, ws_root)

        assert ws.exists()
        # The checked-out commit should match first_commit
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ws, check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert head == first_commit

        # The file should have first-commit content (no "updated" line)
        content = (ws / "hello.txt").read_text()
        assert content == "hello world\n"
        assert not (ws / "readme.md").exists()

    def test_checks_out_second_commit(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _first, second_commit = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()

        ws = create_workspace(str(bare), second_commit, ws_root)

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ws, check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert head == second_commit
        assert (ws / "readme.md").exists()

    def test_idempotent_no_reclone(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        """Calling create_workspace twice returns same path without error."""
        bare, first_commit, _ = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()

        ws1 = create_workspace(str(bare), first_commit, ws_root)
        ws2 = create_workspace(str(bare), first_commit, ws_root)

        assert ws1 == ws2
        assert ws1.exists()


class TestGetPatch:
    """Tests for get_patch."""

    def test_no_changes_returns_empty(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _, second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = create_workspace(str(bare), second, ws_root)

        patch = get_patch(ws)
        assert patch == ""

    def test_returns_diff_after_modification(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _, second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = create_workspace(str(bare), second, ws_root)

        # Modify a file
        (ws / "hello.txt").write_text("hello world\nupdated\nnew line\n")

        patch = get_patch(ws)
        assert "new line" in patch
        assert patch.startswith("diff --git")


class TestApplyPatch:
    """Tests for apply_patch."""

    def test_applies_simple_patch(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _, second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = create_workspace(str(bare), second, ws_root)

        # Create a patch by modifying, getting diff, then resetting
        (ws / "hello.txt").write_text("hello world\nupdated\nextra line\n")
        patch = get_patch(ws)
        assert patch != ""

        # Reset changes
        subprocess.run(["git", "checkout", "."], cwd=ws, check=True, capture_output=True)
        assert get_patch(ws) == ""

        # Apply the patch
        result = apply_patch(ws, patch)
        assert result is True
        assert "extra line" in (ws / "hello.txt").read_text()

    def test_returns_false_on_bad_patch(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _, second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = create_workspace(str(bare), second, ws_root)

        result = apply_patch(ws, "this is not a valid patch")
        assert result is False


class TestRunInWorkspace:
    """Tests for run_in_workspace."""

    def test_runs_command_and_returns_output(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _, second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = create_workspace(str(bare), second, ws_root)

        output, code = run_in_workspace(ws, "echo hello")
        assert code == 0
        assert "hello" in output

    def test_captures_stderr(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _, second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = create_workspace(str(bare), second, ws_root)

        output, code = run_in_workspace(ws, "echo error >&2")
        assert "error" in output

    def test_handles_timeout(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _, second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = create_workspace(str(bare), second, ws_root)

        output, code = run_in_workspace(ws, "sleep 60", timeout=1)
        assert code != 0
        assert "timeout" in output.lower() or "timed out" in output.lower()


class TestCleanupWorkspace:
    """Tests for cleanup_workspace."""

    def test_removes_directory(
        self, tmp_path: Path, local_bare_repo: tuple[Path, str, str]
    ):
        bare, _, second = local_bare_repo
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = create_workspace(str(bare), second, ws_root)
        assert ws.exists()

        cleanup_workspace(ws)
        assert not ws.exists()

    def test_no_error_if_already_removed(self, tmp_path: Path):
        """cleanup_workspace should not raise if dir doesn't exist."""
        nonexistent = tmp_path / "does_not_exist"
        cleanup_workspace(nonexistent)  # Should not raise
