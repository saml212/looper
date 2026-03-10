"""Workspace management for SWE-Bench tasks.

Manages temporary directories where git repos are cloned at specific commits,
so an agent can work on a SWE-Bench task in isolation.
"""

import shutil
import subprocess
from pathlib import Path


def reset_workspace(workspace_dir: Path) -> None:
    """Reset all uncommitted changes in a workspace.

    Discards modified/deleted tracked files and removes untracked files/dirs.
    This ensures a clean state at base_commit before each agent run.
    """
    subprocess.run(
        ["git", "checkout", "--", "."],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "clean", "-fd"],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def create_workspace(
    repo: str,
    base_commit: str,
    workspace_root: Path,
) -> Path:
    """Clone a repo and checkout a specific commit.

    Args:
        repo: GitHub repo in "owner/name" format (e.g. "django/django"),
              or a local path to a bare/regular git repo.
        base_commit: The commit hash to checkout.
        workspace_root: Parent directory for workspaces.

    Returns:
        Path to the workspace directory.

    The workspace directory structure:
        workspace_root / repo_name / base_commit[:8] /  (the cloned repo)

    Idempotent: if the workspace already exists at the right commit,
    resets uncommitted changes and returns the existing path.
    """
    # Derive repo name: last component of the path, minus any .git suffix
    repo_name = Path(repo).stem.removesuffix(".git") if "/" not in repo or repo.startswith("/") else repo.split("/")[-1]
    # Remove .git suffix if present (for bare repos like "bare_repo.git")
    repo_name = repo_name.removesuffix(".git")

    workspace_dir = workspace_root / repo_name / base_commit[:8]

    if workspace_dir.exists():
        # Check if already at the right commit
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip() == base_commit:
            reset_workspace(workspace_dir)
            return workspace_dir

    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Determine the clone URL
    if repo.startswith("/") or repo.startswith(".") or Path(repo).exists():
        clone_url = str(repo)
    else:
        clone_url = f"https://github.com/{repo}.git"

    # Use a reference clone if a cached bare repo exists (much faster)
    ref_dir = workspace_root / ".refs" / repo_name
    clone_cmd = ["git", "clone"]
    if ref_dir.exists():
        clone_cmd += ["--reference", str(ref_dir)]
    clone_cmd += [clone_url, str(workspace_dir)]

    subprocess.run(
        clone_cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    # Checkout the specific commit
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    return workspace_dir


def get_patch(workspace_dir: Path) -> str:
    """Get the git diff of uncommitted changes in the workspace.

    Returns the diff as a string (suitable for SWE-bench predictions).
    Returns empty string if there are no changes.
    """
    result = subprocess.run(
        ["git", "diff"],
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def apply_patch(workspace_dir: Path, patch: str) -> bool:
    """Apply a patch to the workspace.

    Returns True if the patch was applied successfully, False otherwise.
    """
    result = subprocess.run(
        ["git", "apply"],
        cwd=workspace_dir,
        input=patch,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_in_workspace(
    workspace_dir: Path, command: str, timeout: int = 30
) -> tuple[str, int]:
    """Run a shell command in the workspace directory.

    Returns (output, return_code). Combines stdout and stderr.
    Timeout in seconds — returns a timeout message and non-zero code on expiry.
    """
    try:
        result = subprocess.run(
            command,
            cwd=workspace_dir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return "Timed out", 1


def cleanup_workspace(workspace_dir: Path) -> None:
    """Remove the workspace directory. No-op if it doesn't exist."""
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
