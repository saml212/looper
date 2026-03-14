"""Patch verification for SWE-Bench tasks.

Provides:
- verify_patch_tests(): Real verification — applies patch + test_patch, runs
  the FAIL_TO_PASS tests from the curriculum, checks they pass.
  Supports Django (runtests.py), pytest :: format (sphinx, matplotlib,
  scikit-learn, astropy, xarray, pytest), and sympy (plain test names).
- verify_patch_simple(): Lightweight heuristic (file overlap). Too lenient
  for trustworthy results — kept only for fast smoke checks.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path

from looper.models import TaskInfo

logger = logging.getLogger(__name__)

# Repos that use Django's runtests.py
_DJANGO_REPOS = {"django/django"}

# Repos that use pytest :: format (directly runnable)
_PYTEST_REPOS = {
    "sphinx-doc/sphinx",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "astropy/astropy",
    "pydata/xarray",
    "pytest-dev/pytest",
}

# Repos with plain test names (need file discovery from test_patch)
_PLAIN_TEST_REPOS = {"sympy/sympy"}

# Per-repo pip dependencies needed for test execution
_REPO_DEPS: dict[str, list[str]] = {
    "sympy/sympy": ["mpmath"],
    "astropy/astropy": ["numpy", "erfa", "pyerfa"],
    "scikit-learn/scikit-learn": ["numpy", "scipy", "joblib", "threadpoolctl"],
    "matplotlib/matplotlib": ["numpy", "pyparsing", "cycler", "python-dateutil", "kiwisolver", "pillow"],
    "pydata/xarray": ["numpy", "pandas"],
    "sphinx-doc/sphinx": ["Jinja2", "docutils", "Pygments", "snowballstemmer", "babel", "alabaster", "requests", "packaging", "imagesize"],
    "pytest-dev/pytest": ["py", "iniconfig", "attrs", "pluggy", "packaging"],
}


def _extract_files_from_diff(diff: str) -> set[str]:
    """Extract file paths touched by a unified diff."""
    files: set[str] = set()
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", diff, re.MULTILINE):
        files.add(match.group(2))
    return files


def _parse_django_test_id(test_str: str) -> str:
    """Convert SWE-Bench test ID to Django runtests.py argument.

    Input formats:
        "test_paginator_iteration (pagination.tests.PaginationTests)"
        "test_count_distinct_expression (aggregation.tests.AggregateTestCase)"

    Output: "pagination.tests.PaginationTests.test_paginator_iteration"

    For plain dotted paths (no parens), return as-is.
    """
    m = re.match(r"^(\S+)\s+\(([^)]+)\)$", test_str)
    if m:
        test_name, class_path = m.group(1), m.group(2)
        return f"{class_path}.{test_name}"
    return test_str


def _get_test_modules(fail_to_pass: list[str]) -> list[str]:
    """Extract unique top-level test module names for Django runtests.py.

    Django's runtests.py takes module names like "pagination" or "auth_tests",
    not full dotted paths. We extract just the first component.
    """
    modules = set()
    for test_str in fail_to_pass:
        parsed = _parse_django_test_id(test_str)
        # First component is the test module
        module = parsed.split(".")[0]
        modules.add(module)
    return sorted(modules)


def verify_patch_tests(
    task: TaskInfo,
    generated_patch: str,
    workspace_root: Path,
    timeout: int = 600,
) -> dict:
    """Verify a patch by running the FAIL_TO_PASS tests from SWE-Bench.

    Creates/reuses a clean verification workspace under workspace_root/verify/,
    separate from agent workspaces.

    Steps:
    1. Prepare clean workspace at base_commit
    2. Apply generated_patch
    3. Apply test_patch (adds the regression test)
    4. Run FAIL_TO_PASS tests via Django's runtests.py
    5. Parse output to check which FAIL_TO_PASS tests passed

    Returns:
        {
            "resolved": bool,
            "fail_to_pass_passed": int,
            "fail_to_pass_total": int,
            "error": str,
            "test_output": str,  # last 2000 chars of test output
        }
    """
    result = {
        "resolved": False,
        "fail_to_pass_passed": 0,
        "fail_to_pass_total": len(task.fail_to_pass),
        "error": "",
        "test_output": "",
    }

    if not task.fail_to_pass:
        result["error"] = "No FAIL_TO_PASS tests defined"
        return result

    if not generated_patch.strip():
        result["error"] = "Empty patch"
        return result

    # Use a separate verification workspace
    repo_name = task.repo.split("/")[-1]
    verify_dir = workspace_root / "verify" / repo_name / task.base_commit[:8]

    try:
        # Prepare clean workspace
        if verify_dir.exists():
            # Reset existing workspace to clean state
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=verify_dir,
                check=True,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=verify_dir,
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            # Clone fresh — prefer local bare repo, fallback to GitHub
            ref_dir = workspace_root / ".refs" / repo_name
            if ref_dir.exists():
                clone_url = str(ref_dir)
            else:
                clone_url = f"https://github.com/{task.repo}.git"

            verify_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", clone_url, str(verify_dir)],
                check=True, capture_output=True, timeout=timeout,
            )
            subprocess.run(
                ["git", "checkout", task.base_commit],
                cwd=verify_dir,
                check=True,
                capture_output=True,
                timeout=30,
            )

        # Install repo-specific dependencies if needed
        deps = _REPO_DEPS.get(task.repo, [])
        if deps:
            _install_deps_marker = verify_dir / ".deps_installed"
            if not _install_deps_marker.exists():
                logger.info(f"  Installing deps for {task.repo}: {deps}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q"] + deps,
                    cwd=verify_dir,
                    capture_output=True,
                    timeout=120,
                )
                # Also install the repo itself in editable mode if setup.py exists
                if (verify_dir / "setup.py").exists():
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
                        cwd=verify_dir,
                        capture_output=True,
                        timeout=120,
                    )
                _install_deps_marker.touch()

        # Apply agent's generated patch
        apply_result = subprocess.run(
            ["git", "apply", "--allow-empty"],
            input=generated_patch,
            cwd=verify_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if apply_result.returncode != 0:
            result["error"] = f"Patch apply failed: {apply_result.stderr[:500]}"
            return result

        # Apply test patch (adds regression test cases)
        if task.test_patch:
            test_apply = subprocess.run(
                ["git", "apply", "--allow-empty"],
                input=task.test_patch,
                cwd=verify_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if test_apply.returncode != 0:
                result["error"] = f"Test patch apply failed: {test_apply.stderr[:500]}"
                return result

        # Run FAIL_TO_PASS tests — dispatch based on repo type
        if task.repo in _DJANGO_REPOS:
            cmd, env = _build_django_test_cmd(task, verify_dir)
        elif task.repo in _PYTEST_REPOS:
            cmd, env = _build_pytest_test_cmd(task, verify_dir)
        elif task.repo in _PLAIN_TEST_REPOS:
            cmd, env = _build_plain_test_cmd(task, verify_dir)
        else:
            result["error"] = f"Unsupported repo: {task.repo}"
            return result

        logger.info(f"  Running: {' '.join(cmd[-3:])}")
        test_run = subprocess.run(
            cmd,
            cwd=verify_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        output = test_run.stdout + test_run.stderr
        result["test_output"] = output[-2000:]

        # Parse results — check which FAIL_TO_PASS tests passed
        passed = 0
        for test_str in task.fail_to_pass:
            if _test_passed_in_output(test_str, output):
                passed += 1

        result["fail_to_pass_passed"] = passed
        result["resolved"] = (passed == len(task.fail_to_pass))

        return result

    except subprocess.TimeoutExpired:
        result["error"] = f"Tests timed out after {timeout}s"
        return result
    except subprocess.CalledProcessError as e:
        result["error"] = f"Subprocess error: {e}"
        return result
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
        return result
    finally:
        # Reset workspace for next verification
        try:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=verify_dir,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=verify_dir,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass


def _build_django_test_cmd(
    task: TaskInfo, verify_dir: Path
) -> tuple[list[str], dict]:
    """Build test command for Django repos."""
    test_modules = _get_test_modules(task.fail_to_pass)
    cmd = [
        sys.executable, "tests/runtests.py",
        "--settings=test_sqlite",
        "--parallel=1",
        "-v", "2",
    ] + test_modules
    env = dict(subprocess.os.environ)
    env["PYTHONPATH"] = str(verify_dir)
    return cmd, env


def _build_pytest_test_cmd(
    task: TaskInfo, verify_dir: Path
) -> tuple[list[str], dict]:
    """Build test command for repos using pytest :: format.

    FAIL_TO_PASS entries like: tests/test_domain_std.py::test_glossary
    are directly usable as pytest arguments.
    """
    cmd = [sys.executable, "-m", "pytest", "-xvs"] + task.fail_to_pass
    env = dict(subprocess.os.environ)
    env["PYTHONPATH"] = str(verify_dir)
    return cmd, env


def _build_plain_test_cmd(
    task: TaskInfo, verify_dir: Path
) -> tuple[list[str], dict]:
    """Build test command for repos with plain test names (e.g. sympy).

    Maps plain names like 'test_issue_12092' to pytest node IDs by
    extracting the test file path from the test_patch diff.
    """
    # Extract test files from test_patch
    test_files = _extract_files_from_diff(task.test_patch)

    # Build pytest node IDs: file::test_name for each test
    node_ids = []
    for test_name in task.fail_to_pass:
        # Try to find a test file that contains this test (after patch applied)
        matched = False
        for tf in test_files:
            test_path = verify_dir / tf
            if test_path.exists():
                content = test_path.read_text()
                if f"def {test_name}" in content:
                    node_ids.append(f"{tf}::{test_name}")
                    matched = True
                    break
        if not matched:
            # Fallback: use -k filter with all test files
            node_ids.append(f"-k {test_name}")

    # If we have -k filters, combine them differently
    k_filters = [n for n in node_ids if n.startswith("-k ")]
    file_ids = [n for n in node_ids if not n.startswith("-k ")]

    if k_filters and not file_ids:
        # All plain names, use -k with test files
        k_expr = " or ".join(n.replace("-k ", "") for n in k_filters)
        cmd = [sys.executable, "-m", "pytest", "-xvs", "-k", k_expr]
        cmd += list(test_files)
    else:
        cmd = [sys.executable, "-m", "pytest", "-xvs"] + file_ids
        if k_filters:
            k_expr = " or ".join(n.replace("-k ", "") for n in k_filters)
            cmd += ["-k", k_expr]

    env = dict(subprocess.os.environ)
    env["PYTHONPATH"] = str(verify_dir)
    return cmd, env


def _test_passed_in_output(test_str: str, output: str) -> bool:
    """Check if a specific test passed in test output.

    Handles three formats:
    1. Django parens: "test_name (module.Class)" → look for "... ok"
    2. Pytest :: format: "path/test.py::test_name" → look for "PASSED"
    3. Plain name: "test_name" → look for "... ok" or "PASSED"
    """
    # Django parens format
    m = re.match(r"^(\S+)\s+\(([^)]+)\)$", test_str)
    if m:
        test_name, class_path = m.group(1), m.group(2)
        escaped_name = re.escape(test_name)
        escaped_class = re.escape(class_path)
        pattern = escaped_name + r"\s+\(" + escaped_class + r"[^)]*\)\s*\.{3}\s*ok"
        return bool(re.search(pattern, output, re.IGNORECASE))

    # Pytest :: format
    if "::" in test_str:
        escaped = re.escape(test_str)
        # pytest -v output: "path/test.py::test_name PASSED"
        if re.search(escaped + r"\s+PASSED", output):
            return True
        # pytest -s output: just check it's not in FAILED
        if re.search(escaped + r"\s+FAILED", output):
            return False
        # Check for the test name part after last :: with PASSED
        test_func = test_str.split("::")[-1]
        escaped_func = re.escape(test_func)
        if re.search(escaped_func + r"\s+PASSED", output):
            return True
        # Also handle "... ok" from unittest-style output
        if re.search(escaped_func + r"\s*\.{3}\s*ok", output, re.IGNORECASE):
            return True
        return False

    # Plain test name — check both formats
    escaped = re.escape(test_str)
    if re.search(escaped + r"\s+PASSED", output):
        return True
    if re.search(escaped + r"\s*\.{3}\s*ok", output, re.IGNORECASE):
        return True
    return False


def verify_patch_simple(
    task: TaskInfo,
    generated_patch: str,
) -> bool:
    """Simple patch verification: check if the generated patch
    modifies the same files as the ground truth patch.

    WARNING: This is too lenient for real evaluation. Use verify_patch_tests()
    for trustworthy results.

    Returns True if the generated patch touches at least one of the
    same files as the ground truth patch.
    """
    if not generated_patch.strip():
        return False

    truth_files = _extract_files_from_diff(task.patch)
    generated_files = _extract_files_from_diff(generated_patch)

    if not truth_files or not generated_files:
        return False

    return bool(truth_files & generated_files)
