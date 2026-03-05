"""Patch verification for SWE-Bench tasks.

Provides:
- verify_patch_tests(): Real verification — applies patch + test_patch, runs
  the FAIL_TO_PASS tests from the curriculum, checks they pass.
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
            # Clone fresh — use reference clone if available
            ref_dir = workspace_root / ".refs" / repo_name
            clone_cmd = ["git", "clone"]
            if ref_dir.exists():
                clone_cmd += ["--reference", str(ref_dir)]
            clone_cmd += [f"https://github.com/{task.repo}.git", str(verify_dir)]

            verify_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(clone_cmd, check=True, capture_output=True, timeout=timeout)
            subprocess.run(
                ["git", "checkout", task.base_commit],
                cwd=verify_dir,
                check=True,
                capture_output=True,
                timeout=30,
            )

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

        # Run FAIL_TO_PASS tests via Django's runtests.py
        test_modules = _get_test_modules(task.fail_to_pass)
        tests_dir = verify_dir / "tests"

        if not tests_dir.exists():
            result["error"] = "No tests/ directory found"
            return result

        cmd = [
            sys.executable, "tests/runtests.py",
            "--settings=test_sqlite",
            "--parallel=1",
            "-v", "2",
        ] + test_modules

        # Django needs its own root on PYTHONPATH to import itself
        env = dict(subprocess.os.environ)
        env["PYTHONPATH"] = str(verify_dir)

        logger.info(f"  Running tests: {' '.join(test_modules)}")
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


def _test_passed_in_output(test_str: str, output: str) -> bool:
    """Check if a specific test passed in Django's verbose test output.

    SWE-Bench format: "test_paginator_iteration (pagination.tests.PaginationTests)"
    Django -v2 output: "test_paginator_iteration (pagination.tests.PaginationTests.test_paginator_iteration) ... ok"

    We match on test_name + class_path prefix, allowing for the extra method name
    that Django's verbose output appends inside the parens.
    """
    m = re.match(r"^(\S+)\s+\(([^)]+)\)$", test_str)
    if m:
        test_name, class_path = m.group(1), m.group(2)
        # Match: test_name (class_path...) ... ok
        escaped_name = re.escape(test_name)
        escaped_class = re.escape(class_path)
        pattern = escaped_name + r"\s+\(" + escaped_class + r"[^)]*\)\s*\.{3}\s*ok"
        return bool(re.search(pattern, output, re.IGNORECASE))
    else:
        # Plain string — look for it followed by ... ok
        escaped = re.escape(test_str)
        pattern = escaped + r"\s*\.{3}\s*ok"
        return bool(re.search(pattern, output, re.IGNORECASE))


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
