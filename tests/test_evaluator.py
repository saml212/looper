"""Tests for the evaluator module.

Written test-first following TDD. Covers metrics, patch verification,
and results I/O.
"""

import json
from pathlib import Path

import pytest

from looper.models import (
    ExperimentConfig,
    ExperimentResult,
    TaskInfo,
    TaskResult,
)
from looper.evaluators.metrics import (
    avg_steps,
    avg_tokens,
    compare_conditions,
    forward_transfer,
    resolve_rate,
)
from looper.evaluators.patch_verifier import (
    _parse_django_test_id,
    _get_test_modules,
    _test_passed_in_output,
    verify_patch_simple,
    verify_patch_tests,
)
from looper.evaluators.results_io import load_results, results_summary, save_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    task_id: str = "t1",
    condition: str = "base",
    resolved: bool = True,
    steps: int = 10,
    tokens: int = 5000,
    duration: float = 30.0,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        condition=condition,
        resolved=resolved,
        steps=steps,
        tokens=tokens,
        duration_seconds=duration,
    )


def _make_task_info(patch: str, test_patch: str = "") -> TaskInfo:
    return TaskInfo(
        instance_id="django__django-12345",
        repo="django/django",
        base_commit="abc123",
        problem_statement="Fix bug",
        patch=patch,
        test_patch=test_patch,
        difficulty="medium",
        created_at="2025-01-01",
        sequence_position=1,
    )


def _make_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="test_experiment",
        experiment_id="exp-001",
        repo="django/django",
        model_name="qwen2.5-coder:7b",
        train_task_ids=["t1", "t2"],
        test_task_ids=["t3", "t4"],
        strategy="full_replay",
    )


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestResolveRate:
    def test_mixed_results(self):
        results = [
            _make_result(resolved=True),
            _make_result(resolved=False),
            _make_result(resolved=True),
            _make_result(resolved=True),
        ]
        assert resolve_rate(results) == pytest.approx(0.75)

    def test_empty_list(self):
        assert resolve_rate([]) == 0.0


class TestAvgSteps:
    def test_calculation(self):
        results = [
            _make_result(steps=10),
            _make_result(steps=20),
            _make_result(steps=30),
        ]
        assert avg_steps(results) == pytest.approx(20.0)

    def test_empty_list(self):
        assert avg_steps([]) == 0.0


class TestAvgTokens:
    def test_calculation(self):
        results = [
            _make_result(tokens=3000),
            _make_result(tokens=6000),
        ]
        assert avg_tokens(results) == pytest.approx(4500.0)

    def test_empty_list(self):
        assert avg_tokens([]) == 0.0


class TestForwardTransfer:
    def test_positive_transfer(self):
        """Adapter improved resolve rate."""
        base = [
            _make_result(resolved=True),
            _make_result(resolved=False),
            _make_result(resolved=False),
            _make_result(resolved=False),
        ]
        adapted = [
            _make_result(resolved=True),
            _make_result(resolved=True),
            _make_result(resolved=True),
            _make_result(resolved=False),
        ]
        ft = forward_transfer(base, adapted)
        # adapted rate 0.75 - base rate 0.25 = 0.5
        assert ft == pytest.approx(0.5)

    def test_negative_transfer(self):
        """Adapter hurt performance."""
        base = [
            _make_result(resolved=True),
            _make_result(resolved=True),
        ]
        adapted = [
            _make_result(resolved=False),
            _make_result(resolved=False),
        ]
        ft = forward_transfer(base, adapted)
        assert ft == pytest.approx(-1.0)

    def test_zero_transfer(self):
        """No change."""
        base = [_make_result(resolved=True), _make_result(resolved=False)]
        adapted = [_make_result(resolved=True), _make_result(resolved=False)]
        ft = forward_transfer(base, adapted)
        assert ft == pytest.approx(0.0)


class TestCompareConditions:
    def test_two_conditions(self):
        results_by_condition = {
            "base": [
                _make_result(condition="base", resolved=True, steps=10, tokens=5000),
                _make_result(condition="base", resolved=False, steps=20, tokens=8000),
            ],
            "base_lora": [
                _make_result(condition="base_lora", resolved=True, steps=8, tokens=4000),
                _make_result(condition="base_lora", resolved=True, steps=12, tokens=6000),
            ],
        }
        summary = compare_conditions(results_by_condition)

        assert summary["base"]["resolve_rate"] == pytest.approx(0.5)
        assert summary["base"]["avg_steps"] == pytest.approx(15.0)
        assert summary["base"]["avg_tokens"] == pytest.approx(6500.0)

        assert summary["base_lora"]["resolve_rate"] == pytest.approx(1.0)
        assert summary["base_lora"]["avg_steps"] == pytest.approx(10.0)
        assert summary["base_lora"]["avg_tokens"] == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# Patch verifier tests
# ---------------------------------------------------------------------------


_DIFF_A = """\
diff --git a/django/db/models/query.py b/django/db/models/query.py
index abc123..def456 100644
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -10,6 +10,7 @@
+    fix here
"""

_DIFF_B = """\
diff --git a/django/db/models/query.py b/django/db/models/query.py
index abc123..def456 100644
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -20,6 +20,7 @@
+    different fix
"""

_DIFF_C = """\
diff --git a/django/contrib/admin/views.py b/django/contrib/admin/views.py
index 111222..333444 100644
--- a/django/contrib/admin/views.py
+++ b/django/contrib/admin/views.py
@@ -1,3 +1,4 @@
+    unrelated change
"""


class TestParseDjangoTestId:
    def test_standard_format(self):
        result = _parse_django_test_id(
            "test_paginator_iteration (pagination.tests.PaginationTests)"
        )
        assert result == "pagination.tests.PaginationTests.test_paginator_iteration"

    def test_dotted_path_passthrough(self):
        result = _parse_django_test_id("pagination.tests.PaginationTests.test_iter")
        assert result == "pagination.tests.PaginationTests.test_iter"

    def test_description_format(self):
        """Some SWE-Bench tests use plain descriptions like 'Named URLs should be reversible'."""
        result = _parse_django_test_id("Named URLs should be reversible")
        assert result == "Named URLs should be reversible"


class TestGetTestModules:
    def test_single_module(self):
        modules = _get_test_modules([
            "test_paginator_iteration (pagination.tests.PaginationTests)",
        ])
        assert modules == ["pagination"]

    def test_multiple_modules(self):
        modules = _get_test_modules([
            "test_foo (auth_tests.test_validators.FooTest)",
            "test_bar (aggregation.tests.BarTest)",
            "test_baz (auth_tests.test_remote.BazTest)",
        ])
        assert modules == ["aggregation", "auth_tests"]

    def test_empty(self):
        assert _get_test_modules([]) == []


class TestTestPassedInOutput:
    def test_ok_status(self):
        output = "test_paginator_iteration (pagination.tests.PaginationTests) ... ok\n"
        assert _test_passed_in_output(
            "test_paginator_iteration (pagination.tests.PaginationTests)", output
        ) is True

    def test_fail_status(self):
        output = "test_paginator_iteration (pagination.tests.PaginationTests) ... FAIL\n"
        assert _test_passed_in_output(
            "test_paginator_iteration (pagination.tests.PaginationTests)", output
        ) is False

    def test_error_status(self):
        output = "test_paginator_iteration (pagination.tests.PaginationTests) ... ERROR\n"
        assert _test_passed_in_output(
            "test_paginator_iteration (pagination.tests.PaginationTests)", output
        ) is False

    def test_not_present(self):
        output = "some other test ... ok\n"
        assert _test_passed_in_output(
            "test_paginator_iteration (pagination.tests.PaginationTests)", output
        ) is False


class TestVerifyPatchTests:
    def test_empty_patch_returns_error(self):
        task = _make_task_info(
            patch=_DIFF_A,
            test_patch="",
        )
        task = task.model_copy(update={"fail_to_pass": ["test_foo (mod.Test)"]})
        result = verify_patch_tests(task, "", Path("/tmp/nonexistent"))
        assert result["resolved"] is False
        assert result["error"] == "Empty patch"

    def test_no_fail_to_pass_returns_error(self):
        task = _make_task_info(patch=_DIFF_A)
        result = verify_patch_tests(task, _DIFF_B, Path("/tmp/nonexistent"))
        assert result["resolved"] is False
        assert result["error"] == "No FAIL_TO_PASS tests defined"

    def test_clone_failure_returns_error(self, tmp_path):
        """When no ref clone exists and the commit doesn't exist, clone fails."""
        from unittest.mock import patch as mock_patch

        task = _make_task_info(patch=_DIFF_A)
        task = task.model_copy(update={"fail_to_pass": ["test_foo (mod.Test)"]})

        # Mock subprocess.run to simulate git clone failure
        import subprocess as sp

        orig_run = sp.run

        def failing_clone(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if isinstance(cmd, list) and "clone" in cmd:
                raise sp.CalledProcessError(128, "git clone")
            return orig_run(*args, **kwargs)

        with mock_patch("subprocess.run", side_effect=failing_clone):
            result = verify_patch_tests(task, _DIFF_B, tmp_path)
        assert result["resolved"] is False
        assert result["error"] != ""


class TestVerifyPatchSimple:
    def test_same_files_returns_true(self):
        task = _make_task_info(patch=_DIFF_A)
        assert verify_patch_simple(task, _DIFF_B) is True

    def test_different_files_returns_false(self):
        task = _make_task_info(patch=_DIFF_A)
        assert verify_patch_simple(task, _DIFF_C) is False

    def test_empty_generated_patch_returns_false(self):
        task = _make_task_info(patch=_DIFF_A)
        assert verify_patch_simple(task, "") is False


# ---------------------------------------------------------------------------
# Results I/O tests
# ---------------------------------------------------------------------------


class TestResultsIO:
    def test_save_load_roundtrip(self, tmp_path: Path):
        config = _make_experiment_config()
        result = ExperimentResult(
            config=config,
            task_results=[
                _make_result(resolved=True, steps=10, tokens=5000),
                _make_result(resolved=False, steps=20, tokens=8000),
            ],
            forward_transfer=0.25,
            started_at="2025-06-01T00:00:00",
            completed_at="2025-06-01T01:00:00",
        )

        path = tmp_path / "results.json"
        save_results(result, path)
        loaded = load_results(path)

        assert loaded.config.experiment_id == config.experiment_id
        assert len(loaded.task_results) == 2
        assert loaded.forward_transfer == pytest.approx(0.25)
        assert loaded.started_at == "2025-06-01T00:00:00"

    def test_results_summary_contains_key_metrics(self):
        config = _make_experiment_config()
        result = ExperimentResult(
            config=config,
            task_results=[
                _make_result(resolved=True, steps=10, tokens=5000),
                _make_result(resolved=False, steps=20, tokens=8000),
            ],
            forward_transfer=0.25,
            started_at="2025-06-01T00:00:00",
        )
        summary = results_summary(result)
        assert len(summary) > 0
        assert "resolve" in summary.lower()
        assert "0.25" in summary or "forward" in summary.lower()
