#!/usr/bin/env python3
"""32B Framework Run — 15 SWE-Bench tasks with improved agent framework.

Tests the scaling hypothesis with qwen2.5-coder:32b using all framework fixes:
  - line-range read, context pruning, few-shot, loop detection
  - skip-verification prompt
  - code fence stripping

Comparison targets:
  7B+fixes (15 tasks):  3/15 (20.0%) resolved, 86.7% patch rate
  14B+fixes (15 tasks): 4/15 (26.7%) resolved, 100% patch rate
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from looper.agent.ollama_client import ChatMessage, ChatResponse
from looper.collectors.trajectory_store import collect_trajectories
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import ExperimentConfig, ExperimentResult, TaskResult
from looper.tasks.loader import get_repo_tasks, load_curriculum, get_task_by_id

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5-coder:32b"
OLLAMA_URL = "http://localhost:11434"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_framework_32b")

MAX_STEPS = 15
MAX_TOKENS = 4096

# Same 15 tasks as 7B and 14B framework experiments (indices 3-17)
TARGET_TASKS = [
    # Pilot tasks (indices 3-5)
    "django__django-10914",
    "django__django-10999",
    "django__django-11066",
    # Expanded tasks (indices 6-17)
    "django__django-11099",
    "django__django-11119",
    "django__django-11133",
    "django__django-11163",
    "django__django-11179",
    "django__django-11239",
    "django__django-11299",
    "django__django-11433",
    "django__django-11451",
    "django__django-11490",
    "django__django-11555",
    "django__django-11603",
]

# Set PILOT_ONLY=True for initial 3-task validation
PILOT_ONLY = False

# 32B needs longer timeout (19GB model + KV cache on 32GB RAM = slow inference)
OLLAMA_TIMEOUT = 1800.0  # 30 min per call


def chat_32b(
    messages: list[ChatMessage],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_URL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    **kwargs,
) -> ChatResponse:
    """Chat wrapper with extended timeout for 32B model."""
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    last_exc = None
    for attempt in range(3):
        try:
            response = httpx.post(
                f"{base_url}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return ChatResponse(
                content=data["message"]["content"],
                total_tokens=data.get("eval_count", 0)
                + data.get("prompt_eval_count", 0),
                model=data["model"],
            )
        except httpx.ReadTimeout as exc:
            last_exc = exc
            logger.warning(f"32B timeout (attempt {attempt + 1}/3), retrying...")
            time.sleep(5)
    raise last_exc


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "experiment.log"),
        ],
    )


def evaluate_trajectories(trajectories, tasks, workspace_root, condition):
    """Run FAIL_TO_PASS verification on trajectories."""
    task_map = {t.instance_id: t for t in tasks}
    results = []
    for traj in trajectories:
        task = task_map.get(traj.meta.task_id)
        resolved = False
        if task and traj.generated_patch.strip():
            vr = verify_patch_tests(task, traj.generated_patch, workspace_root)
            resolved = vr["resolved"]
            logger.info(
                f"  Verify {traj.meta.task_id}: "
                f"{'PASS' if resolved else 'FAIL'} "
                f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
                + (f" error={vr['error']}" if vr["error"] else "")
            )
        else:
            logger.info(
                f"  Verify {traj.meta.task_id}: SKIP (no patch generated)"
            )
        results.append(
            TaskResult(
                task_id=traj.meta.task_id,
                condition=condition,
                resolved=resolved,
                steps=traj.meta.total_steps,
                tokens=traj.meta.total_tokens,
                duration_seconds=0.0,
            )
        )
    return results


def run_experiment():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    traj_dir = OUTPUT_DIR / "trajectories" / "base"
    traj_dir.mkdir(parents=True, exist_ok=True)

    task_ids = TARGET_TASKS[:3] if PILOT_ONLY else TARGET_TASKS

    logger.info("=" * 60)
    logger.info("32B Framework Run — qwen2.5-coder:32b with All Fixes")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Fixes: line-range read, context pruning, few-shot, loop detection, skip-verification, code-fence-strip")
    logger.info(f"Tasks: {len(task_ids)} {'(PILOT)' if PILOT_ONLY else ''}")
    logger.info(f"Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}, Timeout: {OLLAMA_TIMEOUT}s")
    logger.info("=" * 60)

    # Load tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    tasks = []
    for tid in task_ids:
        task = get_task_by_id(all_tasks, tid)
        if task is None:
            logger.error(f"Task {tid} not found in curriculum!")
            sys.exit(1)
        tasks.append(task)

    logger.info(f"Tasks ({len(tasks)}):")
    for t in tasks:
        logger.info(f"  {t.instance_id} (pos {t.sequence_position})")

    # Run tasks (collect_trajectories has built-in resume support)
    def on_complete(tid, traj):
        logger.info(
            f"  {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, "
            f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
        )

    t_start = time.monotonic()
    trajectories = collect_trajectories(
        tasks=tasks,
        output_dir=traj_dir,
        workspace_root=WORKSPACE_ROOT,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        max_steps=MAX_STEPS,
        max_tokens=MAX_TOKENS,
        on_complete=on_complete,
        chat_fn=chat_32b,
    )
    elapsed = time.monotonic() - t_start

    # Evaluate with FAIL_TO_PASS verification
    logger.info("")
    logger.info("Evaluating with FAIL_TO_PASS verification...")
    task_results = evaluate_trajectories(
        trajectories, tasks, WORKSPACE_ROOT, "base_32b_framework"
    )

    # Compute metrics
    resolved_count = sum(1 for r in task_results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(task_results)
    avg_steps = sum(r.steps for r in task_results) / total if total else 0
    avg_tokens = sum(r.tokens for r in task_results) / total if total else 0

    # Save results
    config = ExperimentConfig(
        name="experiment_framework_32b",
        experiment_id="framework_32b",
        repo="django/django",
        model_name=OLLAMA_MODEL,
        train_task_ids=[],
        test_task_ids=list(task_ids),
        strategy="base_with_framework_fixes_v3",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=task_results,
        forward_transfer=0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "experiment_result.json")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"32B FRAMEWORK RUN COMPLETE {'(PILOT)' if PILOT_ONLY else ''}")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info(f"  Total time:   {elapsed/60:.1f} min")
    logger.info("")

    # Comparison
    logger.info("Comparison (same 15 tasks):")
    logger.info(f"  7B+fixes:   3/15 (20.0%) resolved, 86.7% patch rate")
    logger.info(f"  14B+fixes:  4/15 (26.7%) resolved, 100% patch rate")
    logger.info(f"  32B+fixes:  {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved, {patch_count/total*100:.1f}% patch rate")
    logger.info("")

    # Per-task detail
    logger.info("Per-task results:")
    traj_map = {t.meta.task_id: t for t in trajectories}
    for r in task_results:
        t = traj_map.get(r.task_id)
        has_patch = t.generated_patch.strip() if t else ""
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"steps={r.steps:>2} tokens={r.tokens:>6} "
            f"patch={'yes' if has_patch else 'no':>3}"
        )

    summary = results_summary(result)
    logger.info(f"\n{summary}")

    return result


if __name__ == "__main__":
    run_experiment()
