#!/usr/bin/env python3
"""14B CoT + Few-Shot Experiment — 50 SWE-Bench tasks.

Tests two improvements over the fuzzy_edit baseline:
1. Chain-of-thought <think> blocks before every edit (forces reasoning about WHY)
2. Few-shot example (complete django-11066 trajectory in system prompt)

Hypothesis: Wrong-fix semantics are the bottleneck. Forcing explicit reasoning
about what the test expects, what the code does wrong, and what change fixes it
should improve semantic accuracy of patches.

Uses MLX server on port 8080 (mlx_lm.server with qwen2.5-coder:14b-instruct-q4).
"""

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.agent.ollama_client import openai_chat
from looper.agent.runner import _THINK_RE
from looper.collectors.trajectory_store import collect_trajectories
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import ExperimentConfig, ExperimentResult, TaskResult
from looper.tasks.loader import get_repo_tasks, load_curriculum

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

MLX_MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
MLX_URL = "http://localhost:8080"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_cot_fewshot_14b")

MAX_STEPS = 15
MAX_TOKENS = 4096


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


def analyze_think_blocks(trajectories):
    """Analyze <think> block usage across trajectories."""
    stats = {
        "total_think_blocks": 0,
        "tasks_with_think": 0,
        "tasks_without_think": 0,
        "avg_think_length": 0,
        "think_before_edit": 0,
        "edit_without_think": 0,
    }
    per_task = {}
    sample_thinks = []  # Collect samples for report

    for traj in trajectories:
        task_thinks = 0
        task_edit_with_think = 0
        task_edit_without_think = 0

        for step in traj.steps:
            # Check if reasoning contains a think block
            has_think = bool(_THINK_RE.search(step.reasoning)) if step.reasoning else False
            has_edit = any(tc.tool_name == "edit" for tc in step.tool_calls)
            has_write = any(tc.tool_name == "write" for tc in step.tool_calls)

            if has_think:
                task_thinks += 1
                stats["total_think_blocks"] += 1
                # Collect first 5 samples
                if len(sample_thinks) < 10:
                    think_match = _THINK_RE.search(step.reasoning)
                    if think_match:
                        resolved = None  # filled later
                        sample_thinks.append({
                            "task_id": traj.meta.task_id,
                            "step": step.step_number,
                            "think": think_match.group(1).strip(),
                        })

            if has_edit or has_write:
                if has_think:
                    task_edit_with_think += 1
                    stats["think_before_edit"] += 1
                else:
                    task_edit_without_think += 1
                    stats["edit_without_think"] += 1

        if task_thinks > 0:
            stats["tasks_with_think"] += 1
        else:
            stats["tasks_without_think"] += 1

        per_task[traj.meta.task_id] = {
            "think_blocks": task_thinks,
            "edit_with_think": task_edit_with_think,
            "edit_without_think": task_edit_without_think,
        }

    if stats["total_think_blocks"] > 0:
        # Compute avg length across all think blocks
        all_lengths = []
        for traj in trajectories:
            for step in traj.steps:
                if step.reasoning:
                    for m in _THINK_RE.finditer(step.reasoning):
                        all_lengths.append(len(m.group(1).strip()))
        stats["avg_think_length"] = sum(all_lengths) / len(all_lengths) if all_lengths else 0

    return stats, per_task, sample_thinks


def analyze_tool_usage(trajectories):
    """Analyze edit/write/fuzzy usage across all trajectories."""
    stats = {
        "edit_exact": 0,
        "edit_fuzzy": 0,
        "edit_failed": 0,
        "write": 0,
    }

    for traj in trajectories:
        for step in traj.steps:
            for tc in step.tool_calls:
                if tc.tool_name == "edit":
                    if tc.success and "fuzzy match" in tc.tool_result:
                        stats["edit_fuzzy"] += 1
                    elif tc.success:
                        stats["edit_exact"] += 1
                    else:
                        stats["edit_failed"] += 1
                elif tc.tool_name == "write":
                    stats["write"] += 1

    return stats


def run_experiment():
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    traj_dir = OUTPUT_DIR / "trajectories" / "base"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Load all 50 Django tasks
    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, "django/django")
    tasks = list(all_tasks)

    logger.info("=" * 60)
    logger.info("14B CoT + Few-Shot Experiment — All 50 Django Tasks")
    logger.info(f"Model: {MLX_MODEL} via MLX server at {MLX_URL}")
    logger.info("Improvements over fuzzy_edit baseline:")
    logger.info("  1. Chain-of-thought <think> blocks (reason before edit)")
    logger.info("  2. Few-shot example (django-11066 complete trajectory)")
    logger.info(f"Tasks: {len(tasks)} tasks")
    logger.info(f"Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}")
    logger.info("=" * 60)

    logger.info(f"\nTasks ({len(tasks)}):")
    for t in tasks:
        logger.info(f"  {t.instance_id} (pos {t.sequence_position})")

    # Run tasks via MLX server (openai_chat)
    def on_complete(tid, traj):
        think_count = sum(
            1 for s in traj.steps
            if s.reasoning and _THINK_RE.search(s.reasoning)
        )
        logger.info(
            f"  {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, "
            f"thinks={think_count}, "
            f"patch={'yes' if traj.generated_patch.strip() else 'no'})"
        )

    trajectories = collect_trajectories(
        tasks=tasks,
        output_dir=traj_dir,
        workspace_root=WORKSPACE_ROOT,
        model=MLX_MODEL,
        base_url=MLX_URL,
        max_steps=MAX_STEPS,
        max_tokens=MAX_TOKENS,
        on_complete=on_complete,
        chat_fn=openai_chat,
    )

    # Evaluate with FAIL_TO_PASS verification
    logger.info("")
    logger.info("Evaluating with FAIL_TO_PASS verification...")
    task_results = evaluate_trajectories(
        trajectories, tasks, WORKSPACE_ROOT, "base_14b_cot_fewshot"
    )

    # Compute metrics
    resolved_count = sum(1 for r in task_results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(task_results)
    avg_steps = sum(r.steps for r in task_results) / total if total else 0
    avg_tokens = sum(r.tokens for r in task_results) / total if total else 0

    # Analyze think blocks
    think_stats, think_per_task, sample_thinks = analyze_think_blocks(trajectories)

    # Analyze tool usage
    tool_stats = analyze_tool_usage(trajectories)

    # Save results
    config = ExperimentConfig(
        name="experiment_cot_fewshot_14b",
        experiment_id="cot_fewshot_14b",
        repo="django/django",
        model_name=MLX_MODEL,
        train_task_ids=[],
        test_task_ids=[t.instance_id for t in tasks],
        strategy="base_with_cot_fewshot",
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

    # Save detailed stats
    with open(OUTPUT_DIR / "think_stats.json", "w") as f:
        json.dump({
            "aggregate": think_stats,
            "per_task": think_per_task,
            "samples": sample_thinks,
        }, f, indent=2)

    with open(OUTPUT_DIR / "tool_stats.json", "w") as f:
        json.dump({"aggregate": tool_stats}, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("14B COT + FEW-SHOT EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolved_count/total*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Avg steps:    {avg_steps:.1f}")
    logger.info(f"  Avg tokens:   {avg_tokens:.0f}")
    logger.info("")
    logger.info("Think block stats:")
    logger.info(f"  Total think blocks:   {think_stats['total_think_blocks']}")
    logger.info(f"  Tasks with think:     {think_stats['tasks_with_think']}/{total}")
    logger.info(f"  Think before edit:    {think_stats['think_before_edit']}")
    logger.info(f"  Edit without think:   {think_stats['edit_without_think']}")
    logger.info(f"  Avg think length:     {think_stats['avg_think_length']:.0f} chars")
    logger.info("")
    logger.info("Tool usage:")
    logger.info(f"  Edit (exact match):  {tool_stats['edit_exact']}")
    logger.info(f"  Edit (fuzzy match):  {tool_stats['edit_fuzzy']}")
    logger.info(f"  Edit (failed):       {tool_stats['edit_failed']}")
    logger.info(f"  Write:               {tool_stats['write']}")
    logger.info("")

    # Comparison with prior results
    logger.info("Comparison with prior results:")
    logger.info(f"  14B+fuzzy_edit (baseline): 4/50 (8.0%) resolved, 32/50 (64%) patches")
    logger.info(f"  14B+cot_fewshot:           {resolved_count}/{total} ({resolved_count/total*100:.1f}%) resolved, {patch_count}/{total} ({patch_count/total*100:.1f}%) patches")
    logger.info("")

    # Per-task detail
    logger.info("Per-task results:")
    traj_map = {t.meta.task_id: t for t in trajectories}
    for r in task_results:
        ts = think_per_task.get(r.task_id, {})
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"steps={r.steps:>2} tokens={r.tokens:>6} "
            f"thinks={ts.get('think_blocks', 0)}"
        )

    # Write REPORT.md
    write_report(result, task_results, trajectories, think_stats, think_per_task,
                 sample_thinks, tool_stats)

    summary = results_summary(result)
    logger.info(f"\n{summary}")

    return result


def write_report(result, task_results, trajectories, think_stats, think_per_task,
                 sample_thinks, tool_stats):
    """Write experiment report to REPORT.md."""
    resolved_count = sum(1 for r in task_results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(task_results)
    avg_steps = sum(r.steps for r in task_results) / total if total else 0
    avg_tokens = sum(r.tokens for r in task_results) / total if total else 0

    resolved_tasks = [r.task_id for r in task_results if r.resolved]
    traj_map = {t.meta.task_id: t for t in trajectories}

    report = f"""# 14B CoT + Few-Shot Experiment Report

## Hypothesis
Wrong-fix semantics are the dominant failure mode (confirmed by edit_tool and fuzzy_edit experiments).
Forcing the model to explicitly reason about (1) what the test expects, (2) what the code does wrong,
and (3) what change will fix it — via mandatory `<think>` blocks — should improve semantic accuracy.

A complete few-shot example (django-11066) in the system prompt demonstrates the expected workflow.

## Configuration
- Model: {MLX_MODEL}
- Tasks: {total} Django tasks from SWE-Bench-CL
- Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}
- Interventions: <think> block requirement + django-11066 few-shot example

## Results Summary

| Metric | CoT+FewShot | Fuzzy Edit Baseline | Edit Baseline | Delta vs Fuzzy |
|--------|-------------|---------------------|---------------|----------------|
| Resolve rate | {resolved_count}/{total} ({resolved_count/total*100:.1f}%) | 4/50 (8.0%) | 5/50 (10.0%) | {(resolved_count/total - 4/50)*100:+.1f}pp |
| Patch rate | {patch_count}/{total} ({patch_count/total*100:.1f}%) | 32/50 (64.0%) | 20/50 (40.0%) | {(patch_count/total - 32/50)*100:+.1f}pp |
| Avg steps | {avg_steps:.1f} | 9.8 | 9.2 | {avg_steps - 9.8:+.1f} |
| Avg tokens | {avg_tokens:.0f} | — | — | — |

## Think Block Analysis

| Metric | Value |
|--------|-------|
| Total think blocks | {think_stats['total_think_blocks']} |
| Tasks with think | {think_stats['tasks_with_think']}/{total} ({think_stats['tasks_with_think']/total*100:.0f}%) |
| Think before edit/write | {think_stats['think_before_edit']} |
| Edit/write without think | {think_stats['edit_without_think']} |
| Avg think length | {think_stats['avg_think_length']:.0f} chars |

## Sample Think Blocks

"""
    # Add up to 5 samples, marking resolved ones
    resolved_set = set(resolved_tasks)
    for i, s in enumerate(sample_thinks[:5]):
        status = "RESOLVED" if s["task_id"] in resolved_set else "FAILED"
        report += f"""### Sample {i+1}: {s['task_id']} (step {s['step']}) — {status}
```
{s['think'][:500]}
```

"""

    report += f"""## Tool Usage

| Metric | Count |
|--------|-------|
| Edit (exact match) | {tool_stats['edit_exact']} |
| Edit (fuzzy match) | {tool_stats['edit_fuzzy']} |
| Edit (failed) | {tool_stats['edit_failed']} |
| Write | {tool_stats['write']} |

## Resolved Tasks
{chr(10).join(f'- {t}' for t in resolved_tasks) if resolved_tasks else 'None'}

## Per-Task Breakdown

| Task | Result | Steps | Tokens | Think Blocks |
|------|--------|-------|--------|-------------|
"""
    for r in task_results:
        ts = think_per_task.get(r.task_id, {})
        has_patch = "patch" if (traj_map.get(r.task_id) and traj_map[r.task_id].generated_patch.strip()) else "no patch"
        status = "RESOLVED" if r.resolved else has_patch
        report += (
            f"| {r.task_id} | {status} | {r.steps} | {r.tokens} | "
            f"{ts.get('think_blocks', 0)} |\n"
        )

    report += f"""
## Analysis

### Key Questions
1. **Did think blocks improve semantic accuracy?** Compare resolve rate vs 8% fuzzy baseline.
2. **Did the model actually use think blocks?** {think_stats['tasks_with_think']}/{total} tasks had at least one think block.
3. **Token budget impact?** The few-shot example adds ~500 tokens to every system prompt. Avg tokens: {avg_tokens:.0f}.

### Cumulative Resolved Tasks
Prior unique resolved: 8 tasks (django-9296, 11066, 11099, 11119, 11163, 11451, 13109, 14373)
New from this experiment: {', '.join(t for t in resolved_tasks if t not in ['django__django-9296', 'django__django-11066', 'django__django-11099', 'django__django-11119', 'django__django-11163', 'django__django-11451', 'django__django-13109', 'django__django-14373']) or 'None'}
"""

    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write(report)
    logger.info(f"Report written to {OUTPUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    run_experiment()
