#!/usr/bin/env python3
"""Multi-repo expansion experiment — break through the cold-start problem.

The base 14B model resolves ~8% of Django tasks (2/25). Self-play LoRA training
is blocked because we can't collect enough resolved trajectories from one repo.

This experiment expands to ALL 8 SWE-Bench-CL repos (273 tasks total) to collect
resolved trajectories at scale. At 8% resolve rate, we expect ~22 resolved
trajectories — enough for DPO training.

Phase 1: 25-task pilot on sympy (go/no-go: resolve rate >= 5%)
Phase 2: If pilot passes, run ALL remaining non-django tasks
Phase 3: Aggregate resolved trajectories for DPO training
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from looper.collectors.trajectory_store import collect_trajectories, load_trajectory
from looper.evaluators.patch_verifier import verify_patch_tests
from looper.evaluators.results_io import save_results, results_summary
from looper.models import ExperimentConfig, ExperimentResult, TaskResult
from looper.tasks.loader import get_repo_tasks, load_curriculum

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5-coder:14b"
OLLAMA_URL = "http://localhost:11434"

CURRICULUM = Path("/Volumes/1TB_SSD/looper/datasets/swe-bench-cl-curriculum.json")
WORKSPACE_ROOT = Path("/Volumes/1TB_SSD/looper/cache/workspaces")
OUTPUT_DIR = Path("/Volumes/1TB_SSD/looper/results/experiment_multirepo_expansion")

MAX_STEPS = 15
MAX_TOKENS = 4096
TEMPERATURE = 0.3  # Same as self-play experiment

# All repos in SWE-Bench-CL
ALL_REPOS = [
    "django/django",
    "sympy/sympy",
    "sphinx-doc/sphinx",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "astropy/astropy",
    "pydata/xarray",
    "pytest-dev/pytest",
]

# Pilot repo — sympy has 50 tasks (same as Django), 25 are "<15 min fix"
PILOT_REPO = "sympy/sympy"
PILOT_SIZE = 25  # First 25 tasks (all "<15 min fix")

# Minimum resolve rate to continue to phase 2
MIN_RESOLVE_RATE = 0.05  # 5%


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


def create_reference_clone(repo: str, workspace_root: Path, timeout: int = 600):
    """Create a bare reference clone for faster workspace creation."""
    import subprocess

    repo_name = repo.split("/")[-1]
    ref_dir = workspace_root / ".refs" / repo_name

    if ref_dir.exists():
        logger.info(f"  Reference clone exists: {ref_dir}")
        return

    logger.info(f"  Creating reference clone for {repo}...")
    ref_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(ref_dir)],
        check=True,
        capture_output=True,
        timeout=timeout,
    )
    logger.info(f"  Reference clone created: {ref_dir}")


def run_repo_tasks(
    repo: str,
    tasks,
    condition: str,
    output_dir: Path,
):
    """Run agent on tasks from a single repo, evaluate, return results."""
    traj_dir = output_dir / "trajectories" / repo.replace("/", "_")
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Create reference clone for faster workspace setup
    try:
        create_reference_clone(repo, WORKSPACE_ROOT)
    except Exception as e:
        logger.warning(f"  Failed to create reference clone for {repo}: {e}")
        logger.warning(f"  Will clone from GitHub for each task (slower)")

    def on_complete(tid, traj):
        has_patch = "yes" if traj.generated_patch.strip() else "no"
        logger.info(
            f"  [{repo}] {tid} -> {traj.outcome} "
            f"({len(traj.steps)} steps, patch={has_patch})"
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"Running {len(tasks)} tasks for {repo}")
    logger.info(f"{'='*60}")

    trajectories = collect_trajectories(
        tasks=tasks,
        output_dir=traj_dir,
        workspace_root=WORKSPACE_ROOT,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        max_steps=MAX_STEPS,
        max_tokens=MAX_TOKENS,
        on_complete=on_complete,
    )

    # Evaluate with FAIL_TO_PASS verification
    logger.info(f"\nVerifying patches for {repo}...")
    task_map = {t.instance_id: t for t in tasks}
    results = []
    for traj in trajectories:
        task = task_map.get(traj.meta.task_id)
        resolved = False
        if task and traj.generated_patch.strip():
            try:
                vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
                resolved = vr["resolved"]
                logger.info(
                    f"  Verify {traj.meta.task_id}: "
                    f"{'PASS' if resolved else 'FAIL'} "
                    f"({vr['fail_to_pass_passed']}/{vr['fail_to_pass_total']})"
                    + (f" error={vr['error']}" if vr["error"] else "")
                )
            except Exception as e:
                logger.error(f"  Verify {traj.meta.task_id}: ERROR {e}")
        else:
            logger.info(
                f"  Verify {traj.meta.task_id}: SKIP (no patch)"
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

    return trajectories, results


def load_existing_django_results():
    """Load resolved Django trajectories from prior experiments."""
    resolved = []

    # Check self-play results (most recent 14B Django run)
    selfplay_dir = Path("/Volumes/1TB_SSD/looper/results/experiment_selfplay_14b/trajectories")
    if selfplay_dir.exists():
        collection_dir = selfplay_dir / "collection"
        if collection_dir.exists():
            for traj_file in sorted(collection_dir.glob("*.json")):
                traj = load_trajectory(traj_file)
                resolved.append(("django/django", traj))

    # Check 14B framework results
    for results_dir in [
        Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_full/trajectories/base"),
        Path("/Volumes/1TB_SSD/looper/results/experiment_framework_14b_remaining/trajectories/base"),
    ]:
        if results_dir.exists():
            for traj_file in sorted(results_dir.glob("*.json")):
                traj = load_trajectory(traj_file)
                resolved.append(("django/django", traj))

    return resolved


def aggregate_resolved_trajectories(
    all_trajectories: list[tuple[str, list, list]],
    django_trajs: list[tuple[str, object]],
    curriculum: dict,
):
    """Collect all resolved trajectories across repos."""
    resolved = []
    failed_with_patch = []

    # Get task info for Django trajectories
    django_tasks = get_repo_tasks(curriculum, "django/django")
    django_task_map = {t.instance_id: t for t in django_tasks}

    for repo, traj in django_trajs:
        task = django_task_map.get(traj.meta.task_id)
        if task and traj.generated_patch.strip():
            # Re-verify Django trajectories
            try:
                vr = verify_patch_tests(task, traj.generated_patch, WORKSPACE_ROOT)
                if vr["resolved"]:
                    resolved.append((repo, traj))
                elif traj.generated_patch.strip():
                    failed_with_patch.append((repo, traj))
            except Exception:
                if traj.generated_patch.strip():
                    failed_with_patch.append((repo, traj))

    # New repo results
    for repo, trajectories, results in all_trajectories:
        for traj, result in zip(trajectories, results):
            if result.resolved:
                resolved.append((repo, traj))
            elif traj.generated_patch.strip():
                failed_with_patch.append((repo, traj))

    return resolved, failed_with_patch


def run_pilot():
    """Phase 1: Run 25-task pilot on sympy."""
    setup_logging(OUTPUT_DIR)
    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("MULTI-REPO EXPANSION — PHASE 1: PILOT")
    logger.info(f"Pilot repo: {PILOT_REPO}")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Max steps: {MAX_STEPS}, Max tokens: {MAX_TOKENS}")
    logger.info("=" * 60)

    curriculum = load_curriculum(CURRICULUM)
    all_tasks = get_repo_tasks(curriculum, PILOT_REPO)
    pilot_tasks = all_tasks[:PILOT_SIZE]

    logger.info(f"\nPilot tasks ({len(pilot_tasks)}):")
    for t in pilot_tasks:
        logger.info(f"  {t.instance_id} ({t.difficulty})")

    trajectories, results = run_repo_tasks(
        PILOT_REPO, pilot_tasks, "base_14b_pilot", OUTPUT_DIR,
    )

    resolved_count = sum(1 for r in results if r.resolved)
    patch_count = sum(1 for t in trajectories if t.generated_patch.strip())
    total = len(results)
    resolve_rate = resolved_count / total if total else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("PILOT RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Repo: {PILOT_REPO}")
    logger.info(f"  Resolve rate: {resolved_count}/{total} ({resolve_rate*100:.1f}%)")
    logger.info(f"  Patch rate:   {patch_count}/{total} ({patch_count/total*100:.1f}%)")
    logger.info(f"  Go/No-Go threshold: {MIN_RESOLVE_RATE*100:.0f}%")
    logger.info(f"  Decision: {'GO — continue to phase 2' if resolve_rate >= MIN_RESOLVE_RATE else 'STOP — resolve rate too low'}")

    # Per-task detail
    logger.info("\nPer-task results:")
    for r in results:
        t_idx = next(
            (i for i, t in enumerate(trajectories) if t.meta.task_id == r.task_id),
            -1,
        )
        traj = trajectories[t_idx] if t_idx >= 0 else None
        has_patch = traj.generated_patch.strip() if traj else ""
        logger.info(
            f"  {r.task_id}: "
            f"{'RESOLVED' if r.resolved else 'FAILED':>8} "
            f"steps={r.steps:>2} tokens={r.tokens:>6} "
            f"patch={'yes' if has_patch else 'no':>3}"
        )

    # Save pilot results
    config = ExperimentConfig(
        name="multirepo_expansion_pilot",
        experiment_id="multirepo_pilot",
        repo=PILOT_REPO,
        model_name=OLLAMA_MODEL,
        train_task_ids=[],
        test_task_ids=[t.instance_id for t in pilot_tasks],
        strategy="base_14b_multirepo",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=results,
        forward_transfer=0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "pilot_result.json")

    return resolve_rate, resolved_count, trajectories, results


def run_expansion(curriculum: dict, pilot_repo: str):
    """Phase 2: Run ALL remaining non-pilot, non-django tasks."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("MULTI-REPO EXPANSION — PHASE 2: FULL EXPANSION")
    logger.info("=" * 60)

    started_at = datetime.now(timezone.utc).isoformat()
    all_repo_results = []

    for repo in ALL_REPOS:
        if repo == "django/django":
            logger.info(f"\nSkipping {repo} — already have results from prior experiments")
            continue

        all_tasks = get_repo_tasks(curriculum, repo)
        if repo == pilot_repo:
            # Skip pilot tasks, run remaining
            tasks = all_tasks[PILOT_SIZE:]
            if not tasks:
                logger.info(f"\nSkipping {repo} — all tasks covered by pilot")
                continue
        else:
            tasks = all_tasks

        trajectories, results = run_repo_tasks(
            repo, tasks, f"base_14b_{repo.replace('/', '_')}", OUTPUT_DIR,
        )
        all_repo_results.append((repo, trajectories, results))

    # Save expansion results
    all_task_results = []
    for repo, trajs, results in all_repo_results:
        all_task_results.extend(results)

    config = ExperimentConfig(
        name="multirepo_expansion_full",
        experiment_id="multirepo_full",
        repo="multi",
        model_name=OLLAMA_MODEL,
        train_task_ids=[],
        test_task_ids=[r.task_id for r in all_task_results],
        strategy="base_14b_multirepo",
        seed=0,
    )
    result = ExperimentResult(
        config=config,
        task_results=all_task_results,
        forward_transfer=0.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_results(result, OUTPUT_DIR / "expansion_result.json")

    return all_repo_results


def main():
    # Phase 1: Pilot
    resolve_rate, resolved_count, pilot_trajs, pilot_results = run_pilot()

    curriculum = load_curriculum(CURRICULUM)

    if resolve_rate < MIN_RESOLVE_RATE:
        logger.info(f"\nPilot resolve rate {resolve_rate*100:.1f}% < {MIN_RESOLVE_RATE*100:.0f}% threshold.")
        logger.info("Expanding anyway — we need maximum trajectory coverage.")
        logger.info("Even at 0% on sympy, other repos may have different rates.")

    # Phase 2: Expand to all remaining repos
    all_repo_results = run_expansion(curriculum, PILOT_REPO)

    # Include pilot results in aggregation
    all_repo_results.insert(0, (PILOT_REPO, pilot_trajs, pilot_results))

    # Load existing Django trajectories
    django_trajs = load_existing_django_results()
    logger.info(f"\nLoaded {len(django_trajs)} existing Django trajectories")

    # Phase 3: Aggregate
    resolved, failed_with_patch = aggregate_resolved_trajectories(
        all_repo_results, django_trajs, curriculum,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL AGGREGATION")
    logger.info("=" * 60)
    logger.info(f"  Total resolved trajectories: {len(resolved)}")
    logger.info(f"  Total failed-with-patch:     {len(failed_with_patch)}")
    logger.info(f"  DPO training viable:         {'YES' if len(resolved) >= 10 else 'NO'} (need 10+)")

    if resolved:
        logger.info("\n  Resolved tasks:")
        for repo, traj in resolved:
            logger.info(f"    [{repo}] {traj.meta.task_id} ({traj.meta.total_steps} steps)")

    # Per-repo summary
    logger.info("\n  Per-repo summary:")
    from collections import Counter
    repo_resolved = Counter()
    repo_total = Counter()
    for repo, trajs, results in all_repo_results:
        for r in results:
            repo_total[repo] += 1
            if r.resolved:
                repo_resolved[repo] += 1

    # Add Django
    for repo in ALL_REPOS:
        r = repo_resolved.get(repo, 0)
        t = repo_total.get(repo, 0)
        if repo == "django/django":
            # Count from loaded trajectories
            dr = sum(1 for rp, _ in resolved if rp == repo)
            dt = sum(1 for rp, _ in django_trajs)
            logger.info(f"    {repo:30s}: {dr}/{dt} resolved (from prior experiments)")
        elif t > 0:
            logger.info(f"    {repo:30s}: {r}/{t} resolved ({r/t*100:.1f}%)")

    # Save aggregation summary
    summary = {
        "total_resolved": len(resolved),
        "total_failed_with_patch": len(failed_with_patch),
        "resolved_tasks": [
            {"repo": repo, "task_id": traj.meta.task_id, "steps": traj.meta.total_steps}
            for repo, traj in resolved
        ],
        "dpo_viable": len(resolved) >= 10,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "aggregation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    logger.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
