"""Simple RAG retrieval over agent trajectories.

Uses TF-IDF similarity to find relevant past trajectories for a given task.
No external dependencies beyond scikit-learn (already in the venv).
"""

import logging
from dataclasses import dataclass

from looper.models import AgentTrajectory, TaskInfo

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Context retrieved from past trajectories for RAG injection."""
    task_id: str
    problem_statement: str
    approach_summary: str
    patch_snippet: str
    similarity: float


def _trajectory_to_document(traj: AgentTrajectory, task: TaskInfo | None) -> str:
    """Convert a trajectory into a searchable document string."""
    parts = [f"Task: {traj.meta.task_id}"]
    if task:
        parts.append(f"Problem: {task.problem_statement[:500]}")
    parts.append(f"Outcome: {traj.outcome}")
    parts.append(f"Steps: {traj.meta.total_steps}")

    # Include tool call summaries
    for step in traj.steps[:10]:
        for tc in step.tool_calls[:1]:
            if tc.tool_name == "bash":
                cmd = tc.tool_input.get("input", "")[:100]
                parts.append(f"Command: {cmd}")
            elif tc.tool_name == "read":
                parts.append(f"Read: {tc.tool_input.get('input', '')[:100]}")

    if traj.generated_patch:
        parts.append(f"Patch: {traj.generated_patch[:300]}")

    return "\n".join(parts)


def _trajectory_to_context(traj: AgentTrajectory, task: TaskInfo | None) -> str:
    """Convert a trajectory into context text suitable for injection into a prompt."""
    parts = []
    parts.append(f"--- Past experience: {traj.meta.task_id} ---")
    if task:
        parts.append(f"Problem: {task.problem_statement[:300]}")

    # Summarize the approach
    approach = []
    for step in traj.steps:
        for tc in step.tool_calls:
            if tc.tool_name == "bash":
                approach.append(f"  $ {tc.tool_input.get('input', '')[:80]}")
            elif tc.tool_name == "read":
                approach.append(f"  [read] {tc.tool_input.get('input', '')[:80]}")
            elif tc.tool_name == "write":
                approach.append(f"  [write] {tc.tool_input.get('path', '')[:80]}")

    if approach:
        parts.append("Approach taken:")
        parts.extend(approach[:8])  # Top 8 actions

    if traj.generated_patch:
        parts.append(f"Patch (first 200 chars): {traj.generated_patch[:200]}")

    parts.append(f"Outcome: {traj.outcome}")
    parts.append("---")
    return "\n".join(parts)


def build_rag_index(
    trajectories: list[AgentTrajectory],
    tasks: list[TaskInfo],
) -> tuple[list[AgentTrajectory], list[str], object]:
    """Build a TF-IDF index over trajectories.

    Returns (trajectories, documents, vectorizer_and_matrix) for retrieval.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    task_map = {t.instance_id: t for t in tasks}

    docs = []
    valid_trajs = []
    for traj in trajectories:
        task = task_map.get(traj.meta.task_id)
        doc = _trajectory_to_document(traj, task)
        docs.append(doc)
        valid_trajs.append(traj)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    return valid_trajs, docs, (vectorizer, tfidf_matrix)


def retrieve_context(
    query_task: TaskInfo,
    index: tuple[list[AgentTrajectory], list[str], object],
    tasks: list[TaskInfo],
    top_k: int = 3,
    max_context_chars: int = 3000,
) -> str:
    """Retrieve relevant past trajectory context for a task.

    Returns a string suitable for injecting into the agent's system prompt.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    trajs, docs, (vectorizer, tfidf_matrix) = index
    task_map = {t.instance_id: t for t in tasks}

    query = f"Problem: {query_task.problem_statement[:500]}"
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Get top-k indices (excluding the query task itself)
    ranked = sorted(enumerate(similarities), key=lambda x: -x[1])
    selected = []
    for idx, sim in ranked:
        if trajs[idx].meta.task_id == query_task.instance_id:
            continue
        if sim < 0.01:
            break
        selected.append((idx, sim))
        if len(selected) >= top_k:
            break

    if not selected:
        return ""

    # Build context string
    parts = ["RELEVANT PAST EXPERIENCES (from similar tasks in this repository):"]
    total_chars = 0
    for idx, sim in selected:
        traj = trajs[idx]
        task = task_map.get(traj.meta.task_id)
        ctx = _trajectory_to_context(traj, task)
        if total_chars + len(ctx) > max_context_chars:
            break
        parts.append(ctx)
        total_chars += len(ctx)

    if len(parts) == 1:
        return ""

    return "\n".join(parts)
