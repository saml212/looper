# Looper — Agent Guidelines

## What This Project Is

Looper is an **experimental research framework** that tested whether periodic LoRA consolidation of agent experience produces measurably more capable coding agents. The answer, after 8 experiments across 10 days, is **no** — at least not at current model scales (7B-32B) and data volumes (12-31 resolved tasks).

The most impactful findings were not about LoRA but about the agent framework itself: systematic prompt engineering and tool design improvements tripled the base resolve rate (8% to 27%).

**Key docs:**
1. [LEARNINGS.md](LEARNINGS.md) — Comprehensive results from all experiments
2. [DEEP_AUDIT.md](DEEP_AUDIT.md) — Root cause analysis of why LoRA failed
3. [docs/problem.md](docs/problem.md) — The original thesis: skills vs knowledge
4. [docs/experiments.md](docs/experiments.md) — All 10 pre-registered experiments with results
5. [docs/architecture.md](docs/architecture.md) — Framework components and data flow

## Current Status

**Build:** Complete. 222+ tests passing. All modules implemented.
**Phase 1:** Complete. FT=0 across all conditions (workspace contamination bug invalidated cross-condition comparison).
**Experiments 3, 4, 6, 7:** Complete. No positive results. See LEARNINGS.md.
**Framework fixes:** 8% -> 27% resolve rate via prompt engineering and tool improvements.
**Scaling:** 14B optimal on 32GB Apple Silicon. 32B ties 14B at 5x cost.
**LoRA:** Every training strategy produced zero or negative forward transfer. Core thesis remains untested — needs 100+ resolved tasks or fundamentally different approach.

## Core Framing

This project adds a **new layer** to the agent stack — it does NOT replace context engineering or RAG. The skill adapter sits between the knowledge/retrieval layer and the base model:

```
Context Window    -> what's happening now
Memory/Retrieval  -> knowledge from past sessions
Skill Adapter     -> learned environmental fluency (LoRA)
Base Model        -> general intelligence
```

**Skills vs Knowledge:**
- **Skills** = how to use tools efficiently, navigate environments, recognize patterns, debug instinctively. Encoded in LoRA weights.
- **Knowledge** = facts, docs, specific details. Stays in context/RAG.

## Project Structure

```
looper/
├── looper/              # Python package
│   ├── models.py        # Core Pydantic data models (122 lines)
│   ├── pipeline.py      # Phase 1 orchestration (335 lines)
│   ├── agent/           # Agent execution loop, inference clients, workspace
│   ├── collectors/      # Trajectory save/load
│   ├── synthesizers/    # Trajectory -> training data (including oracle)
│   ├── trainers/        # LoRA training (MLX), EWC, full replay
│   ├── evaluators/      # Patch verification (FAIL_TO_PASS), metrics, RAG
│   ├── integrations/    # OpenClaw parser and experiment runner
│   ├── serving/         # Adapter-to-Ollama conversion
│   ├── tasks/           # SWE-Bench-CL curriculum loader
│   └── analysis/        # Results analysis
├── tests/               # 222+ tests
├── docs/                # Research documentation
├── run_*.py             # Experiment scripts
├── LEARNINGS.md         # Comprehensive results
└── DEEP_AUDIT.md        # Root cause analysis
```

## Technical Stack

- **Python 3.11** with Pydantic v2 for data models
- **LoRA training:** MLX on Apple Silicon (M4 32GB)
- **Inference:** Ollama (base model), MLX in-process (adapted model)
- **Evaluation:** SWE-Bench-CL benchmark (273 tasks, 8 Python repos)
- **Agent protocol:** XML tool tags (`<bash>`, `<read>`, `<write>`, `<edit>`, `<done>`)

## Key Constraints

- **One inference server at a time** — Ollama and MLX cannot run simultaneously on 32GB
- **LoRA fusion breaks quantized models** — adapter must be applied dynamically via `mlx_lm.server --adapter-path` or `mlx_chat()` in-process
- **14B is max viable for adapted inference** — 32B OOMs during MLX LoRA inference
- **Workspace reset** — `create_workspace()` resets uncommitted changes on re-entry (fixed March 9)
- **Code fence stripping** — 14B+ models wrap `<write>` content in markdown fences, stripped by `_strip_code_fences()`
- **Subprocess training** — run LoRA training in subprocess to fully free GPU memory before inference
- **Ollama on SSD** — `OLLAMA_MODELS=/Volumes/1TB_SSD/looper/ollama_models ollama serve`

## Environment

- Python 3.11 venv at `.venv`
- SSD storage: `/Volumes/1TB_SSD/looper/` (datasets, workspaces, results)
- Reference bare repos at `/Volumes/1TB_SSD/looper/cache/workspaces/.refs/`
- Results always written to `/Volumes/1TB_SSD/looper/results/`
