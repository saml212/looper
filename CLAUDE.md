# Looper — Agent Guidelines

## What This Is
Looper tests whether LoRA consolidation of agent experience produces more capable coding agents. Python 3.11 + MLX + Ollama + SWE-Bench-CL. After 8 experiments: LoRA didn't work, but framework engineering tripled resolve rate (8%→27%).

## Documentation (Read Before Working)
- [LEARNINGS.md](LEARNINGS.md) — Results from all 8 experiments
- [docs/problem.md](docs/problem.md) — The skills vs knowledge thesis
- [docs/architecture.md](docs/architecture.md) — Framework components and data flow
- [docs/experiments.md](docs/experiments.md) — All pre-registered experiments with results
- [docs/future-work.md](docs/future-work.md) — Next directions
- [docs/development_process.md](docs/development_process.md) — Workflow, agentic engineering
- [docs/research_landscape.md](docs/research_landscape.md) — Literature survey
- [readinglist.md](readinglist.md) — 100+ annotated papers

## Tech Stack
- **Python 3.11** — Pydantic v2 for data models
- **MLX** — LoRA training on Apple Silicon (M4 32GB)
- **Ollama** — base model inference (~18 sec/request)
- **SWE-Bench-CL** — 273 tasks, 8 Python repos
- **Agent protocol** — XML tool tags (`<bash>`, `<read>`, `<write>`, `<edit>`, `<done>`)
- **Testing** — pytest, ruff for formatting/linting

## Coding Rules
1. Minimal code. Don't add abstractions for hypothetical futures.
2. Simple over clever. Readable beats elegant.
3. Test-driven. Write tests alongside implementation.
4. Incremental. Small changes, one thing at a time.
5. `ruff format` and `ruff check` before every commit. Pre-commit hook enforces this.
6. Don't refactor existing code unless asked.
7. Don't add features beyond what was requested.
8. Pydantic v2 for all data models.
9. No dead code, unused imports, or commented-out code.

## Workflow (Skills)
- `/spec <topic>` — Load and review relevant docs before working.
- `/research <topic>` — Launch research agents. Returns findings. Human decides.
- `/implement <task>` — Implement one scoped task with tests. Uses start→review→resume loop.
- `/review` — Review changes against standards.
- `/test` — Run tests and report results.
- `/codify <learning>` — Document a discovery in the relevant doc.

## The Codify Rule
When you fix a bug, hit a gotcha, or make a design decision:
**update the relevant doc, not just the code.** See LEARNINGS.md for examples
of well-documented findings.

## Subagent Protocol: Start → Review → Resume
Every time you delegate to a subagent:
1. **Start**: Give the subagent a scoped task with clear boundaries.
2. **Review**: When it completes, review its output for scope, quality, minimal code.
3. **Resume**: Send it back to refactor, simplify, reduce code, simplify tests, rerun tests, and do an end-to-end verification (input→output check).
4. **Accept**: Only accept when the code is minimal and all checks pass.

## Key Constraints
- One inference server at a time (Ollama + MLX can't share 32GB)
- LoRA fusion breaks quantized models — use dynamic adapter application
- 14B max for adapted inference on 32GB M4
- Subprocess training essential (MLX holds GPU memory)
- `caffeinate -dims` for overnight runs
- Ollama on SSD: `OLLAMA_MODELS=/Volumes/1TB_SSD/looper/ollama_models ollama serve`

## Environment
- Python 3.11 venv at `.venv`
- SSD storage: `/Volumes/1TB_SSD/looper/` (models, datasets, results, workspaces)
- v1 code preserved on `v1` branch for reference

## Current State
- v1 complete (8 experiments, 222+ tests). Code on `v1` branch.
- Main branch: documentation only. Starting v2 fresh.
- Keep this file under 100 lines.
