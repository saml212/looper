# Development Process

## Philosophy

This project uses **waterfall test-driven development** with **sub-agent delegation**. Each coding task is completed by a single sub-agent that:

1. Writes the test first (TDD)
2. Implements the minimum code to pass the test
3. Refactors for simplicity
4. Documents what it built

The architect agent (main agent) maintains the full project vision and dispatches sub-agents sequentially — never in parallel — to preserve context and catch integration issues early.

## Sub-Agent Protocol

Each sub-agent receives:
- **Exact specification** of what to build (one function, one module, one script)
- **Input/output contracts** with concrete examples
- **Reference to existing code** it depends on
- **Test requirements** — what tests to write first

Each sub-agent must:
1. Write failing tests that define the expected behavior
2. Write the simplest implementation that passes the tests
3. Keep code human-readable — no unnecessary abstraction
4. Avoid AI slop: no excessive comments, no redundant docstrings, no over-engineering

After each sub-agent completes:
- Architect verifies tests pass
- Architect reviews for unnecessary complexity
- If needed, architect dispatches a refactor sub-agent
- Documentation is updated

## Build Order (Phase 1 Focus)

Phase 1 requires running the full pipeline on django/django with Qwen 2.5 Coder 7B.

### Step 0: Environment Setup
- Install Ollama, download Qwen 2.5 Coder 7B
- Set up Python 3.11 venv with dependencies
- Download SWE-Bench-CL dataset
- Create /Volumes/1TB_SSD/looper/ for large files (models, adapters, data)

### Step 1: Core Data Models (`looper/models.py`)
- Pydantic v2 models for: AgentTrajectory, AgentStep, SessionMeta, SynthesizedPair, TrainingExample, ExperimentConfig, ExperimentResult
- Pure data — no logic, no I/O

### Step 2: SWE-Bench-CL Task Loader (`looper/tasks/`)
- Load SWE-Bench-CL curriculum JSON
- Filter by repo (django/django)
- Split into train/test sets
- Provide task metadata (problem statement, repo info, test patches)

### Step 3: Agent Runner (`looper/agent/`)
- Simple agent loop: send problem to model via Ollama API, collect tool calls
- Tool execution in Docker containers (SWE-Bench standard)
- Session recording as JSONL (OpenClaw-compatible format)
- Metrics collection (steps, tokens, time)

### Step 4: Experience Collector (`looper/collectors/`)
- Parse JSONL sessions into AgentTrajectory models
- Extract tool usage patterns, errors, outcomes
- Batch mode only (read completed sessions)

### Step 5: Synthetic Data Generator (`looper/synthesizers/`)
- Convert trajectories to training pairs
- Format A (Simple QA) only for Phase 1
- Confidence scoring
- Deduplication

### Step 6: LoRA Trainer (`looper/trainers/`)
- MLX-based LoRA training for Apple Silicon
- Full replay strategy (Experiment 1)
- Model-agnostic data formatting
- Adapter saving/loading

### Step 7: Evaluation Framework (`looper/evaluators/`)
- Resolve rate (does the patch pass tests?)
- Steps to completion
- Token consumption
- Forward transfer metric
- Compare 4 conditions: base, base+RAG, base+LoRA, base+LoRA+RAG

### Step 8: Pipeline Orchestrator (`looper/pipeline.py`)
- Wire Steps 2-7 into end-to-end Phase 1 flow
- Config-driven (YAML)
- Results logging to structured JSON

### Step 9: Run Phase 1
- Execute the full pipeline
- Collect results
- Generate analysis

### Step 10: Research Paper Agents
- Build agents that read results, analyze data, and write paper sections
- Use readinglist.md for informed literature context

## Storage Strategy

- **Code**: /Users/samuellarson/Experiments/looper/ (git repo)
- **Large files**: /Volumes/1TB_SSD/looper/ (models, adapters, datasets, results)
- **Symlinks**: Link from repo to SSD where needed
- **.gitignore**: Exclude large files, include configs and scripts

## Testing Strategy

- Every module has a corresponding test file in `tests/`
- Tests use fixtures with mock data (no real model calls in unit tests)
- Integration tests are marked and can be skipped in CI
- `pytest` with coverage reporting

## Quality Gates

Before moving to the next step:
1. All tests pass (`pytest`)
2. No lint errors (`ruff check`)
3. Code is simple enough that a human can read it in one pass
4. No dead code or unused imports
