# Development Process

## Philosophy

This project uses **test-driven development** with **sub-agent delegation**. Each coding task is completed by a single sub-agent that:

1. Writes the test first (TDD)
2. Implements the minimum code to pass the test
3. Refactors for simplicity
4. Documents what it built

The architect agent (main agent) maintains the full project vision and dispatches sub-agents sequentially to preserve context and catch integration issues early.

## Current State

- **222+ tests passing** across 14 test modules
- All 8 build order steps complete
- Framework is production-ready for experiments
- Experiments ran March 4-14, 2026

## Build Order (Complete)

| Step | Module | Status |
|------|--------|--------|
| 1 | Core data models (`looper/models.py`) | Done |
| 2 | SWE-Bench-CL task loader (`looper/tasks/`) | Done |
| 3 | Agent runner (`looper/agent/`) | Done |
| 4 | Experience collector (`looper/collectors/`) | Done |
| 5 | Synthetic data generator (`looper/synthesizers/`) | Done |
| 6 | LoRA trainer (`looper/trainers/`) | Done |
| 7 | Evaluation framework (`looper/evaluators/`) | Done |
| 8 | Pipeline orchestrator (`looper/pipeline.py`) | Done |

Post-build additions:
- OpenClaw integration (`looper/integrations/`)
- Patch verification with FAIL_TO_PASS tests (`looper/evaluators/patch_verifier.py`)
- EWC trainer (`looper/trainers/ewc_trainer.py`)
- Edit tool with fuzzy matching (`looper/agent/runner.py`)
- Framework fixes: line-range reads, context pruning, loop detection, code fence stripping

## Storage Strategy

- **Code**: `/Users/samuellarson/Experiments/looper/` (git repo)
- **Large files**: `/Volumes/1TB_SSD/looper/` (models, adapters, datasets, results)
- **Reference repos**: `/Volumes/1TB_SSD/looper/cache/workspaces/.refs/` (bare clones for fast workspace creation)

## Testing Strategy

- Every module has a corresponding test file in `tests/`
- Tests use fixtures with mock data (no real model calls in unit tests)
- `pytest` with coverage reporting
- All tests run in < 6 seconds

## Quality Gates

Before moving to the next step:
1. All tests pass (`pytest`)
2. Code is simple enough that a human can read it in one pass
3. No dead code or unused imports
