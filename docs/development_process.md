# Development Process

## Philosophy

Minimal code, test-driven, incremental progress. Every change is small, tested, and documented. Agents use structured workflows with backpressure mechanisms to maintain quality.

## Agentic Engineering Workflow

Development follows a structured agentic workflow inspired by the [8 Levels of Agentic Engineering](https://www.bassimeledath.com/blog/levels-of-agentic-engineering).

### Core Principles

- **Constraints > instructions**: Boundaries (type system, tests, linters, hooks) over step-by-step checklists. Agents fixate on lists and ignore unlisted items.
- **Backpressure**: Pre-commit hooks, pytest, ruff let agents self-correct without human intervention.
- **Separation of concerns**: Implementer ≠ reviewer. Research ≠ implementation.
- **Codify loop**: When you discover something, update the docs — not just the code.
- **Start → Review → Resume**: Every subagent gets reviewed and sent back to simplify before acceptance.

### Skills (Slash Commands)

| Command | Purpose | Human Input Required? |
|---------|---------|----------------------|
| `/spec <topic>` | Load and review relevant docs | No — context loading |
| `/research <topic>` | Launch research agents, return findings | Yes — human decides |
| `/implement <task>` | Implement one scoped task with tests | Yes — human reviews |
| `/review` | Review changes against standards | No — produces report |
| `/test` | Run tests and summarize results | No — produces report |
| `/codify <learning>` | Document a discovery | No — updates docs |

### The Implement → Codify → Review Loop

```
1. /spec <topic>          ← Load context
2. Human decides task     ← Human chooses
3. /implement <task>      ← Agent implements ONE thing with tests
   ├── start subagent     ← Scoped task
   ├── review output      ← Check scope, quality, minimal code
   ├── resume subagent    ← Refactor, simplify, rerun tests, e2e verify
   └── codify step        ← Document learnings in docs
4. /review                ← Separate review of changes
5. Human approves         ← Human merges or requests changes
```

### The Start → Review → Resume Protocol

This is the key mechanism for keeping subagent output clean:

**Start**: Give the subagent a scoped task. Be specific about boundaries — what to build, what NOT to touch.

**Review**: When the subagent completes, the orchestrator reviews:
- Is the code minimal? Could anything be removed?
- Are tests concise? No redundant setup or assertions?
- Does it follow existing patterns?
- Any dead code or unused imports?

**Resume**: Send the subagent back with specific refinement instructions:
- Reduce total lines of code as much as possible
- Simplify test structure (use fixtures, remove redundancy)
- Remove dead code and unused imports
- Rerun all tests to verify nothing broke
- Do an end-to-end test: create mock/real input, run through the pipeline, verify output
- Only then report back as done

**Accept**: The orchestrator does a final check. Only accept when minimal and clean.

### Backpressure Mechanisms

**Pre-commit hook** (`.githooks/pre-commit`):
- `ruff format --check` — blocks unformatted code
- `ruff check` — blocks lint errors
- `pytest` — blocks failing tests

Enabled via: `git config core.hooksPath .githooks`

**Type system**: Pydantic v2 models validate data at runtime. Use typed models for all data structures.

**Tests**: Required alongside all implementation. No code lands without tests.

### Subagent Architecture

- **Research agents**: Read papers, docs, code. Return findings. Never write code.
- **Implementation agents**: Write code + tests for a specific module. Get reviewed and resumed.
- **Test agents**: Run tests, diagnose failures, fix issues.
- **Review agents**: Separate context from implementer. Check scope, tests, minimal code.
- **Experiment agents**: Run experiments, collect results, report metrics.

**Critical rule**: Orchestrating agents MUST delegate to subagents and preserve their own context for planning, decisions, and progress tracking. This prevents context exhaustion on long-running tasks.

## Code Quality

- **Minimal**: Write the least code that solves the problem.
- **Simple**: Readable beats elegant. No unnecessary abstractions.
- **Small PRs**: Each PR does one thing.
- **No dead code**: No unused imports, commented-out code, or code "for later."

## Testing Strategy

- Every module has tests in `tests/`
- Tests use fixtures with mock data (no real model calls in unit tests)
- `pytest` with `--tb=short` for concise output
- End-to-end smoke tests for pipeline changes
- All tests should run in < 10 seconds

## Storage Strategy

- **Code**: `/Users/samuellarson/Experiments/looper/` (git repo)
- **Large files**: `/Volumes/1TB_SSD/looper/` (models, adapters, datasets, results)
- **Reference repos**: `/Volumes/1TB_SSD/looper/cache/workspaces/.refs/`

## Commit Practices

- Descriptive messages that explain WHY, not just WHAT
- One logical change per commit
- Don't commit generated files, model weights, or secrets
- Normal commit messages — no AI attribution

## Decision Log

Decisions documented in relevant docs with:
- What was decided
- Alternatives considered
- Why this option was chosen
- Links to experiment results or research
