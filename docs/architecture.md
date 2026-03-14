# Framework Architecture

## System Overview

Looper is a pipeline that runs coding agents on SWE-Bench tasks, collects trajectories, synthesizes training data, trains LoRA adapters, and evaluates whether the adapted model outperforms the base.

```
┌──────────────────────────────────────────────────────────────────┐
│                       THE LOOPER PIPELINE                        │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  TASK LOADER │─>│  AGENT LOOP  │─>│  TRAJECTORY  │           │
│  │  (curriculum)│  │  (runner.py) │  │  COLLECTOR   │           │
│  └─────────────┘  └──────────────┘  └──────┬───────┘           │
│                                             │                    │
│                              ┌──────────────┴───────────┐       │
│                              │                          │       │
│                              v                          v       │
│                    ┌──────────────┐            ┌─────────────┐  │
│                    │  SYNTHESIZER │            │  EVALUATOR   │  │
│                    │  (training   │            │  (patch      │  │
│                    │   data gen)  │            │   verifier)  │  │
│                    └──────┬──────┘            └─────────────┘  │
│                           │                                     │
│                           v                                     │
│                    ┌──────────────┐                              │
│                    │  LoRA TRAIN  │                              │
│                    │  (MLX)       │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│                           v                                     │
│                    ┌──────────────┐                              │
│                    │  ADAPTED     │─> Agent loop again ─> Eval  │
│                    │  MODEL       │                              │
│                    └──────────────┘                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Task Loader

**Module:** `looper/tasks/loader.py`

Loads the SWE-Bench-CL curriculum JSON and extracts tasks for a given repo. Tasks are split chronologically (first 25 train, last 25 test) or randomly.

```python
curriculum = load_curriculum(Path("swe-bench-cl-curriculum.json"))
all_tasks = get_repo_tasks(curriculum, "django/django")
train_tasks, test_tasks = split_tasks(all_tasks, train_size=25)
```

Each `TaskInfo` contains: `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch` (gold), `test_patch`, `fail_to_pass`, `pass_to_pass`.

---

## Component 2: Agent Loop

**Module:** `looper/agent/runner.py`

A text-based XML tool protocol for small (7B-32B) models. The system prompt includes available tools, a worked example, and the problem statement. The agent iterates: chat -> parse XML tool calls -> execute in workspace -> feed results back.

### Tools

| Tool | Format | Purpose |
|------|--------|---------|
| `<bash>cmd</bash>` | Shell command | Search, grep, test |
| `<read>path</read>` | Read file | Inspect code |
| `<read>path:100-200</read>` | Line-range read | Large files |
| `<edit path="f">old\n=======\nnew</edit>` | Find-replace | Targeted fixes |
| `<write path="f">content</write>` | Overwrite file | New/small files |
| `<done>` | Signal completion | After writing fix |

### Key features

- **One tool per response** — prevents hallucination of multiple tool calls
- **`<think>` block** — model must analyze before editing (rule 9)
- **Few-shot example** — complete worked example (django-11066) in system prompt
- **Code fence stripping** — `_strip_code_fences()` handles 14B+ markdown wrapping
- **Fuzzy edit matching** — `_fuzzy_find()` with difflib at 0.7 threshold
- **Context pruning** — `prune_messages()` replaces old tool results with summaries
- **Loop detection** — warns after 3 reads of same file; nudges after 3 failed edits
- **Skip verification rule** — prevents pip-install loops

### Inference clients

**Module:** `looper/agent/ollama_client.py`

| Client | Use Case |
|--------|----------|
| `chat()` | Ollama API (base model) |
| `openai_chat()` | OpenAI-compatible API (mlx_lm.server) |
| `mlx_chat()` | In-process MLX inference (adapted model) |

### Workspace management

**Module:** `looper/agent/workspace.py`

- `create_workspace()` — clones repo at `base_commit`, uses `--shared` from local bare repos
- `reset_workspace()` — `git checkout -- . && git clean -fd` (called on every re-entry)
- `get_patch()` — `git diff` of uncommitted changes
- Workspaces at `/Volumes/1TB_SSD/looper/cache/workspaces/<repo>/<commit[:8]>/`

---

## Component 3: Trajectory Collection

**Module:** `looper/collectors/trajectory_store.py`

Runs the agent on each task, saves trajectories as JSON, supports resume (skips existing files).

```python
trajectories = collect_trajectories(
    tasks=test_tasks,
    output_dir=Path("trajectories/base"),
    workspace_root=workspace_root,
    model="qwen2.5-coder:7b",
    max_steps=15,
)
```

Each `AgentTrajectory` contains: `meta` (session info), `steps[]` (reasoning + tool calls), `outcome`, `generated_patch`.

---

## Component 4: Synthesis

**Module:** `looper/synthesizers/synthesizer.py`

Converts trajectories to training data by prompting an LLM to extract instruction/response pairs. Four pair types: `tool_usage`, `error_recovery`, `convention`, `workflow`.

**Finding:** This approach failed. Self-distilled Q&A from failed trajectories produced garbage data ("How do you read a file?" level content). See DEEP_AUDIT.md.

**Module:** `looper/synthesizers/oracle_synthesizer.py`

Alternative approach: uses gold SWE-Bench patches directly as training data (bug report -> unified diff). Also failed — format mismatch with agent's XML tool protocol.

---

## Component 5: LoRA Training

**Module:** `looper/trainers/lora_trainer.py`

MLX-based LoRA training on Apple Silicon.

| Parameter | Value |
|-----------|-------|
| Framework | MLX (`mlx_lm.tuner.train`) |
| Model | mlx-community/Qwen2.5-Coder-7B-Instruct-4bit |
| Rank | 16 (4 for 14B due to OOM) |
| Scale (alpha) | rank * 2.0 |
| Dropout | 0.05 |
| Layers | 16 (8 for 14B) |
| Iterations | 100 |
| Batch size | 1 (memory constraint) |
| Optimizer | Adam, lr=1e-4 |

**Additional trainers:**
- `full_replay.py` — retrains from scratch on all accumulated data
- `ewc_trainer.py` — custom MLX loop with EWC penalty (Fisher information regularization)
- `data_formatter.py` — splits training data into train/valid JSONL

**Critical constraint:** Training runs in a subprocess to fully free GPU memory before inference. The parent process parses metrics JSON from stdout.

---

## Component 6: Evaluation

**Module:** `looper/evaluators/patch_verifier.py`

Real verification: applies generated patch + test patch, runs Django's `runtests.py`, checks FAIL_TO_PASS tests.

```python
result = verify_patch_tests(task, generated_patch, workspace_root)
# result["resolved"] = True iff all FAIL_TO_PASS tests pass
```

**Module:** `looper/evaluators/metrics.py`

Forward transfer = adapted_resolve_rate - base_resolve_rate.

**Module:** `looper/evaluators/rag.py`

TF-IDF RAG retrieval (scikit-learn) for Base+RAG condition.

---

## Component 7: Pipeline Orchestrator

**Module:** `looper/pipeline.py`

Wires all components into `run_phase1()`: load tasks -> run base model -> evaluate -> synthesize -> train LoRA -> run adapted model -> evaluate -> compute metrics -> save results.

Experiment-specific scripts in `run_*.py` files at the project root.

---

## Directory Structure (Actual)

```
looper/
├── looper/
│   ├── __init__.py
│   ├── models.py                    # Core Pydantic data models
│   ├── pipeline.py                  # Phase 1 orchestration
│   ├── agent/
│   │   ├── runner.py                # Agent execution loop
│   │   ├── ollama_client.py         # Ollama, OpenAI, MLX clients
│   │   ├── workspace.py             # Git workspace management
│   │   └── mlx_runner.py            # MLX-specific runner
│   ├── collectors/
│   │   └── trajectory_store.py      # Save/load/batch-collect
│   ├── synthesizers/
│   │   ├── synthesizer.py           # LLM-based synthesis
│   │   ├── oracle_synthesizer.py    # Gold patch synthesis
│   │   └── trajectory_to_text.py    # Trajectory formatting
│   ├── trainers/
│   │   ├── lora_trainer.py          # MLX LoRA training
│   │   ├── ewc_trainer.py           # EWC-LoRA training
│   │   ├── full_replay.py           # Full replay strategy
│   │   └── data_formatter.py        # Train/valid split
│   ├── evaluators/
│   │   ├── patch_verifier.py        # FAIL_TO_PASS verification
│   │   ├── metrics.py               # Forward transfer, etc.
│   │   ├── rag.py                   # TF-IDF RAG
│   │   └── results_io.py            # Results save/load
│   ├── integrations/
│   │   ├── openclaw_parser.py       # OpenClaw JSONL parsing
│   │   ├── openclaw_provider.py     # OpenClaw config management
│   │   └── run_openclaw_experiment.py
│   ├── tasks/
│   │   └── loader.py                # SWE-Bench-CL curriculum
│   ├── serving/
│   │   └── adapter_to_ollama.py     # Adapter serving
│   └── analysis/
│       ├── results_analyzer.py
│       ├── paper_sections.py
│       └── related_work.py
├── tests/                           # 222+ tests
├── docs/                            # Research documentation
├── run_*.py                         # Experiment scripts
├── LEARNINGS.md                     # Comprehensive results
├── DEEP_AUDIT.md                    # Root cause analysis
├── OVERNIGHT_REPORT.md              # Detailed experiment logs
└── CLAUDE.md                        # Agent guidelines
```

---

## Data Flow Summary

```
CURRICULUM.JSON (50 django tasks)
       |
  LOAD + SPLIT (25 train / 25 test)
       |
       +---> RUN AGENT (base, Ollama) on all 50
       |           |
       |     TRAJECTORIES (50 JSON files)
       |           |
       |     +-----+-----+
       |     |           |
       |   TRAIN (25)  TEST (25)
       |     |           |
       |  SYNTHESIZE   VERIFY (FAIL_TO_PASS)
       |     |           |
       |  TRAINING     BASE RESULTS
       |  EXAMPLES       |
       |     |           |
       |  LoRA TRAIN     |
       |  (subprocess)   |
       |     |           |
       |  ADAPTER        |
       |     |           |
       +---> RUN AGENT (adapted, MLX) on test 25
                 |
           VERIFY (FAIL_TO_PASS)
                 |
           ADAPTED RESULTS
                 |
           FORWARD TRANSFER = adapted - base
```
