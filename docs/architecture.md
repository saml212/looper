# Framework Architecture

## System Overview

RECALL is structured as a pipeline with five components. The first (agent loop) and last (serving) use existing infrastructure. The middle three (collection, synthesis, training) are where the framework provides research tooling, and the evaluation layer wraps around everything.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          THE RECALL PIPELINE                                │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │   OPENCLAW    │──▸│  EXPERIENCE  │──▸│  SYNTHETIC   │──▸│   CONTINUAL │ │
│  │   AGENT       │   │  COLLECTOR   │   │  DATA GEN    │   │   TRAINING  │ │
│  │  (existing)   │   │  (build)     │   │  (research)  │   │  (research) │ │
│  └──────────────┘   └──────────────┘   └──────┬───────┘   └──────┬──────┘ │
│         ▲                                      │                  │        │
│         │            ┌──────────────┐          │                  │        │
│         └────────────│  MULTI-LORA  │◂─────────┘──────────────────┘        │
│                      │  SERVING     │                                      │
│                      │  (existing)  │                                      │
│                      └──────────────┘                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     EVALUATION FRAMEWORK                            │   │
│  │  Retention │ Fluency │ Capability │ Baselines │ Experiment Runner   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: OpenClaw Agent (Existing)

RECALL integrates with OpenClaw as the agent framework. We do not build our own agent — we instrument OpenClaw's existing agent loop.

### OpenClaw Session Format

Sessions are stored at `~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl`. Each line is a JSON object:

```jsonl
{"type": "session", "timestamp": "2026-03-01T12:00:00Z", "provider": "telegram", "from": "user123"}
{"type": "message", "timestamp": "...", "message": {"role": "user", "content": [{"type": "text", "text": "..."}]}}
{"type": "message", "timestamp": "...", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_abc", "name": "exec", "input": {"command": "ls"}}]}}
{"type": "message", "timestamp": "...", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "file1\nfile2"}]}}
{"type": "message", "timestamp": "...", "message": {"role": "assistant", "content": [{"type": "text", "text": "Here are your files..."}]}}
```

### OpenClaw Hook System

OpenClaw provides lifecycle hooks for real-time interception:

- `preToolExecution` — Before a tool runs (can modify or block)
- `postToolExecution` — After a tool completes (observation only)
- `postResponse` — After the agent sends a response
- `postCompaction` — After context window compaction

We use `postToolExecution` and `postResponse` hooks for real-time experience collection.

### OpenClaw Custom Providers

OpenClaw supports custom LLM providers via OpenAI-compatible or Anthropic-compatible API endpoints:

```json5
{
  "models": {
    "providers": {
      "recall-adapted": {
        "baseUrl": "http://localhost:8080/v1",
        "api": "openai-completions",
        "models": [{"id": "recall-lora", "name": "RECALL LoRA-Adapted Model"}]
      }
    }
  }
}
```

This is how we serve the LoRA-adapted model back to OpenClaw — as a custom provider pointing to a local vLLM instance.

### OpenClaw Telemetry

The telemetry plugin writes structured events to `~/.openclaw/logs/telemetry.jsonl`:

```json
{"type": "tool.start", "toolName": "exec", "params": {"cmd": "ls"}, "sessionKey": "...", "ts": 1738517700000}
{"type": "tool.end", "toolName": "exec", "duration": 150, "success": true, "sessionKey": "...", "ts": 1738517700150}
{"type": "llm.usage", "model": "claude-sonnet-4-6", "inputTokens": 5000, "outputTokens": 500, "cost": 0.02}
```

We can use this as a supplementary data source for trajectory analysis.

---

## Component 2: Experience Collector

Transforms raw OpenClaw session data into structured trajectories.

### Data Flow

```
~/.openclaw/agents/<agentId>/sessions/<id>.jsonl
                    │
                    ▼
          ┌─────────────────┐
          │  OpenClaw Parser │  recall/collectors/openclaw_parser.py
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ AgentTrajectory  │  recall/models.py
          │ ┌─ SessionMeta  │
          │ ├─ AgentStep[]   │
          │ │  ├─ reasoning  │
          │ │  ├─ tool_call  │
          │ │  └─ result     │
          │ ├─ outcome       │
          │ └─ stats         │
          └─────────────────┘
```

### Two Collection Modes

**Batch mode** (primary for experiments): Read completed session JSONL files from disk. Enumerate available sessions via `sessions.json` index. Parse each into an `AgentTrajectory`.

**Hook mode** (for live deployment): Register an OpenClaw plugin that intercepts `postToolExecution` and `postResponse` hooks, streams events into the collection pipeline in real time.

### Session Store

```python
class SessionStore:
    """Discovers and enumerates OpenClaw sessions."""

    def __init__(self, openclaw_dir: Path = Path.home() / ".openclaw")
    def list_agents(self) -> list[str]
    def list_sessions(self, agent_id: str) -> list[SessionInfo]
    def load_session(self, agent_id: str, session_id: str) -> AgentTrajectory
    def load_all_sessions(self, agent_id: str) -> list[AgentTrajectory]
```

---

## Component 3: Synthetic Data Generator

Converts trajectories into LoRA training data. This is where data quality research happens.

### Data Flow

```
AgentTrajectory
       │
       ▼
┌──────────────────┐
│   Synthesizer    │  recall/synthesizers/
│  ┌─ Prompts (A-E)│  Multiple synthesis formats
│  ├─ LLM API call │  Anthropic / OpenAI / self-model
│  └─ Validation   │  Confidence scoring, dedup
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ SynthesizedPair[]│
│ ┌─ instruction   │
│ ├─ response      │
│ ├─ type          │  (tool_usage | error_recovery | convention | workflow)
│ ├─ confidence    │  0.0 - 1.0
│ └─ source_session│
└──────────────────┘
```

### Pluggable Synthesis Strategies

The synthesizer is an abstract interface. Each synthesis format (A through E from Experiment 6) is a concrete implementation:

```python
class Synthesizer(ABC):
    @abstractmethod
    def synthesize(self, trajectory: AgentTrajectory) -> list[SynthesizedPair]: ...
```

### Environmental Knowledge Focus

The synthesis prompt specifically targets four categories of environmental knowledge:

1. **Tool usage patterns** — How tools are used in this specific environment
2. **Error recovery strategies** — What goes wrong and how to fix it
3. **Code/project conventions** — Naming, structure, style, patterns
4. **Workflow patterns** — Deployment, testing, review, debugging flows

### Deduplication

Before adding new pairs to the training set, we compute embedding similarity against existing pairs and reject duplicates above a threshold. This prevents the dataset from being dominated by frequently-encountered patterns.

---

## Component 4: Continual Training

Model-agnostic LoRA training with pluggable anti-forgetting strategies.

### Data Flow

```
SynthesizedPair[]
       │
       ▼
┌──────────────────┐
│  Data Formatter  │  recall/trainers/data_formatter.py
│  Chat template   │  Model-agnostic tokenizer handling
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│              Training Strategy               │
│  ┌─────────────┐ ┌───────────┐ ┌──────────┐│
│  │ Full Replay  │ │ Partial   │ │ EWC-LoRA ││
│  │             │ │ Replay    │ │          ││
│  └─────────────┘ └───────────┘ └──────────┘│
│  ┌─────────────┐ ┌───────────┐             │
│  │    MoLE     │ │ Adaptive  │             │
│  │             │ │ Rank      │             │
│  └─────────────┘ └───────────┘             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  LoRA Adapter   │  ./adapters/<agent_id>/<session_range>/
              │  (~30-50MB)     │
              └─────────────────┘
```

### Trainer Interface

```python
class Trainer(ABC):
    @abstractmethod
    def train(self, new_data: list[TrainingExample], session_id: str) -> Path:
        """Train/update the adapter. Returns path to saved adapter."""
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Return training metrics (loss, etc.)."""
        ...
```

Each anti-forgetting strategy implements this interface with its own update logic:

| Strategy | How it handles old knowledge |
|----------|------------------------------|
| Full Replay | Retrains on everything every time |
| Partial Replay | Fixed buffer with priority eviction |
| EWC-LoRA | Fisher information penalty on important params |
| MoLE | Separate adapters per knowledge type |
| Adaptive Rank | Dynamic rank + SVD compression of old sessions |

### Model Agnostic Design

The framework doesn't assume a specific base model. Configuration specifies:

```yaml
base_model: "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
max_seq_length: 4096
load_in_4bit: true
lora:
  r: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_alpha: 32
  lora_dropout: 0.05
```

Any HuggingFace-compatible model that works with PEFT/Unsloth can be used.

---

## Component 5: Multi-LoRA Serving (Existing)

We use vLLM for self-hosted serving with multi-LoRA support.

### Serving Architecture

```
OpenClaw Agent
       │
       │  HTTP (OpenAI-compatible API)
       ▼
┌──────────────────┐
│      vLLM        │
│  ┌────────────┐  │
│  │ Base Model  │  │  Loaded once, shared across all requests
│  │ (frozen)    │  │
│  └────────────┘  │
│  ┌────────────┐  │
│  │ Adapter     │  │  Hot-swapped per request
│  │ Registry    │  │  based on agent_id/project
│  └────────────┘  │
└──────────────────┘
```

### Adapter Registry

Maps agent IDs and project contexts to LoRA adapter paths:

```python
class AdapterRegistry:
    def register(self, agent_id: str, project: str, adapter_path: Path) -> None
    def get_adapter(self, agent_id: str, project: str) -> Path | None
    def list_adapters(self) -> list[AdapterInfo]
```

vLLM is started with `--enable-lora` and adapters are specified via the `lora_modules` parameter or loaded dynamically.

---

## Evaluation Layer

Wraps around the entire pipeline, providing measurement at every stage.

### Evaluation Axes

```
┌───────────────────────────────────────────────────────────────────┐
│                      EVALUATION FRAMEWORK                         │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐             │
│  │  Retention   │  │  Fluency    │  │  Capability  │             │
│  │  (does it    │  │  (is it     │  │  (can it     │             │
│  │  remember?)  │  │  better?)   │  │  still code?)│             │
│  └─────────────┘  └─────────────┘  └──────────────┘             │
│                                                                   │
│  ┌─────────────┐  ┌──────────────────────────────────┐          │
│  │  Baselines  │  │  Experiment Runner               │          │
│  │  (RAG, ctx) │  │  Config-driven, reproducible     │          │
│  └─────────────┘  └──────────────────────────────────┘          │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Retention** — Does the adapter retain knowledge from past sessions? Measured by testing accuracy on held-out pairs from each session, plotted as a retention curve over time.

**Environmental Fluency** — Does the adapter make the agent better? Measured by: fewer steps to task completion, fewer errors/retries, less context consumed for environmental setup, better tool selection. This is the primary metric.

**Capability Preservation** — Does the adapter hurt general performance? Measured by standard benchmarks (HumanEval, MBPP, MMLU) before and after adapter application. Acceptable threshold: < 2% degradation.

**Baselines** — Same evaluation run against: no-memory, RAG-only, context-stuffing. The LoRA approach must outperform at least one of these on at least one important axis.

### Experiment Runner

Config-driven experiment orchestration:

```yaml
experiment:
  name: "full_replay_baseline"
  id: "exp_001"
  strategy: "full_replay"
  base_model: "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
  sessions: 50
  consolidation_interval: 5
  lora:
    r: 16
    alpha: 32
  evaluation:
    retention: true
    fluency: true
    capability: true
    baselines: ["rag", "context"]
  seeds: [42, 123, 456]  # Multiple runs for significance
```

Results are logged to structured JSON and optionally to Weights & Biases.

---

## Directory Structure

```
recall/
├── pyproject.toml
├── README.md
├── docs/
│   ├── problem.md                    # Problem statement
│   ├── research_landscape.md         # Literature survey
│   ├── experiments.md                # Experiment definitions
│   └── architecture.md              # This document
├── recall/
│   ├── __init__.py
│   ├── models.py                    # Core data models
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── openclaw_parser.py       # OpenClaw JSONL → AgentTrajectory
│   │   └── session_store.py         # Session discovery and enumeration
│   ├── synthesizers/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract Synthesizer interface
│   │   ├── llm_synthesizer.py       # LLM-based synthesis
│   │   ├── prompts.py              # Synthesis prompt templates
│   │   └── deduplicator.py          # Embedding-based deduplication
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract Trainer interface
│   │   ├── lora_trainer.py          # Basic SFT LoRA training
│   │   ├── full_replay.py           # Full replay strategy
│   │   ├── partial_replay.py        # Prioritized replay buffer
│   │   ├── ewc_lora.py             # EWC regularization
│   │   ├── mole.py                  # Mixture of LoRA Experts
│   │   ├── adaptive_rank.py         # Dynamic rank allocation
│   │   └── data_formatter.py        # Chat template formatting
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── retention.py             # Memory retention scoring
│   │   ├── fluency.py              # Environmental fluency benchmark
│   │   ├── capability.py            # General capability preservation
│   │   ├── baselines.py            # RAG and context-stuffing baselines
│   │   └── metrics.py              # Shared metric functions
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── runner.py                # Experiment orchestration
│   │   └── configs/                 # YAML experiment configs
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── vllm_launcher.py         # vLLM multi-LoRA setup
│   │   └── adapter_registry.py      # Adapter tracking
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── openclaw_skill/
│   │   │   └── SKILL.md
│   │   ├── openclaw_plugin.ts       # Real-time hook plugin
│   │   └── openclaw_provider.py     # Provider config generator
│   ├── pipeline.py                  # End-to-end pipeline
│   └── cli.py                       # CLI interface
├── tests/
│   ├── __init__.py
│   ├── fixtures/                    # Mock data for tests
│   ├── test_models.py
│   ├── test_openclaw_parser.py
│   ├── test_session_store.py
│   ├── test_llm_synthesizer.py
│   ├── test_deduplicator.py
│   ├── test_replay_buffer.py
│   ├── test_data_formatter.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── configs/
│   ├── training_default.yaml
│   └── experiments/
├── notebooks/                       # Analysis and visualization
└── paper/                           # LaTeX source (eventually)
```

---

## Integration Points Summary

| Integration Point | OpenClaw Feature | RECALL Component |
|-------------------|------------------|-------------------|
| Session data ingestion | Session JSONL files | Experience Collector (batch) |
| Real-time collection | Hook system (postToolExecution, postResponse) | OpenClaw Plugin (hook mode) |
| Telemetry augmentation | Telemetry JSONL | Experience Collector (supplementary) |
| Model serving | Custom provider config | vLLM + Adapter Registry |
| User interface | Skill system (SKILL.md) | OpenClaw Skill |
| Memory complement | MEMORY.md + memory search | RAG baseline for comparison |
