# Framework Architecture

## System Overview

Looper is structured as a pipeline with five components. The first (agent loop) and last (serving) use existing infrastructure. The middle three (collection, synthesis, training) are where the framework provides research tooling, and the evaluation layer wraps around everything.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE LOOPER PIPELINE                                  │
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

Looper adds a skill layer to OpenClaw agents. We do not build our own agent — we instrument OpenClaw's existing agent loop and add the consolidation pipeline on top.

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
      "looper-adapted": {
        "baseUrl": "http://localhost:8080/v1",
        "api": "openai-completions",
        "models": [{"id": "looper-lora", "name": "Looper Skill-Adapted Model"}]
      }
    }
  }
}
```

This is how we serve the skill-adapted model back to OpenClaw — as a custom provider pointing to a local vLLM instance with the skill adapter loaded.

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
          │  OpenClaw Parser │  looper/collectors/openclaw_parser.py
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ AgentTrajectory  │  looper/models.py
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

Converts trajectories into skill training data. This is where data quality research happens — extracting the right skills from noisy experience.

### Data Flow

```
AgentTrajectory
       │
       ▼
┌──────────────────┐
│   Synthesizer    │  looper/synthesizers/
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

### Environmental Skill Extraction

The synthesis prompt targets four categories of learnable skills:

1. **Tool usage skills** — How to use the right tools efficiently in this environment
2. **Error recovery skills** — Recognizing failure patterns and applying the correct fix
3. **Convention skills** — Matching project norms for naming, structure, style
4. **Workflow skills** — Navigating deployment, testing, review, and debugging flows

### Deduplication

Before adding new pairs to the training set, we compute embedding similarity against existing pairs and reject duplicates above a threshold. This prevents the dataset from being dominated by frequently-encountered patterns.

---

## Component 4: Continual Training

Model-agnostic LoRA training that consolidates extracted skills into the adapter, with pluggable anti-forgetting strategies.

### Data Flow

```
SynthesizedPair[]
       │
       ▼
┌──────────────────┐
│  Data Formatter  │  looper/trainers/data_formatter.py
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
| MoLE | Separate adapters per skill type |
| Adaptive Rank | Dynamic rank + SVD compression of old sessions |

### Target Models

The framework is model-agnostic — any HuggingFace-compatible model that works with PEFT can be used. For our experiments, we target four models chosen for an M4 Mac Mini with 32GB unified memory:

| Model | Params | Quantized Size | Role |
|-------|--------|----------------|------|
| **Qwen 2.5 Coder 32B** | 32B | ~18GB (Q4_K_M) | Primary coding experiments — strongest open coding model at this size |
| **DeepSeek-R1-Distill-Qwen 32B** | 32B | ~18GB (Q4_K_M) | Reasoning/debugging experiments — distilled from DeepSeek R1 |
| **Command R 35B** | 35B | ~20GB (Q4_K_M) | Tool-calling experiments — built specifically for RAG and tool use |
| **Qwen 2.5 Coder 7B** | 7B | ~4GB (Q4) / ~14GB (FP16) | Fast iteration — runs at full precision, near-instant inference |

The 32B models are the "Goldilocks zone" for 32GB unified memory: ~18-20GB for the model leaves room for the OS, Ollama/LM Studio, and the training pipeline. The 7B model runs fast enough for rapid prototyping and smoke tests.

### Model Configuration

```yaml
base_model: "Qwen/Qwen2.5-Coder-32B-Instruct"  # or any HF-compatible model
quantization: "Q4_K_M"                            # GGUF quantization level
max_seq_length: 4096
lora:
  r: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_alpha: 32
  lora_dropout: 0.05
```

### Local Inference Stack

For experiments, we run models locally via Ollama or LM Studio, which expose an OpenAI-compatible API. OpenClaw connects to this as a custom provider.

```
OpenClaw Agent
       │
       │  HTTP (OpenAI-compatible API)
       ▼
┌──────────────────────────┐
│   Ollama / LM Studio     │
│  ┌────────────────────┐  │
│  │ Base Model (GGUF)   │  │  Loaded once
│  │ + Skill Adapter     │  │  LoRA merged or applied at load time
│  └────────────────────┘  │
└──────────────────────────┘
```

For multi-LoRA serving at scale (multiple adapters hot-swapped per request), vLLM on a cloud GPU is the path. But for local experimentation, Ollama with a single merged adapter is simpler and sufficient.

### Adapter Registry

Maps agent IDs and project contexts to skill adapter paths:

```python
class AdapterRegistry:
    def register(self, agent_id: str, project: str, adapter_path: Path) -> None
    def get_adapter(self, agent_id: str, project: str) -> Path | None
    def list_adapters(self) -> list[AdapterInfo]
```

### Local vs. Cloud Strategy

| Task | Local (M4 32GB) | Cloud (GCP spot L4/A100) |
|------|-----------------|--------------------------|
| Inference / agent sessions | Ollama + GGUF models | vLLM with multi-LoRA |
| LoRA training (7B) | Feasible with MLX or Unsloth | Fast on L4 spot (~$0.70/hr) |
| LoRA training (32B) | Tight but possible with QLoRA + MLX | Needs A100 40GB |
| Synthesis (data gen) | Local 32B model or API (Claude/GPT) | Same |
| Evaluation runs | Local for small suites | Cloud for full benchmark sweeps |

For rapid iteration, run everything locally. Push to cloud only for full experiment sweeps or 32B QLoRA training if local is too slow.

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

**Skill Retention** — Does the adapter retain skills from past sessions? Measured by testing accuracy on held-out pairs from each session, plotted as a retention curve over time.

**Environmental Fluency** — Does the skill layer make the agent better at its job? Measured by: fewer steps to task completion, fewer errors/retries, better first-attempt tool selection, more efficient navigation. This is the primary metric.

**Capability Preservation** — Does the skill layer hurt general performance? Measured by standard benchmarks (HumanEval, MBPP, MMLU) before and after adapter application. Acceptable threshold: < 2% degradation.

**Baselines** — Same evaluation against: no skill layer, knowledge-only (RAG). The skill layer must add measurable value on top of what good knowledge retrieval already provides.

### Experiment Runner

Config-driven experiment orchestration:

```yaml
experiment:
  name: "full_replay_baseline"
  id: "exp_001"
  strategy: "full_replay"
  base_model: "Qwen/Qwen2.5-Coder-7B-Instruct"  # fast iteration model
  # base_model: "Qwen/Qwen2.5-Coder-32B-Instruct"  # full-scale model
  sessions: 50
  consolidation_interval: 5
  lora:
    r: 16
    alpha: 32
  evaluation:
    retention: true
    fluency: true
    capability: true
    baselines: ["rag", "knowledge_only"]
  seeds: [42, 123, 456]  # Multiple runs for significance
```

Results are logged to structured JSON and optionally to Weights & Biases.

---

## Directory Structure

```
looper/
├── pyproject.toml
├── README.md
├── docs/
│   ├── problem.md                    # Problem statement
│   ├── research_landscape.md         # Literature survey
│   ├── experiments.md                # Experiment definitions
│   └── architecture.md              # This document
├── looper/
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

| Integration Point | OpenClaw Feature | Looper Component |
|-------------------|------------------|-------------------|
| Session data ingestion | Session JSONL files | Experience Collector (batch) |
| Real-time collection | Hook system (postToolExecution, postResponse) | OpenClaw Plugin (hook mode) |
| Telemetry augmentation | Telemetry JSONL | Experience Collector (supplementary) |
| Model serving | Custom provider config | vLLM + Adapter Registry |
| User interface | Skill system (SKILL.md) | OpenClaw Skill |
| Memory complement | MEMORY.md + memory search | RAG baseline for comparison |
