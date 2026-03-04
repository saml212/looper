# Looper — Agent Guidelines

## What This Project Is

Looper is an **experimental research framework** for adding a skill layer to AI agents. It tests whether periodic LoRA consolidation of agent experience produces measurably more capable agents over time.

**Read these docs before writing any code:**
1. [docs/problem.md](docs/problem.md) — The core thesis: skills vs knowledge
2. [docs/research_landscape.md](docs/research_landscape.md) — Current state of research
3. [docs/experiments.md](docs/experiments.md) — All 10 experiments with hypotheses and methodology
4. [docs/architecture.md](docs/architecture.md) — Framework components, data flow, OpenClaw integration

## Core Framing

This project adds a **new layer** to the agent stack — it does NOT replace context engineering or RAG. The skill adapter sits between the knowledge/retrieval layer and the base model:

```
Context Window    → what's happening now
Memory/Retrieval  → knowledge from past sessions
Skill Adapter     → learned environmental fluency (LoRA)
Base Model        → general intelligence
```

**Skills vs Knowledge:**
- **Skills** = how to use tools efficiently, navigate environments, recognize patterns, debug instinctively. Encoded in LoRA weights.
- **Knowledge** = facts, docs, specific details. Stays in context/RAG.

LoRA is naturally good at encoding behavioral patterns and naturally bad at memorizing facts. This is a feature, not a bug — it means the technology's strengths align with the skill layer abstraction.

## Project Structure

```
looper/
├── looper/              # Python package
│   ├── models.py        # Core Pydantic data models
│   ├── collectors/      # OpenClaw session parsing
│   ├── synthesizers/    # Trajectory → training data
│   ├── trainers/        # LoRA training strategies
│   ├── evaluators/      # Measurement framework
│   ├── experiments/     # Experiment orchestration
│   ├── serving/         # vLLM multi-LoRA serving
│   └── integrations/    # OpenClaw skill + plugin
├── tests/               # Test suite
├── configs/             # Training and experiment configs
├── docs/                # Research documentation
└── notebooks/           # Analysis and visualization
```

## Technical Stack

- **Python >= 3.11** with Pydantic v2 for data models
- **LoRA training:** PEFT, TRL, Unsloth, MLX
- **Inference:** Ollama / LM Studio (local), vLLM (cloud)
- **Evaluation:** SWE-Bench-CL benchmark (273 tasks, 8 Python repos)
- **Target agent framework:** OpenClaw

## Build Order

The framework is being built incrementally. Each step must have passing tests before the next begins.

1. ~~Project scaffold + research docs~~ (DONE)
2. Core data models (`looper/models.py`)
3. OpenClaw experience collector (`looper/collectors/`)
4. Synthetic data generator (`looper/synthesizers/`)
5. Training pipeline with pluggable strategies (`looper/trainers/`)
6. Evaluation framework (`looper/evaluators/`)
7. OpenClaw integration (`looper/integrations/`, `looper/serving/`)
8. CLI + pipeline orchestrator (`looper/pipeline.py`, `looper/cli.py`)

## Key Constraints

- **Model-agnostic:** All training code must work with any HuggingFace-compatible model via PEFT. Don't hardcode model-specific assumptions.
- **OpenClaw integration:** Sessions come from `~/.openclaw/agents/<agentId>/sessions/<id>.jsonl`. Respect OpenClaw's JSONL format, hook system, and custom provider config.
- **Research rigor:** This is an experimental framework, not a product. Every experiment has pre-registered hypotheses. Design code to support reproducible experiments with configurable parameters.
- **Four evaluation conditions:** Every experiment must compare base, base+RAG, base+LoRA, base+LoRA+RAG.
- **Pluggable strategies:** Anti-forgetting strategies (full replay, partial replay, EWC-LoRA, MoLE, adaptive rank) and synthesis formats (A-E) are pluggable modules behind abstract interfaces.
