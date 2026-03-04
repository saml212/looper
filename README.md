# Looper

An experimental research framework for adding a skill layer to AI agents — turning old context into learned environmental fluency through periodic LoRA consolidation.

## The Idea

Agents today have access to unlimited knowledge but develop zero skills. Everything the industry builds for agent memory — RAG, long context, memory files — operates on the knowledge axis. Nobody is building the skill axis: a system where the agent actually gets better at operating in its environment the longer it works there.

**Looper adds a new layer to the agent stack.** As an agent operates — using tools, navigating code, deploying, debugging — its experience is periodically consolidated into a LoRA adapter. Old context gets turned into trained skills. The agent doesn't just have better notes; it develops the kind of environmental fluency that a human employee builds over weeks on the job.

```
┌─────────────────────────┐
│     Context Window      │  ← What's happening right now
├─────────────────────────┤
│  Memory / Retrieval     │  ← Knowledge from past sessions
├─────────────────────────┤
│     Skill Adapter       │  ← Learned environmental fluency (LoRA)
├─────────────────────────┤
│     Base Model          │  ← General intelligence
└─────────────────────────┘
```

This doesn't replace anything. Context still handles the present. Retrieval still handles knowledge. The base model still provides general intelligence. The skill adapter adds something new: the ability to get better over time.

## What This Framework Does

Looper provides tooling to rigorously test whether skill consolidation actually works:

1. **Experience Collection** — Parses [OpenClaw](https://openclaw.ai/) agent session transcripts into structured trajectories
2. **Synthetic Data Generation** — Extracts environmental skills from trajectories using multiple synthesis strategies (5 formats to compare)
3. **Continual LoRA Training** — Model-agnostic training with 5 pluggable anti-forgetting strategies
4. **Evaluation Framework** — Measures environmental fluency, skill retention, general capability preservation
5. **Experiment Runner** — Config-driven orchestration for 10 pre-registered experiments

## The Experiments

| # | Experiment | Question |
|---|-----------|----------|
| 1 | Full Replay Baseline | What's the upper bound on skill retention? |
| 2 | Partial Replay | Can a fixed buffer achieve 80%+ of full replay? |
| 3 | Mixture of LoRA Experts | Does separating skill types reduce forgetting? |
| 4 | EWC-LoRA | Does Fisher information penalty help in LoRA's low-rank space? |
| 5 | Adaptive Rank | Does dynamic rank allocation + SVD consolidation beat fixed rank? |
| 6 | Synthesis Format | Which data format best captures environmental skills? |
| 7 | Synthesis Budget | How many training pairs per session trajectory? |
| 8 | Self-Synthesis | Can the agent generate its own training data? |
| 9 | Skill + Knowledge Ablation | Does the skill layer add value on top of good retrieval? |
| 10 | Staleness | How do learned skills degrade when the environment changes? |

See [docs/experiments.md](docs/experiments.md) for full hypotheses, methodology, and success criteria.

## Documentation

- [The Problem](docs/problem.md) — Why agents don't learn skills, and what a skill layer would look like
- [Research Landscape](docs/research_landscape.md) — Survey of LoRA, catastrophic forgetting, agent memory, and related work
- [Experiments](docs/experiments.md) — All 10 experiments with pre-registered hypotheses
- [Architecture](docs/architecture.md) — Framework components, data flow, OpenClaw integration

## Status

**Phase: Foundation** — Documentation and core data models.

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/
```

## Project Structure

```
looper/
├── docs/                  # Research documentation
├── looper/
│   ├── models.py          # Core data models
│   ├── collectors/        # OpenClaw session parsing
│   ├── synthesizers/      # Trajectory → training data
│   ├── trainers/          # LoRA training strategies
│   ├── evaluators/        # Measurement framework
│   ├── experiments/       # Experiment orchestration
│   ├── serving/           # vLLM multi-LoRA serving
│   └── integrations/      # OpenClaw skill + plugin
├── tests/                 # Test suite
├── configs/               # Training and experiment configs
└── notebooks/             # Analysis and visualization
```

## Key Papers

- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) — The foundation
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (Biderman et al., 2024) — The fundamental tradeoff
- [S-LoRA](https://arxiv.org/abs/2311.03285) (Sheng et al., 2024) — Multi-tenant serving
- [EWC](https://arxiv.org/abs/1612.00796) (Kirkpatrick et al., 2017) — Classic anti-forgetting
- [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023) — Context-based agent memory (baseline)

## License

MIT
