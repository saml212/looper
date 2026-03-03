# RECALL: Research on Experience-Consolidated Adaptive LoRA Learning

An experimental research framework for testing whether periodic LoRA consolidation of agent experience improves environmental fluency for AI agents.

## The Thesis

AI agents today are stuck on Day 1. Every session, they start from scratch — no memory of what worked, what failed, or how the user's environment is structured. The industry's current answer is **context engineering**: stuffing retrieved memories and documents into the prompt. It works, but it doesn't let the agent build genuine competence.

**RECALL tests a different approach:** periodically consolidate an agent's experience into LoRA adapter weights, encoding procedural knowledge — tool usage patterns, debugging strategies, codebase conventions, deployment workflows — directly into the model. The agent shouldn't just have better notes; it should actually get better at its job over time.

This is not about replacing context. Context handles what's happening *right now*. LoRA handles how the agent *operates* — the kind of environmental fluency that a human employee builds over weeks and months on the job.

## What This Framework Does

RECALL provides the tooling to rigorously test this hypothesis:

1. **Experience Collection** — Parses [OpenClaw](https://openclaw.ai/) agent session transcripts into structured trajectories
2. **Synthetic Data Generation** — Converts trajectories into LoRA training data using multiple synthesis strategies (5 formats to compare)
3. **Continual LoRA Training** — Model-agnostic training with 5 pluggable anti-forgetting strategies (full replay, partial replay, EWC-LoRA, Mixture of LoRA Experts, adaptive rank allocation)
4. **Evaluation Framework** — Measures environmental fluency, knowledge retention, general capability preservation, and comparison against RAG/context baselines
5. **Experiment Runner** — Config-driven orchestration for 10 pre-registered experiments

## The Experiments

| # | Experiment | Question |
|---|-----------|----------|
| 1 | Full Replay Baseline | What's the upper bound on retention? |
| 2 | Partial Replay | Can a fixed buffer achieve 80%+ of full replay? |
| 3 | Mixture of LoRA Experts | Does separating knowledge types reduce forgetting? |
| 4 | EWC-LoRA | Does Fisher information penalty help in LoRA's low-rank space? |
| 5 | Adaptive Rank | Does dynamic rank allocation + SVD consolidation beat fixed rank? |
| 6 | Synthesis Format | Which data format best captures environmental knowledge? |
| 7 | Synthesis Budget | How many training pairs per session trajectory? |
| 8 | Self-Synthesis | Can the agent generate its own training data? |
| 9 | Hybrid Ablation | Does LoRA + RAG beat either alone? |
| 10 | Staleness | How does adapter quality degrade when the environment changes? |

See [docs/experiments.md](docs/experiments.md) for full hypotheses, methodology, and success criteria.

## The Honest Framing

If the LoRA approach can't beat a well-implemented RAG system on any important axis — task success rate, steps to completion, context efficiency, inference latency — then the whole approach doesn't justify its complexity. **That's a legitimate finding worth publishing.** This framework is designed to produce rigorous empirical answers, not optimistic demos. Negative results are documented alongside positive ones.

## Documentation

- [The Problem](docs/problem.md) — Why agent memory matters, the competence vs. recall distinction
- [Research Landscape](docs/research_landscape.md) — Survey of LoRA, catastrophic forgetting, context engineering, and related work
- [Experiments](docs/experiments.md) — All 10 experiments with pre-registered hypotheses
- [Architecture](docs/architecture.md) — Framework components, data flow, OpenClaw integration

## Status

**Phase: Foundation** — Documentation and core data models. See the build plan for current progress.

## Quick Start

```bash
# Install core dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# (Coming soon) Collect experience from OpenClaw sessions
# recall collect --agent-id <agent_id> --output trajectories/

# (Coming soon) Synthesize training data
# recall synthesize --input trajectories/ --format chain_of_thought

# (Coming soon) Train LoRA adapter
# recall train --strategy full_replay --config configs/training_default.yaml

# (Coming soon) Run an experiment
# recall experiment --config configs/experiments/full_replay_baseline.yaml
```

## Project Structure

```
recall/
├── docs/                  # Research documentation
├── recall/
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
├── notebooks/             # Analysis and visualization
└── paper/                 # LaTeX source
```

## Key Papers

- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) — The foundation
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (Biderman et al., 2024) — The fundamental tradeoff
- [S-LoRA](https://arxiv.org/abs/2311.03285) (Sheng et al., 2024) — Multi-tenant serving
- [EWC](https://arxiv.org/abs/1612.00796) (Kirkpatrick et al., 2017) — Classic anti-forgetting
- [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023) — Context-based agent memory (our baseline)

## License

MIT
