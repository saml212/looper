# Looper

An experimental research framework for adding a skill layer to AI agents — testing whether periodic LoRA consolidation of agent experience produces measurably more capable coding agents.

## Result

**LoRA skill consolidation does not work at current model scales.** After 8 experiments over 10 days (March 4-14, 2026), every LoRA training strategy produced zero or negative forward transfer. The adapted model was always the same or worse than the base model.

What *did* work was systematic improvements to the agent framework itself: prompt engineering, tool design, and loop detection tripled the resolve rate from 8% to 27%.

See [LEARNINGS.md](LEARNINGS.md) for the full story.

## The Idea

Agents today have access to unlimited knowledge but develop zero skills. Everything built for agent memory — RAG, long context, memory files — operates on the knowledge axis. Nobody is building the skill axis: a system where the agent actually gets better at operating in its environment the longer it works there.

Looper tests whether a LoRA adapter trained on agent experience — the skill layer — adds measurable value on top of the knowledge layer.

```
Context Window    -> What's happening right now
Memory/Retrieval  -> Knowledge from past sessions
Skill Adapter     -> Learned environmental fluency (LoRA)   <-- this layer
Base Model        -> General intelligence
```

## What We Found

### LoRA experiments (all negative)

| Experiment | Forward Transfer | What Happened |
|-----------|-----------------|---------------|
| Phase 1: trajectory synthesis | 0.0 | Training data was garbage Q&A from failed trajectories |
| Oracle SFT (gold patches) | -0.08 | Adapter learned diffs, agent needs tool calls — format mismatch |
| Correct-format trajectories | negative | Overfitting on 18 examples, hallucination on large files |
| Trajectory collection (65 examples) | negative | Learned "finish quickly" pattern, not debugging skill |
| MoLE (mixture of experts) | -0.10 | Merged experts identical to single adapter |
| EWC-LoRA (elastic weight consolidation) | -0.10 | Nothing worth remembering — data quality bottleneck |

### Framework fixes (all positive)

| Fix | Impact |
|-----|--------|
| Few-shot example in system prompt | Eliminated aimless exploration |
| Line-range reads (`<read>file:100-200</read>`) | Prevented context saturation |
| Context pruning | Kept conversation under token budget |
| Loop detection | Stopped re-reading same file 13+ times |
| Skip-verification prompt rule | Eliminated pip-install loops |
| Code fence stripping | Fixed 14B's 0% resolve rate (was 0%, became 27%) |
| `<edit>` tool with fuzzy matching | Targeted fixes without file truncation |

### Model scaling

| Model | Resolve Rate (with fixes) | Optimal? |
|-------|--------------------------|----------|
| 7B | 3/15 (20%) | Fast iteration |
| 14B | 4/15 (27%) | Best tradeoff |
| 32B | 4/15 (27%) | 5x slower, same result |

## Why LoRA Failed

Three compounding reasons:

1. **Not enough data.** The 7B base model resolves 8% of tasks. You can't bootstrap skill from a 92% failure rate. The minimum viable training set appears to be 100+ unique resolved tasks.

2. **Format mismatch.** Training data (Q&A pairs, diffs) never matched the inference format (multi-turn XML tool calls). Even gold-standard patches in diff format produced negative transfer because the agent needs `<bash>`, `<read>`, `<write>` — not diffs.

3. **Overfitting.** When we finally got format-matched data (18 correct-format examples), the model memorized the surface pattern (grep -> read -> write -> done in 4 steps) instead of learning generalizable debugging strategies.

The core thesis — that LoRA can encode useful agent skills — remains untested. We never had enough data in the right format to give it a fair shot.

## Documentation

- [LEARNINGS.md](LEARNINGS.md) — All results and findings
- [DEEP_AUDIT.md](DEEP_AUDIT.md) — Root cause analysis of FT=0
- [The Problem](docs/problem.md) — Why agents don't learn skills
- [Experiments](docs/experiments.md) — 10 pre-registered experiments with results
- [Architecture](docs/architecture.md) — Framework components and data flow
- [Research Landscape](docs/research_landscape.md) — Literature survey
- [Future Work](docs/future-work.md) — Where to go from here

## Technical Stack

- **Python 3.11**, Pydantic v2, 222+ tests passing
- **LoRA training:** MLX on Apple Silicon (M4 32GB)
- **Inference:** Ollama (7B/14B/32B), MLX in-process (adapted models)
- **Benchmark:** SWE-Bench-CL (273 tasks, 8 Python repos)
- **Agent protocol:** XML tool tags (`<bash>`, `<read>`, `<write>`, `<edit>`, `<done>`)

## Quick Start

```bash
python -m venv .venv --python=python3.11
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
