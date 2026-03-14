# Looper: What We've Learned

**Date range:** 2026-03-04 to 2026-03-14
**Core question:** Can periodic LoRA consolidation of agent experience produce measurably more capable agents?
**Answer so far:** No. Every LoRA training strategy we tried either had zero effect or made the model worse.

---

## The Numbers

### Every experiment, one table

| # | Experiment | Date | FT | Base Rate | Adapted Rate | Verdict |
|---|-----------|------|-----|-----------|--------------|---------|
| 1 | Phase 1: trajectory synthesis LoRA (7B) | 03-05 | 0.0 | 2/25 (8%) | 2/25 (8%) | No effect |
| 2 | Phase 1: +RAG | 03-05 | 0.0 | 2/25 (8%) | 2/25 (8%) | No effect |
| 3 | MoLE (4 configs, 7B) | 03-11 | -0.10 | 1/10 (10%) | 0/10 (0%) | Regression |
| 4 | EWC-LoRA (3 lambdas, 7B) | 03-11 | -0.10 | 1/10 (10%) | 0/10 (0%) | Regression |
| 5 | Oracle SFT (gold patches, 7B) | 03-09 | -0.08 | 2/25 (8%) | 0/25 (0%) | Regression |
| 6 | Correct-format LoRA (XML trajectories, 14B) | 03-12 | — | 7/7 (100%) | 5/7 (71%) | Regression |
| 7 | Trajectory collection LoRA (12 tasks, 14B) | 03-13 | — | 5/50 (10%) | 1/50 (2%) | Regression |
| 8 | Self-play 14B (pilot) | 03-13 | — | 2/25 (8%) | not run | Stopped (too few resolved) |

**Forward transfer is zero or negative in every single experiment.** LoRA never helped. It always either did nothing or actively damaged the model.

### Scaling (without LoRA, with framework fixes)

| Model | No Fixes | With Fixes | Patch Rate | Runtime (15 tasks) |
|-------|----------|-----------|------------|-------------------|
| 7B | 2/25 (8%) | 3/15 (20%) | 87% | ~30 min |
| 14B | 0/25 (0%) | 4/15 (27%) | 100% | ~45 min |
| 32B | 0/3 (0%) | 4/15 (27%) | 87% | ~217 min |

14B is the sweet spot. 32B ties 14B but takes 5x longer and OOMs. The "inverse scaling" we saw initially (7B > 14B > 32B) was entirely a framework artifact — code fence corruption and read-loop pathology, not a model problem.

Union across all three models: 5/15 (33%). Each model resolves at least one task the others can't.

---

## The Five Bugs That Wasted the Most Time

### 1. Workspace contamination (discovered 03-09)

`create_workspace()` returned dirty workspaces without resetting. When conditions ran sequentially (Base, then RAG, then LoRA), later conditions inherited the previous condition's patches. The "all conditions resolve the same 2 tasks" result was an artifact — the LoRA and RAG conditions never actually ran on clean state.

**Cost:** Invalidated the entire Phase 1 cross-condition comparison. We thought FT=0 meant "LoRA doesn't help." In reality we never actually tested LoRA in isolation.

### 2. Loss logging bug (discovered 03-05)

`_MetricsCallback` checked for `train_info["loss"]` but MLX sends `train_info["train_loss"]`. Every experiment reported `train_loss=0.0`. We thought the model was trivially memorizing all training data. In reality, training was working fine (loss ~1.0), we just couldn't see it.

**Cost:** Wasted time investigating "memorization" that wasn't happening. All synthesis experiments (format, budget) were evaluated on a broken metric.

### 3. Code fence corruption (discovered 03-12)

14B wraps all `<write>` content in `` ```python ... ``` ``. This silently corrupts every source file. We thought 14B was worse than 7B (inverse scaling). In reality, 14B was generating correct fixes but the fences made them invalid.

**Cost:** 14B was dismissed as worse than 7B for a week. The fix (`_strip_code_fences()`) immediately took 14B from 0% to 50% on a pilot.

### 4. Read-loop pathology (discovered 03-11)

32B re-reads the same file 13+ times without making changes. The model gets stuck in a loop because the context grows until it can't reason about what to do next. This made 32B appear worse than 7B.

**Cost:** 32B was dismissed. Fix (loop detection + context pruning + few-shot example) restored normal scaling.

### 5. Verification-loop pathology (discovered 03-11)

After writing a fix, the model spends 10+ steps trying to `pip install` dependencies and run tests. It never calls `<done>`, so patches exist in the workspace but get marked as `max_steps` failures.

**Cost:** Deflated resolve rates. Fix (system prompt rule: "don't verify, just call done") doubled effective patch rate.

---

## What Actually Improved Performance

Only two things moved the resolve rate: framework fixes and model scaling. LoRA never did.

### Framework fixes: 8% -> 27%

Four changes to `runner.py` collectively more than tripled the resolve rate:

1. **Line-range reads** (`<read>file.py:100-200</read>`) — prevents context saturation from reading entire large files
2. **Context pruning** (`prune_messages()`) — replaces old tool results with summaries to stay under token budget
3. **Few-shot example** — a complete worked example in the system prompt showing grep → read → fix → done
4. **Loop detection** — warns the model after 3 reads of the same file without a write

Plus two follow-ups:
5. **"Skip verification" prompt rule** — eliminates pip-install loops
6. **Code fence stripping** — fixes 14B's markdown wrapping

Each fix addresses a specific pathology. Together they transform a model that wanders aimlessly into one that follows a clean grep → read → fix → done pattern.

### Model scaling: 7B -> 14B

14B is a strict superset of 7B on the 15-task overlap. It resolves everything 7B resolves plus django-11119 (template autoescape). 32B ties 14B but resolves different tasks — 32B uniquely solves django-11603 (aggregation pipeline, requires deeper reasoning) but regresses on django-11066 (generates wrong fix).

14B is the practical ceiling on 32GB Apple Silicon. The 19GB 32B model leaves too little headroom for KV cache, causing OOM on task 4 and 15-min write steps.

---

## What We Tried With LoRA and Why It Failed

### Attempt 1: Self-distilled Q&A (Phase 1)
**Data:** 86 Q&A pairs synthesized by the same 7B model from its own (failed) trajectories.
**Failure:** The data is trivial ("How do you read a file?"), contains no tool-call structure, and comes from 92% failed sessions. The adapter shifted the output distribution toward degenerate repetition.

### Attempt 2: Oracle SFT (gold patches)
**Data:** 20 correct SWE-Bench patches formatted as "bug report → unified diff."
**Failure:** The adapter learned the mapping faithfully — it outputs raw diffs. But the agent loop expects XML tool calls. Every step produces a hallucinated diff with zero parseable tool calls. Even perfect content in the wrong format = negative transfer.

### Attempt 3: Correct-format trajectories (14B)
**Data:** 18 per-step examples from 4 resolved trajectories, with XML tool calls in the right format.
**Failure:** Severe overfitting (train_loss 0.005, val_loss 0.419 on 18 examples). The model learned to rigidly follow the exact grep → read → write → done pattern from training, including hallucinating entire file contents for large files. Patch rate regressed from 100% to 71%.

### Attempt 4: Scaled trajectory collection (14B)
**Data:** 65 examples from 31 resolved trajectories (12 unique tasks) across multi-attempt collection at temp=0.7.
**Failure:** Model learned "finish quickly" — 32% of tasks call `<done>` in ≤3 steps with no patch. LoRA encoded the surface completion pattern, not the debugging skill. Resolve rate dropped from 10% to 2%.

### Attempt 5: MoLE (mixture of experts)
**Data:** Same trajectory data split into 3 experts (search/read/modify).
**Failure:** Merged experts perform identically to single adapter. "Successful-only" filtering produces models that mimic the structure of success (quick completion) without the substance.

### Attempt 6: EWC-LoRA (continual learning)
**Data:** 169 examples across 5 sequential batches with Fisher Information regularization.
**Failure:** EWC prevents forgetting, but there's nothing worth remembering. The bottleneck is data quality, not catastrophic forgetting.

### Why it all fails

Every attempt runs into the same fundamental problem: **you can't bootstrap skill from an 8% success rate.** The model doesn't solve enough tasks to generate useful training data. When you train on the few successes, LoRA overfits to their surface patterns (file paths, step counts, completion behavior) rather than learning generalizable debugging strategies.

The minimum viable training set for LoRA to generalize appears to be 100+ unique resolved tasks with diverse strategies. At 8% base resolve rate, that requires ~1,250 task attempts — and even then, the resolved tasks would be the easiest ones, biasing the adapter toward trivial fixes.

---

## The Edit Tool Saga

The `<edit>` tool was added to solve 14B's file truncation problem (rewriting 500+ line files hits max_tokens). Three iterations:

| Version | Patch Rate | Resolve Rate | Problem |
|---------|-----------|-------------|---------|
| Write only | 96% | 4/25 (16%) | File truncation on large files |
| Edit only | 40% | 5/50 (10%) | Model retries failed exact matches 7-14x |
| Edit + fuzzy matching | 64% | 4/50 (8%) | Fuzzy saves rescue edits but patches are still wrong |

The edit tool fixes the tool problem (no more truncation) but reveals the model problem (wrong fixes). Fuzzy matching at 0.7 threshold rescued 24% of failed edits but didn't improve resolve rate — the model generates the wrong fix regardless of how reliably it can write it.

The current hybrid approach (system prompt: use `<edit>` for 50+ line files, `<write>` for small/new files) is the best tradeoff.

---

## The Solution Leakage Problem

The two tasks that every model resolves (django-12304, django-13410) have the answer embedded in the bug report:

- **django-12304:** Bug report says "The easy solution would be to declare `do_not_call_in_templates = True`." The fix is literally that one line.
- **django-13410:** Bug report includes a complete `diff --git` patch.

No other test task has this. The 8% base resolve rate on the 25-task test set is entirely explained by solution leakage, not by genuine debugging ability.

This means the true "reasoning-required" resolve rate for 7B is closer to 0%, and the 27% rate for 14B+fixes reflects ~4 tasks where the model can actually diagnose and fix a bug from scratch.

---

## Multi-Repo Expansion (in progress)

To break the cold-start bottleneck, we expanded to all 8 SWE-Bench-CL repos:
- Django, Sympy, Sphinx, Matplotlib, Scikit-learn, Astropy, Xarray, Pytest
- All reference bare clones created at `/Volumes/1TB_SSD/looper/cache/workspaces/.refs/`

**Blocking issue:** Old repo versions (2017-2021) are incompatible with Python 3.11. Sympy pilot: 7/25 patches generated but 0/25 verified because `collections.Mapping` was removed in 3.10. Docker containers needed for rigorous multi-repo verification.

---

## Infrastructure Lessons

| Lesson | Context |
|--------|---------|
| **One inference server at a time** | Ollama + MLX simultaneously causes severe slowdowns on 32GB M4 |
| **LoRA fusion breaks quantized models** | Adapter trained on 4-bit MLX produces garbage when fused. Must use dynamic application (`--adapter-path`) |
| **MLX is 15-20x slower than Ollama** | MLX server: ~5 min/request. Ollama: ~18 sec/request. In-process `mlx_chat()` is slightly better but still slow |
| **14B OOMs at rank > 4 on 32GB** | rank=4 with 8 layers is the max for 14B 4-bit LoRA training on Apple Silicon |
| **Subprocess training is essential** | MLX training holds GPU memory. Run in subprocess so it's freed before inference |
| **`caffeinate -dims`** | Prevents Mac sleep during overnight runs |
| **Ollama on SSD** | `OLLAMA_MODELS=/Volumes/1TB_SSD/looper/ollama_models ollama serve` — brew services doesn't support env vars reliably |

---

## Where Things Stand

### What works
- The agent framework (runner.py with all fixes) is solid: 27% resolve rate with 14B, 100% patch rate
- Patch verification (FAIL_TO_PASS test runner) is correct and reliable for Django
- The codebase is clean: 222+ tests passing

### What doesn't work
- Every LoRA training strategy has negative or zero transfer
- Self-distillation from trajectories is a dead end at <10% base resolve rate
- Multi-repo verification is blocked by Python version incompatibility

### The core thesis is still untested
The thesis — "periodic LoRA consolidation of agent experience produces more capable agents" — requires a base model that resolves enough tasks to generate useful training data. At 8-27% resolve rate with 12-31 unique resolved tasks, we're below the minimum viable scale for LoRA generalization.

The path forward is either:
1. **Scale up the base model** — Use a model with 40%+ resolve rate so there's enough successful experience to consolidate
2. **Scale up the task diversity** — Multi-repo + Docker verification to reach 100+ unique resolved tasks
3. **Change the training approach** — DPO with positive/negative trajectory pairs, or distillation from a stronger model's trajectories, instead of SFT
4. **Accept the negative result** — LoRA skill consolidation may not work for coding agents at current model scales. The framework fixes (prompting, tools, loop detection) are the actual "skill layer" — they just live in the system prompt instead of the weights.
