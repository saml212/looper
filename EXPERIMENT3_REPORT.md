# Experiment 3: Mixture of LoRA Experts (MoLE)

**Date:** 2026-03-11 (13:53 - 17:37 PST, ~3.75 hours)
**Model:** qwen2.5-coder:7b (base/Ollama) / mlx-community/Qwen2.5-Coder-7B-Instruct-4bit (adapted/MLX)
**Training data:** XML tool-call trajectories (trajectory_synthesizer.py, format-matched per DEEP_AUDIT)
**Scope:** 10-task pilot (test tasks 26-35 from django/django)
**Results:** /Volumes/1TB_SSD/looper/results/experiment3_mole/

---

## Hypotheses Tested

1. **MoLE hypothesis**: Separating training data by tool-call type (search/read/modify) into specialized expert adapters and merging them reduces inter-skill interference vs. a single monolithic adapter.

2. **Successful-only hypothesis** (critical untested gap from DEEP_AUDIT): Training ONLY on successful trajectories (8 patch-generating trajectories, 30 examples) avoids self-distillation from the 92% failure rate.

---

## Experimental Design

### Training Data

| Source | Trajectories | Per-Step Examples |
|--------|-------------|-------------------|
| All train (25 trajectories) | 25 | 169 |
| Successful-only (patch_generated) | 8 | 30 |

### Skill Categories (for MoLE split)

| Category | Matcher | All Data | Successful Only |
|----------|---------|----------|-----------------|
| **search** | `<bash>` in assistant msg | 48 | 5 |
| **read** | `<read>` in assistant msg | 105 | 9 |
| **modify** | `<write>` or `<done>` in assistant msg | 16 | 16 |

Note: the "modify" category has identical counts because ALL 16 write/done examples come from successful trajectories (failed trajectories never reach the write stage).

### Configurations

| Config | Rank | Examples | Description |
|--------|------|----------|-------------|
| single_all | 16 | 169 | Single adapter, all data |
| single_success | 16 | 30 | Single adapter, successful-only |
| mole_3_all | 5 (x3) | 169 | 3 experts merged, all data |
| mole_3_success | 5 (x3) | 30 | 3 experts merged, successful-only |

MoLE merge: uniform weight averaging of LoRA A and B matrices across experts.

---

## Training Results

| Condition | Rank | #Ex | Train Loss | Val Loss |
|-----------|------|-----|------------|----------|
| single_all | 16 | 169 | 0.4818 | 0.6423 |
| single_success | 16 | 30 | 0.1507 | 0.0431 |
| mole_3_all_search | 5 | 48 | 0.0466 | 0.3129 |
| mole_3_all_read | 5 | 105 | 0.0753 | 0.1534 |
| mole_3_all_modify | 5 | 16 | 0.0371 | 0.0500 |
| mole_3_success_search | 5 | 5 | **0.0054** | **0.0000** |
| mole_3_success_read | 5 | 9 | 0.0320 | 0.0000 |
| mole_3_success_modify | 5 | 16 | 0.0387 | 0.0425 |

**Key observation:** single_all has much higher loss (0.48) than single_success (0.15) — the failed trajectory data introduces noise. The success-only experts (especially search with 5 examples) are severely overfitted (val_loss=0.0).

### Sanity Checks (all PASS)

All 4 adapters produce XML tool calls. Notable quality differences:
- **mole_3_all**: Cleanest output — natural reasoning + `<bash>` tag, no degeneration
- **mole_3_success**: Direct `<bash>` tag, minimal but functional
- **single_all**: `<read>` tag + degenerate repetition after
- **single_success**: `<read>` tag + severe `<|im_end|>` token leakage (overfitting artifact)

---

## Evaluation Results

| Condition | Resolved | Patches | Avg Steps | Avg Tokens | FT vs Base |
|-----------|----------|---------|-----------|------------|------------|
| **base** | **1/10 (10%)** | **3/10 (30%)** | **8.4** | **18,402** | — |
| single_all | 0/10 (0%) | 0/10 (0%) | 10.0 | 94,427 | -0.10 |
| single_success | 0/10 (0%) | 1/10 (10%) | **4.6** | **18,476** | -0.10 |
| mole_3_all | 0/10 (0%) | 0/10 (0%) | 10.0 | 74,910 | -0.10 |
| mole_3_success | 0/10 (0%) | 0/10 (0%) | 10.0 | 77,735 | -0.10 |

**All adapted conditions: 0/10 resolved. FT = -0.10 across the board.**

Base resolved django-12304 (answer embedded in problem statement, as documented in DEEP_AUDIT).

---

## Analysis

### Finding 1: MoLE provides no measurable benefit

Both MoLE configurations (mole_3_all, mole_3_success) produce 0% resolve rate and 0% patch rate — identical to the single-adapter "all data" configuration. The weight-averaging merge does not preserve the specialized expert knowledge in a useful way.

**Why:** At rank 5, each expert has only 33% of the parameter budget of the single rank-16 adapter. When averaged, the merged adapter is a diluted blend that doesn't strongly reflect any individual expert's specialization. The "separate and merge" approach only helps when experts learn genuinely distinct skills — but with a 7B model on these simple trajectories, all experts learn similar low-level patterns.

### Finding 2: Successful-only training produces qualitatively different behavior

`single_success` stands out from all other adapted conditions:

| Metric | single_all | single_success |
|--------|-----------|----------------|
| Avg steps | 10.0 | **4.6** |
| Avg tokens | 94,427 | **18,476** |
| Patches generated | 0/10 | **1/10** |
| Tasks with "completed" outcome | 0/10 | **7/10** |

The model trained on successful trajectories learned the **pattern of success**: quick completion in 3-4 steps (read → write → done). Most tasks complete in 3 steps. But the completions are premature — the model writes incorrect patches or emits `<done>` without fixing the bug.

**Interpretation:** The model learned the *structure* of successful trajectories (explore briefly, modify, finish) but not the *content* (what the right fix is). With only 30 training examples from 8 tasks, this is expected — there's not enough signal to learn Django debugging skills, only enough to learn the shape of a successful trajectory.

### Finding 3: Token consumption confirms adapter damage

| Condition | Tokens/task | vs Base |
|-----------|-------------|---------|
| base | 18,402 | 1.0x |
| single_success | 18,476 | 1.0x |
| mole_3_all | 74,910 | 4.1x |
| mole_3_success | 77,735 | 4.2x |
| single_all | 94,427 | 5.1x |

Adapters trained on all data (including failures) cause massive token bloat — the model produces verbose, repetitive output without useful tool calls. The successful-only adapter (single_success) has token consumption matching the base model.

### Finding 4: The bottleneck remains data quality and quantity

The experiment confirms the converging finding from Experiments 1-4:
- **8% base resolve rate** → only 2 verifiable successful trajectories in the train set
- Training on these trajectories (even with format-matched XML data) teaches pattern mimicry, not debugging skill
- Neither MoLE (architecture), partial replay (buffer), nor EWC (regularization) can compensate for insufficient training signal

---

## Conclusions

### MoLE: Not warranted for full experiment

The MoLE architecture provides no benefit in this setting. Full experiment (25/25 tasks) is not warranted — the pilot result is decisive.

### Successful-only training: Directionally promising but insufficient

Training on successful-only trajectories is qualitatively better than training on all trajectories:
- Produces clean, efficient agent behavior (4.6 vs 10 steps)
- Token consumption matches base model (no degeneration)
- Only condition to generate any patches

But with only 8 successful trajectories (30 examples), the signal is too weak to teach actual debugging skills. The prerequisite for positive forward transfer remains: a higher base resolve rate (20%+) or access to external gold-standard trajectories.

### Updated experiment priority

| Experiment | Status | Recommendation |
|------------|--------|----------------|
| Exp 1 (Full Replay) | Complete | FT=0.0 |
| Exp 2 (Partial Replay) | Complete | FT=-0.08 to -0.10 |
| **Exp 3 (MoLE)** | **Complete (pilot)** | **FT=-0.10, no benefit** |
| Exp 4 (EWC-LoRA) | Complete | FT=-0.10, no benefit |
| Exp 5 (Adaptive Rank) | Not started | Skip — same data quality problem |
| Exp 6 (Format) | Complete | Format matters but doesn't fix resolve rate |
| Exp 7 (Budget) | Complete | Budget 5 is optimal |
| Exp 9 (Ablation) | Complete | FT=0.0, all conditions equal |

**All training strategy experiments (1-5) converge on the same conclusion:** the bottleneck is training data quality (8% base resolve rate → training on 92% failures), not the training strategy. No architecture or regularization technique can extract useful signal from trajectories that don't contain solutions.

**Next steps should focus on improving training data quality**, not training strategies:
1. Use a stronger model (32B) for base runs to get higher resolve rate
2. Curate gold-standard trajectories from external sources (SWE-Bench solutions)
3. Test with a model that has higher base resolve rate on SWE-Bench-CL
