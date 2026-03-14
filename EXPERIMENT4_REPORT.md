# Experiment 4: EWC-LoRA (Elastic Weight Consolidation for LoRA)

**Date:** 2026-03-11
**Status:** Pilot complete. Full experiment not warranted (see findings).
**Model:** qwen2.5-coder:7b (base) / mlx-community/Qwen2.5-Coder-7B-Instruct-4bit (adapted)
**Training data:** XML tool-call trajectories (trajectory_synthesizer.py, per-step format)
**Total training examples:** 169 (across 5 sequential batches)
**Evaluation:** 10 test tasks from SWE-Bench-CL Django curriculum

## Hypothesis

> H4: EWC regularization on LoRA parameters during sequential updates will reduce
> catastrophic forgetting compared to naive sequential training (lambda=0).
>
> Counter-hypothesis: LoRA's low-rank constraint already provides sufficient
> implicit regularization, making EWC redundant.

## Design

Sequential LoRA training across 5 batches of trajectories. After each batch:
1. Compute diagonal Fisher Information Matrix over LoRA parameters
2. Save parameter snapshot
3. On next batch, add EWC penalty to loss: `lambda * sum(F_i * (theta_i - theta_i*)^2)`
4. Accumulate Fisher online across all previous batches

**Lambda sweep:** [0 (naive sequential), 100, 1000]

**Training hyperparameters:** LR=5e-5, iters=50/batch, batch_size=1, max_seq_length=1024, rank=16, 16 LoRA layers, Fisher samples=50

## Training Results

| Condition | Lambda | B1 Loss | B2 Loss | B3 Loss | B4 Loss | B5 Loss (final) |
|-----------|--------|---------|---------|---------|---------|------------------|
| naive_seq | 0      | 0.012   | 0.222   | 0.041   | 0.621   | 0.243            |
| ewc_100   | 100    | 0.049   | 0.374   | 0.159   | 0.797   | 0.440            |
| ewc_1000  | 1000   | 0.015   | 0.328   | 0.212   | 0.591   | 0.494            |

All conditions completed 5 batches with 0 NaN steps (50/50 valid steps per batch).

**Observations:**
- Loss fluctuation across batches (B1 low → B2 high → B3 low → B4 high) reflects the model adapting to different trajectory distributions in each batch
- Higher lambda produces higher final loss (0.24 → 0.44 → 0.49), as expected — EWC penalty constrains weight movement
- No NaN instability at LR=5e-5 (earlier attempts at LR=1e-4 with 100 iters caused NaN on batch 2)

## Evaluation Results

| Condition | Resolved | Patch Rate | Avg Steps | Avg Tokens | FT vs Base |
|-----------|----------|------------|-----------|------------|------------|
| base      | 1/10 (10%) | 3/10 (30%) | 8.4     | 18,402     | —          |
| naive_seq (λ=0) | 0/10 (0%) | 0/10 (0%) | 2.2  | 5,240      | -0.10      |
| ewc_100 (λ=100) | 0/10 (0%) | 0/10 (0%) | 10.0 | 62,326     | -0.10      |
| ewc_1000 (λ=1000) | 0/10 (0%) | 0/10 (0%) | 10.0 | 83,926   | -0.10      |

**Forward transfer: -0.10 for all adapted conditions** (all worse than base).

## Two Distinct Failure Modes

The adapted models fail in qualitatively different ways depending on lambda:

### naive_seq (λ=0): Premature termination
- Completes in 1-2 steps with no patches
- Model learned to generate tool calls but terminates too quickly
- Likely overfits to short successful trajectories in training data
- Same failure mode as Experiment 2 (partial replay)

### ewc_100, ewc_1000 (λ>0): Stuck in loops
- Hits max_steps (10) on every single task
- Generates 10 steps but never produces a patch
- Uses 3-5x more tokens than base (62K-84K vs 18K)
- EWC constraint keeps weights closer to initialization, producing verbose but unproductive behavior
- The model is "trying harder" but not making useful edits

## Key Findings

1. **EWC provides no benefit over naive sequential training.** All conditions resolve 0/10 tasks. The question of whether EWC prevents catastrophic forgetting is moot when the adapted model can't resolve tasks regardless.

2. **LoRA's low-rank constraint hypothesis is untestable at this base resolve rate.** With base at 10% (1/10) and all adapted at 0%, we can't distinguish "EWC helps with forgetting" from "LoRA training degrades the model." The signal is below the noise floor.

3. **Higher lambda changes failure mode but doesn't improve outcomes.** λ=0 fails fast (premature termination), λ>0 fails slow (stuck loops). Neither produces patches.

4. **Training on failed trajectories remains the root cause.** Base resolves only 8% of tasks (Phase 1 verified). Training on 92% failed trajectories teaches the model to fail in structured ways rather than succeed.

5. **Consistent with Experiment 2 findings.** Format-matched XML training produces models that generate valid tool calls but cannot solve tasks. The format mismatch fix was necessary but not sufficient.

## Decision: No Full Experiment

The pilot shows 0% resolve rate across all 3 lambda values with no differentiation. Running the full experiment (4 lambdas × 25 tasks) would not change the conclusion. The bottleneck is upstream (training data quality), not forgetting prevention.

## Implications for Research Agenda

- **Experiments 1-4 converge on the same conclusion:** LoRA training on low-quality trajectories (8% success rate) cannot produce positive forward transfer, regardless of replay strategy (full, partial, EWC) or synthesis format.
- **Prerequisite for Experiment 5+ (MoLE, adaptive rank):** Need higher base resolve rate or curated successful-only trajectories. Consider: (a) stronger base model, (b) filtering to only successful trajectories, (c) human-curated training examples.
- **EWC implementation is verified and available** for future use when training data quality is solved. The `looper/trainers/ewc_trainer.py` module works correctly (0 NaN steps, proper Fisher computation, online accumulation).

## Files

- `detailed_results.json` — Per-task evaluation results
- `experiment_result.json` — Summary metrics
- `all_training_metrics.json` — Per-batch training losses
- `adapters/` — Saved adapter weights for all conditions
- `batch_data/` — Per-batch JSONL training files
- `trajectories/` — Per-condition evaluation trajectories
