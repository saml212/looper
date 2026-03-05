# Looper Overnight Report
**Date:** 2026-03-05
**Session:** Continued from Phase 1 completion (2026-03-04)

---

## Summary

Phase 1 pilot is complete. The full experiment pipeline works end-to-end: task loading, agent inference, trajectory collection, synthesis, LoRA training, adapted inference, and patch verification. Key findings so far establish the baseline and identify critical bottlenecks.

**Experiments completed this session:**
1. Phase 1 analysis (verified results review)
2. Experiment 7: Synthesis Budget Sweep (complete)

**Experiments in progress:**
- Condition 3 (Base+LoRA) running via `resume_condition3.py` — estimated 15-20 hours remaining

---

## Phase 1 Results (Verified)

### Base Model Performance (Qwen2.5-Coder-7B on django/django)

| Metric | Train (25) | Test (25) | Total (50) |
|--------|-----------|-----------|------------|
| Resolve rate | 2/25 (8%) | 2/25 (8%) | 4/50 (8%) |
| Patches generated | 7/25 | 7/25 | 14/50 |
| Patch accuracy | 2/7 (29%) | 2/7 (29%) | 4/14 (29%) |
| Avg steps (resolved) | 4.0 | 3.5 | 3.8 |
| Hit max_steps | 18/25 | 16/25 | 34/50 |

**Resolved tasks:** django-9296, django-11099 (train), django-12304, django-13410 (test) — all completed in 3-4 steps.

### Key Observations

1. **Bimodal behavior:** Tasks either resolve quickly (3-4 steps, ~6K tokens) or fail completely (15 steps, ~30K tokens). There is no middle ground — no tasks resolve in 6-14 steps.

2. **Low base rate limits the experiment:** 8% resolve rate means we need large sample sizes to detect forward transfer. With 25 test tasks, we'd need the adapter to change at least 3 tasks from fail to pass to achieve statistical significance.

3. **Patch quality matters:** 14/50 tasks generated patches, but only 4 actually fixed the issue. The old file-overlap verifier would have claimed 16% resolve rate — the FAIL_TO_PASS verification halved that to 8%.

4. **Synthesis coverage:** Only 18/25 train trajectories produced valid synthesis pairs. 7 trajectories generated invalid JSON or prose responses instead of structured data. The same problematic trajectories (django-10880, django-11490, django-11603, django-11820, django-11880, django-12125, django-12209) fail across all synthesis attempts.

---

## Experiment 7: Synthesis Budget Sweep

**Hypothesis:** More pairs per trajectory provides diminishing returns; the curve flattens between 5-10 pairs.

### Results

| Budget (requested) | Actual pairs | Pairs/traj | Avg resp length | Types distribution | Train loss |
|---|---|---|---|---|---|
| 3 | 54 | 2.2 | 150 chars | tool:24 err:16 conv:12 wf:2 | 0.0 |
| 5 | 94 | 3.8 | 176 chars | tool:36 err:21 conv:19 wf:18 | 0.0 |
| 10 | 163 | 6.5 | 146 chars | tool:59 err:30 conv:55 wf:19 | 0.0 |
| 20 | 362 | 14.5 | 134 chars | tool:74 err:71 conv:186 wf:31 | 0.0 |

### Findings

1. **Yield is ~70% of requested:** The 7B synthesis model reliably produces pairs but drops ~30% of trajectories due to JSON formatting failures (wrapping in markdown fences, returning prose instead of JSON).

2. **Response quality decreases at higher budgets:** Average response length drops from 176 chars (budget 5) to 134 chars (budget 20), suggesting the model generates shorter, less detailed responses when asked for more pairs.

3. **Type distribution shifts:** At budget 3-5, tool_usage dominates (44%). At budget 20, conventions dominate (51%). Higher budgets force the model to generate more convention-type pairs, which are lower quality filler.

4. **Training loss is uninformative:** All budgets achieve 0.0 training loss at 100 iterations. The model perfectly memorizes datasets of 54-362 examples. Differentiation requires inference evaluation (running each adapter on test tasks).

5. **Optimal budget appears to be 5:** Best combination of yield (3.8/traj), response length (176 chars), and balanced type distribution. Budget 3 produces too few pairs per trajectory; budget 10+ degrades quality.

### Limitations

- Training loss cannot differentiate adapters — all reach 0.0. Need inference evaluation.
- Same 7 trajectories fail synthesis across all budgets (systematic, not budget-dependent).
- No repetition (experiment should be run 3x for statistical significance per docs/experiments.md).

---

## Condition 3 Status (In Progress)

Running `resume_condition3.py` — tests the adapted model (base + LoRA adapter) on 25 test tasks.

- **Started:** 10:44 AM, 2026-03-05
- **MLX server:** Port 8080 with adapter from `/Volumes/1TB_SSD/looper/results/phase1/adapter`
- **Speed:** ~4 min/step, 15 max steps/task
- **Estimated completion:** Late evening / overnight
- **Previous attempt failed:** Running Ollama simultaneously caused resource contention (all 25 tasks timed out). Current run has MLX as the only GPU consumer.

---

## Technical Lessons

### Resource Contention
Running Ollama (qwen2.5-coder:7b, ~5GB) and MLX server (7B 4-bit, ~4.5GB) simultaneously on 32GB M4 causes severe MLX slowdowns. Each inference call that should take 3-4 minutes takes 10+ minutes and eventually times out. **Rule: Only run one inference server at a time.**

### LoRA Fusion Still Broken
LoRA adapters trained on 4-bit quantized MLX models cannot be fused and converted to GGUF for Ollama. The adapter must be used dynamically via `mlx_lm.server --adapter-path`. This limits evaluation to MLX-based inference, which is slower than Ollama.

### Synthesis Model Limitations
The 7B model fails to produce valid JSON ~30% of the time. Common failures:
- Wrapping JSON in markdown code fences (```json...```)
- Generating prose summaries instead of structured pairs
- Truncating long JSON arrays

The `_extract_json_array()` parser handles some of these, but markdown fences with truncated content are unrecoverable. A more robust approach would retry failed synthesis or use a stronger model.

---

## What's Next

### Immediate (when condition 3 completes)
1. Verify condition 3 patches with FAIL_TO_PASS tests
2. Compute forward transfer: base test resolve rate vs adapted test resolve rate
3. If FT > 0: run condition 4 (LoRA + RAG) for the full 4-condition ablation

### Short Term
4. Run Experiment 6 (Synthesis Format Comparison) — scripts are ready
5. Re-run Experiment 7 with inference evaluation (using each budget's adapter on test tasks)
6. Phase 2: Cross-validation with 5 splits to test robustness

### Blocking Issues
- **MLX inference speed:** ~4 min/step makes full 50-task runs take 25+ hours. Consider using a 1.5B or 3B model for faster iteration, or switching to cloud inference.
- **Low base resolve rate (8%):** The 7B model barely solves any tasks, making it hard to measure improvement. Phase 3 with 32B models may show clearer results.
- **Synthesis quality:** 30% JSON failure rate. Need better prompt engineering or a fallback to a stronger synthesis model.

---

## File Locations

| Item | Path |
|------|------|
| Phase 1 verified results | `/Volumes/1TB_SSD/looper/results/phase1_verified/experiment_result.json` |
| Phase 1 OpenClaw pilot | `/Volumes/1TB_SSD/looper/results/phase1_openclaw/experiment_result.json` |
| Experiment 7 results | `/Volumes/1TB_SSD/looper/results/experiment7_budget/experiment_result.json` |
| Condition 3 log | `/Volumes/1TB_SSD/looper/results/phase1_full/condition3_resume.log` |
| Condition 3 trajectories | `/Volumes/1TB_SSD/looper/results/phase1_full/trajectories/base_lora/` |
| Trained adapter | `/Volumes/1TB_SSD/looper/results/phase1/adapter/` |
| Experiment scripts | `run_experiment6_format.py`, `run_experiment7_budget.py` |

---

## Git Status

Branch: `v1`
All work committed (219 tests passing).
Recent commits:
- `b350ecc` Add Experiment 6/7 scripts
- `ee09e10` Complete build order steps 2-8 (219 tests)
- `9f72be8` Reframe anti-forgetting section
