# Looper Overnight Report
**Date:** 2026-03-05
**Sessions:** Continued from Phase 1 completion (2026-03-04), resumed 2026-03-05

---

## Summary

Phase 1 full 3-condition ablation is complete, plus two synthesis experiments. The LoRA skill adapter shows **zero forward transfer** on the test set — the adapted model resolves exactly the same tasks as the base model. Synthesis experiments (format and budget) reveal that the 7B model memorizes all training data (loss=0.0), making training-time metrics uninformative.

**Experiments completed:**
1. Phase 1 Condition 3 (Base+LoRA) — 25 test tasks with adapted model
2. Experiment 6: Synthesis Format Comparison — 4 prompt formats
3. Experiment 7: Synthesis Budget Sweep — 4 budget levels (prior session)

---

## Phase 1 Full Results (3 Conditions)

### Test Set Performance (25 django/django tasks)

| Condition | Resolved | Resolve Rate | Resolved Tasks |
|-----------|----------|-------------|----------------|
| Base | 2/25 | 8% | django-12304, django-13410 |
| Base+RAG | 2/25 | 8% | django-12304, django-13410 |
| Base+LoRA | 2/25 | 8% | django-12304, django-13410 |

**Forward Transfer = 0.0** — All three conditions resolve the exact same two tasks. Neither RAG nor LoRA changes the outcome.

### Condition 3 Details
- **Adapter:** Trained on 25 train trajectories (budget=5, format=simple_qa)
- **Max steps:** 10 (reduced from 15 to avoid GPU OOM)
- **All 25 tasks hit max_steps** — no task resolved early
- **Patches generated:** 6/25 (24%), vs 7/25 for base at 15 steps
- **Verified patches:** 2/25 resolved (django-12304, django-13410 — same as base)
- **Avg tokens/task:** ~36K (consistent across tasks at 10 steps × ~3.6K/step)
- **Runtime:** ~2 hours (25 tasks × ~290s/task)

### Why Forward Transfer = 0

1. **Training data quality:** The LoRA adapter was trained on synthesis pairs extracted from trajectories where the base model only resolved 2/25 tasks. Most training data comes from failed trajectories — the model is learning from its own mistakes, not from successful strategies.

2. **Memorization, not generalization:** Training loss reaches 0.0 — the model perfectly memorizes 89 examples. But memorizing instruction/response pairs from failed Django debugging sessions doesn't help with different Django tasks.

3. **Base rate too low:** At 8% resolve rate, the 7B model lacks the fundamental capability to solve most tasks. LoRA fine-tuning on behavioral patterns can't compensate for insufficient base reasoning ability.

4. **Step budget confound:** Condition 3 used max_steps=10 vs 15 for conditions 1-2. However, all resolved tasks complete in 3-4 steps, so this shouldn't affect resolvable tasks.

---

## Experiment 7: Synthesis Budget Sweep

**Hypothesis:** More pairs per trajectory provides diminishing returns; the curve flattens between 5-10 pairs.

| Budget | Actual Pairs | Pairs/Traj | Avg Resp Length | Type Distribution | Train Loss |
|--------|-------------|-----------|-----------------|-------------------|------------|
| 3 | 54 | 2.2 | 150 chars | tool:24 err:16 conv:12 wf:2 | 0.0 |
| 5 | 94 | 3.8 | 176 chars | tool:36 err:21 conv:19 wf:18 | 0.0 |
| 10 | 163 | 6.5 | 146 chars | tool:59 err:30 conv:55 wf:19 | 0.0 |
| 20 | 362 | 14.5 | 134 chars | tool:74 err:71 conv:186 wf:31 | 0.0 |

### Findings
1. **Yield ~70% of requested** — 7 trajectories consistently fail JSON parsing
2. **Response quality decreases at higher budgets** — avg length drops from 176→134 chars
3. **Type distribution shifts** — higher budgets generate more "filler" convention pairs
4. **Budget 5 is optimal** for balanced quality and quantity
5. **Training loss uninformative** — all reach 0.0. Need inference evaluation.

---

## Experiment 6: Synthesis Format Comparison

**Hypothesis:** Different synthesis formats produce qualitatively different training data and adapters.

| Format | Pairs | Avg Response Length | Synthesis Time | Training Time | Dominant Type |
|--------|-------|----|----|---|---|
| A_simple_qa | 89 | 150 chars | 753s | 91s | balanced |
| B_chain_of_thought | 88 | **394 chars** | 982s | 139s | tool_usage (78%) |
| D_reflexion | 97 | 282 chars | 920s | 134s | error_recovery (88%) |
| E_contextual | 99 | 294 chars | 880s | 122s | convention (94%) |

### Findings

1. **Format strongly influences output characteristics:**
   - Chain-of-thought produces 2.6x longer responses than simple QA
   - Reflexion generates more pairs (97 vs 88-89) with error_recovery focus
   - Contextual produces the most pairs (99) with convention focus
   - Simple QA produces the most balanced type distribution

2. **Same 7 trajectories fail across all formats** — the JSON failure is trajectory-dependent, not format-dependent

3. **Training time correlates with response length** — longer responses = longer training (91s → 139s)

4. **All formats: train_loss=0.0** — same memorization problem. The 7B model with rank-16 LoRA has more than enough capacity for 88-99 examples.

5. **Format C (DPO) was excluded** — requires paired preference data not yet implemented

---

## Technical Lessons

### GPU OOM on MLX
Running condition 3 with max_steps=15 and max_tokens=4096 caused `kIOGPUCommandBufferCallbackErrorOutOfMemory` after step ~11. Context grows linearly with steps (~3.6K tokens/step). Fix: max_steps=10, max_tokens=512. This reduced per-step time from ~4 min to ~28 sec.

### Resource Contention
Running Ollama + MLX server simultaneously on 32GB M4 causes severe MLX slowdowns and timeouts. **Rule: only one inference server at a time.**

### LoRA Fusion Still Broken
Adapters trained on 4-bit quantized MLX models cannot be fused and converted to GGUF. Must use `mlx_lm.server --adapter-path` for dynamic application.

### Synthesis Model Limitations
7B model fails valid JSON ~30% of the time. Same 7 trajectories fail across all budgets and formats: django-10880, django-11490, django-11603, django-11820, django-11880, django-12125, django-12209.

---

## Cross-Experiment Analysis

### The Core Problem: Training Signal Quality

All three experiments converge on the same issue: **the training signal is too weak to produce meaningful adaptation.**

1. **Low base resolve rate (8%)** means most training trajectories are from failed sessions
2. **Training loss = 0.0** across all configurations (54-362 examples, 4 formats, 4 budgets)
3. **Forward transfer = 0.0** with the best adapter (budget=5, simple_qa format)

This suggests the bottleneck is not synthesis format or budget — it's the **quality of the source trajectories**. The model needs to succeed more often to generate useful training data.

### Potential Paths Forward

1. **Stronger base model** (Phase 3): 32B+ models have higher base resolve rates (~20-30% on SWE-Bench). More successful trajectories = better training signal.

2. **Oracle trajectories**: Train on gold-standard patches from SWE-Bench rather than model-generated ones. Tests whether LoRA *can* help when given good data.

3. **Curriculum filtering**: Only synthesize from resolved trajectories (currently 2/25). This means very few training examples but higher quality.

4. **Multi-repo training**: Accumulate trajectories across many repos to increase the number of resolved tasks in the training set.

5. **Inference evaluation**: Run each adapter (from Exp 6 and 7) on test tasks to differentiate. This is the critical missing piece — training metrics are uninformative. Estimated cost: ~2 hours per adapter × 8 adapters = ~16 hours.

---

## What's Next

### Immediate Priority
1. **Run condition 4 (LoRA+RAG)** to complete the 4-condition ablation (~2 hours)
2. **Inference evaluation** for at least one Experiment 6 adapter (chain-of-thought, the most differentiated)

### Short Term
3. **Oracle trajectory experiment**: Train LoRA on gold patches to test upper bound
4. Phase 2: Cross-validation with 5 splits (but FT=0.0 makes this less urgent)
5. Phase 3: Test with stronger model (Qwen2.5-Coder-32B or similar)

### Blocking Issues
- **MLX inference speed**: ~5 min/task limits evaluation throughput
- **Base resolve rate**: 8% provides insufficient training signal
- **Training loss uninformative**: Need inference eval, which is slow

---

## File Locations

| Item | Path |
|------|------|
| Phase 1 verified results | `/Volumes/1TB_SSD/looper/results/phase1_verified/experiment_result.json` |
| Condition 3 results | `/Volumes/1TB_SSD/looper/results/phase1_full/condition3_results.json` |
| Condition 3 trajectories | `/Volumes/1TB_SSD/looper/results/phase1_full/trajectories/base_lora/` |
| Experiment 6 results | `/Volumes/1TB_SSD/looper/results/experiment6_format/experiment_result.json` |
| Experiment 7 results | `/Volumes/1TB_SSD/looper/results/experiment7_budget/experiment_result.json` |
| Trained adapters (Exp 6) | `/Volumes/1TB_SSD/looper/results/experiment6_format/{format}/adapter/` |
| Phase 1 base trajectories | `/Volumes/1TB_SSD/looper/results/phase1/trajectories/base/` |

---

## Git Status

Branch: `v1`
Recent commits:
- `295a84a` Add OVERNIGHT_REPORT.md
- `b350ecc` Add Experiment 6/7 scripts
- `ee09e10` Complete build order steps 2-8 (219 tests)
