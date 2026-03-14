# Future Work

Based on findings from 8 experiments (March 4-14, 2026). See [LEARNINGS.md](../LEARNINGS.md) for full context.

---

## 1. Opus-Supervised Agentic Loop (Guided Skill Acquisition)

**Problem:** The cold-start problem blocks self-play: the model resolves too few tasks to generate useful training data. Self-distillation from failures teaches failure patterns.

**Approach:** Run the small model (7B/14B with LoRA) in a full agentic loop, but with a stronger model (Claude Opus) as an oversight layer that:
1. Approves or rejects the agent's proposed actions before execution
2. Provides feedback on why something was rejected
3. Suggests better approaches when the agent is stuck
4. Generates rich training signal — granular behavioral feedback, not just pass/fail

**Training loop:**
1. Small model attempts task in agentic loop with tool access
2. Opus oversees, approves/rejects/guides in real time
3. Successful trajectories (including Opus corrections) become LoRA training data
4. Measure: does the small model need less Opus intervention over time?

**Why this might work where self-play failed:**
- Provides correct trajectories in the right format (XML tool calls in context)
- Bypasses the cold-start problem entirely — Opus can solve tasks the small model can't
- Training data is diverse (Opus approaches problems differently each time)
- The small model sees >100 unique resolved tasks, crossing the apparent minimum viable scale

**Key metric:** Opus intervention rate over time. If LoRA is working, the small model should need fewer corrections.

---

## 2. Richer Tool/Environment Benchmarks

**Problem:** SWE-Bench tasks are "read code, write patch." The agent's environment is minimal — no tool discovery, no complex workflows, no state management across sessions. This limits our ability to measure environmental fluency.

**What we actually need:** A benchmark that:
- Requires tool discovery and selection (not just "edit file")
- Has a rich environment the agent must learn to navigate
- Rewards efficiency gains over repeated tasks
- Measures behavioral pattern development, not just correctness

**Candidates:**
- OpenClaw-native tasks (configure channels, debug gateway issues, manage plugins)
- DevOps/SRE benchmarks with real infrastructure
- Multi-tool coding tasks requiring git, CI, testing frameworks, deployment
- Custom benchmark around a realistic developer workflow

---

## 3. Docker-Based Multi-Repo Verification

**Problem:** SWE-Bench-CL covers 8 repos, but old repo versions (2017-2021) are incompatible with Python 3.11. Our FAIL_TO_PASS verifier only works reliably on Django.

**Solution:** Use Docker containers matching each repo's original Python/dependency versions. This would:
- Enable rigorous verification across all 8 repos (273 tasks total)
- Increase the pool of verifiable resolved tasks (currently ~30 across Django only)
- Potentially cross the 100+ unique resolved task threshold needed for LoRA generalization

---

## 4. DPO/Reward-Weighted Training

**Problem:** SFT on resolved trajectories teaches surface completion patterns, not debugging skill. The model learns "finish in 4 steps" not "diagnose the bug correctly."

**Approach:** Train with DPO (Direct Preference Optimization) using positive/negative trajectory pairs:
- Positive: trajectories that resolved the task
- Negative: trajectories that failed on the same task
- The model learns to prefer the behavioral patterns that lead to success

A DPO training script (`run_dpo_training.py`, 678 lines) is already written but not yet tested due to insufficient positive/negative pairs.

---

## 5. Ensemble/Majority-Vote Across Model Sizes

**Finding:** The union of 7B + 14B + 32B resolves 5/15 tasks (33%), more than any single model (27%). Each model resolves at least one task the others can't.

**Approach:** Run all three models on each task, take majority vote or union. This requires no LoRA training — it's a pure inference-time improvement. Could be implemented as a simple wrapper around the existing agent loop.

---

## 6. Scaling Factor Investigation

**Finding:** LoRA scaling factor (alpha/rank = 2.0) may be too aggressive for 4-bit quantized models. The adapter produced degenerate output (repetitive gibberish, leaked `<|im_end|>` tokens) in Phase 1, suggesting the distribution shift overwhelmed the quantized weights.

**Test:** Sweep alpha/rank from 0.1 to 2.0 and measure output quality. A lower scaling factor might allow the adapter to shift behavior without destabilizing the model.

---

*Originally added March 5, 2026. Updated March 14, 2026 with findings from all experiments.*
