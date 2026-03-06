# Failure Analysis — Looper Overnight Run (March 4–5, 2026)

## Summary

The Claude Code agent built an impressive amount of working code but **never completed the overnight experiment** because the MLX LoRA inference server is catastrophically slow (~4–5 minutes per request). The experiment got stuck in Condition 3 (Base + LoRA) and the agent process eventually died — likely due to Claude Code session timeout or context exhaustion — while waiting for MLX to grind through 25 tasks × 15 steps.

No `OVERNIGHT_REPORT.md` was written because the experiment never reached completion.

---

## What the Agent Accomplished

The codebase is in excellent shape. The agent built the entire Looper framework from scaffold to working experiments:

### Completed Components (all with passing tests — **219/219 tests pass**)
- `looper/models.py` — Core Pydantic data models
- `looper/agent/runner.py` — XML-tool-calling agent loop for 7B models
- `looper/agent/ollama_client.py` — Chat clients (Ollama, OpenAI-compat, MLX native)
- `looper/agent/workspace.py` — Git workspace management
- `looper/collectors/trajectory_store.py` — Trajectory save/load/batch-collect with resume support
- `looper/synthesizers/` — Trajectory → training data pipeline
- `looper/trainers/` — LoRA training (full replay strategy)
- `looper/evaluators/` — Patch verification, RAG, metrics, results I/O
- `looper/integrations/` — OpenClaw parser, provider, experiment runner
- `looper/tasks/loader.py` — SWE-Bench-CL curriculum loader
- `looper/serving/` — Adapter-to-Ollama conversion
- `looper/analysis/` — Results analysis and paper sections
- `looper/pipeline.py` — Pipeline orchestrator

### Completed Experiment Runs
1. **Phase 1 basic** (`/Volumes/1TB_SSD/looper/results/phase1/`) — Full run with base model trajectories, synthesis, adapter training. Has `experiment_result.json`.
2. **Phase 1 OpenClaw** (`/Volumes/1TB_SSD/looper/results/phase1_openclaw/`) — OpenClaw integration pilot (3 tasks). Completed with results.
3. **Phase 1 Full — Conditions 1+2** (`/Volumes/1TB_SSD/looper/results/phase1_full/`) — Base (2/25 resolved) and Base+RAG (2/25 resolved) conditions completed with all 25 trajectories saved.

### Where It Got Stuck
4. **Condition 3: Base + LoRA** — Started at ~00:17 on Mar 5, the agent attempted this condition **three separate times** (original `run_phase1_full.py`, then `run_phase1_full_resume.py` at ~02:05, then `resume_condition3.py` at ~03:44). Each time it stalled on the same problem: MLX inference is ~4–5 minutes per chat completion request.

**The `base_lora/` trajectory directory is empty** — not a single task completed across any of the three attempts.

---

## Root Cause: MLX Inference Speed

The MLX server logs tell the story clearly:

```
03:44:00 - POST /v1/chat/completions 200    (first request)
03:47:28 - POST /v1/chat/completions 200    (3.5 min later)
03:51:23 - POST /v1/chat/completions 200    (4 min later)
03:55:39 - POST /v1/chat/completions 200    (4 min later)
04:00:07 - POST /v1/chat/completions 200    (4.5 min later)
04:04:56 - POST /v1/chat/completions 200    (5 min later)
04:10:04 - POST /v1/chat/completions 200    (5 min later)
04:15:33 - POST /v1/chat/completions 200    (5.5 min later — last seen)
```

**Math:** 25 tasks × 15 steps/task = 375 requests. At 4–5 min each = **25–31 hours** for Condition 3 alone. The overnight window was ~5 hours. The agent would need **50+ hours** for Conditions 3+4 combined.

By contrast, Ollama handled the same workload in ~16 minutes for Condition 2 (25 tasks, 00:00 to 00:17).

### Why MLX is so slow
The MLX server log shows prompt sizes of 2048–4000+ tokens being processed in batches, with prompt processing taking ~15 seconds per step. The 4-bit quantized Qwen 7B model on Apple Silicon via MLX is orders of magnitude slower than Ollama's optimized inference for the same model. The chat completions are also generating long responses (the context grows with each agent step).

---

## Why the Agent Process Died

The agent (Claude Code) created increasingly sophisticated resume scripts across three attempts, each re-running Conditions 1+2 verification (fast) before hitting the MLX wall again on Condition 3. The last log entry is at **04:15:33**. Most likely causes:

1. **Claude Code session/API timeout** — The process was in a long-running Python subprocess. If the Claude Code session had a maximum duration or the API connection dropped, the process would die silently.
2. **macOS sleep/power management** — At ~4 AM with no user interaction, the Mac mini may have entered sleep or reduced power state, killing the MLX server or the Python process.
3. **Memory pressure** — MLX loads the full model into unified memory. Combined with Ollama, the workspace git clones, and Claude Code itself, the system may have hit memory limits.

The most likely cause is **#1 or #2** — the agent ran for approximately 4.5 hours (from ~00:00 to ~04:15) and simply ran out of time or got disconnected.

---

## What's Missing or Broken

### Nothing is broken in the code
- All 219 tests pass
- All imports work cleanly
- The code is well-structured with proper `__init__.py` files everywhere
- Dependencies are installed (mlx, mlx-lm, httpx, pydantic)

### What's missing for experiment completion
1. **Condition 3 results** (Base + LoRA) — `/Volumes/1TB_SSD/looper/results/phase1_full/trajectories/base_lora/` is empty
2. **Condition 4 results** (Base + LoRA + RAG) — directory doesn't even exist yet
3. **Final comparison metrics** and `experiment_result.json` for the 4-condition experiment
4. **OVERNIGHT_REPORT.md** — never written because experiment never completed

---

## Recommended Fixes Before Restarting

### 1. Replace MLX Server with Ollama for LoRA Conditions (Critical)

The MLX server is 15–20x slower than Ollama for this workload. Two options:

**Option A: Use the adapter-to-Ollama converter** (already built at `looper/serving/adapter_to_ollama.py`)
```bash
# Convert the MLX LoRA adapter to an Ollama model
python -c "
from looper.serving.adapter_to_ollama import convert_adapter
convert_adapter(
    '/Volumes/1TB_SSD/looper/results/phase1/adapter',
    'qwen2.5-coder:7b',
    'qwen2.5-coder-looper:7b'
)
"
# Then use ollama with the adapted model name instead of MLX
```

**Option B: Use `mlx_chat()` (in-process MLX) instead of `openai_chat()` (server)**
The `ollama_client.py` already has `load_mlx_model()` and `mlx_chat()` which run inference in-process without the HTTP overhead. This might be faster than the server.

**Option C: Use `llama.cpp` / `llama-server` with GGUF**
Convert the adapter to GGUF format and serve with llama.cpp, which is typically much faster than MLX server.

### 2. Reduce the Workload for LoRA Conditions

If MLX must be used, reduce the test set size for Conditions 3+4:
- Use `adapted_test_size=10` instead of 25
- Or reduce `MAX_STEPS` from 15 to 8 for LoRA conditions

### 3. Add Timeout/Progress Tracking to Resume Script

The resume scripts already have good progress logging but should add:
- Per-task timeout (kill after 30 min instead of waiting for all 15 steps at 5 min each)
- Write partial results to disk after each condition completes
- Save a `progress.json` that tracks which task/condition was active when the process died

### 4. Keep Mac Awake

```bash
caffeinate -dims &  # Prevent sleep during overnight runs
```

### 5. Run via `nohup` or `tmux`

```bash
# Don't tie the experiment to a Claude Code session
cd ~/experiments/looper
tmux new -d -s looper '.venv/bin/python run_phase1_full_resume.py 2>&1 | tee run.log'
```

---

## Current State Summary

| Component | Status |
|-----------|--------|
| Codebase | ✅ Complete, 219/219 tests passing |
| Phase 1 Basic | ✅ Complete with results |
| Phase 1 OpenClaw | ✅ Complete (pilot, 3 tasks) |
| Condition 1 (Base) | ✅ 2/25 resolved |
| Condition 2 (Base+RAG) | ✅ 2/25 resolved |
| Condition 3 (Base+LoRA) | ❌ 0/25 — MLX too slow |
| Condition 4 (Base+LoRA+RAG) | ❌ Not started |
| Final comparison | ❌ Blocked on Conditions 3+4 |
| Ollama | ✅ Running (PID 1772) |
| MLX Server | ⏹️ Not running (was terminated) |
| LoRA Adapter | ✅ Trained at `/Volumes/1TB_SSD/looper/results/phase1/adapter/` |
| fused_model/ | ✅ Exists at `~/experiments/looper/fused_model/` |

---

## Fixes Applied

No code fixes were needed — the codebase is clean and all tests pass. The failure was purely an infrastructure/performance problem with MLX inference speed, not a code bug.

---

*Analysis completed: 2026-03-05 08:31 PST*
