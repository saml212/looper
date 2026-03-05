# Future Work

## 1. Richer Tool/Environment Benchmarks

**Problem:** SWE-Bench-CL tasks are primarily "read code, write patch." The agent's environment is minimal — it doesn't need to discover tools, navigate complex workflows, manage state across sessions, or build environmental familiarity. This limits our ability to measure whether LoRA skill layers help agents gain efficiency in their *working environment*, not just at code patching.

**What we actually care about:** Agents forming *habits* — learned behavioral patterns that make them faster and more effective in a specific environment over time. Like a developer who learns the codebase's testing conventions, knows which tools to reach for, remembers where the config files live. SWE-Bench tasks don't require much of this environmental fluency.

**Direction:** Find or build a benchmark that:
- Requires tool discovery and selection (not just "edit file")
- Has a rich environment the agent must learn to navigate (e.g., OpenClaw itself, with its CLI, plugins, configs, APIs)
- Rewards efficiency gains over repeated tasks in the same environment
- Measures whether the agent develops useful patterns (habits/skills) vs. just memorizing answers

**Candidates to explore:**
- OpenClaw-native tasks (configure channels, debug gateway issues, manage plugins) — the agent's actual working environment
- DevOps/SRE benchmarks with real infrastructure
- Multi-tool coding tasks that require git, CI, testing frameworks, deployment
- Build a custom benchmark around a realistic developer workflow in a specific codebase

## 2. Opus-Supervised Agentic Loop (Guided Skill Acquisition)

**Problem:** Current approach synthesizes training data from trajectories after the fact. In the real world, a junior developer doesn't just attempt tasks solo and then get graded — they work *with* a senior developer who provides real-time guidance, approves/rejects approaches, and shapes their habits as they go.

**Approach:** Run the small open-source model (e.g., Qwen 7B/32B with LoRA) in a full agentic loop — calling tools, navigating the environment, solving tasks — but with a stronger model (Claude Opus) acting as an oversight layer that:
1. **Approves or rejects** the agent's proposed actions before execution
2. **Provides feedback** on why something was rejected ("don't grep the whole repo, use the project's search index")
3. **Suggests better approaches** when the agent is stuck
4. Generates **rich training signal** — not just "solved/didn't solve" but granular behavioral feedback

**Training loop:**
1. Small model attempts task in agentic loop with tool access
2. Opus oversees, approves/rejects/guides in real time
3. Successful trajectories (including Opus corrections) become LoRA training data
4. Over iterations, the small model internalizes the patterns Opus taught it
5. Measure: does the small model need less Opus intervention over time? (= skill acquisition)

**Why this is better than pure distillation:**
- Distillation copies outputs. This copies *working patterns in context*.
- The training data captures the agent navigating a real environment with real tools, not just generating text.
- It mirrors how humans actually learn on the job — guided practice, not lectures.
- The LoRA learns environment-specific habits, not just general coding ability.

**Key metric:** Opus intervention rate over time. If the LoRA is working, the small model should need fewer corrections as it accumulates experience in a given codebase/environment.

---

*Added March 5, 2026 — Sam's notes, written up by Gumi*
