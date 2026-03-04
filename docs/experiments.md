# Experiment Definitions

All experiments are pre-registered here with hypotheses, methodology, and success criteria defined before results are observed. This prevents post-hoc rationalization.

---

## Evaluation Framework: SWE-Bench-CL

All experiments use **SWE-Bench-CL** (Joshi et al., 2025) as the primary benchmark. SWE-Bench-CL was built for exactly our use case: 273 real GitHub issues from 8 Python repositories, organized chronologically, designed to measure whether coding agents improve through continual learning in a fixed environment.

### Repositories

| Repository | Tasks | Difficulty |
|------------|-------|------------|
| django/django | 50 | All easy |
| sympy/sympy | 50 | 25 easy, 25 medium |
| sphinx-doc/sphinx | 44 | 22 easy, 17 medium, 5 hard |
| matplotlib/matplotlib | 34 | 15 easy, 19 medium |
| scikit-learn/scikit-learn | 32 | 13 easy, 18 medium, 1 hard |
| astropy/astropy | 22 | 4 easy, 15 medium, 3 hard |
| pydata/xarray | 22 | 5 easy, 15 medium, 2 hard |
| pytest-dev/pytest | 19 | 8 easy, 8 medium, 3 hard |

### Metrics (from continual learning literature)

| Metric | What it measures | Formula |
|--------|-----------------|---------|
| **Forward Transfer (FT)** | Does past experience help with new tasks? | Accuracy on task i+1 after training through task i, minus baseline |
| **Forgetting (F)** | Does learning new skills degrade old ones? | Average drop from peak performance on each old task |
| **CL-F-beta** | Single score balancing learning vs. forgetting | Harmonic mean of plasticity and stability |
| **Resolve Rate** | Does the agent actually fix the issue? | Proportion of patches that pass all tests |
| **Steps to Completion** | Is the agent more efficient? | Number of tool calls to reach solution |
| **Token Consumption** | Is it cheaper? | Total input+output tokens per task |
| **Tool-Use Efficiency** | Does it pick the right tools? | Ratio of successful action time to total time |

### General Capability Preservation

After each LoRA consolidation, run **HumanEval** (164 problems, pass@1) and **MBPP** (427 problems, pass@1) to verify the skill adapter hasn't degraded general coding ability. Threshold: < 2% drop from base model. This follows the protocol established by Biderman et al. (2024) in "LoRA Learns Less and Forgets Less."

### Conditions (run for every experiment)

Every experiment is evaluated under four conditions using the same source data:

1. **Base model** — no skill adapter, no retrieved context
2. **Knowledge layer only** — base model + RAG over past session trajectories
3. **Skill layer only** — base model + LoRA skill adapter, no retrieval
4. **Both layers** — base model + LoRA skill adapter + RAG

---

## Experimental Phases

The experiments are structured in three phases, each building on the previous. No single phase produces conclusive evidence on its own — the phases are designed so that combined, they provide statistically meaningful results across multiple repos, models, and task orderings.

### Phase 1: Pilot (single repo, single model)

**Goal:** Get the full pipeline working end-to-end. Produce directional results. Identify obvious failure modes before investing in large-scale runs.

**Setup:**
- **Repo:** django/django (50 tasks, all easy — lowest variance)
- **Model:** Qwen 2.5 Coder 7B (fast iteration)
- **Split:** Train on tasks 1-25, test on tasks 26-50
- **Training strategy:** Full replay (Experiment 1 — simplest, establishes upper bound)

**Protocol:**
1. Run base model through all 50 tasks. Record resolve rate, steps, tokens per task.
2. Run base model through tasks 1-25. Collect session trajectories.
3. Synthesize trajectories into LoRA training data.
4. Train LoRA skill adapter on synthesized data.
5. Run adapted model through tasks 26-50. Record same metrics.
6. Compare adapted vs. base on tasks 26-50.

**What this tells us:** Does the pipeline work at all? Is there any directional signal? What breaks? This is NOT statistically significant — it's a single split on a single repo with a single model. It's a smoke test.

**Estimated time:** ~2 hours local compute (7B model).

### Phase 2: Cross-Validation (single repo, multiple splits, single model)

**Goal:** Control for task ordering effects. Determine whether the pilot result is robust or an artifact of the specific split.

**Setup:**
- **Repo:** django/django (same as pilot)
- **Model:** Qwen 2.5 Coder 7B (same as pilot)
- **Splits:**
  - Split A: Train 1-25, test 26-50 (same as pilot)
  - Split B: Train 26-50, test 1-25 (reversed)
  - Split C: Train odd tasks, test even tasks
  - Split D: Random 25/25 split (seed=42)
  - Split E: Random 25/25 split (seed=123)

**Protocol:** Same as Phase 1, run independently for each split. Report mean and variance of Forward Transfer and Forgetting across all 5 splits.

**What this tells us:** Is the effect consistent across orderings, or did we get lucky with the pilot split? If Forward Transfer is positive across 4/5 or 5/5 splits, the effect is robust for this repo+model. If it's 2/5 or 3/5, the effect is weak and ordering-dependent. If 0/5 or 1/5, the skill layer isn't working.

**Statistical test:** Paired t-test or Wilcoxon signed-rank test on per-task metrics (adapted vs. base) across all splits. We need p < 0.05 to claim significance.

**Estimated time:** ~10 hours local compute (5x Phase 1).

### Phase 3: Multi-Repo, Multi-Model (full rigor)

**Goal:** Establish whether the skill layer generalizes across different codebases and different models, or is specific to one repo/model pairing.

**Setup:**
- **Repos:** All 8 SWE-Bench-CL repositories
- **Models:** All 4 target models:
  - Qwen 2.5 Coder 7B (baseline small)
  - Qwen 2.5 Coder 32B (primary large)
  - DeepSeek-R1-Distill-Qwen 32B (reasoning variant)
  - Command R 35B (tool-calling variant)
- **Splits:** Best 2 splits from Phase 2 (the split designs that showed least variance)

**Protocol:** For each repo x model x split combination:
1. Run base model on train split, collect trajectories
2. Synthesize + train LoRA
3. Evaluate on test split under all 4 conditions
4. Run HumanEval/MBPP capability check

**What this tells us:** The full picture:
- Does the skill layer help across repos of different sizes and difficulties?
- Do larger models benefit more or less than smaller ones?
- Does the reasoning model (DeepSeek-R1) learn differently than the coding model (Qwen Coder)?
- Does the tool-calling model (Command R) show more improvement on tool-use efficiency?

**Analysis:** Two-way ANOVA (repo x model) on Forward Transfer. Report effect sizes, not just p-values. Include negative results prominently.

**Matrix:** 8 repos x 4 models x 2 splits = 64 experimental runs. Each run involves training + evaluation on ~25 tasks.

**Estimated time:**
- 7B runs (8 repos x 2 splits = 16 runs): ~32 hours local
- 32B runs (8 repos x 3 models x 2 splits = 48 runs): cloud recommended, ~96 GPU-hours

### Phase Summary

| Phase | Repos | Models | Splits | Total Runs | Purpose |
|-------|-------|--------|--------|------------|---------|
| 1. Pilot | 1 | 1 | 1 | 1 | Does it work at all? |
| 2. Cross-Val | 1 | 1 | 5 | 5 | Is it robust to ordering? |
| 3. Full | 8 | 4 | 2 | 64 | Does it generalize? |

Phase 1 can produce a result in an afternoon. Phase 2 in a weekend. Phase 3 is a multi-week effort, potentially the core of a paper.

---

## Anti-Forgetting Strategy Experiments

The following experiments (1-5) test different LoRA training strategies. Each is run within the Phase 2 or Phase 3 framework above — same repos, same splits, same metrics. The only variable is the training strategy.

---

## Experiment 1: Full Replay Baseline

### Hypothesis
Retraining the LoRA adapter from scratch on ALL accumulated synthetic data at each consolidation step will produce near-perfect retention of past skills, establishing an upper bound on retention and a lower bound on training efficiency.

### Methodology
- Accumulate synthetic training pairs from sessions 1 through N
- At each consolidation checkpoint (every 5 sessions), reinitialize LoRA weights and retrain on the entire accumulated dataset
- Measure retention curve: after consolidation at session N, test accuracy on pairs from sessions 1, N/4, N/2, 3N/4, N

### Variables
- **Independent:** Number of accumulated sessions (10, 20, 50, 100)
- **Dependent:** Retention accuracy per session origin, training wall time, general capability delta
- **Controlled:** Base model, LoRA rank (16), learning rate, synthesis format

### Success Criteria
- Retention accuracy > 90% across all session origins (flat retention curve)
- General capability degradation < 2% on HumanEval/MBPP
- Establishes clear upper bound for comparison with other strategies

### Expected Outcome
Near-perfect retention but linearly growing training cost. This approach should work well for 10-50 sessions but become impractical at 500+ sessions, motivating the more efficient strategies.

### Compute Estimate
- Per consolidation: 10-60 minutes on L4 GPU depending on dataset size
- Total experiment: ~20 GPU-hours

---

## Experiment 2: Partial Replay with Prioritized Sampling

### Hypothesis
A fixed-size replay buffer with intelligent prioritization can achieve 80%+ of full replay's retention at a fraction of the training cost. The priority function determines how much retention is lost.

### Methodology
- Maintain a replay buffer of fixed size (ablate across sizes)
- When buffer is full, evict lowest-priority examples
- At each consolidation, train on new data mixed with replay buffer samples
- Test four priority schemes:
  - **Recency**: newer examples get higher priority
  - **Difficulty**: examples the model currently gets wrong get higher priority (measured by loss)
  - **Diversity**: maximize embedding-space coverage across buffer
  - **Confidence**: higher-confidence synthesized pairs get higher priority

### Variables
- **Independent:** Buffer size (500, 1000, 2000, 5000), priority scheme (4 schemes), mixing ratio (new:replay)
- **Dependent:** Retention curve, training time, forgetting rate vs. full replay
- **Controlled:** Base model, LoRA rank, total training steps per consolidation

### Success Criteria
- Best configuration achieves > 80% of full replay retention at < 30% of training cost
- Clear ranking of priority schemes (identifies which heuristic matters most)
- Identified knee of the buffer-size vs. retention curve

### Expected Outcome
Difficulty-based priority will outperform recency because it focuses capacity on skills the model is actively forgetting. Buffer size 2000 will likely be the sweet spot. Mixing ratio matters more than expected.

### Compute Estimate
- 5 buffer sizes x 4 priority schemes x 3 mixing ratios = 60 configurations
- ~30 minutes each = ~30 GPU-hours total

---

## Experiment 3: Mixture of LoRA Experts (MoLE)

### Hypothesis
Separating skills into specialized adapters (e.g., tool usage, error recovery, code conventions, workflow patterns) reduces inter-skill interference, resulting in less catastrophic forgetting than a single monolithic adapter with equivalent total parameter count.

### Methodology
- Define 3-5 skill categories from the synthesized data types
- Train a separate LoRA adapter for each category
- Train a lightweight router (small MLP on query embeddings) that selects which adapter(s) to activate per query
- At inference time, merge the selected adapters' deltas: W' = W + sum(w_i * B_i @ A_i)
- Compare forgetting against single adapter with the same total rank budget (e.g., 4 experts at rank 4 each vs. 1 expert at rank 16)

### Variables
- **Independent:** Number of experts (3, 4, 5), category taxonomy, router architecture, top-k selection (1, 2, 3)
- **Dependent:** Per-category retention, cross-category interference, routing accuracy, overall fluency score
- **Controlled:** Total parameter budget (rank sum constant), base model, training procedure

### Success Criteria
- MoLE shows measurably less forgetting than single adapter at equivalent parameter count
- Router correctly classifies query type > 85% of the time
- No individual expert category shows > 10% degradation after 50 sessions

### Expected Outcome
MoLE will show a significant advantage for heterogeneous skills (tool usage vs. code conventions are quite different) but less advantage for homogeneous skills. Router accuracy will be the limiting factor.

### Compute Estimate
- 3 expert configurations x 3 top-k values = 9 configurations
- ~1 hour each = ~9 GPU-hours

---

## Experiment 4: EWC-LoRA (Elastic Weight Consolidation for LoRA)

### Hypothesis
Applying EWC regularization to LoRA parameters during sequential updates will reduce catastrophic forgetting. However, the effectiveness may be limited because the low-rank constraint already provides implicit regularization, potentially making EWC's explicit penalty redundant.

### Methodology
- After training on session batch t, compute the diagonal Fisher Information Matrix for all LoRA parameters
- When training on session batch t+1, add EWC penalty: lambda * sum(F_i * (theta_i - theta_i*)^2)
- Use online EWC (accumulate Fisher across all previous tasks)
- Sweep lambda values to find optimal penalty strength

### Variables
- **Independent:** EWC lambda (0, 10, 100, 1000, 10000), Fisher estimation samples (50, 100, 200), accumulation strategy (replace vs. running average)
- **Dependent:** Retention curve, forgetting rate, training time overhead (Fisher computation), general capability delta
- **Controlled:** Base model, LoRA rank, dataset, consolidation frequency

### Success Criteria
- At optimal lambda, EWC-LoRA reduces forgetting rate by > 25% compared to naive sequential updates
- Fisher computation overhead is < 50% of training time
- General capability degradation remains < 3%

### Expected Outcome
This is genuinely uncertain. EWC was designed for full-parameter training where the parameter space is vast. In LoRA's constrained low-rank space, the Fisher penalty might be too restrictive (preventing any learning) or too weak (the low-rank structure already prevents catastrophic changes). Finding the right lambda will be critical.

**If EWC provides no benefit over naive sequential LoRA:** That's a significant finding — it would suggest that low-rank constraint provides sufficient implicit regularization, and explicit anti-forgetting mechanisms need to operate at a different level (e.g., data selection rather than weight protection).

### Compute Estimate
- 5 lambda values x 3 Fisher sample sizes x 2 accumulation strategies = 30 configurations
- Fisher computation adds ~50% overhead
- ~45 minutes each = ~22 GPU-hours

---

## Experiment 5: Adaptive Rank Allocation with SVD Consolidation

### Hypothesis
Dynamically allocating rank budget based on session information content — and compressing old adapters via truncated SVD to free capacity for new sessions — will outperform fixed-rank approaches at the same total parameter budget. This mimics human memory consolidation: recent experiences in high fidelity, old experiences compressed to gist.

### Methodology
- For each new session, determine minimum sufficient rank by probing (train at rank 1, 2, 4, 8, 16; find the knee where validation loss stops improving)
- When total allocated rank exceeds budget, merge the oldest N session adapters and compress via truncated SVD to a lower rank
- Track information retention per session before and after compression

### Variables
- **Independent:** Total rank budget (32, 64, 128), compression trigger threshold, SVD target rank for old sessions (1, 2, 4), number of sessions to merge per compression
- **Dependent:** Information per parameter (retention / total rank), retention curve shape, compression loss
- **Controlled:** Base model, synthesis format, consolidation frequency

### Success Criteria
- Adaptive allocation retains > 15% more information per parameter than fixed-rank approaches
- SVD compression preserves > 70% of compressed sessions' skills at 50% rank reduction
- The retention curve shows a graceful decay (power law) rather than cliff-edge forgetting

### Expected Outcome
Sessions vary wildly in information content. A session where the agent learns a completely new deployment pipeline should need more rank than one where it does routine debugging. Adaptive allocation should be strictly better than fixed allocation. SVD compression will work well for procedural/behavioral skill but poorly for any specific facts that slipped into the adapter.

### Compute Estimate
- 3 budgets x 3 SVD targets x 2 merge strategies = 18 configurations
- ~1 hour each = ~18 GPU-hours

---

## Experiment 6: Synthesis Format Comparison

### Hypothesis
The format of synthesized training data significantly affects what kind of environmental skill the adapter encodes. Chain-of-thought and reflexion-style formats will outperform simple QA for procedural skills, while contextual formats will better preserve environment-specific details.

### Methodology
Train separate adapters from the same trajectory data, synthesized into five formats:

**Format A — Simple QA:**
```
Q: "How do you deploy in this project?"
A: "Run `make deploy` which triggers the GitHub Actions workflow..."
```

**Format B — Chain-of-Thought:**
```
Q: "How do you deploy in this project?"
A: "Let me think through the deployment process. First, the project uses...
    The pipeline starts with... Here's why each step matters..."
```

**Format C — DPO Preference Pairs** (requires both successful and failed trajectories):
```
Q: "How should you handle database migrations?"
Chosen: [approach that worked]
Rejected: [approach that failed]
```

**Format D — Reflexion-style Self-Critique:**
```
Q: "I tried running migrations with `alembic upgrade head` directly and it failed."
A: "That approach fails in this project because we use a custom migration
    wrapper. The correct approach is... because..."
```

**Format E — Contextual Memory** (includes project metadata):
```
Q: "In the payments-api project using FastAPI 0.104 with PostgreSQL, how
    are database connections managed?"
A: "This project uses SQLAlchemy async with a connection pool configured in..."
```

### Variables
- **Independent:** Synthesis format (A, B, C, D, E)
- **Dependent:** Environmental fluency score, retention accuracy by skill type, adapter training loss, downstream task performance
- **Controlled:** Same source trajectories, same base model, same LoRA config, same number of training tokens per format

### Success Criteria
- Statistically significant difference in fluency score between at least 2 formats (p < 0.05)
- Clear pattern of which format works best for which skill type
- At least one format outperforms the others by > 10% on the environmental fluency benchmark

### Expected Outcome
Format D (reflexion) will likely perform best for error recovery skills because it explicitly encodes the failure→diagnosis→fix pattern. Format E (contextual) may overfit to specific project details. Format C (DPO) requires both good and bad trajectories but should produce the strongest behavioral shifts.

### Compute Estimate
- 5 formats x 3 repetitions (for statistical significance) = 15 training runs
- ~30 minutes each = ~8 GPU-hours

---

## Experiment 7: Synthesis Budget Sweep

### Hypothesis
There is a diminishing-returns curve for the number of synthesized training pairs per trajectory. Beyond a certain point, additional pairs provide redundant information and may introduce noise.

### Methodology
- Take the same set of 50 session trajectories
- For each budget level (3, 5, 10, 20, 50 pairs per session), synthesize training data
- Train LoRA adapters on each dataset
- Measure retention, fluency, and capability preservation

### Variables
- **Independent:** Pairs per session (3, 5, 10, 20, 50)
- **Dependent:** Retention score, fluency score, training loss, noise ratio (manually scored on 100-pair sample)
- **Controlled:** Source trajectories, synthesis model, synthesis prompt, base model, LoRA config

### Success Criteria
- Identify the knee of the retention-vs-budget curve
- Determine optimal pairs-per-session for cost-effective synthesis
- Characterize the noise ratio at each budget level

### Expected Outcome
The curve will flatten between 5-10 pairs per session. Beyond 10, additional pairs are increasingly redundant or noisy. The cost-optimal budget will be lower than expected because agent trajectories contain more repetition than novel information.

### Compute Estimate
- 5 budget levels x 3 repetitions = 15 training runs
- ~20 minutes each + synthesis API costs (~$5 total) = ~5 GPU-hours

---

## Experiment 8: Self-Synthesis

### Hypothesis
An agent equipped with its own LoRA adapter can generate higher-quality training data about its environment than an external model, because it has already internalized some environmental context. However, this risks a feedback loop where errors are amplified.

### Methodology
- Phase 1: Use external model (Claude Sonnet) to synthesize first 20 sessions
- Phase 2: Use the adapted agent itself to synthesize sessions 21-40
- Phase 3: Compare adapter quality from external-synthesis vs. self-synthesis
- Also test: can the agent identify and correct errors in externally-synthesized data?

### Variables
- **Independent:** Synthesis source (external model, adapted self-model, hybrid)
- **Dependent:** Synthesis quality (human-scored on 200-pair sample), adapter retention, fluency score, error amplification rate
- **Controlled:** Source trajectories, base model, LoRA config

### Success Criteria
- Self-synthesis quality is within 10% of external synthesis quality
- No measurable error amplification (feedback loop) over 20 sessions
- If self-synthesis works: eliminates external API dependency entirely

### Expected Outcome
Self-synthesis will be lower quality initially (first 5 sessions) but improve as the adapter accumulates competence. Error amplification will be detectable but manageable if we include a validation step (generate, then verify with the base model without adapter).

### Risk
This is the highest-risk experiment. If error amplification is severe, self-synthesis is unviable and the pipeline permanently depends on an external strong model. That's still a useful finding.

### Compute Estimate
- 3 synthesis sources x 40 sessions x 3 repetitions = negligible training (same as other experiments)
- Primary cost is synthesis API calls for the external baseline (~$20)

---

## Experiment 9: Skill Layer + Knowledge Layer Ablation

### Hypothesis
A hybrid system that uses a LoRA skill layer for procedural/behavioral learning and a knowledge layer (RAG) for episodic/factual retrieval will outperform either in isolation. The skill adapter reduces the context budget needed for stable environmental competence, freeing more context for the specific facts and details that actually need to be in the prompt.

### Methodology
Run the same task suite under four conditions:
1. **No memory** — Base model, no skill adapter, no retrieved context
2. **Knowledge layer only** — Base model + RAG (trajectory-derived documents retrieved into context)
3. **Skill layer only** — Skill-adapted model, no retrieved context
4. **Both layers** — Skill-adapted model + RAG

Use the same source data for both the knowledge index and the skill training data, so any differences are purely from the mechanism, not the information available.

### Variables
- **Independent:** Layer configuration (4 conditions above)
- **Dependent:** Task success rate, steps to completion, context tokens used, errors/retries, inference latency
- **Controlled:** Source data (identical), base model, task suite, evaluation procedure

### Success Criteria
- Both layers together outperform knowledge-only on at least one metric by > 10%
- Skill-only outperforms no-memory on task success rate
- Clear characterization of what the skill layer handles vs. what the knowledge layer handles

### Expected Outcome
- Knowledge layer will win on factual retrieval (specific details from past sessions)
- Skill layer will win on behavioral fluency (fewer steps, better tool selection, more efficient navigation)
- Both layers together will win overall because skills and knowledge are complementary
- Skill-only will use significantly less context than knowledge-only for equivalent environmental competence

**If knowledge-only matches or beats the combined approach:** That's a clean negative result. It means the skill layer doesn't add value on top of good knowledge retrieval. That should be published — it saves the community from building something that doesn't work.

### Compute Estimate
- 4 conditions x 50 tasks x 3 repetitions = 600 task evaluations
- Primary cost is inference tokens, not training
- ~$50-100 in inference costs depending on model

---

## Experiment 10: Skill Staleness and Environmental Drift

### Hypothesis
When the environment changes (e.g., codebase refactored, deployment pipeline modified), a stale skill adapter will initially degrade performance compared to no adapter, because the adapter's learned skills conflict with the new reality. However, after re-consolidation on new sessions, the adapter should recover — the agent relearns the new patterns.

### Methodology
- Train a skill adapter on 30 sessions in Environment v1 (specific codebase, deployment pipeline, conventions)
- Modify the environment to v2 (rename key files, change deployment target, alter conventions)
- Measure skill adapter performance in v2 without retraining (stale skills)
- Measure how quickly the adapter recovers after consolidation on new v2 sessions (1, 5, 10, 20 sessions)

### Variables
- **Independent:** Environment change severity (minor/moderate/major), recovery sessions (0, 1, 5, 10, 20)
- **Dependent:** Task success rate in v2, adapter-context conflict rate (how often the adapter's behavior contradicts current context), recovery speed
- **Controlled:** Base model, LoRA config, task suite (adapted for v2)

### Success Criteria
- Quantified degradation from stale skills (measured, not assumed)
- Recovery curve: after N sessions of retraining in v2, skill performance returns to pre-change levels
- Identified indicators that could trigger automatic skill adapter invalidation

### Expected Outcome
Minor environment changes (renaming a few files) will cause minimal degradation because LoRA encodes patterns, not specific paths. Major changes (switching from REST to GraphQL) will cause significant degradation — the adapter will actively fight the new context. Recovery should be fast (5-10 sessions) because the base model's general capabilities are preserved and just the adapter needs updating.

### Implications
If staleness degradation is severe and recovery is slow, the system needs automatic staleness detection — perhaps monitoring when the skill adapter's behaviors conflict with current environmental signals and invalidating when drift exceeds a threshold.

### Compute Estimate
- 3 change severities x 5 recovery checkpoints x 3 repetitions = 45 evaluations
- ~20 GPU-hours for training + evaluation

---

## Execution Order

Experiments are ordered by phase and dependency:

### During Phase 1 (Pilot)
1. **Experiment 1** (Full Replay) — uses the simplest training strategy to get end-to-end results

### During Phase 2 (Cross-Validation)
2. **Experiment 6** (Synthesis Format) — determines which format to use going forward
3. **Experiment 7** (Synthesis Budget) — determines optimal pairs-per-session
4. **Experiment 9** (Skill + Knowledge Ablation) — the fundamental question, now with cross-validated rigor

### During Phase 3 (Multi-Repo, Multi-Model)
5. **Experiment 2** (Partial Replay) — main forgetting mitigation, run across repos/models
6. **Experiment 4** (EWC-LoRA) — novel anti-forgetting contribution
7. **Experiment 3** (MoLE) — alternative architecture, compare against 2 and 4
8. **Experiment 5** (Adaptive Rank) — most novel, uses insights from 2/4/3
9. **Experiment 10** (Staleness) — practical concern, run once best strategy is identified
10. **Experiment 8** (Self-Synthesis) — highest risk, run last with stable pipeline

---

## Target Models

Experiments run on an M4 Mac Mini with 32GB unified memory. Four models chosen for different experimental roles:

| Model | Size | Role in Experiments |
|-------|------|---------------------|
| **Qwen 2.5 Coder 32B** (Q4_K_M) | ~18GB | Primary skill training target — strongest open coding model |
| **DeepSeek-R1-Distill-Qwen 32B** (Q4_K_M) | ~18GB | Reasoning/debugging skill experiments |
| **Command R 35B** (Q4_K_M) | ~20GB | Tool-calling skill experiments — built for RAG and tool use |
| **Qwen 2.5 Coder 7B** (FP16) | ~14GB | Fast iteration, smoke tests, rapid prototyping |

Most experiments use the 7B model first for fast iteration, then validate findings on 32B models. The 7B model runs fast enough for quick feedback loops; the 32B models are for confirming results at scale.

### Inference Stack

Models run locally via **Ollama** or **LM Studio**, which expose an OpenAI-compatible API. OpenClaw connects to this as a custom provider. No cloud inference needed for experimentation.

### Training Stack

- **7B LoRA training:** Feasible locally on M4 via MLX or Unsloth (minutes per run)
- **32B QLoRA training:** Possible locally with MLX but tight on memory. Fall back to GCP spot L4/A100 if needed.
- **Synthesis (data generation):** Run locally using the 32B model, or use Anthropic/OpenAI API for comparison

### Compute Budget

| Experiment | Local Est. | Cloud Fallback | API Costs |
|------------|-----------|----------------|-----------|
| 1. Full Replay | ~5 hrs | 20 GPU-hrs | $5 |
| 2. Partial Replay | ~8 hrs | 30 GPU-hrs | $5 |
| 3. MoLE | ~3 hrs | 9 GPU-hrs | $5 |
| 4. EWC-LoRA | ~6 hrs | 22 GPU-hrs | $5 |
| 5. Adaptive Rank | ~5 hrs | 18 GPU-hrs | $5 |
| 6. Synthesis Format | ~2 hrs | 8 GPU-hrs | $15 |
| 7. Synthesis Budget | ~1 hr | 5 GPU-hrs | $5 |
| 8. Self-Synthesis | ~1 hr | 5 GPU-hrs | $20 |
| 9. Skill+Knowledge Ablation | ~3 hrs | 10 GPU-hrs | $50 |
| 10. Staleness | ~5 hrs | 20 GPU-hrs | $10 |

Local estimates are for 7B model experiments. 32B experiments will be slower locally or use cloud. API costs are for synthesis using Claude/GPT when comparing against local synthesis quality.
