# Experiment Definitions

All experiments are pre-registered here with hypotheses, methodology, and success criteria defined before results are observed. This prevents post-hoc rationalization.

---

## Experiment 1: Full Replay Baseline

### Hypothesis
Retraining the LoRA adapter from scratch on ALL accumulated synthetic data at each consolidation step will produce near-perfect retention of past knowledge, establishing an upper bound on retention and a lower bound on training efficiency.

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
Difficulty-based priority will outperform recency because it focuses capacity on knowledge the model is actively forgetting. Buffer size 2000 will likely be the sweet spot. Mixing ratio matters more than expected.

### Compute Estimate
- 5 buffer sizes x 4 priority schemes x 3 mixing ratios = 60 configurations
- ~30 minutes each = ~30 GPU-hours total

---

## Experiment 3: Mixture of LoRA Experts (MoLE)

### Hypothesis
Separating knowledge into specialized adapters (e.g., tool usage, error recovery, code conventions, workflow patterns) reduces inter-knowledge interference, resulting in less catastrophic forgetting than a single monolithic adapter with equivalent total parameter count.

### Methodology
- Define 3-5 knowledge categories from the synthesized data types
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
MoLE will show a significant advantage for heterogeneous knowledge (tool usage vs. code conventions are quite different) but less advantage for homogeneous knowledge. Router accuracy will be the limiting factor.

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
- SVD compression preserves > 70% of compressed sessions' knowledge at 50% rank reduction
- The retention curve shows a graceful decay (power law) rather than cliff-edge forgetting

### Expected Outcome
Sessions vary wildly in information content. A session where the agent learns a completely new deployment pipeline should need more rank than one where it does routine debugging. Adaptive allocation should be strictly better than fixed allocation. SVD compression will work well for procedural/behavioral knowledge but poorly for any specific facts that slipped into the adapter.

### Compute Estimate
- 3 budgets x 3 SVD targets x 2 merge strategies = 18 configurations
- ~1 hour each = ~18 GPU-hours

---

## Experiment 6: Synthesis Format Comparison

### Hypothesis
The format of synthesized training data significantly affects what kind of environmental knowledge the adapter encodes. Chain-of-thought and reflexion-style formats will outperform simple QA for procedural knowledge, while contextual formats will better preserve environment-specific details.

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
- **Dependent:** Environmental fluency score, retention accuracy by knowledge type, adapter training loss, downstream task performance
- **Controlled:** Same source trajectories, same base model, same LoRA config, same number of training tokens per format

### Success Criteria
- Statistically significant difference in fluency score between at least 2 formats (p < 0.05)
- Clear pattern of which format works best for which knowledge type
- At least one format outperforms the others by > 10% on the environmental fluency benchmark

### Expected Outcome
Format D (reflexion) will likely perform best for error recovery knowledge because it explicitly encodes the failure→diagnosis→fix pattern. Format E (contextual) may overfit to specific project details. Format C (DPO) requires both good and bad trajectories but should produce the strongest behavioral shifts.

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

## Experiment 9: Hybrid Ablation (LoRA + RAG vs. Either Alone)

### Hypothesis
A hybrid system that uses LoRA for procedural/behavioral knowledge and RAG for episodic/factual knowledge will outperform either system in isolation. The LoRA adapter reduces context budget consumption for stable environmental knowledge, freeing more context for the RAG-retrieved specifics that actually need to be in the prompt.

### Methodology
Run the same task suite under four conditions:
1. **No memory** — Base model with no adapter and no retrieved context
2. **RAG only** — Base model with trajectory-derived documents retrieved into context
3. **LoRA only** — Adapted model with no retrieved context
4. **LoRA + RAG** — Adapted model with retrieved context

Use the same source data for both the RAG index and the LoRA training data, so any differences are purely from the storage/access mechanism, not the information available.

### Variables
- **Independent:** Memory configuration (4 conditions above)
- **Dependent:** Task success rate, steps to completion, context tokens used, errors/retries, inference latency
- **Controlled:** Source data (identical), base model, task suite, evaluation procedure

### Success Criteria
- LoRA + RAG outperforms RAG-only on at least one metric by > 10%
- LoRA-only outperforms no-memory on task success rate
- Clear characterization of what LoRA handles vs. what RAG handles

### Expected Outcome
- RAG will win on factual recall (specific details from past sessions)
- LoRA will win on behavioral fluency (fewer steps, better tool selection, more idiomatic code)
- Hybrid will win overall because it gets both advantages
- LoRA-only will use significantly less context than RAG-only for equivalent background knowledge

**If RAG-only matches or beats the hybrid:** That's a clean negative result. It means weight-based memory doesn't add value over context-based retrieval, and the complexity of the training pipeline is unjustified. This should be published as a negative result.

### Compute Estimate
- 4 conditions x 50 tasks x 3 repetitions = 600 task evaluations
- Primary cost is inference tokens, not training
- ~$50-100 in inference costs depending on model

---

## Experiment 10: Staleness and Environmental Drift

### Hypothesis
When the environment changes (e.g., codebase refactored, deployment pipeline modified), an outdated LoRA adapter will initially degrade performance compared to no adapter, because the adapter's encoded patterns conflict with the new reality. However, after re-consolidation on new sessions, the adapter should recover.

### Methodology
- Train an adapter on 30 sessions in Environment v1 (specific codebase, deployment pipeline, conventions)
- Modify the environment to v2 (rename key files, change deployment target, alter conventions)
- Measure adapter performance in v2 without retraining
- Measure how quickly the adapter recovers after consolidation on new v2 sessions (1, 5, 10, 20 sessions)

### Variables
- **Independent:** Environment change severity (minor/moderate/major), recovery sessions (0, 1, 5, 10, 20)
- **Dependent:** Task success rate in v2, adapter-context conflict rate (how often the adapter's behavior contradicts current context), recovery speed
- **Controlled:** Base model, LoRA config, task suite (adapted for v2)

### Success Criteria
- Quantified degradation from stale adapters (measured, not assumed)
- Recovery curve: after N sessions of retraining in v2, performance returns to pre-change levels
- Identified indicators that could trigger automatic adapter invalidation

### Expected Outcome
Minor environment changes (renaming a few files) will cause minimal degradation because LoRA encodes patterns, not specific paths. Major changes (switching from REST to GraphQL) will cause significant degradation — the adapter will actively fight the new context. Recovery should be fast (5-10 sessions) because the base model's general capabilities are preserved and just the adapter needs updating.

### Implications
If staleness degradation is severe and recovery is slow, the system needs an automatic staleness detection mechanism — perhaps comparing adapter predictions with actual environment state and invalidating when drift exceeds a threshold.

### Compute Estimate
- 3 change severities x 5 recovery checkpoints x 3 repetitions = 45 evaluations
- ~20 GPU-hours for training + evaluation

---

## Experiment Execution Order

Experiments are ordered by dependency and information value:

1. **Experiment 1** (Full Replay) — Establishes upper bound, required for all comparisons
2. **Experiment 6** (Synthesis Format) — Determines which format to use for all subsequent experiments
3. **Experiment 7** (Synthesis Budget) — Determines optimal pairs-per-session for data generation
4. **Experiment 9** (Hybrid Ablation) — The fundamental question: does LoRA help at all?
5. **Experiment 2** (Partial Replay) — Main forgetting mitigation approach
6. **Experiment 4** (EWC-LoRA) — Novel anti-forgetting contribution
7. **Experiment 3** (MoLE) — Alternative architecture for forgetting mitigation
8. **Experiment 5** (Adaptive Rank) — Most novel experiment, depends on insights from 2/4
9. **Experiment 10** (Staleness) — Practical concern, run once best training strategy is identified
10. **Experiment 8** (Self-Synthesis) — Highest risk, run last after pipeline is stable

---

## Total Compute Budget

| Experiment | GPU-Hours | API Costs |
|------------|-----------|-----------|
| 1. Full Replay | 20 | $5 |
| 2. Partial Replay | 30 | $5 |
| 3. MoLE | 9 | $5 |
| 4. EWC-LoRA | 22 | $5 |
| 5. Adaptive Rank | 18 | $5 |
| 6. Synthesis Format | 8 | $15 |
| 7. Synthesis Budget | 5 | $5 |
| 8. Self-Synthesis | 5 | $20 |
| 9. Hybrid Ablation | 10 | $100 |
| 10. Staleness | 20 | $10 |
| **Total** | **~147** | **~$175** |

At GCP spot L4 pricing (~$0.70/hr), total GPU cost is approximately **$103**. Combined with API costs, the full experiment suite costs roughly **$280**.
