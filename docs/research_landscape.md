# Research Landscape

A survey of the current state of research across the areas that Looper draws on and contributes to.

---

## 1. Parameter-Efficient Fine-Tuning (PEFT)

### LoRA: Low-Rank Adaptation

**LoRA** (Hu et al., 2021) is the foundation of this project. Instead of updating all parameters during fine-tuning, LoRA freezes the pre-trained weights and injects small trainable rank-decomposition matrices into transformer layers.

For a pre-trained weight matrix W of dimensions d x k, the update is:

```
W' = W + BA
```

where B is d x r and A is r x k, with rank r << min(d, k) (typically r = 8 to 64). This reduces trainable parameters by 1000x or more while achieving performance competitive with full fine-tuning on many tasks.

**Key properties for our use case:**
- Adapter files are small (30-50MB for a 7B model at rank 16)
- Multiple adapters can be hot-swapped over a frozen base model
- Training is fast — minutes to hours on a single GPU for small datasets
- LoRA excels at encoding behavioral patterns, style, and domain expertise
- LoRA struggles with precise factual memorization (see "LoRA Learns Less and Forgets Less" below)

### QLoRA

**QLoRA** (Dettmers et al., 2023) combines 4-bit quantization of the base model with 16-bit LoRA adapters. This allows fine-tuning a 65B parameter model on a single 48GB GPU. For our purposes, it means we can experiment with larger base models on commodity hardware.

### Variants and Extensions

- **DoRA** (Weight-Decomposed Low-Rank Adaptation) — decomposes weight updates into magnitude and direction components, showing improved learning capacity for the same rank
- **AdaLoRA** — adaptively allocates rank budget across layers based on importance scores
- **LoRA+** — uses different learning rates for the A and B matrices

### The Fundamental Limitation: LoRA Learns Less and Forgets Less

**Biderman et al. (2024)** conducted the most thorough empirical study of LoRA versus full fine-tuning. Their key findings:

1. Full fine-tuning produces weight changes with effective rank 10-100x higher than typical LoRA configurations
2. LoRA is effective at learning within a domain (adapting behavior) but substantially underperforms at learning new factual knowledge
3. However, LoRA also forgets less — the low-rank constraint acts as implicit regularization
4. This creates a fundamental tradeoff: LoRA adapters are good at shifting behavioral patterns but poor at memorizing specific facts

**Implication for Looper:** This is exactly why LoRA is the right technology for a skill layer. LoRA is naturally good at encoding the things that constitute environmental skill — tool usage patterns, debugging strategies, workflow conventions — and naturally bad at the things that should stay in the knowledge layer (specific facts, episodic details). The technology's strengths align with the abstraction.

---

## 2. Catastrophic Forgetting in Continual Learning

Catastrophic forgetting is the phenomenon where a neural network trained on task B loses its ability to perform task A. This is the core research challenge for any system that sequentially updates an adapter with new experience.

### Classical Approaches

**Elastic Weight Consolidation (EWC)** — Kirkpatrick et al., 2017. After learning task A, compute the Fisher Information Matrix to identify which parameters were most important. When learning task B, add a penalty that prevents large changes to those important parameters. The penalty is:

```
L_ewc = lambda * sum(F_i * (theta_i - theta_i*)^2)
```

where F_i is the Fisher information for parameter i and theta_i* is the optimal value from task A.

**Online EWC** extends this to sequential tasks by accumulating Fisher information across all previous tasks.

**Progressive Neural Networks** — Rusu et al., 2016. Freeze the network after each task and add a new column of parameters. Avoids forgetting entirely but grows linearly with tasks.

**Replay-based methods** — Store a subset of old training data and mix it with new data during updates. Simple but effective. The key question is what to store (prioritization) and how much (buffer size).

### Forgetting in the LoRA Setting

The LoRA setting introduces unique dynamics because the trainable parameter space is extremely constrained.

**O-LoRA** (Wang et al., 2023) — Attempts to use orthogonal subspaces for different tasks, so updating one doesn't interfere with another. Each task gets an orthogonal projection of the LoRA parameter space.

**AM-LoRA** (2024) — Proved mathematically that orthogonality in LoRA parameter space is insufficient to prevent forgetting in nonlinear networks. The nonlinear activations mean that orthogonal weight updates can still produce interfering feature representations. This is a significant negative result that rules out a clean geometric solution.

**I-LoRA** — Uses a dual-memory system with replay. Still shows approximately 19% forgetting rate, suggesting the problem remains substantially unsolved.

**Online-LoRA** (WACV 2025) — Continual learning framework for LoRA that combines experience replay with adaptive rank allocation. Closest existing work to our adaptive rank experiment.

### Scaling Laws for Forgetting

**Kalajdzievski (2024)** showed that forgetting follows a power law: as you train on more data for a new task, forgetting of the old task increases as a power function. Critical finding: this relationship cannot be circumvented by early stopping or learning rate adjustment. The power law appears to be fundamental to gradient-based optimization in finite-capacity networks.

**Implication for Looper:** This suggests there may be an inherent ceiling on how many skills a fixed-rank adapter can retain across sessions. Our experiments should characterize where this ceiling is, not just try to push past it.

---

## 3. The Existing Knowledge Layer

The skill layer that Looper proposes sits on top of — not instead of — the existing knowledge infrastructure. Understanding what already exists is important because Looper needs to add value beyond what these systems already provide.

### RAG (Retrieval-Augmented Generation)

The dominant paradigm for giving LLMs access to external knowledge. Store documents as embeddings, retrieve the most relevant chunks at query time, inject them into context. This is a **knowledge** tool — it gives the agent information it can reference. It doesn't teach the agent how to use that information efficiently.

### Long Context Windows

Context windows have expanded from 4K to 10M+ tokens in three years. Google's Gemini 1.5 supports 10M tokens; Claude supports 200K; many open models support 128K+. **Many-Shot ICL** (Agarwal et al., 2024) showed that stuffing hundreds of examples into the prompt achieves performance that previously required fine-tuning. Long context is powerful for knowledge. But even with unlimited context, the agent doesn't develop skills — it's given more to reference, not made more fluent.

### Agent Memory Systems (All Knowledge-Layer)

**Reflexion** (Shinn et al., 2023) — Agents reflect on failures and store verbal self-critiques in a memory buffer injected into future prompts. Context-based. This is our primary comparison baseline.

**MemGPT** (Packer et al., 2023) — OS-inspired virtual memory for LLM context. Extends effective context indefinitely but all knowledge remains as text.

**ExpeL** (Zhao et al., 2023) — Extracts trajectory insights as natural language rules, retrieved into future prompts. Similar to Looper's synthesis step but keeps everything in the knowledge layer.

**Voyager** (Wang et al., 2023) — Minecraft agent that writes reusable code functions stored in a skill library. The closest analog to what Looper does, but implemented as code retrieval rather than weight updates. Voyager builds a library of skills as text; Looper trains skills into the model.

**OpenClaw's built-in memory** — File-based markdown memory with hybrid vector + BM25 search. MEMORY.md for curated long-term knowledge, daily logs for recent context. This is the knowledge layer that Looper's skill layer sits on top of.

### The Gap

All of these systems operate on the knowledge axis. They give the agent more information to reference. None of them change the model itself. The agent with perfect knowledge retrieval is still the same Day 1 employee — just one with better notes. Looper tests whether adding a skill layer on top of the knowledge layer produces a measurably more capable agent.

---

## 4. Synthetic Data for Post-Training

Converting experience into training data is an active research area.

**SWE-Gym** (Pan et al., 2025) — Trains coding agents on their own trajectories. Demonstrates that agents can improve from self-generated experience. Closest existing work to Looper's synthesis pipeline, but focuses on general capability improvement rather than per-instance environmental skill.

**AutoRefine** (arXiv 2601.22758) — Converts agent trajectories into reusable expertise for continual refinement. The most directly relevant existing work to Looper's synthesis component. Key gap: doesn't address catastrophic forgetting or per-instance adaptation.

**Context Distillation** — The general technique of converting in-context knowledge into weight updates. Used extensively in RLHF pipelines where human feedback is distilled into reward models and then into policy weights.

**Self-Play / Self-Improvement** — Constitutional AI, self-instruct, and similar approaches where models generate their own training data. The risk is feedback loops where errors are amplified. Our synthesis pipeline faces the same risk if the agent synthesizes training data from flawed trajectories.

### Data Quality as the Bottleneck

A critical insight: the hardest part of the Looper pipeline is not the LoRA training (well-understood) or the serving (solved by S-LoRA/vLLM). It's the synthetic data generation step.

Agent trajectories are noisy. They contain false starts, tool errors, backtracking, and verbose observations. The synthesizer must:
- Correctly attribute failures to the right causes (was approach A wrong, or did it fail due to a transient error?)
- Identify which strategies are genuinely good vs. which happened to work by luck
- Distinguish the agent's correct reasoning from its confabulations
- Extract generalizable environmental skills from specific instances

Bad synthesis → bad training data → bad adapter → the agent learns bad habits that are hard to detect and fix (unlike a bad document in the knowledge layer, which is easy to find and delete).

---

## 5. Multi-LoRA Serving Infrastructure

The "every instance has different weights" problem has been solved at the infrastructure level.

**S-LoRA** (Sheng et al., 2024) — Scalable serving of thousands of LoRA adapters over a single shared base model. Key innovations: unified paging for adapter weights in GPU/CPU memory, a tensor parallelism strategy that partitions adapters across GPUs, and a custom CUDA kernel for batched LoRA computations. Can serve up to 2,000 adapters simultaneously on a single machine.

**Punica** — Similar multi-tenant LoRA serving with a focus on the CUDA kernel for efficient batched LoRA matrix multiplications.

**vLLM** — The most widely used open-source inference engine. Supports multi-LoRA serving natively since v0.4, with adapter hot-swapping at request time.

**Commercial options:** Fireworks AI, Together AI, and Anyscale all offer multi-LoRA serving as a managed service.

**Storage math:** A rank-16 LoRA adapter for a 7B model is ~30-50MB. 10,000 adapters = 500GB, costing about $10/month on cloud object storage. This is negligible.

**Implication for Looper:** The serving infrastructure is not a research problem. We use vLLM for self-hosted experiments and can deploy to Fireworks or similar for production. The per-adapter overhead is small enough that per-project or even per-user skill adapters are economically viable.

---

## 6. Memory-Augmented Architectures

Some recent work explores architectural alternatives to LoRA for encoding memory.

**Titans: Learning to Memorize at Test Time** (Google, 2025) — Introduces a neural long-term memory module that learns to memorize at test time. The module is a small neural network that gets updated during inference (not training) to store relevant context. Novel approach that blurs the line between context and weights.

**Memory Layers at Scale** (Meta, 2024) — Replaces some transformer layers with explicit key-value memory layers containing up to 128B parameters. The memory layers act like a learned database within the model. Shows promise for factual retrieval without full model updates.

**MemoryLLM** (ICML 2024) — Self-updatable memory pools that the model can read from and write to during inference. Another architectural approach to persistent memory.

**Doc-to-LoRA** (Sakana AI, 2026) — A hypernetwork that converts a document directly into LoRA adapter weights in sub-second time, without any gradient-based training. Demonstrated on Gemma-2-2B, achieving 83.5% of in-context performance. If this scales to larger models and higher accuracy, it could make per-session consolidation instantaneous. Currently limited by: small model (2B), accuracy gap (17% below in-context), and no evidence of scaling to 7B+.

**Implication for Looper:** These architectural approaches are fascinating but require training from scratch or modifying the base model. Looper deliberately targets existing pre-trained models with standard LoRA — no architecture changes, no custom pre-training. It's a layer you add on top, not a new model you build from scratch.

---

## 7. Key Papers Reference List

### Must-Read (Directly Relevant)

| Paper | Year | Key Finding |
|-------|------|-------------|
| LoRA (Hu et al.) | 2021 | Low-rank adaptation achieves near full fine-tuning performance |
| LoRA Learns Less and Forgets Less (Biderman et al.) | 2024 | LoRA is better for behavioral adaptation than factual learning; forgetting tradeoff |
| S-LoRA (Sheng et al.) | 2024 | Multi-tenant LoRA serving at scale — 2000+ adapters per machine |
| Doc-to-LoRA (Sakana AI) | 2026 | Hypernetwork converts documents to LoRA weights in sub-second time |
| SWE-Gym (Pan et al.) | 2025 | Training coding agents on their own trajectories |
| Reflexion (Shinn et al.) | 2023 | Context-based agent self-reflection memory (our primary baseline) |
| SWE-Bench-CL (Joshi et al.) | 2025 | Continual learning benchmark for coding agents — 273 tasks, 8 repos, chronological ordering |

### Anti-Forgetting Strategies

| Paper | Year | Key Finding |
|-------|------|-------------|
| EWC (Kirkpatrick et al.) | 2017 | Fisher information penalty prevents overwriting important weights |
| Online-LoRA (WACV) | 2025 | Continual learning framework combining replay with adaptive rank |
| Scaling Laws for Forgetting (Kalajdzievski) | 2024 | Forgetting follows an irreducible power law |
| O-LoRA (Wang et al.) | 2023 | Orthogonal subspaces for task separation in LoRA |
| AM-LoRA | 2024 | Proves orthogonality insufficient in nonlinear networks |
| Overcoming Catastrophic Forgetting (arXiv:2401.14448) | 2024 | EWC + replay combinations for continual fine-tuning |

### Context Engineering (Comparison Baselines)

| Paper | Year | Key Finding |
|-------|------|-------------|
| Lost in the Middle (Liu et al.) | 2023 | LLMs attend poorly to mid-context information |
| Many-Shot ICL (Agarwal et al.) | 2024 | Hundreds of in-context examples approach fine-tuning performance |
| MemGPT (Packer et al.) | 2023 | OS-inspired virtual memory management for LLM context |
| ExpeL (Zhao et al.) | 2023 | Extract and store trajectory insights as retrievable rules |
| Voyager (Wang et al.) | 2023 | Skill library as procedural memory via code retrieval |

### Agent Self-Improvement

| Paper | Year | Key Finding |
|-------|------|-------------|
| AutoRefine (arXiv 2601.22758) | 2025 | Converting agent trajectories to reusable expertise |
| Self-Instruct (Wang et al.) | 2023 | Models generating their own instruction-tuning data |
| Constitutional AI (Bai et al.) | 2022 | Self-supervised alignment through self-critique |

### Architecture Alternatives

| Paper | Year | Key Finding |
|-------|------|-------------|
| Titans (Google) | 2025 | Neural memory modules that learn at test time |
| Memory Layers at Scale (Meta) | 2024 | 128B parameter explicit memory layers in transformers |
| MemoryLLM (ICML) | 2024 | Self-updatable memory pools during inference |
