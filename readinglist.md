# Paper Reading List: LLM Memory via Per-Instance Weight Updates

An annotated bibliography organized as a learning curriculum. Papers are grouped by topic and ordered within each section from foundational to frontier. Priority ratings indicate reading order for the specific project of building a per-instance LoRA memory system for agents.

**Priority key:**
- ★★★ = Read before writing any code. Essential foundation.
- ★★ = Read during Phase 1 (building the baseline pipeline).
- ★ = Read during Phase 2–3 (forgetting experiments and synthesis research).
- ☆ = Read for depth/context when you hit specific problems.

---

## Section 1: How Transformers and Post-Training Work

*Read this first if you want to understand what post-training actually does to the weights mechanistically.*

### 1.1 — Attention Is All You Need
- **Authors:** Vaswani et al. (Google Brain)
- **Venue:** NeurIPS 2017
- **Link:** https://arxiv.org/abs/1706.03762
- **Priority:** ★★★
- **Why read it:** The foundational architecture. You need to understand the transformer block (self-attention + FFN), residual connections, and layer normalization to reason about where LoRA adapters attach and why certain layers matter more than others. If you already understand transformers well, skim sections 3.1–3.3.

### 1.2 — Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
- **Authors:** Ouyang et al. (OpenAI)
- **Venue:** NeurIPS 2022
- **Link:** https://arxiv.org/abs/2203.02155
- **Priority:** ★★★
- **Why read it:** The paper that established the RLHF post-training paradigm. Explains the three-stage pipeline: supervised fine-tuning (SFT), reward model training, and PPO optimization. You need this to understand what "post-training" means mechanically — it's not just one thing, it's multiple distinct phases that modify weights differently. Pay attention to Section 3 (methods) and the discussion of alignment tax.

### 1.3 — Direct Preference Optimization: Your Language Model Is Secretly a Reward Model (DPO)
- **Authors:** Rafailov et al. (Stanford)
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2305.18290
- **Priority:** ★★
- **Why read it:** DPO replaced PPO as the dominant alignment technique. It eliminates the reward model entirely and directly optimizes on preference pairs. Crucial for your project because DPO preference pairs from agent trajectories (successful approach vs. failed approach) are one of the most promising synthetic data formats. Read the derivation in Section 4 to understand why it works.

### 1.4 — A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity
- **Authors:** Lee et al. (Anthropic)
- **Venue:** ICML 2024
- **Link:** https://arxiv.org/abs/2401.01967
- **Priority:** ★★
- **Why read it:** Shows what DPO actually does to model weights at a mechanistic level. Key finding: alignment creates a distributed offset across layers rather than localized changes, and toxic capabilities are suppressed but not removed. This matters for your project because it tells you what kinds of changes LoRA can and can't make — small distributed shifts (behavior) vs. large localized changes (factual knowledge).

### 1.5 — Scaling Laws for Neural Language Models
- **Authors:** Kaplan et al. (OpenAI)
- **Venue:** arXiv 2020
- **Link:** https://arxiv.org/abs/2001.08361
- **Priority:** ☆
- **Why read it:** Background on how model size, data, and compute relate. Useful context for understanding why smaller LoRA adapters face fundamental capacity limits. Skim if you're already familiar with scaling laws.

---

## Section 2: Parameter-Efficient Fine-Tuning (PEFT)

*The technical foundation for everything you'll build. Read all ★★★ papers here before starting experiments.*

### 2.1 — LoRA: Low-Rank Adaptation of Large Language Models
- **Authors:** Hu et al. (Microsoft)
- **Venue:** ICLR 2022
- **Link:** https://arxiv.org/abs/2106.09685
- **Priority:** ★★★
- **Why read it:** The foundational paper for your entire project. Introduces the W' = W + BA low-rank decomposition, demonstrates it matches full fine-tuning on GPT-3 175B, and provides the intrinsic rank hypothesis that motivates the approach. Read every section. Pay special attention to Section 4.2 (which weight matrices to adapt — their recommendation of Q and V only is now known to be suboptimal) and Section 7 (understanding the low-rank updates).

### 2.2 — QLoRA: Efficient Finetuning of Quantized LLMs
- **Authors:** Dettmers et al. (UW)
- **Venue:** NeurIPS 2023 (Oral)
- **Link:** https://arxiv.org/abs/2305.14314
- **Priority:** ★★★
- **Why read it:** This is what you'll actually use for training. QLoRA combines 4-bit quantization of the base model with 16-bit LoRA adapters, enabling fine-tuning of 65B models on a single 48GB GPU. The Guanaco model trained with QLoRA reached 99.3% of ChatGPT performance. Read Section 3 (methods) and Section 4 (experiments) carefully.

### 2.3 — DoRA: Weight-Decomposed Low-Rank Adaptation
- **Authors:** Liu et al. (NVIDIA)
- **Venue:** ICML 2024 (Oral)
- **Link:** https://arxiv.org/abs/2402.09353
- **Priority:** ★★
- **Why read it:** Decomposes pretrained weights into magnitude and direction components, training them separately. Consistently outperforms LoRA across tasks. You may want to use DoRA instead of vanilla LoRA as your adapter method. Read the analysis of how LoRA and full fine-tuning differ in their magnitude/direction update patterns (Section 3).

### 2.4 — LoRA+: Efficient Low Rank Adaptation of Large Models
- **Authors:** Hayou et al.
- **Venue:** ICML 2024
- **Link:** https://arxiv.org/abs/2402.12354
- **Priority:** ★
- **Why read it:** Shows that using different learning rates for the A and B matrices improves LoRA by 1–2% and up to 2× speedup. Simple to implement, worth incorporating. Useful theory connecting LoRA to infinite-width neural network analysis.

### 2.5 — LoRA Learns Less and Forgets Less
- **Authors:** Biderman et al. (EleutherAI/Databricks)
- **Venue:** TMLR 2024
- **Link:** https://arxiv.org/abs/2405.09673
- **Priority:** ★★★
- **Why read it:** THE most important paper for your project. Establishes that LoRA's learning capacity is fundamentally limited compared to full fine-tuning (weight perturbations have effective rank 10–100× lower), but it also forgets less. This is the stability-plasticity tradeoff at the heart of your research. The paper shows LoRA is good at "changing tone and behavior in the same domain" but poor at learning new factual knowledge. This directly informs what you can and can't expect from per-instance LoRA memory. Read the entire paper.

### 2.6 — LoRA Without Regret
- **Authors:** Thinking Machines Lab (2025)
- **Venue:** arXiv 2025
- **Link:** https://thinkingmachines.ai/blog/lora/
- **Priority:** ★★
- **Why read it:** The most comprehensive recent analysis of LoRA best practices. Key findings: (1) MLP-only LoRA outperforms attention-only LoRA at the same parameter count; (2) best practice is to apply LoRA to all linear layers; (3) RL with LoRA matches full fine-tuning even at rank 1 because RL absorbs ~1000× less information per token than SFT. This last finding is critical — it suggests behavioral adaptation via RL needs far less adapter capacity than factual memorization via SFT.

### 2.7 — LISA: Layerwise Importance Sampled AdamW
- **Authors:** Pan et al.
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2403.17919
- **Priority:** ★
- **Why read it:** Reveals that LoRA's layerwise weight norms are uncommonly skewed — the bottom embedding layer and top head layer absorb the majority of updates, while middle layers barely change. LISA exploits this by always updating embedding + head + 2 random middle layers, outperforming LoRA by 10–35% on MT-Bench. Directly relevant to your adaptive rank allocation experiment.

### 2.8 — Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey
- **Authors:** Han et al.
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2403.14608
- **Priority:** ★★
- **Why read it:** Survey covering the full PEFT landscape: adapters, prompt tuning, prefix tuning, LoRA variants, (IA)³, VeRA, ReFT, and more. Read this to understand the alternatives to LoRA and whether any might be better suited for memory encoding. Focus on the comparison tables and the taxonomy.

### 2.9 — The Expressive Power of Low-Rank Adaptation
- **Authors:** Zeng & Lee
- **Venue:** ICLR 2024
- **Link:** https://arxiv.org/abs/2310.17513
- **Priority:** ★
- **Why read it:** Theoretical analysis of what LoRA can and cannot represent. Proves that rank-r LoRA can approximate any model achievable by full fine-tuning when r is large enough, but characterizes the approximation error as a function of rank. Important for understanding the theoretical ceiling of your approach.

---

## Section 3: Multi-LoRA Serving Infrastructure

*Read these to understand how per-user adapters get served in production.*

### 3.1 — S-LoRA: Serving Thousands of Concurrent LoRA Adapters
- **Authors:** Sheng et al. (UC Berkeley)
- **Venue:** MLSys 2024
- **Link:** https://arxiv.org/abs/2311.03285
- **Priority:** ★★★
- **Why read it:** The key infrastructure paper. Demonstrates serving 2,000+ concurrent LoRA adapters on a single GPU with unified memory paging and custom CUDA kernels. This proves the "every user gets their own adapter" model is operationally feasible. Read Sections 3–4 (system design) and Section 5 (evaluation).

### 3.2 — Punica: Multi-Tenant LoRA Serving
- **Authors:** Chen et al.
- **Venue:** MLSys 2024
- **Link:** https://arxiv.org/abs/2310.18547
- **Priority:** ★★
- **Why read it:** Introduces the SGMV (Segmented Gather Matrix-Vector) kernel that enables batching requests across different LoRA adapters with only ~2ms overhead per token. Complementary to S-LoRA. Read if you want to understand the GPU kernel-level details of multi-adapter serving.

### 3.3 — vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Authors:** Kwon et al. (UC Berkeley)
- **Venue:** SOSP 2023
- **Link:** https://arxiv.org/abs/2309.06180
- **Priority:** ★★
- **Why read it:** vLLM is the inference engine you'll likely use. PagedAttention is the key innovation. The multi-LoRA support built on top of this is what S-LoRA and Punica enable. Read to understand the serving system you'll deploy on.

---

## Section 4: Catastrophic Forgetting and Continual Learning

*This is your core research area. Read all of these.*

### 4.1 — Overcoming Catastrophic Forgetting in Neural Networks (EWC)
- **Authors:** Kirkpatrick et al. (DeepMind)
- **Venue:** PNAS 2017
- **Link:** https://arxiv.org/abs/1612.00796
- **Priority:** ★★★
- **Why read it:** The foundational paper on Elastic Weight Consolidation. Introduces the Fisher Information Matrix approach to identifying important parameters and penalizing changes to them. This is the basis for your EWC-LoRA experiment. The biological analogy to synaptic consolidation is directly relevant to your memory consolidation framing. Read sections 1–3 carefully.

### 4.2 — Scaling Laws for Forgetting When Fine-Tuning Large Language Models
- **Authors:** Kalajdzievski
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2401.05605
- **Priority:** ★★★
- **Why read it:** Establishes the quantitative relationship between fine-tuning and forgetting. Key finding: forgetting follows a shifted power law in both parameter count and training steps, and there is a strong inverse linear relationship between fine-tuning performance and forgetting. This means forgetting cannot be avoided through early stopping or parameter count adjustment alone — fundamental result for your project.

### 4.3 — Continual Learning of Large Language Models: A Comprehensive Survey
- **Authors:** Shi et al.
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2404.16789
- **Priority:** ★★
- **Why read it:** Comprehensive survey of all continual learning approaches applied to LLMs: replay-based, regularization-based, architecture-based, and representation-based methods. Provides taxonomy and comparison that will help you position your work. Read the taxonomy (Section 3) and the section on PEFT-based continual learning (Section 4).

### 4.4 — Parameter-Efficient Continual Fine-Tuning: A Survey
- **Authors:** Various
- **Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2504.13822
- **Priority:** ★★
- **Why read it:** Specifically covers the intersection of PEFT and continual learning — exactly your research area. More focused than 4.3 on the specific setting you're working in.

### 4.5 — O-LoRA: Orthogonal Low-Rank Adaptation of Large Language Models
- **Authors:** Wang et al.
- **Venue:** arXiv 2023
- **Link:** https://arxiv.org/abs/2312.01898
- **Priority:** ★★
- **Why read it:** Proposes constraining new LoRA updates to orthogonal subspaces to prevent interference with previous tasks. Important baseline for your forgetting experiments. However, AM-LoRA (2024) later proved that orthogonality alone is insufficient in nonlinear models — read both together.

### 4.6 — Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning
- **Authors:** Various
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2402.18865
- **Priority:** ★★
- **Why read it:** Directly analyzes catastrophic forgetting specifically in PEFT methods (LoRA, adapters, prompt tuning). Proposes mitigation strategies. This is the closest existing work to your core research question.

### 4.7 — Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation
- **Authors:** Various
- **Venue:** WACV 2025
- **Link:** https://arxiv.org/abs/2411.05663
- **Priority:** ★★★
- **Why read it:** Directly relevant — proposes continual learning with LoRA that monitors loss plateaus to detect distribution shifts and introduces new parameters accordingly. This is one of the most direct precedents for your adaptive rank allocation experiment. Read carefully and compare their approach to yours.

### 4.8 — How Much is Too Much? Exploring LoRA Rank Trade-offs for Retaining Knowledge and Domain Robustness
- **Authors:** Various
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2512.15634
- **Priority:** ★
- **Why read it:** Empirically explores the relationship between LoRA rank and knowledge retention vs. new domain adaptation. Higher rank = more learning capacity but more forgetting. Directly informs your adaptive rank allocation experiment.

---

## Section 5: Agent Self-Improvement and Experience Learning

*Papers on training agents from their own trajectories — the data generation side of your pipeline.*

### 5.1 — STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning
- **Authors:** Zelikman et al. (Stanford)
- **Venue:** NeurIPS 2022
- **Link:** https://arxiv.org/abs/2203.14465
- **Priority:** ★★
- **Why read it:** Established the paradigm of generating chain-of-thought rationales, fine-tuning on correct ones, and iterating. Achieved +35.9% on CommonsenseQA. Directly relevant to your synthetic data generation step — you're doing essentially the same thing but with agent trajectories instead of reasoning chains.

### 5.2 — Self-Instruct: Aligning Language Models with Self-Generated Instructions
- **Authors:** Wang et al.
- **Venue:** ACL 2023
- **Link:** https://arxiv.org/abs/2212.10560
- **Priority:** ★★
- **Why read it:** Shows LLMs can bootstrap their own instruction-tuning data. Achieved 33% improvement on Super-NaturalInstructions from self-generated data. Important precedent for your "self-synthesis" experiment where the agent generates its own training data.

### 5.3 — Reinforced Self-Training (ReST) for Language Modeling
- **Authors:** Gulcehre et al. (Google DeepMind)
- **Venue:** arXiv 2023
- **Link:** https://arxiv.org/abs/2308.08998
- **Priority:** ★★
- **Why read it:** Formalizes the generate-filter-train loop as growing-batch offline RL. Sample multiple outputs, rank with reward function, fine-tune on best, repeat. This is the closest algorithmic framework to your proposed pipeline.

### 5.4 — Reflexion: Language Agents with Verbal Reinforcement Learning
- **Authors:** Shinn et al.
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2303.11366
- **Priority:** ★★★
- **Why read it:** Critical comparison baseline. Reflexion achieves 91% on HumanEval through verbal self-reflection stored in episodic memory — with NO weight updates. The agent reflects on failures in natural language and stores those reflections for future reference. This is the context-based approach your weight-based approach needs to beat. If you can't outperform Reflexion on downstream tasks, your LoRA memory system doesn't justify its complexity.

### 5.5 — Voyager: An Open-Ended Embodied Agent with Large Language Models
- **Authors:** Wang et al. (NVIDIA)
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2305.16291
- **Priority:** ★★
- **Why read it:** Builds a lifelong learning Minecraft agent with an ever-growing skill library of executable code — no fine-tuning at all, purely context-based learning plus code storage. Another important comparison baseline showing that weight updates aren't necessary for agent self-improvement.

### 5.6 — ExpeL: LLM Agents Are Experiential Learners
- **Authors:** Zhao et al.
- **Venue:** AAAI 2024
- **Link:** https://arxiv.org/abs/2308.10144
- **Priority:** ★★
- **Why read it:** Extracts generalizable insights from both successful and failed trajectories as natural language rules, then uses them in-context. Directly comparable to your approach but without weight updates. The "experience extraction" step is essentially what your synthetic data generator needs to do.

### 5.7 — AutoRefine: From Trajectories to Reusable Expertise for Continual LLM Agent Refinement
- **Authors:** Various
- **Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2601.22758
- **Priority:** ★★★
- **Why read it:** Closest paper to your full pipeline. Converts agent trajectories into reusable expertise for continual refinement. Read this paper very carefully — it may be the most direct prior work to your project. Understand what they do, what they don't do, and where you can improve.

### 5.8 — SWE-Gym: An Environment for Training Software Engineering Agents
- **Authors:** Pan et al.
- **Venue:** ICML 2025
- **Link:** https://arxiv.org/abs/2410.06992
- **Priority:** ★★
- **Why read it:** First environment for training real-world SWE agents on their own trajectories. Fine-tuning on just 491 successful trajectories improved SWE-Bench by 12–14%. Shows that agent experience data is genuinely useful for training. You could use SWE-Gym as your test environment.

### 5.9 — SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
- **Authors:** Wei et al. (Meta/FAIR)
- **Venue:** NeurIPS 2025
- **Link:** https://arxiv.org/abs/2502.18449
- **Priority:** ★★
- **Why read it:** Scaled RL for software engineering, achieving 41% on SWE-Bench Verified (SOTA for open models <100B). Demonstrates that RL on agent trajectories significantly outperforms SFT on the same data. This suggests your pipeline should explore DPO/RL training formats, not just SFT.

---

## Section 6: Memory Architectures for LLMs

*Understand the alternatives to LoRA-based memory — both context-based and architecture-based.*

### 6.1 — MemGPT: Towards LLMs as Operating Systems
- **Authors:** Packer et al.
- **Venue:** arXiv 2023
- **Link:** https://arxiv.org/abs/2310.08560
- **Priority:** ★★★
- **Why read it:** The foundational paper for context-based LLM memory management. Treats the LLM like an OS with virtual memory — main context (RAM), retrieval memory (disk), and archival memory. Now productionized as Letta. This is the primary competitor/comparison system for your approach. Read carefully to understand what context engineering already handles well.

### 6.2 — Titans: Learning to Memorize at Test Time
- **Authors:** Google Research
- **Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2501.00663
- **Priority:** ★★
- **Why read it:** The most ambitious weight-based memory architecture. Introduces a learnable long-term memory module that updates its own parameters at test time using a "surprise metric." Outperformed GPT-4 on BABILong, scaling to 2M+ tokens. This is the architectural approach to the same problem you're tackling with LoRA — read to understand the alternative paradigm.

### 6.3 — Memory Layers at Scale
- **Authors:** Meta/FAIR
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2412.09764
- **Priority:** ★
- **Why read it:** Replaces feed-forward layers with trainable key-value lookup memories scaled to 128 billion parameters. Outperforms dense models with 2× the compute. Demonstrates that massive dedicated memory capacity in the architecture can substitute for model scale. A different approach to the problem.

### 6.4 — MemoryLLM: Towards Self-Updatable Large Language Models
- **Authors:** Wang et al.
- **Venue:** ICML 2024
- **Link:** https://arxiv.org/abs/2402.04624
- **Priority:** ★★
- **Why read it:** Adds a fixed-size learnable memory pool (~1B parameters) to Llama2-7B with a self-update mechanism. Long-term retention persists after ~10⁶ updates. The closest architectural approach to what you're trying to achieve with LoRA adapters. Compare their retention characteristics to what you observe with LoRA updates.

### 6.5 — HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models
- **Authors:** Gutierrez et al.
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2405.14831
- **Priority:** ★★
- **Why read it:** Mimics hippocampal indexing theory for LLM memory — uses a knowledge graph as a schemaless indexing structure, outperforming standard RAG by up to 20% on multi-hop QA. Important because it shows that even within context-based approaches, architecture matters enormously. Your RAG comparison baseline should use HippoRAG or similar structured retrieval, not naive vector search.

### 6.6 — Towards Large Language Models with Human-like Episodic Memory
- **Authors:** Various
- **Venue:** Trends in Cognitive Sciences, 2025
- **Link:** https://doi.org/10.1016/j.tics.2025.07.007
- **Priority:** ★
- **Why read it:** Position paper arguing that current LLMs have semantic memory (general knowledge in weights) but lack true episodic memory (binding of specific experiences with contextual features). Provides the theoretical framework from cognitive science for understanding why your project matters. The distinction between semantic and episodic memory maps directly to the distinction between what LoRA encodes well (semantic/procedural) and what it doesn't (episodic).

### 6.7 — Memory-Augmented Transformers: A Systematic Review from Neuroscience Principles to Technical Solutions
- **Authors:** Various
- **Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2508.10824
- **Priority:** ★
- **Why read it:** Comprehensive survey mapping neuroscience memory concepts to transformer architectures. Useful reference for understanding the full landscape of memory-augmented approaches and positioning your work within it.

---

## Section 7: Context Engineering and the Competition

*Understand what you're competing against. Context-based approaches are the incumbent.*

### 7.1 — Effective Context Engineering for AI Agents
- **Authors:** Anthropic
- **Venue:** Blog post, 2025
- **Link:** https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- **Priority:** ★★
- **Why read it:** Anthropic's guide on the dominant paradigm. Context engineering has become the standard approach precisely because it works reliably. Your project needs to demonstrate advantages over this.

### 7.2 — Many-Shot In-Context Learning
- **Authors:** Agarwal et al. (Google DeepMind)
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2404.11018
- **Priority:** ★★
- **Why read it:** Shows that inserting hundreds or thousands of examples into the prompt achieves performance that previously required fine-tuning. This is the strongest argument against your approach — if you can just stuff everything into context, why bother with weight updates? Read to understand the limits (where performance saturates, cost scaling, latency).

### 7.3 — Lost in the Middle: How Language Models Use Long Contexts
- **Authors:** Liu et al. (Stanford)
- **Venue:** TACL 2024
- **Link:** https://arxiv.org/abs/2307.03172
- **Priority:** ★★
- **Why read it:** Demonstrates that LLMs have a U-shaped attention pattern — they attend well to the beginning and end of context but poorly to the middle. This is the key weakness of context-based memory and a potential advantage for weight-based approaches. If important memories end up in the middle of a long context, they may be effectively lost. Weight-based encoding avoids this entirely.

---

## Section 8: Doc-to-LoRA and Hypernetwork Approaches

*The most directly relevant recent breakthrough for your pipeline.*

### 8.1 — Doc-to-LoRA: Instant LLM Updates with Doc-to-LoRA and Text-to-LoRA
- **Authors:** Sakana AI
- **Venue:** arXiv 2026
- **Link:** https://pub.sakana.ai/doc-to-lora/
- **Priority:** ★★★
- **Why read it:** The most exciting paper for your project. Uses a 309M-parameter Perceiver-based hypernetwork that generates a rank-8 LoRA adapter from any document in a single sub-second forward pass — no gradient updates. Achieves 83.5% of full in-context performance on SQuAD with constant ~50MB memory regardless of document length. If you could apply this to agent trajectory summaries, you'd bypass the entire gradient-based training bottleneck. Read the full paper and the companion Text-to-LoRA paper (ICML 2025).

### 8.2 — HyperNetworks
- **Authors:** Ha et al. (Google Brain)
- **Venue:** ICLR 2017
- **Link:** https://arxiv.org/abs/1609.09106
- **Priority:** ★
- **Why read it:** The original hypernetwork paper — a network that generates weights for another network. Doc-to-LoRA is a hypernetwork. If you want to extend or modify the Doc-to-LoRA approach, understanding the original hypernetwork framework is important.

### 8.3 — LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
- **Authors:** Huang et al.
- **Venue:** arXiv 2023
- **Link:** https://arxiv.org/abs/2307.13269
- **Priority:** ★
- **Why read it:** Composes multiple pre-trained LoRA adapters via learned coefficients to handle new tasks. Relevant to your Mixture of LoRA Experts (MoLE) experiment — instead of training separate adapters per domain, you could compose existing task-specific adapters dynamically.

---

## Section 9: Synthetic Data Generation

*The data quality side of your pipeline.*

### 9.1 — Textbooks Are All You Need (Phi-1)
- **Authors:** Gunasekar et al. (Microsoft)
- **Venue:** arXiv 2023
- **Link:** https://arxiv.org/abs/2306.11644
- **Priority:** ★★
- **Why read it:** Demonstrated that a 1.3B model trained on high-quality synthetic textbook data outperforms models 10× its size. The key insight: data quality matters far more than quantity. Directly relevant to your synthetic data generation step — a small number of high-quality synthesized pairs will outperform a large number of noisy ones.

### 9.2 — Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN)
- **Authors:** Chen et al.
- **Venue:** ICML 2024
- **Link:** https://arxiv.org/abs/2401.01335
- **Priority:** ★
- **Why read it:** Uses self-play to generate training data where the model plays against itself. Relevant to your self-synthesis experiment. The model generates responses, a distinguisher identifies which are LLM-generated vs. human, and the model trains on the distinction.

### 9.3 — WizardLM: Empowering Large Language Models to Follow Complex Instructions
- **Authors:** Xu et al. (Microsoft)
- **Venue:** arXiv 2023
- **Link:** https://arxiv.org/abs/2304.12244
- **Priority:** ★
- **Why read it:** Introduces Evol-Instruct, a method for evolving simple instructions into complex ones to generate high-quality training data. Could be applied to your trajectory synthesis step — evolve simple trajectory summaries into more nuanced training pairs.

---

## Section 10: Evaluation and Benchmarks

*Know how to measure what you're building.*

### 10.1 — LoCoMo: Long Context Multi-Turn Benchmark
- **Authors:** Various
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2402.07209
- **Priority:** ★★
- **Why read it:** Benchmark specifically for evaluating long-term memory in conversational AI. Tests memory retention across multi-turn conversations. Use this or adapt it for your evaluation framework.

### 10.2 — SWE-Bench: Can Language Models Resolve Real-World GitHub Issues?
- **Authors:** Jimenez et al. (Princeton)
- **Venue:** ICLR 2024
- **Link:** https://arxiv.org/abs/2310.06770
- **Priority:** ★★
- **Why read it:** The standard benchmark for evaluating coding agents. If you use a coding agent as your test bed, you'll want to evaluate on SWE-Bench to demonstrate that memory improves downstream performance.

---

## Suggested Reading Order

**Week 1 (Foundations):**
1. Skim 1.1 (Attention Is All You Need) — refresh on transformer architecture
2. Read 2.1 (LoRA) — the core technique
3. Read 2.2 (QLoRA) — what you'll use in practice
4. Read 2.5 (LoRA Learns Less and Forgets Less) — the fundamental limitation

**Week 2 (The Forgetting Problem):**
5. Read 4.1 (EWC) — the classic solution
6. Read 4.2 (Scaling Laws for Forgetting) — the quantitative landscape
7. Read 4.7 (Online-LoRA) — the closest prior work to your approach
8. Read 4.3 (Continual Learning Survey) — skim for breadth

**Week 3 (Agent Learning + Memory):**
9. Read 5.4 (Reflexion) — your primary comparison baseline
10. Read 5.7 (AutoRefine) — closest to your full pipeline
11. Read 6.1 (MemGPT) — the context-based competition
12. Read 8.1 (Doc-to-LoRA) — potential breakthrough for your pipeline

**Week 4 (Infrastructure + Depth):**
13. Read 3.1 (S-LoRA) — serving infrastructure
14. Read 1.2 (InstructGPT) — understand post-training mechanistically
15. Read 5.8 (SWE-Gym) — potential test environment
16. Read 7.3 (Lost in the Middle) — context weaknesses you can exploit

**Ongoing (as needed):**
- Read 2.6 (LoRA Without Regret) before configuring your LoRA hyperparameters
- Read 1.3 (DPO) before implementing preference-based training from trajectories
- Read 6.2 (Titans) and 6.4 (MemoryLLM) for architectural alternatives
- Read 9.1 (Phi-1) before designing your synthetic data quality experiments

---

## Quick Reference: arXiv IDs

For fast lookup, here are all papers with their arXiv identifiers:

| Paper | arXiv ID |
|-------|----------|
| Attention Is All You Need | 1706.03762 |
| InstructGPT | 2203.02155 |
| DPO | 2305.18290 |
| Mechanistic Understanding of DPO | 2401.01967 |
| Scaling Laws | 2001.08361 |
| LoRA | 2106.09685 |
| QLoRA | 2305.14314 |
| DoRA | 2402.09353 |
| LoRA+ | 2402.12354 |
| LoRA Learns Less and Forgets Less | 2405.09673 |
| LISA | 2403.17919 |
| PEFT Survey | 2403.14608 |
| Expressive Power of LoRA | 2310.17513 |
| S-LoRA | 2311.03285 |
| Punica | 2310.18547 |
| vLLM | 2309.06180 |
| EWC | 1612.00796 |
| Scaling Laws for Forgetting | 2401.05605 |
| Continual Learning Survey | 2404.16789 |
| PEFT Continual Learning Survey | 2504.13822 |
| O-LoRA | 2312.01898 |
| Catastrophic Forgetting in PEFT | 2402.18865 |
| Online-LoRA | 2411.05663 |
| LoRA Rank Tradeoffs | 2512.15634 |
| STaR | 2203.14465 |
| Self-Instruct | 2212.10560 |
| ReST | 2308.08998 |
| Reflexion | 2303.11366 |
| Voyager | 2305.16291 |
| ExpeL | 2308.10144 |
| AutoRefine | 2601.22758 |
| SWE-Gym | 2410.06992 |
| SWE-RL | 2502.18449 |
| MemGPT | 2310.08560 |
| Titans | 2501.00663 |
| Memory Layers at Scale | 2412.09764 |
| MemoryLLM | 2402.04624 |
| HippoRAG | 2405.14831 |
| Memory-Augmented Transformers Survey | 2508.10824 |
| Many-Shot ICL | 2404.11018 |
| Lost in the Middle | 2307.03172 |
| HyperNetworks | 1609.09106 |
| LoRAHub | 2307.13269 |
| Phi-1 | 2306.11644 |
| SPIN | 2401.01335 |
| WizardLM | 2304.12244 |
| LoCoMo | 2402.07209 |
| SWE-Bench | 2310.06770 |

