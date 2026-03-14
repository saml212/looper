# The Problem: Agents Don't Learn Skills

## Every Session Is Day One

Modern AI agents operate in stateless sessions. Each time a session begins, the agent has the same general capabilities it always had — but zero learned skill in the specific environment it's operating in. It doesn't know which tools work best for which tasks in this project. It doesn't know the deployment patterns. It doesn't know that the last three times it tried approach A, approach B worked better. Every session, it figures it all out from scratch.

This is like hiring a contractor who's brilliant but has amnesia. They show up every morning with the same raw talent, but they never get faster. They never develop the instincts that come from doing the same kind of work in the same environment day after day.

## Skills vs. Knowledge

There's an important distinction that gets lost when people talk about "agent memory."

**Knowledge** is information. Facts. What's in the docs. "This project uses PostgreSQL." "The API is deployed on Cloud Run." "Tests are in the `tests/` directory." Knowledge can be looked up. You can write it down, retrieve it, paste it into a prompt. Context windows, RAG, project config files — these are all good tools for giving an agent knowledge.

**Skills** are different. Skills are how you use tools efficiently. How you navigate an environment. The instinct to check the logs first when a deployment fails, because you've seen that pattern before. The fluency that lets you go straight to the right file instead of searching. The behavioral pattern of running the linter before committing because you've been burned by CI failures.

A human employee on Day 1 can be given all the knowledge they need — hand them the docs, the wiki, the onboarding guide. But they still won't have the skills. Skills come from doing the work. From making mistakes and adjusting. From building up patterns through repetition until the right approach becomes instinctive.

**Agents today have access to unlimited knowledge but develop zero skills.**

Everything the industry does for agent memory — RAG, long context, memory files, system prompts — operates on the knowledge axis. And it works. But nobody is working on the skill axis. Nobody is building a system where the agent actually gets better at operating in its environment the longer it works there.

## How Humans Acquire Skills

Consider a software engineer working on a specific project:

**Week 1:** They have all the knowledge — docs, codebase access, team wiki. But they're slow. They search for things they'll later find instantly. They try deployment approaches that the team abandoned months ago. They run into errors that experienced team members would avoid reflexively. Their raw intelligence is the same as it will be on Week 12, but their environmental skill is near zero.

**Week 4:** They've internalized the project's rhythms. They know which files to check for which kinds of bugs. They've developed a muscle memory for the deployment flow. When something breaks, they have a mental decision tree of things to check — not because they memorized a checklist, but because they've seen these failure modes before. They're faster, not because they're smarter or have more information, but because they've developed **skills specific to this environment**.

**Week 12:** They're fluent. They navigate the codebase instinctively. They recognize patterns across systems ("this looks like the same race condition we had in the billing service"). Their tool usage is efficient — they reach for the right tool first, not after trying three others. This fluency isn't knowledge. It's the accumulated result of operating in the same environment repeatedly.

Every day is like a context window. At the end of the day, the specifics of what happened fade. But the skills — the patterns, the instincts, the fluency — those consolidate. The engineer wakes up the next morning without remembering every detail of yesterday, but they're measurably more skilled than they were the day before.

## The Missing Layer

The modern agent stack looks roughly like this:

```
┌─────────────────────────┐
│     Context Window      │  ← What's happening right now
├─────────────────────────┤
│  Memory / Retrieval     │  ← Knowledge from past sessions
├─────────────────────────┤
│     Base Model          │  ← General intelligence
└─────────────────────────┘
```

Knowledge lives in the memory/retrieval layer. General capability lives in the base model. But there's no layer for **learned skills** — the accumulated environmental fluency that should develop through experience.

Looper adds that layer:

```
┌─────────────────────────┐
│     Context Window      │  ← What's happening right now
├─────────────────────────┤
│  Memory / Retrieval     │  ← Knowledge from past sessions
├─────────────────────────┤
│     Skill Adapter       │  ← Learned environmental fluency (LoRA)
├─────────────────────────┤
│     Base Model          │  ← General intelligence
└─────────────────────────┘
```

The skill adapter is a thin LoRA layer trained on the agent's own experience. As the agent operates in an environment — using tools, navigating code, deploying, debugging — that experience is periodically consolidated into the adapter. Old context gets turned into trained skills.

This isn't replacing anything. The context window still handles the present. Memory/retrieval still handles knowledge. The base model still provides general intelligence. The skill adapter adds something new: the ability to get better at operating in a specific environment over time.

## What LoRA Is Good At (And What It Isn't)

This framing matters because LoRA adapters have specific strengths that align with skills, not knowledge.

Research shows that LoRA is excellent at encoding:
- Behavioral patterns (how to approach problems)
- Tool usage strategies (which tools to use when)
- Style and convention adherence (matching project norms)
- Domain-specific heuristics (debugging decision trees)

And poor at encoding:
- Precise facts ("the database password is xyz")
- Specific episodic details ("in session 47, the API returned a 403")
- Rapidly changing information

This maps perfectly onto the skills vs. knowledge distinction. LoRA is a skill encoder, not a knowledge store. Using it as a knowledge store would be fighting its strengths. Using it as a skill layer is working with the grain of the technology.

## The Consolidation Loop

The skill adapter doesn't update in real time. Like human skill acquisition, it works through a consolidation cycle:

1. **Experience** — The agent operates in its environment across multiple sessions, producing trajectories of actions, observations, successes, and failures
2. **Synthesis** — Periodically, those trajectories are processed to extract the environmental skills embedded in them: what tools worked, what approaches succeeded, what patterns emerged
3. **Training** — The extracted skills are consolidated into the LoRA adapter through efficient fine-tuning
4. **Application** — On the next session, the agent operates with the updated skill adapter, and the cycle repeats

Each cycle makes the agent slightly more fluent in its environment. Not by giving it more information (that's what the context window and memory layer are for) but by making it inherently better at operating — the same way sleeping on a problem makes a human better at it the next day, even without new information.

## What We're Testing

The core question is simple: **does this actually work?**

Specifically:
- Does an agent with a skill adapter complete tasks faster (fewer steps, fewer retries)?
- Does it make better first-attempt tool selections?
- Does it need less context to operate effectively in a familiar environment?
- Does skill accumulate over time, or does catastrophic forgetting erase old skills as new ones are learned?
- And critically: does the skill adapter add measurable value on top of what good memory/retrieval already provides?

If the answer to the last question is no — if good knowledge retrieval alone accounts for all the gains — then the skill layer doesn't justify its complexity. That's a legitimate finding. The goal is truth, not confirmation.

---

## What We Found (March 2026)

After 8 experiments over 10 days, the answer so far is **no** — LoRA skill consolidation does not work at current model scales (7B-32B) with available training data (12-31 resolved tasks).

Every LoRA training strategy produced zero or negative forward transfer. The adapted model was always the same or worse than the base model. Three compounding reasons:

1. **Cold-start problem.** The 7B base model resolves only 8% of SWE-Bench tasks. You can't bootstrap skill from a 92% failure rate. The minimum viable training set appears to be 100+ unique resolved tasks with diverse strategies.

2. **Format mismatch.** LoRA faithfully learns whatever format you train on. Q&A pairs produce a Q&A model. Diffs produce a diff model. Neither produces an agent that uses XML tool calls. Training data must exactly match the inference format.

3. **Overfitting at small scale.** When we finally got format-matched data (18 correct-format examples), the model memorized the surface pattern (grep -> read -> write -> done in 4 steps) instead of learning generalizable debugging strategies.

What *did* work was improving the agent framework itself: prompt engineering (few-shot examples, skip-verification rule), tool design (line-range reads, edit tool, code fence stripping), and loop detection. These changes tripled the resolve rate from 8% to 27%.

The framework fixes are, in a sense, the "skill layer" — they just live in the system prompt instead of the LoRA weights. This raises a deeper question: is the distinction between skills-in-weights and skills-in-prompt meaningful? Or is the system prompt simply the more effective encoding for agent behavioral patterns?

See [LEARNINGS.md](../LEARNINGS.md) for the full results.
