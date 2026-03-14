Launch deep research on a topic relevant to Looper development.

Usage: /research <topic>

Topic: $ARGUMENTS

Steps:
1. Read `CLAUDE.md` and relevant docs to understand what we already know.
2. Check `readinglist.md` — it has 100+ annotated papers. Don't duplicate research that's already documented.
3. Launch one or more research subagents to investigate the topic:
   - Search the web for current research (2025-2026 papers, implementations)
   - Look for production implementations and real-world experience
   - Find counter-arguments and failure modes
   - Check if anyone has solved the specific problem we're investigating
4. Synthesize findings:
   - What was found
   - How it relates to our existing findings in LEARNINGS.md
   - Options with tradeoffs
   - Specific recommendations
5. Present findings and WAIT for human decision before taking action.

Do NOT write code. Do NOT make implementation decisions. Research only.
