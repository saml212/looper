Review recent changes against project standards.

Usage: /review

Steps:

**1. Load Context**
- Read `CLAUDE.md` for coding rules.
- Read `docs/development_process.md` for workflow expectations.

**2. Review the Changes**
- Run `git diff` (or `git diff HEAD~1` for last commit).
- Evaluate:

**Scope:**
- Does this change do ONE thing? Flag scope creep.
- Are there unrelated changes?

**Code Quality:**
- Is the code minimal? Could anything be removed?
- Is it simple? Pydantic models clean? No unnecessary abstractions?
- Does it follow existing patterns in the codebase?
- No dead code, unused imports, or commented-out code?

**Tests:**
- Are there tests for new code?
- Do tests actually test the right things?
- Are tests concise? No redundant setup or assertions?
- Do tests use fixtures and mock data appropriately?

**End-to-End:**
- Can the change be verified with a smoke test?
- Does input → processing → output make sense?

**Undocumented Learnings:**
- Were design decisions made that aren't recorded?
- Were there gotchas that should go in LEARNINGS.md?

**3. Report**
- **Looks good**: Well done items
- **Issues**: Must fix before merging
- **Suggestions**: Optional (clearly marked)
- **Doc updates needed**: Learnings to codify

Be direct. No padding.
