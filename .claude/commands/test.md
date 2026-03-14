Run the test suite and report results.

Usage: /test [optional: specific test file or pattern]

Test filter: $ARGUMENTS

Steps:

1. If a specific test was requested: `pytest tests/<filter> -v`
   Otherwise: `pytest tests/ -v --tb=short`

2. Also run `ruff check looper/ tests/` for lint issues.

3. If tests fail:
   - Show failing test names and errors
   - Read failing test code to understand what's tested
   - Identify likely cause
   - Present options (don't auto-fix without asking)

4. If tests pass:
   - Report count (passed/failed/skipped)
   - Note any lint warnings

Keep output concise — summarize, don't dump raw pytest output.
