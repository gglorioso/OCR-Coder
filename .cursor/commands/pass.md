**Purpose:** When the user types something like `/pass` the assistant should:
- Summarize the current session’s work, and
- Update the project context files so a fresh instance (Cursor or Claude) can quickly get up to speed.

---

## What the Assistant Should Do

When asked to run `/pass` (or equivalent):

1. **Summarize the session**
   - What we worked on and why (high level)
   - Key decisions and rationale
   - Files touched and what changed (briefly)

2. **Update context markdowns**
   - `Context/WORKSPACE_NOTES.md`
     - Update “Last updated” date
     - Mark relevant checklist items as done / in-progress
     - Add a short bullet list under “Next Actions” that reflects the new reality
   - `Context/PHASE2_PLAN.md`
     - If we materially changed Phase 2 execution (e.g., new data pipeline, new script), append a line to the Changelog (Section 15).
   - `Context/claude.md`
     - Update “Last Updated” date
     - Refresh “Quick Status” and “Next Steps” based on current work.

3. **Emit a copy-pasteable Handoff Block**

Return a markdown block in the chat that looks like:

```md
## Session Handoff

**Date:** 2026-02-10 23:15 (local)
**Branch:** [branch name or “unknown” if not available]

### Context
- [1–3 bullets: what we were working on and why]

### Progress Made
- [Completed item 1]
- [Completed item 2]

### Files Modified
`path/to/file.py` – [short description]
`Context/WORKSPACE_NOTES.md` – [short description]

### Key Decisions
- [Decision + rationale]

### Incomplete Work
- [What’s started but not finished, and why]

### Next Steps
1. [Immediate next action]
2. [Follow-up]
3. [Any validation / tests to run]

### Blockers / Risks
- [Any obstacles or open questions]
```

The goal is: **if a fresh instance only sees this block + the existing context files, it can continue seamlessly.**
