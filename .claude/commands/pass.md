Session handoff - LIGHTWEIGHT MODE (minimal context usage).

## 1. Identify What Changed
Ask user: "What changed this session?" (they respond with 1-2 bullets)
Or infer from conversation if obvious.

## 2. Targeted Updates Only

### Always update:
- `Context/claude.md` - Find "## Quick Status" section and update:
  - Update "Last Updated" date
  - Update "Current Phase" if changed
  - Update first 2 "Next Steps" items under "Current Step"

### Update if phase/architecture changed:
- `Context/WORKSPACE_NOTES.md` - Find "## Next Actions" section:
  - Check first item under "Next Actions" and update if different
  - Add new completed items at the top if needed

### Update if plan/hyperparameters changed:
- `Context/PHASE2_PLAN.md` - Find changelog section (search for "Changelog" or "Section 15"):
  - Append 1 line with date and brief summary

**Use codebase_search to locate sections, then use targeted Edit calls - do NOT read full files before editing**

## 3. Emit Ultra-Brief Handoff

Return terse handoff (max 8 lines):

```
Session: [date]
Done: [1-line summary of what was completed]
Next: [1 immediate action]
Files: [comma-separated list]
Blocker: [if any, else "None"]
```
