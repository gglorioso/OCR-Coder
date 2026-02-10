Session handoff - LIGHTWEIGHT MODE (minimal context usage).

## 1. Identify What Changed
Ask user: "What changed this session?" (they respond with 1-2 bullets)
Or infer from conversation if obvious.

## 2. Targeted Updates Only

### Always update:
- `Context/claude.md` - Quick Status section only (lines 8-36)
  - Update "Last Updated" date
  - Update "Current Phase" if changed
  - Update first 2 "Next Steps" items

### Update if phase/architecture changed:
- `Context/WORKSPACE_NOTES.md` - Next Actions section only (lines 260-275)
  - Check first item under "Next Actions" and update if different

### Update if plan/hyperparameters changed:
- `Context/PHASE2_PLAN.md` - Changelog section only (append 1 line to Section 15)

**Use targeted Edit calls - do NOT read full files before editing**

## 3. Emit Ultra-Brief Handoff

Return terse handoff (max 8 lines):

```
Session: [date]
Done: [1-line summary of what was completed]
Next: [1 immediate action]
Files: [comma-separated list]
Blocker: [if any, else "None"]
```
