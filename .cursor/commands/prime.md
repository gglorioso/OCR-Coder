Bootstrap a fresh instance - LIGHTWEIGHT MODE (minimal context usage).

## 1. Read Essential Sections Only
Use codebase_search to find and read:
- `Context/claude.md` - Search for "## Quick Status" section and read through "## Key Context" section
- `Context/WORKSPACE_NOTES.md` - Search for "## Next Actions" section and read that entire section

## 2. Quick File Check (no full reads)
```bash
ls models/vision_encoder.pt 2>/dev/null && echo "✓ Vision encoder ready" || echo "✗ Vision encoder missing"
ls -d coder_vl/ 2>/dev/null && echo "✓ Implementation exists" || echo "✗ Implementation missing"
ls Data\ Crawling/output/manifests/*.jsonl 2>/dev/null | wc -l | xargs -I{} echo "✓ {} manifest files"
# Check git status (informational only)
git status --short 2>/dev/null | head -5 || echo "Note: Check git status manually"
```

## 3. Output (Brief)
Just say: **"Ready."** + current phase + immediate next step (max 2 lines total)
