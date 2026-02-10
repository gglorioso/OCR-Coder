Bootstrap a fresh instance - LIGHTWEIGHT MODE (minimal context usage).

## 1. Read Essential Sections Only
- `Context/claude.md` lines 1-75 (Quick Status + Key Context sections)
- `Context/WORKSPACE_NOTES.md` lines 260-316 (Next Actions section)

## 2. Quick File Check (no full reads)
```bash
ls models/vision_encoder.pt 2>/dev/null && echo "✓ Vision encoder ready" || echo "✗ Vision encoder missing"
ls -d coder_vl/ 2>/dev/null && echo "✓ Implementation exists" || echo "✗ Implementation missing"
ls Data\ Crawling/output/manifests/*.jsonl 2>/dev/null | wc -l | xargs -I{} echo "✓ {} manifest files"
```

## 3. Output (Brief)
Just say: **"Ready."** + current phase + immediate next step (max 2 lines total)
