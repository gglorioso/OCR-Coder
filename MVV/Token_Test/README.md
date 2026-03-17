# Token_Test

## What was tested

How many DeepSeek-Coder-V2-Lite-Instruct tokens a typical 40-line code snippet
contains, using a random sample of 500 entries from the Phase 1.9a
`ground_truth.jsonl` manifest.

Each snippet is constructed exactly as the image renderer does: read 40 lines
from `anchor_line`, apply `expandtabs(4)` and truncate each line to 80
characters, then tokenize with `add_special_tokens=False`.

## Files

```
Token_Test/
  token_count.py        # tokenization script
  run_token_test.sh     # SLURM job (teaching partition, 16 GB RAM)
  results/
    token_stats.json    # summary statistics + histogram (output)
```

## Script

`token_count.py` loads the tokenizer from
`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, samples 500 entries
(seed=42) from the manifest, reads each source file from `Scraped Repos/`,
slices the 40-line window, and records the token count. Results are written
to `results/token_stats.json`.

Run via SLURM:
```bash
sbatch MVV/Token_Test/run_token_test.sh
```

## Results (n=500, seed=42, skipped=0)

| Stat   | Tokens |
|--------|-------:|
| Min    |     41 |
| Max    |  2,749 |
| Mean   |    322 |
| Median |    323 |
| Std    |    160 |
| P95    |    529 |

Distribution (bucket : count):

| Bucket   | Count |
|----------|------:|
| 0–100    |    24 |
| 100–200  |    48 |
| 200–500  |   394 |
| 500–1k   |    33 |
| 1k+      |     1 |

79% of snippets fall in the 200–500 token range.

## Interpretation

The median 40-line snippet costs **323 text tokens**. Our visual representation
of the same snippet uses **256 visual tokens** (one per SigLIP patch after
tiling). Visual encoding is therefore **~1.26x more token-efficient** at the
median and improves further toward the P95 (529 tokens → 256 visual, ~2x).

This confirms the original motivation: replacing text tokens with a fixed-size
visual embedding frees context budget, especially for larger files where a
single 40-line window already approaches 500+ tokens in text form. The
efficiency gain compounds when fitting many such windows into a single
LLM context (the target use case for SWE-bench file retrieval).
