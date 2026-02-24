Good framing — two separate tracks with different timelines and goals. Here's how I'd structure them:

Track 1: MICS paper (15 days) — Q-Former ablation

Implement Q-Former as a drop-in replacement for the MLP adapter, train one run, compare directly. The paper becomes an architectural ablation:

Connector	ROUGE-L	Sensitivity	Notes
MLP (2-layer)	0.31	0.34	current
Q-Former	TBD	TBD	proposed
Even if Q-Former doesn't fully fix things, the comparison is the contribution. This is ~3 days of implementation + 1 ROSIE run.

Track 2: Rosie competition — full pretraining

This is the right vehicle for the more ambitious work. The competition rewards heavy cluster use, which means multi-node distributed training is actually a feature here, not overhead.

The plan that makes sense for your existing infrastructure:

Stage 0 — Expand the alignment dataset. You already have scraped repos and a rendering pipeline. Before touching model architecture, generate 10-20× more code image/annotation pairs from the repos you already have. You're sitting on Django, Flask, FastAPI, Pandas source trees. Render more files, more chunks, more themes. Getting from 8K to 80-100K training pairs is feasible without any new data collection and would directly address the data volume gap.

Stage 1 — Visual alignment pretraining. Freeze the coder model entirely. Train only the connector (Q-Former or MLP) on the expanded dataset with a simple next-token prediction objective. This is what LLaVA Stage 1 does — the LLM "sees" projected visual tokens paired with correct text thousands of times until it learns to attend to them. Run this on a DGX node, potentially multi-GPU with DDP.

Stage 2 — LoRA instruction tuning. Same as Phase 2b but on a model whose weights have already been calibrated to visual tokens. Much smaller dataset needed at this stage.

Practical split:


Now → MICS deadline (15 days):
  - Text-only baseline (2 hrs)
  - Retrieval baseline (2 hrs)  
  - Q-Former implementation + training run (1 week)
  - Write paper

In parallel / after MICS:
  - Expand dataset (rendering pipeline already exists)
  - Stage 1 pretraining on Rosie (multi-GPU, multi-day)
  - Stage 2 fine-tuning
  - Rosie competition submission
The expanded dataset work can actually start now since it's just running the existing render/annotate pipeline on more source files — no model changes needed. Want to start by checking how many additional files you could render from the scraped repos?