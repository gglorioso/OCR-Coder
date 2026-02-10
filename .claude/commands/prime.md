Bootstrap a fresh instance by reading project context files and understanding the current state.

## 1. Scan the Repo Structure
List top-level dirs/files to get bearings:
- `README.md`
- `Context/` (all files)
- `Data Crawling/` (especially `output/` and scripts)

## 2. Read Core Context Files
- `README.md` – high-level description of DeepSeek-Coder-VL and goals
- `Context/WORKSPACE_NOTES.md` – most recent progress, phases, and "Next Actions"
- `Context/PHASE2_PLAN.md` – Phase 2 objectives, training strategy, and gates
- `Context/claude.md` – "Quick Status" and "Next Steps" (the latest snapshot)

## 3. Check Data + Scripts
Verify presence of:
- `Data Crawling/simple_data_gen.py`
- `Data Crawling/simple_data_gen.sh`
- `Data Crawling/output/manifests/{train,val,test}.jsonl`

Optionally peek at a few manifest lines to understand data schema.

## 4. Check Git State (Optional)
If git is available:
- Run `git status` to see uncommitted changes
- Run `git log --oneline -10` to see recent commits for extra context

## 5. Produce a Concise Prime Summary
After reading all the context, return the statement:

**"Ready to Roll!"**

**Goal:** A brand new Claude instance should be able to meaningfully answer: "What is this project, what just happened, and what should we do next?"
