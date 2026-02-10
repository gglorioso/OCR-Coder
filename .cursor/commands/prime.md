# /prime – Fresh Instance Bootstrapping Command (Cursor & Claude)

**Purpose:** When the user types `/prime`  the assistant should quickly rebuild context by reading the right files and summarizing the current state of the project.

---

## What the Assistant Should Do

When asked to run `/prime`:

1. **Scan the repo structure**
   - List top-level dirs/files to get bearings:
     - `README.md`
     - `Context/` (all files)
     - `Data Crawling/` (especially `output/` and scripts)

2. **Read core context files**
   - `README.md` – high-level description of DeepSeek-Coder-VL and goals.
   - `Context/WORKSPACE_NOTES.md` – most recent progress, phases, and “Next Actions”.
   - `Context/PHASE2_PLAN.md` – Phase 2 objectives, training strategy, and gates.
   - `Context/claude.md` – “Quick Status” and “Next Steps” (the latest snapshot).

3. **Check data + scripts**
   - Verify presence of:
     - `Data Crawling/simple_data_gen.py`
     - `Data Crawling/simple_data_gen.sh`
     - `Data Crawling/output/manifests/{train,val,test}.jsonl`
   - Optionally peek at a few manifest lines to understand data schema.

4. **(Optional) Check git state**
   - If git is available, conceptually:
     - `git status` → see uncommitted changes
     - `git log --oneline -10` → recent commits for extra context

5. **Produce a concise “Prime Summary”**

Return the statement "Ready to Roll!" when complete. 

The goal is for a brand new Cursor/Claude instance to run `/prime`, follow this playbook, and then be able to meaningfully answer: **“What is this project, what just happened, and what should we do next?”**
