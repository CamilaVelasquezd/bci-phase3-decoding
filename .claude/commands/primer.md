# Prime Context for Claude Code — bci-phase3-decoding

Use the command `tree` to get an understanding of the project structure.

Start by reading this file:
1. `CONTEXT.md` — this is the primary context file and the source of truth for this repo. Read it fully before doing anything else.

Then explore the codebase:
- List the contents of `decoding/` if it exists
- List the contents of `notebooks/` if it exists
- Check if `.agents/PRPs/` exists and list any PRPs already created

Explain back to me:
- Project purpose and Phase 3 goals
- Current state of the repo (what has been implemented vs what is pending)
- The decoding pipeline structure: feature extraction → dimensionality reduction → models
- The correct API to use: `from bci_decoding_dataset import DatasetLoader`
- Any known bugs or gotchas documented in CONTEXT.md
- What datasets are available and which to use for regression vs classification
- What the evaluation metrics are for comparing dimensionality reduction methods