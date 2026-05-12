# Create PRP — bci-phase3-decoding

## Feature file: $ARGUMENTS

Generate a complete PRP for a Phase 3 decoding feature with thorough research. Ensure context is passed to the AI agent to enable self-validation and iterative refinement. Read the feature file first to understand what needs to be created.

The AI agent only gets the context you append to the PRP and its training data. Assume the AI agent has access to the codebase and the same knowledge cutoff as you — include your research findings directly in the PRP. The agent has web search capabilities, so pass URLs to documentation and examples.

## Research Process

1. **Read CONTEXT.md first**
   - Understand the V8 schema, dataset quirks, and known bugs
   - Note the exact S3 connection pattern to copy
   - Note the coding conventions (shuffle=False, no hardcoding, sklearn interface, etc.)
   - Note the gotchas section — these must be referenced in the PRP

2. **Codebase Analysis**
   - Search for existing patterns in `decoding/` if it exists
   - Check `notebooks/` for reference notebook style
   - Identify files to reference in the PRP
   - Note existing conventions to follow

3. **External Research**
   - Search for sklearn documentation for the relevant estimator
   - Search for implementation examples (GitHub, papers, blogs)
   - Include specific URLs in the PRP
   - Note common pitfalls for the specific algorithm

4. **User Clarification** (if needed)
   - Which dataset(s) to use?
   - Regression (vx/vy) or classification (8 directions)?
   - Which dimensionality reduction to apply before the model?

## PRP Generation

Use `.agents/PRPs/templates/prp_base.md` as template if it exists, otherwise structure the PRP as:

### Critical Context to Include
- **CONTEXT.md reference**: always remind the agent to read CONTEXT.md
- **S3 connection pattern**: copy exactly from CONTEXT.md section 5
- **Schema**: V8 variable shapes and dtypes from CONTEXT.md section 4
- **Dataset quirks**: which datasets to use/avoid for this specific task
- **Documentation URLs**: sklearn, scipy, or other relevant docs
- **Code examples**: real snippets from existing notebooks or decoding/ package
- **Gotchas**: from CONTEXT.md section 10, plus any algorithm-specific ones

### Implementation Blueprint
- Start with pseudocode showing the full approach
- Reference real files for patterns
- Include error handling strategy
- List tasks in the order they should be completed

### Validation Gates (must be executable)
```bash
# Import check
python -c "from decoding import DecodingPipeline; print('OK')"

# Synthetic data test
python -m pytest tests/ -v

# Manual smoke test
python -c "
import numpy as np
from decoding.feature_extraction import compute_binned_counts
# ... minimal smoke test
"
```

### Coding Conventions to Enforce
- `shuffle=False` in ALL train/test splits and cross-validation
- No hardcoded session IDs, dataset names, or S3 paths
- `from __future__ import annotations` at top of every `.py` file
- Numpy-style docstrings on every public function
- `DecodingPipeline.fit/predict/score` must match sklearn API
- Do NOT use `velocity` from `DANDI_000140` (vx/vy duplication bug)
- Filter to `task_type in ('center_out', 'center_out_maze')` before `compute_direction_labels()`

*** CRITICAL: AFTER RESEARCHING, ULTRATHINK ABOUT THE PRP BEFORE WRITING IT ***

## Output
Save as: `.agents/PRPs/feature-request/FR00X-NAME/{feature-name}.md`

## Quality Checklist
- [ ] CONTEXT.md reference included
- [ ] S3 connection pattern copied exactly
- [ ] V8 schema documented
- [ ] Dataset quirks and known bugs noted
- [ ] Validation gates are executable
- [ ] shuffle=False enforced
- [ ] No hardcoded paths or session IDs
- [ ] sklearn interface respected
- [ ] Clear implementation path with ordered tasks

Score the PRP on a scale of 1-10 (confidence level to succeed in one-pass implementation).

Remember: The goal is one-pass implementation success through comprehensive context.
