# Execute PRP — bci-phase3-decoding

Implement a feature using the PRP file.

## PRP File: $ARGUMENTS

## Execution Process

1. **Load context**
   - Read `CONTEXT.md` fully before doing anything else
   - Read the specified PRP file
   - Understand all requirements, schema, and known bugs
   - Do more codebase exploration and web searches as needed

2. **ULTRATHINK**
   - Think hard before executing. Create a comprehensive plan.
   - Break down complex tasks into smaller steps using TodoWrite.
   - Identify implementation patterns from existing code in `decoding/` or `notebooks/`.
   - Confirm: does the plan respect all conventions from CONTEXT.md section 9?

3. **Execute the plan**
   - Implement all the code
   - Follow coding conventions strictly:
     - `shuffle=False` everywhere
     - No hardcoded session IDs or S3 paths
     - `from __future__ import annotations` at top of every `.py`
     - Numpy-style docstrings on all public functions
     - sklearn-compatible `fit/predict/score` interface

4. **Validate**
   - Run each validation gate from the PRP
   - Fix any failures
   - Re-run until all pass
   - Verify the quality checklist in CONTEXT.md section 13

5. **Complete**
   - Ensure all checklist items done
   - Run final validation suite
   - Report completion status
   - Re-read the PRP to confirm everything was implemented

Note: If validation fails, use the error patterns and gotchas in CONTEXT.md section 10 and the PRP to fix and retry.
