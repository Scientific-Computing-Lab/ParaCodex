# Output Format
- Update code in `{kernel_dir}` for `{file_listing}` (or `{file_list_str}` when provided).
- Write/refresh artifacts mentioned in the prompt:
  - supervisor_output.txt
  - correctness_verification.md (written on PASS — use template in SKILL.md)
- Preserve the candidate's optimized structure while adding compact correctness instrumentation.
- Prefer checksums over contiguous logical buffers rather than per-row/per-element host loops when both represent the same final state.
