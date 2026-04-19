# Optimization Plan

## Current Metrics
- Profile source: `profile.log`
- GPU kernels: 2 total
- GPU kernel time: 5,312 ns total
- CUDA memcpy time: 6,975 ns total
- CUDA stream sync time: 15,739 ns total
- CUDA launch time: 134,658 ns total
- Host wait time: 539,014,526 ns total
- Host fread time: 46,047,453 ns total
- Host poll time: 100,615,821 ns total
- Host ioctl time: 73,614,636 ns total

## Structural Check
- The timed path already uses one fused GPU kernel per command path, so the offload unit is not fragmented into many tiny kernels.
- Data transfer volume is tiny: the profile shows only a few kilobytes of HtoD/DtoH traffic, so the current data strategy is not the primary problem.
- The runtime is execution-structure limited on the host side, with wait/poll/I/O dominating far more than GPU execution.
- Conclusion: keep the step-1 offload/data strategy and focus on lower-risk compute-path tightening.

## Fusion Opportunities
- No additional kernel fusion is required before tuning because each command already maps to a single target region.
- The profitable tuning surface is inside the tree traversal helpers: reduce per-node comparison work and cache repeated node fields locally.

## Iteration Loop
1. Rebuild after each change with the NVC OpenMP GPU path.
2. Re-run the bundled correctness path and verify the output remains unchanged.
3. Re-profile the bundled input and compare combined GPU kernel + memcpy + sync cost, not kernel time alone.
4. If a change helps kernels but worsens end-to-end runtime, revert or refine it.
5. Validate at least one larger query batch before finalizing if the runtime budget allows.

## SpMV Inner Loop Decision
- Not applicable to b+tree.

## Strategy
- Keep the fused per-command GPU offload structure from step 1.
- Replace linear key scans inside tree nodes with a faster search strategy.
- Cache `num_keys`, key values, and child pointers locally in the traversal helpers to reduce repeated indirect loads.
- Preserve correctness and avoid adding extra kernel launches or host/device synchronization.

## Final Summary
- Baseline bundled profile combined GPU kernel + memcpy + sync cost: `28,026 ns`
- Final bundled profile combined GPU kernel + memcpy + sync cost: `23,689 ns`
- Speedup on the measured GPU path: `1.18x` overall, about `15.5%` lower combined device-side cost
- Baseline bundled GPU kernel time: `5,312 ns`
- Final bundled GPU kernel time: `4,608 ns`
- Baseline bundled memcpy + sync time: `22,714 ns`
- Final bundled memcpy + sync time: `19,081 ns`
- Optimizations applied:
  - Replaced the linear in-node search with binary search in both GPU traversal helpers.
  - Cached the query key and leaf node pointer locally to cut repeated loads.
- Optimizations reverted:
  - None.
- Key insights:
  - The step-1 offload/data strategy was already structurally sound.
  - The bottleneck is still execution-structure limited on the host side; GPU work is tiny relative to wait/poll/I/O.
  - Larger-input profiling with a `k 65535` batch confirmed the same conclusion: the runtime remains dominated by host overhead, not by transfer volume or kernel fragmentation.
- Step-1 offload/data strategy changed?
  - No. The code keeps the fused per-command GPU offload because the measured transfer volume is already low and the structure is stable.
- Larger-input profiling changed the optimization decision?
  - No. It confirmed the same execution-structure-limited diagnosis and did not justify a data-residency rewrite.
- Final result classification:
  - Execution-structure limited but stable.
