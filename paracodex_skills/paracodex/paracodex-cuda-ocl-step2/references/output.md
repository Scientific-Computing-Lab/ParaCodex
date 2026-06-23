# Performance Analysis - CUDA to OpenCL

## Current Metrics
- Total runtime: [X]s
- Main kernel: [name], [Y]ms total, [Z] calls
- Memory transfer: [W]ms, [V]MB total
- Combined GPU+mem time: [X]ns
- Kernel compilation time: [U]ms (if significant)
- Host/setup overhead: [description]

## Scalability Check
- Default correctness size: [value]
- Larger practical profiling size: [value or rule]
- Why this size materially exercises the GPU: [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- Hardware basis from `system_info_summary.txt`: [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]
- Small-input sensitive? [YES/NO]
- Larger-input profiling changed the decision? [YES/NO]

## CUDA vs OpenCL Comparison
- CUDA runtime: [X]s (baseline)
- OpenCL runtime: [Y]s (current)
- Slowdown: [Y/X]x
- Target: <1.3x slowdown (acceptable for portability)

## Bottleneck Analysis

### [ ] 1. Work-Group Size Suboptimal
- Current: [X] work-items per group
- Device max: [check CL_DEVICE_MAX_WORK_GROUP_SIZE]
- Device preferred multiple: [check CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE]
- **Fix:** Tune local_work_size to device characteristics
- Expected gain: [X]%

### [ ] 2. Memory Transfer Overhead
- Transfer time: [X]% of total
- If >50%: Consider:
  - CL_MEM_ALLOC_HOST_PTR for pinned memory
  - Async transfers with events
  - Reduce transfer frequency
- Expected gain: [X]%

### [ ] 3. Math Function Precision
- Using sin/cos (precise) vs native_sin/native_cos (fast)
- Using pow vs native_powr
- **Fix:** Replace with native_* variants if precision allows
- Expected gain: [X]%

### [ ] 4. Local Memory Bank Conflicts
- Check __local memory access patterns
- Stride should avoid 32-word (AMD) or 64-word (NVIDIA) conflicts
- **Fix:** Pad __local arrays or change access pattern
- Expected gain: [X]%

### [ ] 5. Kernel Launch Overhead
- Kernel calls: [N]
- If many small kernels: Consider fusion
- If compilation in timing: Move clBuildProgram outside timer
- Expected gain: [X]%

### [ ] 8. Wrong Kernel / Program Structure
- If kernel time is fine but total runtime is poor, check setup/build cost, tiny kernels, or bad split.
- **Fix:** Structural rewrite before micro-tuning.
- Expected gain: [X]%

### [ ] 6. Global Memory Coalescing
- Check access patterns (same as CUDA requirements)
- Sequential threads should access sequential memory
- **Fix:** Restructure access patterns
- Expected gain: [X]%

### [ ] 7. Barrier Overhead
- barrier() calls: [count and locations]
- Check if CLK_LOCAL_MEM_FENCE sufficient (vs CLK_GLOBAL_MEM_FENCE)
- Consider algorithm restructuring to reduce barriers
- Expected gain: [X]%

## Optimization Strategy (priority order)
1. [ACTION]: [description] - expected [X]% gain
2. [ACTION]: [description] - expected [X]% gain

## Target Performance
- Target runtime: [X]s (<1.3x CUDA baseline)
- Target slowdown: [X]x
- Target combined GPU+mem cost: [X]ns

---

# Final Performance Summary

## Baseline (CUDA)
- Runtime: [X]s
- Main kernel: [Y]ms

## Initial OpenCL
- Runtime: [A]s
- Slowdown vs CUDA: [A/X]x
- Main kernel: [B]ms

## Optimized OpenCL
- Runtime: [C]s
- Slowdown vs CUDA: [C/X]x (target: <1.3x)
- Speedup vs initial OpenCL: [A/C]x
- Main kernel: [D]ms
- Combined GPU+mem cost improved: [YES/NO]

## Optimizations Applied
1. [X] [ACTION]: [description] → [±X%]
2. [X] [ACTION]: [description] → [±X%]
3. [ ] [ACTION]: REVERTED (broke correctness or slower)

## Micro-Optimizations
1. [X] Native math functions → [±X%]
2. [X] Work-group size tuned → [±X%]
3. [X] Memory access optimized → [±X%]

## Key Insights
- [Most impactful optimization]
- [Remaining performance gap vs CUDA: X%]
- [OpenCL-specific bottlenecks]
- [Limitations compared to CUDA]
- [Whether end-to-end runtime improved, not just kernel metrics]
- [Whether the final structure stayed plausible at the larger practical size]

## Device-Specific Notes
- Device: [name]
- Max work-group size: [X]
- Local memory: [Y] KB
- Optimal work-group size found: [Z]
