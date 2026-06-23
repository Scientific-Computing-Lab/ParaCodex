# Performance Tuning

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Do not change data strategy from used in the code**

**Required:**
- **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

## EARLY EXIT CHECK
If current runtime is within 5% of expected optimal (based on nsys kernel times):
- Document current metrics in optimization_plan.md
- Skip optimization - code is already well-tuned
- Focus only on micro-optimizations (const, restrict, cache locals)

## Workflow

### 1. Verify Baseline 
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt | grep -E "Verification|SUCCESSFUL|FAILED"
```

If results differ, fix Step 2 first.
If there are any errors, fix them before continuing.

### 2. Analyze Profile and Create Plan
 1.1. Read profile data:
 ```bash
# Try to find kernel information (OpenMP kernels may not appear in standard sections)
cat {profile_log_path} | grep -A20 "cuda_gpu_kern_sum" || echo "No cuda_gpu_kern_sum found - kernels may not be offloading to GPU"
cat {profile_log_path} | grep -A10 "cuda_api_sum"
cat {profile_log_path} | grep -A10 "cuda_gpu_mem_time_sum"
# Also check for any GPU activity
cat {profile_log_path} | grep -i "gpu\|kernel\|target" | head -20
```
 1.2. Run 
 ```bush
 nvidia-smi --query-gpu=name,compute_cap --format=csv
 ```
 roughly estimate the GPU saturation threshold
---

2. Create optimization_plan.md in {kernel_dir}:
```markdown
# Performance Analysis

## Current Metrics
- Runtime: [X]s
- Main kernel: [name], [Y]% GPU, [Z] instances
- Memory transfer: [%] time, [MB] total
- Kernel launches: [count]

## Fusion Opportunities:

### Identified Fusions:
- Lines X-Y: init → FUSE (same bounds)
- Lines A-B: compute+reduce → FUSE (register value)

## Iteration Loop (if present):
- Main: lines [X-Y], [N] iters
- SpMV line Z: [N] times
- Update line W: [N] times
- Total: [N×M] ops

## SpMV Inner Loop Decision
- Avg nonzeros per row (NONZER): [value from code/headers]
- If NONZER < 50: Keep inner loop SERIAL
- If NONZER > 100: Add `#pragma omp loop reduction`

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Data transfers | >30% transfer time | Move to Strategy C, use is_device_ptr |
| Launch overhead | instances >> iterations | Inline helper functions |
| Over-parallelization | Type C slow, outer saturated | Remove inner pragmas |
| Hot kernel | One kernel >50% time | collapse, simd, cache locals |
| Stage parallelization | FAIL verification | Remove pragma from stage loops |


## Strategy (priority)
1. [ACTION]: [what] - [why] - expect [gain]
2. [ACTION]: [what] - [why] - expect [gain]

## Micro-opts
[ ] const, restrict, firstprivate, cache locals

## Target
- Runtime: [X]s
- Kernels: ~[N] for [M] iters
- Memory: <[X]%
```
### Fusion rules

**Fuse when:**
- Adjacent independent, same bounds
- Producer-consumer
- Multi-vector ops

**Don't fuse:**
- Different bounds
- Intermediate sync required

### 3. Execute Optimization Plan
- Apply changes and document in optimization_plan.md

### 4. Optimization Actions

### 4A. Fix Data Movement

- Hoist target data outside loops
- omp_target_alloc + is_device_ptr for scratch
- Remove map inside target data
- Wrap functions: present,alloc
- Host init: target update to after

### 4B. Optimize Hot Kernel

- Use combined target teams loop
- Type B: Add inner #pragma omp loop reduction(+:sum)
- collapse(N) on nested dense loops
- Add #pragma omp simd to innermost
- Cache array accesses (SpMV/CSR):

```c
int tmp1, tmp2, tmp3;  // Function scope
#pragma omp target teams loop is_device_ptr(...)
for (int i = 0; i < nrows; i++) {
  tmp1 = d_rowptr[i];
  tmp2 = d_rowptr[i+1];
  double sum = 0.0;
  #pragma omp loop reduction(+:sum)
  for (int k = tmp1; k < tmp2; k++) {
    tmp3 = d_col[k];
    sum += d_val[k] * d_x[tmp3];
  }
  d_y[i] = sum;
}
```

### 4C. Launch Overhead

**Rule:** If kernel instances >> iteration count, inline helper functions in the main loop.
- Keep reduction helpers (dot, norm) - they return scalars
- Inline SpMV, vector updates, scaling operations
- Fuse adjacent loops with same bounds

### 4D. Fix Type C1 (Multi-Stage)

Outer loops: collapse(2) on spatial dimensions
Inner stage loops: Remove all pragmas (must be serial)

### 4E. Increase Parallelism

- Increase collapse depth
-  Use tile sizes(32, 32)
- Remove manual num_teams/thread_limit

### 5. Final Summary
Update optimization_plan.md:
```markdown
# Final Performance Summary

### Baseline (Step 2)
- Runtime: [X]s
- Main kernel: [Y] instances, [Z]ms total

### Final (Step 3)
- Runtime: [X]s
- Speedup: [X]x
- Main kernel: [Y] instances, [Z]ms total

### Optimizations Applied
1. [] [ACTION]: [description] → [±X%]
2. [] [ACTION]: REVERTED (slower)

### Micro-optimizations Applied
1. [] [MICRO-OPT]: [description] → [±X%]
2. [] [MICRO-OPT]: REVERTED (slower)

### Key Insights
- [Most impactful optimization]
- [Remaining bottlenecks]
```

**Reference: Available Opts**
## Bottlenecks (mark applicable)
### [ ] 1. Data Management Issue (CRITICAL - fix first!)
- Transfer ratio: [actual/expected] = [X]x
- If >2.5x: Data management wrong
- Root cause: [from data_plan.md verification]
- Fix: [specific action - e.g., offload missing functions, move scratch to device]
- Expected gain: [X]x speedup

### [ ] 2. Kernel Launch Overhead
- Kernel instances: [count]
- Expected: ~[N] for [N] iterations
- If instances >> N: Helper functions called in loop
- Root cause: [which functions - e.g., device_spmv, device_axpy]
- Fix: Inline operations in loop (ACTION 4C)
- Expected gain: [X]x (reduce [Y] launches to [Z])

### [ ] 3. Memory Transfer Bottleneck
- Transfer time: [X]% of total time
- If >50% AND ratio <2x: Transfers correct but dominant
- Fix: Optimize data movement (ACTION 4A)
- Expected gain: [X]%

### [ ] 4. Hot Kernel Performance
- Kernel: [name] takes [X]% GPU time, [Y]ms avg
- Root cause: [inefficient algorithm/missing optimization]
- Fix: [collapse/simd/cache/etc.] (ACTION 4B)
- Expected gain: [X]% faster kernel

### [ ] 5. Type C Parallelization Error
- Verification: [PASS/FAIL]
- If FAIL: Wrong stage loop parallelization
- Fix: Remove inner pragmas (ACTION 4D)

[ ] 6. Over-Parallelization (saturated outer loops)
- Outer parallelized iterations: [K × J = ?]
- Saturation threshold: [Saturation threshold]
- IF saturated AND inner has pragma → REMOVE inner pragmas
- Symptoms: Type C kernel slower after (or before) "optimization", GPU over-saturated
- Fix: Remove collapse/omp loop from inner/stage/writeback loops
- Expected gain: [X]%

## Profiling
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
# Check for kernel information (OpenMP kernels may appear in cuda_gpu_kern_sum or with different names)
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found - check if code is offloading to GPU"
```

### Deliverables
- optimization_plan.md - Complete analysis and results
- Optimized source code
- Final profile: {profile_log_path}

#**RULES** BRAKING A RULE = FAILURE.
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- You may create backup/output files (*.bak, *.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}
```

