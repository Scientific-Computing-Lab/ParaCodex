# Serial to OpenMP Translation Prompts

This document contains the current prompts used for serial-to-OpenMP translation and optimization.

## Initial Translation Prompt

This prompt is used in `initial_translation_codex.py` for the analysis phase before translation.

```markdown
# Loop Classification for GPU Offload - Analysis Phase

## Task
Analyze loops in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. Copy source files unmodified to `{kernel_dir}/`.

**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Process

### 0. COPY THE SOURCE FILES - {file_listing} TO THE KERNEL DIRECTORY {kernel_dir}

### 1. Find All Loops
```bash
# Find main compute loop
grep -n "for.*iter\|for.*it\|while\|main(" *.c *.cpp 2>/dev/null | head -50

# List all loop-containing functions
grep -n "for\s*(" *.c *.cpp 2>/dev/null | head -100
```

Prioritize functions called in main compute loop:
- Every iteration → CRITICAL/IMPORTANT
- Once at setup → SECONDARY/AVOID

### 2. Classify Priority
For each loop: `iterations × ops/iter = total work`

- **CRITICAL:** >50% runtime OR called every iteration with O(N) work
- **IMPORTANT:** 5-50% runtime OR called every iteration with small work
- **SECONDARY:** Called once at setup
- **AVOID:** Setup/IO/RNG OR <10K iterations

### 3. Determine Loop Type (Decision Tree)

```
Q0: Nested inside another loop? → Note parent
Q1: Writes A[idx[i]] with varying idx? → Type D (Histogram)
Q2: Reads A[i-1] or accumulates across iterations? → Type E (Recurrence - CPU only)
Q3: Stage loop where L+1 depends on L?
    - Scratch swap (tmp1↔tmp2)? → C1 (FFT/Butterfly)
    - Level traversal with stencil calls? → C2 (Multigrid)
Q4: Inner bound varies with outer index? → Type B (Sparse)
Q5: Accumulates to scalar? → Type F (Reduction)
Q6: Accesses neighbors? → Type G (Stencil)
Default → Type A (Dense)
```

**Special Case - Outer A + Inner E:**
When outer loop iterates over INDEPENDENT samples and inner has RNG:
- Mark outer as Type A (CRITICAL) - parallelizable with per-thread RNG
- Mark inner RNG as Type E - sequential WITHIN each thread
- Note: "RNG replicable: YES - each sample can compute its own seed"

### 4. Type Reference

| Type | Pattern | Parallelizable |
|------|---------|----------------|
| A | Dense, constant bounds | YES |
| B | Sparse (CSR), inner bound varies | Outer only |
| C1 | FFT/Butterfly, scratch swap | Outer only |
| C2 | Multigrid, hierarchical calls | Outer only |
| D | Histogram, indirect write | YES + atomic |
| E | Recurrence, loop-carried dep | NO |
| F | Reduction to scalar | YES + reduction |
| G | Stencil, neighbor access | YES |

### 5. Data Analysis
For each array:
- Definition: flat vs pointer-to-pointer
- Allocation: static vs dynamic
- Struct members accessed?
- Global variables used?

### 6. Flag Issues
- Variable bounds
- Reduction needed
- Atomic required
- Stage dependency
- RNG in loop
- <10K iterations

## Output: analysis.md

### Loop Nesting Structure
```
- outer_loop (line:X) Type A
  └── inner_loop_1 (line:Y) Type E
- standalone_loop (line:Z) Type A
```

### Loop Details
For each CRITICAL/IMPORTANT/SECONDARY loop:
```
## Loop: [function] at [file:line]
- **Iterations:** [count]
- **Type:** [A-H] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [inner loops or none]
- **Dependencies:** [none / reduction:vars / stage / recurrence]
- **Nested bounds:** [constant / variable]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW)]
- **Issues:** [flags]
```

### Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|

### Data Details
- **Dominant compute loop:** [main timed loop]
- **Arrays swapped between functions?:** YES/NO
- **Scratch arrays?:** YES/NO
- **Mid-computation sync?:** YES/NO
- **RNG in timed loop?:** YES/NO (only if inside timer)

## Constraints
- Find all loops in functions called from main compute loop
- Document only - no pragmas or code modifications
- When uncertain between B and C, choose C
- Copy all source files unmodified to `{kernel_dir}/`
```

---

## Optimization Step 1: GPU Offload with OpenMP

This prompt is used in `optimize_codex.py` for the first optimization step (GPU offload implementation).

```markdown
# GPU Offload with OpenMP

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for all runs
- DO NOT use `distribute parallel for`

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline (CLASS A/S)
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > baseline_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > baseline_output.txt 2>&1
grep -E "Verification|SUCCESSFUL|FAILED" baseline_output.txt

DO NOT SKIP THIS STEP.
```

### 2. Choose Data Strategy
Walk through IN ORDER, stop at first match:

```
RULE 1: Type B (Sparse/CSR)?              → STRATEGY A/C
RULE 2: Type C1 (Iterative Solvers/Butterfly)?→ STRATEGY C
RULE 3: Type C2 (Multigrid)?              → STRATEGY A
RULE 4: Outer A + inner E (per-thread RNG)?→ STRATEGY A
RULE 5: Multiple independent kernels?     → STRATEGY B
RULE 6: Otherwise                         → STRATEGY A
```

### 2.5. Create Data Management Plan
MANDATORY: Create data_plan.md in {kernel_dir} before implementation

Analyze ALL arrays and functions in timed region:

```markdown

# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| [name] | [bytes] | working/scratch/const/index | host/device | R/W/RO |

**Types:** working (main data), scratch (temp), const (read-only), index (maps)

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| [name] | [list] | per-iteration/once | device/host |

## Data Movement Strategy

**Chosen Strategy:** [A/B/C]

**Device Allocations (once):**
```
Strategy C: d_[array]: [size] via omp_target_alloc
Strategy A: [arrays] in target data region
```

**Host→Device Transfers:**
- When: [before iterations/once at start]
- Arrays: [array1]→d_[array1] ([size] MB)
- Total H→D: ~[X] MB

**Device→Host Transfers:**
- When: [after iterations/once at end]
- Arrays: d_[array1]→[array1] ([size] MB)
- Total D→H: ~[Y] MB

**Transfers During Iterations:** [YES/NO]
- If YES: [which arrays and why]
- If NO: All data stays on device

## Critical Checks (for chosen strategy)

**Strategy A:**
- [ ] Functions inside target data use `present,alloc` wrapper?
- [ ] Scratch arrays use enter/exit data OR omp_target_alloc?

**Strategy C:**
- [ ] ALL functions in iteration loop use is_device_ptr?
- [ ] Scratch arrays allocated on device (not host)?
- [ ] No map() clauses (only is_device_ptr)?

**Common Mistakes:**
-  Some functions on device, others on host (causes copying)
-  Scratch as host arrays in Strategy C
-  Forgetting to offload ALL functions in loop

## Expected Transfer Volume
- Total: ~[X+Y] MB for entire execution
- **Red flag:** If actual >2x expected → data management wrong

## Additional Parallelization Notes
- **RNG Replicable?** [YES/NO] - If YES, use `#pragma omp declare target` on RNG function
- **Outer Saturation?** [outer iters]
- **Sparse Matrix NONZER?** [value]
- **Histogram Strategy?** For small bin counts: use per-thread local array + atomic merge (NO scratch arrays needed!)

**Summary:** [num] arrays ([num] scratch, [num] working), [num] functions, Strategy [A/B/C]. Expected: ~[X] MB H→D, ~[Y] MB D→H.
```

### 2.6. Implement Data Plan

**Use data_plan.md as implementation guide**

### Step 1: Setup Data Structures
From "Arrays Inventory" and "Data Movement Strategy":
- Declare device arrays/pointers as needed for chosen strategy
- Create allocation/initialization functions based on strategy:
  - **Strategy A:** Setup target data regions with map clauses from plan
  - **Strategy B:** Prepare depend clauses for async operations
  - **Strategy C:** Create omp_target_alloc calls using sizes from plan

### Step 2: Implement Transfers
From "H→D Transfers" and "D→H Transfers" sections:
- Implement each transfer listed with timing specified in plan
- Use method appropriate for strategy (map clauses, omp_target_memcpy, update, etc.)

### Step 3: Offload Functions
Use "Functions in Timed Region" table:
- For each function where "Must Run On" = device:
  - Add appropriate pragma for strategy
  - Include arrays from "Arrays Accessed" column
  - Follow strategy-specific patterns from Step 2

### Step 4: Main Program Flow
Follow "Data Movement Strategy" timing:
```
[setup from plan]
[H→D transfers at specified time]
[timed computation - call functions]
[D→H transfers at specified time]
[cleanup]
```

### Step 5: Verify Implementation
Check ALL items in "Critical Checks" section for YOUR strategy:
- [ ] Verify each checkpoint matches implementation
- [ ] Cross-reference "Functions in Timed Region" table
- [ ] Confirm transfer timing matches plan

**Common errors:** Mismatched array names, missing functions from table, wrong transfer timing

**Ready when:** All strategy-specific checks ✓ and compiles
---

## Strategy Details

### STRATEGY A: target data Region

**Map Clause Selection:**
| Scenario | Map Clause | Why |
|----------|------------|-----|
| Device-init arrays (zero(), fill()) | `alloc` | Avoid copying garbage |
| Host RNG init then sync | `alloc` + `update to` | Explicit sync after host init |
| Read + modify + write | `tofrom` | Bidirectional |
| Read-only | `to` | One-way |

**Functions Called Inside target data:**
Wrap with `present,alloc`/'to,tofrom', then use bare `target teams loop`:
```c
void compute(double *u, double *v, int n) {
  #pragma omp target data map(present,alloc:u[0:n],v[0:n])
  {
    #pragma omp target teams loop
    for (int i = 0; i < n; i++) { ... }
  }
}
```

**RNG replicable:**
```c
#pragma omp target teams loop reduction(+:sum1, sum2) firstprivate(seed_base, params)
for (int sample = 0; sample < N; ++sample) {
  double rng_state = compute_seed_for_sample(sample);  // Per-thread seed
  double local_hist[BINS] = {0};  // Per-thread histogram
  
  // Type E (RNG) is sequential WITHIN this thread
  for (int i = 0; i < work_per_sample; ++i) {
    double r = my_rng(&rng_state, A);
    int bin = compute_bin(r);
    local_hist[bin] += 1.0;
    sum1 += ...; sum2 += ...;  // Reduction handles these
  }
  
  // Atomic merge histogram at end
  for (int b = 0; b < BINS; ++b) {
    if (local_hist[b] != 0.0) {
      #pragma omp atomic update
      global_hist[b] += local_hist[b];
    }
  }
}
```

**Scratch Arrays (two options):**

- **Option 1: enter/exit data**
```c
double scratch[N];
#pragma omp target enter data map(alloc:scratch[0:n])
#pragma omp target data map(present,alloc:in[0:n])
{
  #pragma omp target teams loop
  for (...) { /* use scratch */ }
}
#pragma omp target exit data map(delete:scratch[0:n])
```

- **Option 2: omp_target_alloc**
```c
double *scratch = (double*)omp_target_alloc(n*sizeof(double), 0);
#pragma omp target data map(present,alloc:in[0:n])
{
  #pragma omp target teams loop is_device_ptr(scratch)
  for (...) { ... }
}
omp_target_free(scratch, 0);
```

**Mid-computation sync:**
```c
#pragma omp target update from(result)
host_compute(result);
#pragma omp target update to(indices)
```

### STRATEGY B: Asynchronous Offload
Use when: Overlapping compute/transfer possible
```c
#pragma omp target teams loop nowait depend(out:x[0])
for (i = 0; i < N; i++) { x[i] = init(i); }

#pragma omp target teams loop nowait depend(in:x[0]) depend(out:y[0])
for (i = 0; i < N; i++) { y[i] = compute(x[i]); }

#pragma omp taskwait
```

STRATEGY C: Global Device State (Iterative Solvers)
Use omp_target_alloc + is_device_ptr for all device arrays.

**Pattern:**
```c
// Device pointers: static double *d_arr
allocate_device_arrays();  // omp_target_alloc once
copy_to_device();          // omp_target_memcpy once

for (iter ...) {
  #pragma omp target teams is_device_ptr(d_arr1, d_arr2, ...)
  {
  #pragma omp loop            // Outer parallelism
  for (k ...) {
    #pragma omp loop          // Middle parallelism (if needed)
    for (j ...) {
      for (stage ...) { ... }  // NO pragma - stages must be serial!
    }
  }
  }
}

free_device_arrays();
```

**Key Rules:**
- Use `is_device_ptr` everywhere (no map clauses in hot path)
- Reduction helpers (dot, norm) OK - they return scalars
- stage loops: parallelize outer k,j; keep stage loop L serial
- Iterative solvers: inline SpMV, updates in main loop
---

### 3. Map Globals & Functions
```c
#pragma omp declare target
double helper_func() { ... };
#pragma omp end declare target

#pragma omp declare target(global_var)
```
---

## 4. Parallelize loops

**Parallelization patterns:**

**Type A (Dense):**
```c
#pragma omp target teams loop collapse(2)
for (i = 0; i < N; i++)
  for (j = 0; j < M; j++) ...
```

**Type B (Sparse/CSR) - Nested Parallelism:**
```c
int tmp1, tmp2, tmp3;  // Function scope
#pragma omp target teams loop is_device_ptr(...)
for (int row = 0; row < nrows; row++) {
  tmp1 = rowptr[row];
  tmp2 = rowptr[row+1];
  double sum = 0.0;
  ***#pragma omp loop reduction(+:sum)***  // Parallelize inner *based on GPU saturation* 
  for (int k = tmp1; k < tmp2; k++) {
    tmp3 = colidx[k];
    sum += A[k] * x[tmp3];
  }
  y[row] = sum;
}
```

**Type C1 (Iterative Solvers) - Serial Inner:**
```c
#pragma omp target teams is_device_ptr(...)
{
#pragma omp loop collapse(2)
  for (k = 0; k < K; k++) {
    for (j = 0; j < J; j++) {
      for (stage = 0; stage < S; stage++) { ... }  // No pragma - keep inner serial!
    }
  }
}
**Rationale:** K×J teams already saturate GPU. Inner serial = better register reuse, no barriers.
```

**Type C2 (Multigrid):** Wrap with `present,alloc`; each stencil call gets `target teams loop`.

**Type C special rule:** Stage-dependent algorithms (multigrid, iterative stages) 
should NEVER have inner parallelism, regardless of GPU. The barrier overhead between 
stages exceeds any benefit from inner thread parallelism.

**Type D (Histogram):** Add `#pragma omp atomic` on indirect writes.

**Type F (Reduction):** `reduction(+:sum)`

**Type G (Stencil):** `collapse(2)` on spatial dimensions.

**Type A+E (Outer parallel, inner RNG):** 
**When analysis says "RNG replicable: YES":**
- Add `declare target` on RNG function - GPU callable.
- Parallelize over samples, each thread has private RNG + histogram
- Atomic merge histogram at the end

## Histogram Optimization 
If histogram bins ≤ 100:
```c
// GOOD: Per-thread local array (80 bytes for 10 bins)
#pragma omp target teams loop reduction(+:sx, sy)
for (int k = 0; k < N; ++k) {
  double q_local[BINS] = {0};  // Thread-private
  // ... accumulate into q_local ...
  for (int b = 0; b < BINS; ++b) {
    if (q_local[b] != 0.0) {
      #pragma omp atomic update
      q[b] += q_local[b];
    }
  }
}
```
**DO NOT** create large scratch arrays for small histograms - the atomic overhead is negligible compared to memory transfer costs.
**Key:** Each thread replicates the RNG state for its sample. Type E becomes parallelizable at the OUTER level.

## 5. Compile and Test (CLASS A/S)
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {correctness_run_cmd} > gpu_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > gpu_output.txt 2>&1
```

If timeout/segfault: Remove `#pragma omp loop` from Type C inner loops.

## 6. Verify Correctness
```bash
diff baseline_output.txt gpu_output.txt
```

## 8. Profile (CLASS B/C)
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
grep "cuda_gpu_kern" {profile_log_path} | head -5
```

#**RULES** BRAKING A RULE = FAILURE.
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- DO NOT CHANGE/EDIT FILES OTHER THEN {file_listing}
```

---

## Optimization Step 2: Performance Tuning

This prompt is used in `optimize_codex.py` for the second optimization step (performance tuning).

```markdown
# Performance Tuning

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Do not change data strategy from used in the code**

## EARLY EXIT CHECK
If current runtime is within 5% of expected optimal (based on nsys kernel times):
- Document current metrics in optimization_plan.md
- Skip optimization - code is already well-tuned
- Focus only on micro-optimizations (const, restrict, cache locals)

## Workflow

### 1. Verify Baseline (CLASS A/S)
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
cat {profile_log_path} | grep -A20 "cuda_gpu_kern_sum"
cat {profile_log_path} | grep -A10 "cuda_api_sum"
cat {profile_log_path} | grep -A10 "cuda_gpu_mem_time_sum"
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

## Profiling (CLASS B/C)
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
grep "cuda_gpu_kern" {profile_log_path} | head -5
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
- DO NOT CHANGE FILES OTHER THEN {file_listing}
```

---

## Notes

- All prompts contain template variables (e.g., `{kernel_dir}`, `{file_listing}`, `{clean_cmd_str}`, etc.) that are filled in at runtime by the Python scripts.
- The prompts are designed to guide the AI through a systematic process of analysis, translation, and optimization.
- Template variables are replaced with actual values when the prompts are used in the codebase.

