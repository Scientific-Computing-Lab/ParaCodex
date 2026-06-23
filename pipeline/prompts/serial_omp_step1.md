# GPU Offload with OpenMP

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for all runs
- DO NOT use `distribute parallel for`
- **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

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

**FIRST: Check if original algorithm can be simplified for GPU:**
- Large scratch arrays for intermediate results → Can per-thread locals replace them?
- Block-based iteration (for cache) → REMOVE blocking, use single parallel loop over ALL work items
- Multi-stage with host sync → Can everything run in one kernel?

**Rule:** If scratch arrays exist ONLY to avoid atomics on small data (<1KB), 
DELETE them and use per-thread locals + atomic merge instead.

**Block elimination:** If code has `for (blk = 0; blk < numblks; blk++)` with scratch arrays,
this is a CPU cache optimization. For GPU: remove blocking, parallelize over all N items directly.

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
- **Histogram Strategy?** For small bin (≤ 100) counts: use per-thread local array + atomic merge (NO scratch arrays needed!)

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

## 8. Profile
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
# Check for kernel information (OpenMP kernels may appear in cuda_gpu_kern_sum or with different names)
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found - check if code is offloading to GPU"
```

#**RULES** BRAKING A RULE = FAILURE.
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- You may create backup/output files (*.bak, *.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}
```

