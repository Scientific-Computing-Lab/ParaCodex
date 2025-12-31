'''# Loop Classification for OMP Migration - Analysis Phase

## Task
Analyze CUDA kernels in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. Copy source files to `{kernel_dir}/` with suffix conversion (.cu → .c or .cpp).

**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Process

### 0. COPY SOURCE FILES WITH SUFFIX CONVERSION
- Copy `{file_listing}` from `{source_dir}/` to `{kernel_dir}/`
- Convert suffixes: `.cu` → `.c` (for C code) or `.cpp` (for C++ code). You can inspecct the makefile in {kernel_dir}/ to see the expected file names.
- Get baseline output. Run {clean_cmd_str} and `{run_cmd_str} > baseline_output.txt 2>&1` in {source_dir}/. Copy the baseline output to {kernel_dir}/baseline_output.txt.
- Preserve all file content exactly - no code modifications
- Document mapping: `original.cu → converted.c` in analysis.md
- Convert header includes in {file_listing}. Make sure the code can be compiled with the converted files.
- DO NOT MODIFY FILES OTHER THAN {file_listing}.

### 1. Find All CUDA Kernels and Loops
```bash
# Find CUDA kernels
grep -n "__global__\|__device__" *.cu 2>/dev/null

# Find kernel launch sites
grep -n "<<<.*>>>" *.cu 2>/dev/null

# Find device loops (inside kernels)
grep -n "for\s*(" *.cu 2>/dev/null | head -100

# Find host loops calling kernels
grep -n "for.*iter\|for.*it\|while" *.cu 2>/dev/null | head -50
```

Prioritize by execution pattern:
- Kernel called every iteration → CRITICAL/IMPORTANT
- Kernel called once at setup → SECONDARY/AVOID
- Device loops inside kernels → analyze work per thread

### 2. Classify Priority
For each kernel/loop: `grid_size × block_size × device_iterations × ops = total work`

- **CRITICAL:** >50% runtime OR called every iteration with O(N) work
- **IMPORTANT:** 5-50% runtime OR called every iteration with small work
- **SECONDARY:** Called once at setup
- **AVOID:** Setup/IO/memory allocation OR <10K total threads

### 3. Determine Kernel/Loop Type (Decision Tree)

```
Q0: Is this a __global__ kernel or host loop? → Note context
Q1: Writes A[idx[i]] with varying idx (atomicAdd)? → Type D (Histogram)
Q2: Uses __syncthreads() or shared memory dependencies? → Type E (Block-level recurrence)
Q3: Multi-stage kernel pattern?
    - Separate kernels for stages with global sync? → C1 (FFT/Butterfly)
    - Hierarchical grid calls? → C2 (Multigrid)
Q4: Block/thread indexing varies with outer dimension? → Type B (Sparse)
Q5: Uses atomicAdd to scalar (reduction pattern)? → Type F (Reduction)
Q6: Accesses neighboring threads' data? → Type G (Stencil)
Default → Type A (Dense)
```

**CUDA-Specific Patterns:**
- **Kernel with thread loop:** Outer grid parallelism + inner device loop
  - Mark grid dimension as Type A (CRITICAL) - maps to OMP parallel
  - Mark device loop by standard classification
  - Note: "Grid-stride loop" if thread loops beyond block size

- **Atomic operations:** 
  - atomicAdd → requires OMP atomic/reduction
  - Race conditions → document carefully

- **Shared memory:**
  - __shared__ arrays → maps to OMP private/firstprivate
  - __syncthreads() → limited OMP equivalent, may need restructuring

### 4. Type Reference

| Type | CUDA Pattern | OMP Equivalent | Notes |
|------|--------------|----------------|-------|
| A | Dense kernel, regular grid | YES - parallel for | Direct map |
| B | Sparse (CSR), varying bounds | Outer only | Inner sequential |
| C1 | Multi-kernel, global sync | Outer only | Barrier between stages |
| C2 | Hierarchical grid | Outer only | Nested parallelism tricky |
| D | Histogram, atomicAdd | YES + atomic | Performance loss expected |
| E | __syncthreads, shared deps | NO | Requires restructuring |
| F | Reduction, atomicAdd scalar | YES + reduction | OMP reduction clause |
| G | Stencil, halo exchange | YES | Ghost zone handling |

### 5. CUDA-Specific Data Analysis
For each array:
- Memory type: __global__, __shared__, __constant__, host
- Transfer pattern: cudaMemcpy direction and frequency
- Allocation: cudaMalloc vs managed memory
- Device pointers vs host pointers
- Struct members on device?

CUDA constructs to document:
- Thread indexing: threadIdx, blockIdx, blockDim, gridDim
- Synchronization: __syncthreads(), kernel boundaries
- Memory access patterns: coalesced vs strided
- Atomic operations and their locations

### 6. Flag OMP Migration Issues
- __syncthreads() usage (no direct OMP equivalent)
- Shared memory dependencies (complex privatization)
- Atomics (performance penalty in OMP)
- Reduction patterns (may need manual implementation)
- <10K total threads (overhead concern)
- Dynamic parallelism (not in OMP)
- Warp-level primitives (no OMP equivalent)

## Output: analysis.md

### File Conversion Mapping
```
original.cu → converted.c
kernel_utils.cu → kernel_utils.cpp
```

### Kernel/Loop Nesting Structure
```
- host_loop (line:X) calls kernel1 
  └── kernel1<<<grid,block>>> (line:Y) Type A
      └── device_loop (line:Z) Type A
- kernel2<<<grid,block>>> (line:W) Type D
```

### Kernel/Loop Details
For each CRITICAL/IMPORTANT/SECONDARY kernel or loop:
```
## Kernel/Loop: [name] at [file:line]
- **Context:** [__global__ kernel / host loop / __device__ function]
- **Launch config:** [grid_size × block_size] or [iterations]
- **Total threads/iterations:** [count]
- **Type:** [A-G] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [device loops or none]
- **Dependencies:** [none / atomicAdd / __syncthreads / reduction]
- **Shared memory:** [YES/NO - size and usage]
- **Thread indexing:** [pattern used]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW) - memory type]
- **OMP Migration Issues:** [flags]
```

### Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|-----------------|------|----------|---------|------------|--------------|------------|

### CUDA-Specific Details
- **Dominant compute kernel:** [main timed kernel]
- **Memory transfers in timed loop?:** YES/NO
- **Shared memory usage:** [total bytes, patterns]
- **Synchronization points:** [__syncthreads locations]
- **Atomic operations:** [locations and variables]
- **Reduction patterns:** [manual vs atomicAdd]

### OMP Migration Strategy Notes
- **Direct kernel → parallel for:** [list]
- **Requires restructuring:** [list with reasons]
- **Performance concerns:** [atomics, false sharing, etc.]
- **Data management:** [allocation changes needed]

## Constraints
- Find all kernels and loops called from main compute section
- Document CUDA-specific constructs for migration planning
- Copy all source files with suffix conversion (.cu → .c/.cpp)
- No code modifications - documentation only
- Identify __syncthreads() patterns (critical for OMP feasibility)
'''



'''# CUDA to OpenMP Migration

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for all runs
- DO NOT use `distribute parallel for`

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline
```bash
Baseline cuda outpuut is in baseline_output.txt in {kernel_dir}/
```

### 2. Choose Data Strategy
Walk through IN ORDER, stop at first match:

```
RULE 1: Type B (Sparse/CSR)?              → STRATEGY A/C
RULE 2: Type C1 (Iterative Solvers/Butterfly)?→ STRATEGY C
RULE 3: Type C2 (Multigrid)?              → STRATEGY A
RULE 4: Multiple independent kernels?     → STRATEGY B
RULE 5: Otherwise                         → STRATEGY A
```

### 2.5. Create Data Management Plan
MANDATORY: Create data_plan.md in {kernel_dir} before implementation

**FIRST: Understand CUDA memory model and map to OMP:**
- cudaMalloc + device pointers → omp_target_alloc OR target data map(alloc)
- cudaMemcpy H→D → map(to) OR omp_target_memcpy OR update to
- cudaMemcpy D→H → map(from) OR omp_target_memcpy OR update from
- Kernel launches in loops → target teams loop with is_device_ptr

**CUDA Pattern Recognition:**
```
Pattern 1: cudaMalloc once → kernel loop → cudaFree
  → Strategy C: omp_target_alloc + is_device_ptr

Pattern 2: Single kernel launch with data transfer
  → Strategy A: target data region

Pattern 3: Multiple kernels with dependencies
  → Strategy B: nowait + depend clauses
```

Analyze ALL arrays and kernels in timed region:

```markdown
# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| d_[name] | cudaMalloc | [bytes] | H→D once/D→H once/both |
| [name] | host array | [bytes] | source/destination |

**CUDA Operations:**
- cudaMalloc calls: [list with sizes]
- cudaMemcpy H→D: [list with timing]
- cudaMemcpy D→H: [list with timing]
- Kernel launches: [list with frequency]

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| kernel_name<<<G,B>>> | grid=[X], block=[Y] | per-iteration/once | [list] |

**Kernel Launch Patterns:**
- In outer loop? → Multiple target teams loop
- Sequential kernels? → Multiple target regions OR nowait+depend
- Conditional launch? → target if clause

## OMP Data Movement Strategy

**Chosen Strategy:** [A/B/C]

**Rationale:** [Map CUDA pattern to strategy]

**Device Allocations (OMP equivalent):**
```
CUDA: cudaMalloc(&d_arr, size)
OMP Strategy C: d_arr = omp_target_alloc(size, 0)
OMP Strategy A: #pragma omp target data map(alloc:arr[0:n])
```

**Host→Device Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice)
OMP Strategy C: omp_target_memcpy(d_arr, h_arr, size, 0, 0, 0, omp_get_initial_device())
OMP Strategy A: map(to:arr[0:n]) OR #pragma omp target update to(arr[0:n])
```
- When: [before iterations/once at start]
- Arrays: [list with sizes]
- Total H→D: ~[X] MB

**Device→Host Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost)
OMP Strategy C: omp_target_memcpy(h_arr, d_arr, size, 0, 0, omp_get_initial_device(), 0)
OMP Strategy A: map(from:arr[0:n]) OR #pragma omp target update from(arr[0:n])
```
- When: [after iterations/once at end]
- Arrays: [list with sizes]
- Total D→H: ~[Y] MB

**Transfers During Iterations:** [YES/NO]
- If YES: [which arrays and why - may indicate wrong strategy]

## Kernel to OMP Mapping (short)
- Replace each CUDA kernel launch with a `#pragma omp target teams loop` over the same *logical* work domain.
- Replace `blockIdx/threadIdx` indexing with the loop induction variable.
- Keep bounds checks; keep inner device loops as normal C loops inside the offloaded loop body.

## Critical Migration Issues

**From analysis.md "OMP Migration Issues":**
- [ ] __syncthreads() usage: [locations and resolution strategy]
- [ ] Shared memory: [convert to private/firstprivate]
- [ ] Atomics: [verify OMP atomic equivalents]
- [ ] Dynamic indexing: [verify OMP handles correctly]

**__syncthreads() Resolution:**
- Within single kernel → May need to split into multiple target regions
- At kernel boundaries → Natural OMP barrier between target regions
- Strategy: [describe approach]

**Shared memory / barriers:**
- No direct equivalent for CUDA `__shared__` + `__syncthreads()`; refactor and document your approach.

## Expected Performance
- CUDA kernel time: [X] ms (from profiling if available)
- OMP expected: [Y] ms (may be slower due to __syncthreads elimination)
- Red flag: If >3x slower → wrong strategy or missing parallelism

**Summary:** [num] kernels, [num] device arrays, Strategy [A/B/C]. 
CUDA pattern: [describe]. OMP approach: [describe].
Expected: ~[X] MB H→D, ~[Y] MB D→H.
```

### 2.6. Implement Data Plan

**Use data_plan.md as implementation guide**

### Step 1: Remove CUDA API Calls
From "CUDA Memory Analysis":
- Remove all cudaMalloc/cudaFree calls
- Remove all cudaMemcpy calls
- Remove kernel launch syntax <<<grid, block>>>
- Keep all kernel BODY code (will convert to functions)

### Step 2: Convert Kernels to Functions
From "Kernel Inventory":
```
CUDA:
  __global__ void kernel_name(double *arr, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = ...;
  }}

OMP:
  void kernel_name(double *arr, int n) {{
    #pragma omp target teams loop is_device_ptr(arr)
    for (int idx = 0; idx < n; idx++) {{  
      arr[idx] = ...;
    }}
  }}
```

### Step 3: Setup Data Structures
From "OMP Data Movement Strategy":
- Create OMP allocations based on chosen strategy
- For Strategy C: Add omp_target_alloc calls
- For Strategy A: Setup target data regions

### Step 4: Implement Transfers
From "Host→Device" and "Device→Host" sections:
- Implement transfers using method for chosen strategy
- Match timing from original CUDA code

### Step 5: Convert Thread Indexing
From "Thread Indexing Conversion":
- Replace blockIdx/threadIdx with loop iterator
- Remove if (idx < N) guards (loop bounds handle this)
- Convert grid-stride loops to simple loops

### Step 6: Handle Special CUDA Constructs
From "Critical Migration Issues":
- **atomicAdd** → `#pragma omp atomic update`
- **__syncthreads()** → Split kernel OR remove if not critical
- **Shared memory** → Per-thread private OR elimination
- **Reduction in kernel** → `reduction(op:var)` clause

### Step 7: Verify Implementation
Check ALL items in "Critical Migration Issues":
- [ ] All kernels converted to OMP functions
- [ ] Thread indexing removed
- [ ] Memory management matches strategy
- [ ] Special constructs handled

**Common errors:** 
- Forgot to remove <<<>>> syntax
- Left blockIdx/threadIdx in code
- Missed cudaMemcpy conversions
- Wrong is_device_ptr usage

**CRITICAL: OpenMP Clause Syntax Limitation**
OpenMP pragma clauses (`is_device_ptr`, `use_device_addr`, `map`) do NOT support struct member access.
You MUST extract struct members to local pointer variables first.

WRONG (will not compile):
```c
#pragma omp target teams loop is_device_ptr(data.arr1, data.arr2)
```

CORRECT:
```c
double *d_arr1 = data.arr1;
double *d_arr2 = data.arr2;
#pragma omp target teams loop is_device_ptr(d_arr1, d_arr2)
for (int i = 0; i < n; i++) {{
    // use d_arr1[i], d_arr2[i] inside the loop
}}
```

When converting CUDA code that passes structs to kernels, extract ALL device pointer members
to local variables BEFORE the pragma, then use those local variables in the clause AND loop body.

**Ready when:** Compiles and runs with OMP flags, no CUDA API calls remain

---

## Strategy / Pattern Notes (short)
- Strategy A: `target data map(...)` for simpler flows (few kernels).
- Strategy C: `omp_target_alloc` + `omp_target_memcpy` + `is_device_ptr` for persistent device pointers (CUDA-like).
- Device helpers: former `__device__` helpers typically need `#pragma omp declare target`.

## 5. Compile and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > gpu_output.txt 2>&1
```

If timeout/segfault: Check for unconverted CUDA constructs.
If core dumped/Aborted: run compute sanitizer.

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

## RULES - BREAKING A RULE = FAILURE
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- DO NOT CHANGE/EDIT FILES OTHER THAN {file_listing}
- REMOVE ALL CUDA API CALLS (cudaMalloc, cudaMemcpy, cudaFree, kernel<<<>>>)
- CONVERT ALL __global__ FUNCTIONS TO REGULAR FUNCTIONS
- REMOVE ALL CUDA-SPECIFIC SYNTAX (blockIdx, threadIdx, __syncthreads, __shared__)
'''




'''
# Performance Tuning - CUDA to OMP Migration

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Do not change data strategy from used in the code**

## EARLY EXIT CHECK
If current runtime is within 5% of expected optimal (based on nsys kernel times):
- Document current metrics in optimization_plan.md
- Skip optimization - code is already well-tuned
- Focus only on micro-optimizations (const, restrict, cache locals)

## Context: CUDA to OMP Migration
The code was migrated from CUDA to OMP. Key differences affect optimization:
- CUDA kernels → OMP target teams loop
- cudaMemcpy → OMP map clauses or omp_target_memcpy
- __syncthreads() → May have been split into multiple target regions
- Shared memory → Converted to private or eliminated
- atomicAdd → OMP atomic

**Common migration bottlenecks:**
1. Excessive data transfers (lost explicit CUDA control)
2. Over-decomposed kernels (from __syncthreads() elimination)
3. Missing collapse on nested loops (CUDA had 2D/3D grids)
4. Suboptimal thread mapping (CUDA grid-stride → OMP loop)

## Workflow

### 1. Verify Baseline
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt | grep -E "Verification|SUCCESSFUL|FAILED"
```

If results differ, fix Step 2 first.
If there are any errors, fix them before continuing.

### 2. Analyze Profile and Create Plan

2.1. Read profile data:
```bash
# Try to find kernel information (OpenMP kernels may not appear in standard sections)
cat {profile_log_path} | grep -A20 "cuda_gpu_kern_sum" || echo "No cuda_gpu_kern_sum found - kernels may not be offloading to GPU"
cat {profile_log_path} | grep -A10 "cuda_api_sum"
cat {profile_log_path} | grep -A10 "cuda_gpu_mem_time_sum"
# Also check for any GPU activity
cat {profile_log_path} | grep -i "gpu\|kernel\|target" | head -20
```

2.2. Check GPU capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```
Roughly estimate the GPU saturation threshold

2.3. Compare with original CUDA performance (if available):
- CUDA kernel time: [X]ms
- OMP target teams loop time: [Y]ms
- Ratio: [Y/X]
- If >2x slower: Major optimization opportunity

---

3. Create optimization_plan.md in {kernel_dir}:
```markdown
# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: [X]s
- Main kernel: [name], [Y]% GPU, [Z] instances
- Memory transfer: [%] time, [MB] total
- Kernel launches: [count]

## Bottleneck Hypothesis (pick 1–2)
- [ ] Transfers too high (CUDA avoided transfers in loop)
- [ ] Too many kernels / target regions (launch overhead)
- [ ] Missing collapse vs CUDA grid dimensionality
- [ ] Hot kernel needs micro-opts

## Actions (1–3 max)
1. [ACTION]: [what] - [why] - expected [gain]
2. [ACTION]: ...
```

### Fusion Rules

**Fuse when:**
- CUDA had single kernel for operations
- Adjacent independent, same bounds
- Producer-consumer in CUDA
- Multi-vector ops in one CUDA kernel

**Don't fuse:**
- Different bounds
- CUDA had separate kernels with cudaDeviceSynchronize()
- __syncthreads() required synchronization

### 3. Execute Optimization Plan
- Apply changes and document in optimization_plan.md

### 4. Optimization Actions (short)
- **Transfers high**: hoist data; use `omp_target_alloc` + `is_device_ptr` for persistent arrays; avoid per-iteration mapping
- **Too many target regions**: fuse adjacent target loops; inline helper kernels when safe
- **Grid shape mismatch**: add `collapse(N)` to mirror CUDA grid dimensionality
- **Kernel micro-opts**: `const`, `restrict`, cache locals, reduce recomputation

### 5. Final Summary
Update optimization_plan.md:
```markdown
# Final Performance Summary - CUDA to OMP Migration

### Baseline (from CUDA)
- CUDA Runtime: [X]s (if available)
- CUDA Main kernel: [Y] launches, [Z]ms total

### OMP Before Optimization
- Runtime: [X]s
- Slowdown vs CUDA: [X]x
- Main kernel: [Y] instances, [Z]ms total

### OMP After Optimization
- Runtime: [X]s
- Slowdown vs CUDA: [X]x (target <1.5x)
- Speedup vs initial OMP: [X]x
- Main kernel: [Y] instances, [Z]ms total

### Optimizations Applied
1. [X] [ACTION]: [description] → [±X%] [recovered CUDA pattern Y]
2. [X] [ACTION]: REVERTED (slower)

### CUDA→OMP Recovery Status
- [X] Restored 2D/3D grid mapping with collapse
- [X] Matched CUDA kernel fusion structure
- [X] Eliminated excessive transfers (matched CUDA pattern)
- [ ] Still missing: [any CUDA optimizations that couldn't be recovered]

### Micro-optimizations Applied
1. [X] [MICRO-OPT]: [description] → [±X%]
2. [X] [MICRO-OPT]: REVERTED (slower)

### Key Insights
- [Most impactful optimization - relate to CUDA pattern]
- [Remaining bottlenecks vs CUDA]
- [OMP limitations compared to CUDA]
```

## Optimization Checklist (short)
- [ ] Transfers dominate: hoist data; `omp_target_alloc` + `is_device_ptr`; avoid per-iter mapping
- [ ] Too many kernels/regions: fuse adjacent target loops; inline helper kernels when safe
- [ ] Missing CUDA grid shape: add `collapse(N)`
- [ ] Hot kernel: `const`, `restrict`, cache locals, reduce recomputation (and `simd` where safe)

## Profiling
```bash
{clean_cmd_str}
# Fallback: {build_cmd_str} run > {profile_log_path} 2>&1
# Check for kernel information (OpenMP kernels may appear in cuda_gpu_kern_sum or with different names)
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found - check if code is offloading to GPU"
```

### Deliverables
- optimization_plan.md - Complete analysis including CUDA comparison
- Optimized source code
- Final profile: {profile_log_path}

**REMINDER: OpenMP Clause Syntax**
OpenMP clauses (`is_device_ptr`, `use_device_addr`, `map`) require bare pointer variables.
Extract struct members to local variables before the pragma:
```c
double *d_arr = data.arr;  // Extract first
#pragma omp target teams loop is_device_ptr(d_arr)  // Use local var
```

## RULES - BREAKING A RULE = FAILURE
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- DO NOT CHANGE FILES OTHER THAN {file_listing}
- PRESERVE CORRECTNESS - diff against baseline after each change
'''