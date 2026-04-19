# CUDA to OpenMP Migration

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for all runs
- DO NOT use `distribute parallel for`
- **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

** IMPORTANT ** YOU MAY MODIFY THE MAKEFILE TO ADD ANYTHING YOU NEED TO RUN THE CODE.

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline
```bash
Baseline cuda output is in baseline_output.txt in {kernel_dir}/
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

|| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
||---------------|-----------------|------|------------------|
|| d_[name] | cudaMalloc | [bytes] | H→D once/D→H once/both |
|| [name] | host array | [bytes] | source/destination |

**CUDA Operations:**
- cudaMalloc calls: [list with sizes]
- cudaMemcpy H→D: [list with timing]
- cudaMemcpy D→H: [list with timing]
- Kernel launches: [list with frequency]

## Kernel Inventory
|| Kernel Name | Launch Config | Frequency | Arrays Used |
||-------------|---------------|-----------|-------------|
|| kernel_name<<<G,B>>> | grid=[X], block=[Y] | per-iteration/once | [list] |

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
  __global__ void kernel_name(double *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = ...;
  }

OMP:
  void kernel_name(double *arr, int n) {
    #pragma omp target teams loop is_device_ptr(arr)
    for (int idx = 0; idx < n; idx++) {  
      arr[idx] = ...;
    }
  }
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
for (int i = 0; i < n; i++) {
    // use d_arr1[i], d_arr2[i] inside the loop
}
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
- You may create backup/output files (*.bak, *.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}
- REMOVE ALL CUDA API CALLS (cudaMalloc, cudaMemcpy, cudaFree, kernel<<<>>>)
- CONVERT ALL __global__ FUNCTIONS TO REGULAR FUNCTIONS
- REMOVE ALL CUDA-SPECIFIC SYNTAX (blockIdx, threadIdx, __syncthreads, __shared__)
```

