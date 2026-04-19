# Loop Classification for OMP Migration - Analysis Phase

## Task
Analyze CUDA kernels in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. Copy source files to `{kernel_dir}/` with suffix conversion (.cu → .c or .cpp).

**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Process

> **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

### 0. COPY SOURCE FILES WITH SUFFIX CONVERSION
- Copy `{file_listing}` from `{source_dir}/` to `{kernel_dir}/`
- Convert suffixes: `.cu` → `.c` (for C code) or `.cpp` (for C++ code). You can inspect the makefile in {kernel_dir}/ to see the expected file names.
- Get baseline output. Run {clean_cmd_str} and `{run_cmd_str} > baseline_output.txt 2>&1` in {source_dir}/. Copy the baseline output to {kernel_dir}/baseline_output.txt.
- Preserve all file content exactly - no code modifications
- Document mapping: `original.cu → converted.c` in analysis.md
- Convert header includes in {file_listing}. Make sure the code can be compiled with the converted files.
- You may create documentation/output files (analysis.md, baseline_output.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}

## Create Environment
**You need** to create an environment to run the code in {kernel_dir}.
That means:
- Create any header files, util files, etc. that are needed to run the code.
- Create a Makefile called Makefile.nvc in {kernel_dir}/ that can be used to run the code. The compiler that needs to be used is nvc++.

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

|| Type | CUDA Pattern | OMP Equivalent | Notes |
||------|--------------|----------------|-------|
|| A | Dense kernel, regular grid | YES - parallel for | Direct map |
|| B | Sparse (CSR), varying bounds | Outer only | Inner sequential |
|| C1 | Multi-kernel, global sync | Outer only | Barrier between stages |
|| C2 | Hierarchical grid | Outer only | Nested parallelism tricky |
|| D | Histogram, atomicAdd | YES + atomic | Performance loss expected |
|| E | __syncthreads, shared deps | NO | Requires restructuring |
|| F | Reduction, atomicAdd scalar | YES + reduction | OMP reduction clause |
|| G | Stencil, halo exchange | YES | Ghost zone handling |

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
|| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
||-----------------|------|----------|---------|------------|--------------|------------|

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
```
