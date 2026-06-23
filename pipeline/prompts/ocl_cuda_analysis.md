# OpenCL to CUDA Migration - Analysis Phase

## Task
Analyze OpenCL kernels in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. Copy source files to `{kernel_dir}/` with suffix conversion (.cl → .cu, host code → .cu).

**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Process

> **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

### 0. COPY SOURCE FILES WITH SUFFIX CONVERSION
- Copy `{file_listing}` from `{source_dir}/` to `{kernel_dir}/`
- Convert suffixes:
  - `.cl` kernel files → `.cu` (CUDA source)
  - Host `.c/.cpp` files → `.cu` (unified CUDA source)
  - Check Makefile in `{kernel_dir}/` to determine expected file names
  - if Makefile is unavailable - default is 1 `.cu` file for both host code and device code. 
- Update `#include` statements if converting headers:
  - Change `#include "foo.h"` → `#include "foo.cuh"` (if converting to .cuh)
- Get baseline output. Run {clean_cmd_str} and `{run_cmd_str} > baseline_output.txt 2>&1` in {source_dir}/. Copy the baseline output to {kernel_dir}/baseline_output.txt.
- Preserve all kernel logic - no algorithmic modifications
- Document mapping: `kernels.cl + host.cpp → combined.cu` in analysis.md
- You may create documentation/output files (analysis.md, baseline_output.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}

### 1. Find All OpenCL Kernels and Analyze Structure
```bash
# Find OpenCL kernels
grep -n "__kernel\|kernel void" *.cl 2>/dev/null

# Find kernel enqueue sites
grep -n "clEnqueueNDRangeKernel" *.c *.cpp 2>/dev/null

# Find work-item loops (inside kernels)
grep -n "for\s*(" *.cl 2>/dev/null | head -100

# Find host loops calling kernels
grep -n "for.*iter\|for.*it\|while" *.c *.cpp 2>/dev/null | head -50

# Find synchronization points
grep -n "barrier\|clFinish" *.cl *.c *.cpp 2>/dev/null
```

Prioritize by execution pattern:
- Kernel called every iteration → CRITICAL/IMPORTANT
- Kernel called once at setup → SECONDARY/AVOID
- Work-item loops inside kernels → analyze work per item

### 2. Classify Priority
For each kernel: `num_groups × local_size × device_iterations × ops = total work`

- **CRITICAL:** >50% runtime OR called every iteration with O(N) work
- **IMPORTANT:** 5-50% runtime OR called every iteration with small work
- **SECONDARY:** Called once at setup
- **AVOID:** Setup/IO/memory allocation OR <10K total work-items

### 3. Determine Kernel Type (Decision Tree)

```
Q0: Is this a __kernel or regular function? → Note context
Q1: Writes A[idx[i]] with varying idx (atomic_add)? → Type D (Histogram)
Q2: Uses barrier() or __local dependencies? → Type E (Work-group synchronization)
Q3: Multi-stage kernel pattern?
    - Separate kernels for stages with clFinish? → C1 (FFT/Butterfly)
    - Hierarchical enqueues? → C2 (Multigrid)
Q4: Work-group/item indexing varies with outer dimension? → Type B (Sparse)
Q5: Uses atomic_add to scalar (reduction pattern)? → Type F (Reduction)
Q6: Accesses neighboring work-items' data? → Type G (Stencil)
Q7: Uses image/sampler objects? → Type H (Image-based)
Default → Type A (Dense)
```

**OpenCL-to-CUDA Specific Patterns:**
- **barrier():** Maps to `__syncthreads()` - direct equivalent
- **__local memory:** Maps to `__shared__` - same semantics
- **Atomic operations:** OpenCL atomics → CUDA atomics (syntax differs)
- **Image objects:** Map to texture memory (different API)
- **Work-item functions:** get_global_id, etc. → thread indexing math
- **Sub-groups:** Map to warp-level primitives (CUDA has more features)
- **Pipes:** No direct CUDA equivalent - requires restructuring

### 4. Type Reference

|| Type | OpenCL Pattern | CUDA Equivalent | Notes |
||------|----------------|-----------------|-------|
|| A | Dense kernel, regular NDRange | YES - kernel launch | Direct map |
|| B | Sparse (CSR), varying bounds | Outer only | Inner sequential |
|| C1 | Multi-kernel, clFinish sync | Multiple kernels | Same pattern |
|| C2 | Hierarchical enqueues | Multiple launches | No device enqueue |
|| D | Histogram, atomic_add | YES + atomicAdd | Direct map |
|| E | barrier, __local | YES - __syncthreads, __shared__ | Direct map |
|| F | Reduction, atomic_add scalar | YES + atomicAdd | Direct map |
|| G | Stencil, work-group sharing | YES | Same pattern |
|| H | Image objects | Texture memory | Different API |

### 5. OpenCL-Specific Data Analysis
For each buffer/image:
- **Memory type:** __global, __local, __constant, host
- **Transfer pattern:** clEnqueueWriteBuffer/ReadBuffer frequency
- **Allocation:** clCreateBuffer flags (READ_WRITE, READ_ONLY, etc.)
- **Image objects:** Format, dimensions, sampler properties

OpenCL constructs to document:
- **Work-item indexing:** get_local_id(), get_group_id(), get_local_size(), get_num_groups()
  - Maps to: threadIdx, blockIdx, blockDim, gridDim
- **Synchronization:** barrier(), clFinish()
  - Maps to: __syncthreads(), cudaDeviceSynchronize()
- **Memory qualifiers:** __local, __constant, __global
  - Maps to: __shared__, __constant__, (no qualifier)
- **Atomic operations:** atomic_add, atomic_max, etc.
  - Maps to: atomicAdd, atomicMax, etc.

### 6. Flag CUDA Migration Issues
- **Sub-groups with explicit size** - CUDA warps are architecture-dependent
- **Device-side enqueue** - NOT supported in CUDA (requires restructuring)
- **Pipes** - NO CUDA equivalent
- **Image objects with samplers** - Different texture API
- **cl_khr_fp16 half types** - CUDA has native half support but different syntax
- **Atomic operations on floats** - CUDA has atomicAdd for float (OpenCL may use atomic_xchg workaround)
- **Local memory size queries** - Different API
- **printf in kernels** - Both support but CUDA is easier

## Output: analysis.md

### File Conversion Mapping
```
kernels.cl → combined.cu (kernel code as __global__)
host.cpp → combined.cu (host code in same file)
utils.h → utils.cuh (or keep as .h)
```

### Kernel Structure
```
- host_loop (line:X) enqueues kernel1
  └── kernel1 NDRange (line:Y) Type A
      └── work-item loop (line:Z) Type A
      └── barrier() (line:W) - maps to __syncthreads()
- kernel2 NDRange (line:V) Type D
    └── atomic_add operations - maps to atomicAdd()
```

### Kernel Details
For each CRITICAL/IMPORTANT/SECONDARY kernel:
```
## Kernel: [name] at [file:line]
- **Context:** [__kernel function]
- **NDRange config:** [global_work_size / local_work_size]
- **Work dimensions:** [1D/2D/3D]
- **Total work-items:** [count]
- **Type:** [A-H] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [work-item loops or none]
- **Dependencies:** [none / atomic_add / barrier / reduction]
- **Local memory:** [YES/NO - size bytes]
- **Work-item indexing pattern:** [1D/2D/3D]
- **Private vars:** [list]
- **Buffers:** [name(R/W/RW) - memory type]
- **CUDA Migration Issues:** [flags from section 6]
```

### CUDA Mapping Table
|| OpenCL Construct | CUDA Equivalent | Complexity |
||------------------|-----------------|------------|
|| __kernel void kernel() | __global__ void kernel() | Trivial |
|| get_local_id(0) | threadIdx.x | Trivial |
|| get_group_id(0) | blockIdx.x | Trivial |
|| get_local_size(0) | blockDim.x | Trivial |
|| get_num_groups(0) | gridDim.x | Trivial |
|| get_global_id(0) | blockIdx.x*blockDim.x + threadIdx.x | Trivial |
|| barrier(CLK_LOCAL_MEM_FENCE) | __syncthreads() | Trivial |
|| __local float arr[N] | __shared__ float arr[N] | Trivial |
|| __constant | __constant__ | Trivial |
|| atomic_add(&var, val) | atomicAdd(&var, val) | Simple |
|| clCreateBuffer | cudaMalloc | Moderate |
|| clEnqueueWriteBuffer | cudaMemcpy H→D | Moderate |
|| clEnqueueNDRangeKernel | kernel<<<grid,block>>>() | Moderate |
|| Image objects | Texture memory | Moderate |
|| Sub-groups (explicit) | Warp primitives | Moderate |
|| Device enqueue | NO EQUIVALENT | Complex |
|| Pipes | NO EQUIVALENT | Complex |

### Summary Table
|| Kernel/Function | Type | Priority | NDRange Config | Total Work | Dependencies | CUDA Issues |
||-----------------|------|----------|----------------|------------|--------------|-------------|

### OpenCL-Specific Details
- **Dominant compute kernel:** [main timed kernel]
- **Memory transfers in timed loop?:** YES/NO
- **Local memory usage:** [total bytes, patterns]
- **Synchronization points:** [barrier locations, clFinish calls]
- **Atomic operations:** [locations and variables]
- **Image/sampler usage:** [formats and access patterns]
- **Sub-group operations:** [locations - consider warp-level optimization]
- **Device enqueue:** [YES/NO - requires restructuring for CUDA]

### CUDA Migration Strategy Notes
- **Direct kernel conversion:** [list - simple __kernel → __global__]
- **Requires restructuring:** [list with reasons - device enqueue, pipes]
- **Performance opportunities:** [CUDA warp primitives, faster texture cache]
- **Memory management simplification:** [CUDA API simpler than OpenCL]
- **API setup simplification:** [No context/queue/program compilation boilerplate]
- **Expected complexity:** [LOW/MEDIUM/HIGH based on issues]

## Constraints
- Find all kernels and their enqueue sites
- Document OpenCL-specific constructs for migration planning
- Copy all source files with suffix conversion (.cl → .cu)
- No code modifications - documentation only
- Identify device-side enqueue (CRITICAL - requires algorithm restructuring)
- Note image object usage (requires CUDA texture API)
- Flag sub-group operations (consider warp-level optimizations in CUDA)
```

---

## Translation Step 1: OpenCL to CUDA Implementation
