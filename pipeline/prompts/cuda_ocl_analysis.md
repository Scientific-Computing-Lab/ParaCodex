# CUDA to OpenCL Migration - Analysis Phase

## Task
Analyze CUDA kernels in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. Copy source files to `{kernel_dir}/` with suffix conversion (.cu → .cl for kernels, .cu → .c/.cpp for host code).

**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Process

> **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

### 0. COPY SOURCE FILES WITH SUFFIX CONVERSION
- Copy `{file_listing}` from `{source_dir}/` to `{kernel_dir}/`
- Convert suffixes:
  - `.cu` kernel files → `.cl` (OpenCL kernel source)
  - `.cu` host files → `.c` or `.cpp` (based on Makefile expectations)
  - Check Makefile in `{kernel_dir}/` to determine expected file names
  - if Makefile is unavailable - default is 1 `.cpp` or `.c` file, and 1 `.cl` file. 
- Update `#include` statements in all files:
  - Change `#include "foo.cuh"` → `#include "foo.h"`
- Get baseline output. Run {clean_cmd_str} and `{run_cmd_str} > baseline_output.txt 2>&1` in {source_dir}/. Copy the baseline output to {kernel_dir}/baseline_output.txt.
- Preserve all kernel logic - no algorithmic modifications
- Document mapping: `original.cu → kernel.cl + host.cpp` in analysis.md
- You may create documentation/output files (analysis.md, baseline_output.txt, etc.)
- ONLY EDIT SOURCE CODE IN: {file_listing}

### 1. Find All CUDA Kernels and Analyze Structure
```bash
# Find CUDA kernels
grep -n "__global__\|__device__" *.cu 2>/dev/null

# Find kernel launch sites
grep -n "<<<.*>>>" *.cu 2>/dev/null

# Find device loops (inside kernels)
grep -n "for\s*(" *.cu 2>/dev/null | head -100

# Find host loops calling kernels
grep -n "for.*iter\|for.*it\|while" *.cu 2>/dev/null | head -50

# Find synchronization points
grep -n "__syncthreads\|cudaDeviceSynchronize" *.cu 2>/dev/null
```

Prioritize by execution pattern:
- Kernel called every iteration → CRITICAL/IMPORTANT
- Kernel called once at setup → SECONDARY/AVOID
- Device loops inside kernels → analyze work per thread

### 2. Classify Priority
For each kernel: `grid_size × block_size × device_iterations × ops = total work`

- **CRITICAL:** >50% runtime OR called every iteration with O(N) work
- **IMPORTANT:** 5-50% runtime OR called every iteration with small work
- **SECONDARY:** Called once at setup
- **AVOID:** Setup/IO/memory allocation OR <10K total threads

### 3. Determine Kernel Type (Decision Tree)

```
Q0: Is this a __global__ kernel or __device__ function? → Note context
Q1: Writes A[idx[i]] with varying idx (atomicAdd)? → Type D (Histogram)
Q2: Uses __syncthreads() or __shared__ dependencies? → Type E (Block-level synchronization)
Q3: Multi-stage kernel pattern?
    - Separate kernels for stages with global sync? → C1 (FFT/Butterfly)
    - Hierarchical grid calls? → C2 (Multigrid)
Q4: Block/thread indexing varies with outer dimension? → Type B (Sparse)
Q5: Uses atomicAdd to scalar (reduction pattern)? → Type F (Reduction)
Q6: Accesses neighboring threads' data? → Type G (Stencil)
Q7: Uses texture/surface memory? → Type H (Texture-based)
Default → Type A (Dense)
```

**CUDA-to-OpenCL Specific Patterns:**
- **__syncthreads():** Maps to `barrier(CLK_LOCAL_MEM_FENCE)` - direct equivalent
- **__shared__ memory:** Maps to `__local` - size must be known at kernel enqueue
- **Atomic operations:** CUDA atomics → OpenCL atomics (syntax differs)
- **Warp-level primitives:** No direct OpenCL equivalent - requires restructuring
- **Dynamic parallelism:** Not supported in OpenCL - must flatten
- **Texture memory:** Maps to OpenCL images (different API)

### 4. Type Reference

|| Type | CUDA Pattern | OpenCL Equivalent | Notes |
||------|--------------|-------------------|-------|
|| A | Dense kernel, regular grid | YES - NDRange | Direct map |
|| B | Sparse (CSR), varying bounds | Outer only | Inner sequential |
|| C1 | Multi-kernel, global sync | Multiple kernels | Explicit command queue ordering |
|| C2 | Hierarchical grid | Multiple kernels | No nested dispatch |
|| D | Histogram, atomicAdd | YES + atomic | atomic_add() |
|| E | __syncthreads, __shared__ | YES - barrier() | Local memory + barrier |
|| F | Reduction, atomicAdd scalar | YES + reduction | May need local reduction |
|| G | Stencil, halo exchange | YES | Work-group handling |
|| H | Texture memory | Image objects | Requires sampler objects |

### 5. CUDA-Specific Data Analysis
For each array:
- **Memory type:** __global__, __shared__, __constant__, host
- **Transfer pattern:** cudaMemcpy direction and frequency
- **Allocation:** cudaMalloc vs managed memory
- **Device vs host pointers**
- **Texture/surface bindings**

CUDA constructs to document:
- **Thread indexing:** threadIdx, blockIdx, blockDim, gridDim
  - Maps to: get_local_id(), get_group_id(), get_local_size(), get_num_groups()
- **Synchronization:** __syncthreads(), cudaDeviceSynchronize()
  - Maps to: barrier(), clFinish()
- **Memory qualifiers:** __shared__, __constant__
  - Maps to: __local, __constant
- **Atomic operations:** atomicAdd, atomicMax, etc.
  - Maps to: atomic_add, atomic_max, etc.

### 6. Flag OpenCL Migration Issues
- **Warp-level primitives** (__shfl, __ballot, etc.) - NO OpenCL equivalent
- **Dynamic parallelism** - NOT supported in OpenCL
- **Texture memory** - Different API (images + samplers)
- **Dynamic __shared__ memory** - Must specify size at enqueue
- **Cross-block synchronization** - Requires multiple kernel launches
- **CUDA-specific math functions** - Check OpenCL equivalents
- **Half-precision ops** - OpenCL extension required (cl_khr_fp16)
- **Cooperative groups** - NO OpenCL equivalent

## Output: analysis.md

### File Conversion Mapping
```
original.cu → kernels.cl (kernel code)
original.cu → host.cpp (host code)
utils.cuh → utils.h
```

### Kernel Structure
```
- host_loop (line:X) launches kernel1
  └── kernel1<<<grid,block>>> (line:Y) Type A
      └── device_loop (line:Z) Type A
      └── __syncthreads() (line:W) - maps to barrier()
- kernel2<<<grid,block>>> (line:V) Type D
    └── atomicAdd operations - maps to atomic_add()
```

### Kernel Details
For each CRITICAL/IMPORTANT/SECONDARY kernel:
```
## Kernel: [name] at [file:line]
- **Context:** [__global__ kernel / __device__ function]
- **Launch config:** [grid_size × block_size]
- **Total work-items:** [count]
- **Type:** [A-H] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [device loops or none]
- **Dependencies:** [none / atomicAdd / __syncthreads / reduction]
- **Shared memory:** [YES/NO - size (dynamic/static)]
- **Thread indexing pattern:** [1D/2D/3D]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW) - memory type]
- **OpenCL Migration Issues:** [flags from section 6]
```

### OpenCL Mapping Table
|| CUDA Construct | OpenCL Equivalent | Complexity |
||----------------|-------------------|------------|
|| __global__ void kernel() | __kernel void kernel() | Trivial |
|| threadIdx.x | get_local_id(0) | Trivial |
|| blockIdx.x | get_group_id(0) | Trivial |
|| blockDim.x | get_local_size(0) | Trivial |
|| gridDim.x | get_num_groups(0) | Trivial |
|| __syncthreads() | barrier(CLK_LOCAL_MEM_FENCE) | Trivial |
|| __shared__ float arr[N] | __local float arr[N] | Trivial |
|| __constant__ | __constant | Trivial |
|| atomicAdd(&var, val) | atomic_add(&var, val) | Simple |
|| cudaMalloc | clCreateBuffer | Moderate |
|| cudaMemcpy | clEnqueueWriteBuffer/Read | Moderate |
|| kernel<<<G,B>>>() | clEnqueueNDRangeKernel | Moderate |
|| __shfl_down | NO EQUIVALENT | Complex |
|| Dynamic __shared__ | Local mem + clSetKernelArg | Moderate |

### Summary Table
|| Kernel/Function | Type | Priority | Launch Config | Total Work | Dependencies | OpenCL Issues |
||-----------------|------|----------|---------------|------------|--------------|---------------|

### CUDA-Specific Details
- **Dominant compute kernel:** [main timed kernel]
- **Memory transfers in timed loop?:** YES/NO
- **Shared memory usage:** [total bytes (static/dynamic), patterns]
- **Synchronization points:** [__syncthreads locations, kernel boundaries]
- **Atomic operations:** [locations and variables]
- **Texture/surface usage:** [bindings and access patterns]
- **Warp-level operations:** [locations - CRITICAL migration issue]
- **Dynamic parallelism:** [YES/NO - requires restructuring]

### OpenCL Migration Strategy Notes
- **Direct kernel conversion:** [list - simple __global__ → __kernel]
- **Requires restructuring:** [list with reasons - warp ops, dynamic parallelism]
- **Performance concerns:** [atomics overhead, local memory size, barriers]
- **Memory management changes:** [cudaMalloc → clCreateBuffer, etc.]
- **API setup overhead:** [context, command queue, program compilation]
- **Expected complexity:** [LOW/MEDIUM/HIGH based on issues]

## Constraints
- Find all kernels and device functions
- Document CUDA-specific constructs for migration planning
- Copy all source files with suffix conversion (.cu → .cl + .c/.cpp)
- No code modifications - documentation only
- Identify warp-level primitives (CRITICAL - may require algorithm changes)
- Note texture memory usage (requires OpenCL image API)
- Flag dynamic __shared__ memory (size must be set at kernel enqueue)
```

---

## Translation Step 1: CUDA to OpenCL Implementation
