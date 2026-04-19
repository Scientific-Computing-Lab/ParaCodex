# Kernel Classification Guide

## Decision Tree
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

## CUDA-to-OpenCL Specific Patterns
- **__syncthreads():** Maps to `barrier(CLK_LOCAL_MEM_FENCE)` - direct equivalent.
- **__shared__ memory:** Maps to `__local` - size must be known at kernel enqueue.
- **Atomic operations:** CUDA atomics → OpenCL atomics (syntax differs).
- **Warp-level primitives:** No direct OpenCL equivalent - requires restructuring.
- **Dynamic parallelism:** Not supported in OpenCL - must flatten.
- **Texture memory:** Maps to OpenCL images (different API).
- **Kernel decomposition:** Preserve CUDA kernel boundaries only when they still make sense after OpenCL setup/build overhead is accounted for.

## Type Reference
| Type | CUDA Pattern | OpenCL Equivalent | Notes |
|------|--------------|-------------------|-------|
| A | Dense kernel, regular grid | YES - NDRange | Direct map |
| B | Sparse (CSR), varying bounds | Outer only | Inner sequential |
| C1 | Multi-kernel, global sync | Multiple kernels | Explicit command queue ordering |
| C2 | Hierarchical grid | Multiple kernels | No nested dispatch |
| D | Histogram, atomicAdd | YES + atomic | atomic_add() |
| E | __syncthreads, __shared__ | YES - barrier() | Local memory + barrier |
| F | Reduction, atomicAdd scalar | YES + reduction | May need local reduction |
| G | Stencil, halo exchange | YES | Work-group handling |
| H | Texture memory | Image objects | Requires sampler objects |

---

## Code Examples

### grep commands for CUDA kernel analysis
```bash
grep -n "__global__\|__device__" *.cu 2>/dev/null
grep -n "<<<.*>>>" *.cu 2>/dev/null
grep -n "__syncthreads\|cudaDeviceSynchronize" *.cu 2>/dev/null
grep -n "__shared__" *.cu 2>/dev/null
grep -n "atomicAdd\|atomicMax\|atomicMin" *.cu 2>/dev/null
grep -n "__shfl\|__ballot\|__any\|__all" *.cu 2>/dev/null   # Warp primitives — no OCL equiv
grep -n "texture\|tex1D\|tex2D\|cudaBindTexture" *.cu 2>/dev/null  # Texture — flag
```

### Syntax Mapping Quick Reference
| CUDA | OpenCL | Complexity |
|------|--------|------------|
| `__global__ void k(...)` | `__kernel void k(...)` | Trivial |
| `threadIdx.x` | `get_local_id(0)` | Trivial |
| `blockIdx.x` | `get_group_id(0)` | Trivial |
| `blockDim.x` | `get_local_size(0)` | Trivial |
| `gridDim.x` | `get_num_groups(0)` | Trivial |
| `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` | Trivial |
| `__shared__ T arr[N]` | `__local T arr[N]` | Trivial |
| `atomicAdd(&x, v)` | `atomic_add(&x, v)` | Simple |
| `__shfl_down_sync(...)` | **NO EQUIVALENT** | Complex — flag |
| Dynamic parallelism | **NOT SUPPORTED** | Complex — flag |

## Structural recommendation rule
- If the CUDA code already has a sensible kernel split, preserve it.
- If the hot path is fragmented into many tiny kernels or helpers, recommend fusion or simplification in the analysis report.
