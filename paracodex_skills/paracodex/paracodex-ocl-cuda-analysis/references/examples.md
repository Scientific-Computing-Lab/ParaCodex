# Kernel Classification Guide

## Decision Tree
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

## OpenCL-to-CUDA Specific Patterns
- **barrier():** Maps to `__syncthreads()` - direct equivalent
- **__local memory:** Maps to `__shared__` - same semantics
- **Atomic operations:** OpenCL atomics → CUDA atomics (syntax differs)
- **Image objects:** Map to texture memory (different API)
- **Work-item functions:** `get_global_id` etc. → thread indexing math
- **Sub-groups:** Map to warp-level primitives (CUDA has more features)
- **Pipes:** No direct CUDA equivalent - requires restructuring
- **Kernel/enqueue structure:** Preserve OpenCL kernel boundaries only when they still make performance sense in CUDA.

## Type Reference
| Type | OpenCL Pattern | CUDA Equivalent | Notes |
|------|----------------|-----------------|-------|
| A | Dense kernel, regular NDRange | YES - kernel launch | Direct map |
| B | Sparse (CSR), varying bounds | Outer only | Inner sequential |
| C1 | Multi-kernel, clFinish sync | Multiple kernels | Same pattern |
| C2 | Hierarchical enqueues | Multiple launches | No device enqueue |
| D | Histogram, atomic_add | YES + atomicAdd | Direct map |
| E | barrier, __local | YES - __syncthreads, __shared__ | Direct map |
| F | Reduction, atomic_add scalar | YES + atomicAdd | Direct map |
| G | Stencil, work-group sharing | YES | Same pattern |
| H | Image objects | Texture memory | Different API |

## Structural recommendation rule
- If OpenCL used a sensible kernel split, keep that logical unit in CUDA.
- If OpenCL host code fragmented the hot path into many tiny enqueues, recommend fusion or simplification in the analysis report.
