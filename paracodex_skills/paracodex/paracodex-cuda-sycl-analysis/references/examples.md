# Analysis Decision Tree

## Q1: Is it portable?
- Check for `nvcc` specific flags.
- Check for `__CUDA_ARCH__`.

## Q2: Unsupported Features?
- **Texture/Surface Memory:** SYCL 1.2.1 Images are optional. `dpct` often converts to Bindless Images or explicit sampling.
- **Dynamic Parallelism:** Not supported in standard SYCL (requires `enqueue_kernel` extension).
- **Flag as:** High Complexity.

## Q3: Libraries?
- `cuBLAS` -> `oneMKL` (BLAS)
- `cuFFT` -> `oneMKL` (DFT)
- Prop libraries? (Flag as blocker).

## Q4: Warp Shuffle?
- `__shfl_sync` -> `sub_group::shuffle`.
- **Note:** Check if warp size 32 is hardcoded. SYCL sub-groups can be 16, 32, 64.

## Q5: Is the CUDA submission structure worth preserving?
- If the CUDA hot path is already one or a few good kernels, preserve that logical decomposition.
- If the source has helper-level fragmentation, recommend a simpler SYCL submission structure in the analysis report.

---

## Code Examples

### grep commands for SYCL migration analysis
```bash
# Kernels and device code
grep -n "__global__\|__device__\|__shared__" *.cu 2>/dev/null

# Warp shuffles → sub_group::shuffle
grep -n "__shfl\|__ballot\|__any\|__all" *.cu 2>/dev/null

# Texture memory → flag as high complexity
grep -n "texture\|tex1D\|tex2D\|cudaBindTexture\|cudaCreateTextureObject" *.cu 2>/dev/null

# Dynamic parallelism → NOT supported in SYCL
grep -n "cudaLaunchKernel\|cudaLaunchDevice\|__device__.*<<<" *.cu 2>/dev/null

# CUDA library usage
grep -n "cublas\|cufft\|cusolver\|curand" *.cu *.cpp *.h 2>/dev/null -i
```

### API Migration Quick Reference
| CUDA | SYCL | Notes |
|------|------|-------|
| `cudaMalloc` | `sycl::malloc_device` | USM preferred |
| `cudaMemcpy H→D` | `q.memcpy(d, h, bytes)` | Returns event |
| `cudaMemcpyAsync` | `q.memcpy(...).depends_on(ev)` | Event chaining |
| `cudaFree` | `sycl::free(ptr, q)` | |
| `kernel<<<G,B>>>` | `q.parallel_for(nd_range<1>{G,B}, ...)` | |
| `__syncthreads()` | `it.barrier(sycl::access::fence_space::local_space)` | |
| `__shared__ T arr[N]` | `sycl::local_accessor<T,1> arr(N, h)` | |
| `__shfl_down_sync` | `sg.shuffle_down(val, offset)` | Sub-group |
| `atomicAdd` | `sycl::atomic_ref<T,...>::fetch_add` | |
| `cuBLAS` | `oneMKL BLAS` | Different API |
| `cuFFT` | `oneMKL DFT` | Different API |
