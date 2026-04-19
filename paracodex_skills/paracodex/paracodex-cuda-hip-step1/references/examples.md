# Q1: How to use hipify-perl?
> **Command:**
> `hipify-perl input.cu > output.hip.cpp`
>
> **Pattern - In-place:**
> `hipify-perl -inplace input.cu` (Creates .prehip backup)

# Q2: Common API Mappings?
| CUDA | HIP | Notes |
| :--- | :--- | :--- |
| `cudaMalloc(&ptr, size)` | `hipMalloc(&ptr, size)` | Direct map |
| `cudaMemcpy(...)` | `hipMemcpy(...)` | Direct map |
| `cudaDeviceSynchronize()` | `hipDeviceSynchronize()` | Direct map |
| `__syncthreads()` | `__syncthreads()` | Same |
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` | Check mask support (HIP often ignores mask on AMD) |

# Q3: Dealing with Warp Size (32 vs 64)?
**Problem:** CUDA is always 32. AMD wavefront is architecture-dependent:
- **CDNA2/CDNA3** (MI200/MI300 series): wavefront = **64**
- **RDNA3/RDNA4** (RX 7000 series): wavefront = **32**
- `__AMDGCN_WAVEFRONT_SIZE` macro **removed in ROCm 7.x** — do NOT use it.

**Solution:** Always use `warpSize` built-in at runtime:
```cpp
// Bad — hardcoded, wrong on CDNA or RDNA depending on guess
int lane = threadIdx.x % 32;

// Good — portable across all AMD architectures
#include <hip/hip_runtime.h>
int lane = threadIdx.x % warpSize;

// Reduction pattern — portable
for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);  // HIP: no mask argument on AMD
```

# Q4: Kernel Launch with Stream + hipMallocAsync (Best Practice)?
```cpp
// hipMallocAsync available since ROCm 5.2 — preferred for dynamic allocations
hipStream_t stream;
hipStreamCreate(&stream);
float *d_A, *d_B, *d_C;
HIP_CHECK(hipMallocAsync(&d_A, size, stream));
HIP_CHECK(hipMallocAsync(&d_B, size, stream));
HIP_CHECK(hipMallocAsync(&d_C, size, stream));

// Async H→D
hipMemcpyAsync(d_A, h_A, size, hipMemcpyHostToDevice, stream);
hipMemcpyAsync(d_B, h_B, size, hipMemcpyHostToDevice, stream);

// 256 = safe block size for both CDNA (4×64) and RDNA (8×32)
dim3 block(256);
dim3 grid((N + block.x - 1) / block.x);
myKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);

// Async D→H
hipMemcpyAsync(h_C, d_C, size, hipMemcpyDeviceToHost, stream);
hipStreamSynchronize(stream);

HIP_CHECK(hipFreeAsync(d_A, stream));
HIP_CHECK(hipFreeAsync(d_B, stream));
HIP_CHECK(hipFreeAsync(d_C, stream));
hipStreamDestroy(stream);
```

# Q5: Error Handling?
```cpp
// CUDA
cudaError_t err = cudaGetLastError();

// HIP
hipError_t err = hipGetLastError();
if (err != hipSuccess) {
    fprintf(stderr, "HIP error: %s\n", hipGetErrorString(err));
    exit(1);
}

// Macro
#define HIP_CHECK(call) { \
    hipError_t e = (call); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } }
HIP_CHECK(hipMalloc(&d_A, size));
```
