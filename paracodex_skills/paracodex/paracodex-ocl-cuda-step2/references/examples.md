# CUDA Optimization Examples

## 4A. Occupancy-Driven Block Size
```c
// Let CUDA choose the best block size for this kernel
int minGrid, bestBlock;
cudaOccupancyMaxPotentialBlockSize(&minGrid, &bestBlock, myKernel, 0, 0);
int grid = (N + bestBlock - 1) / bestBlock;
myKernel<<<grid, bestBlock>>>(d_A, d_B, d_C, N);

// Or manually check achieved occupancy
int activeWarps, maxWarps;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeWarps, myKernel, 256, 0);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
float occupancy = (float)(activeWarps * 256 / 32) /
                  (float)prop.maxThreadsPerMultiProcessor;
printf("Occupancy: %.2f%%\n", occupancy * 100);
```

## 4B. Fast Math Intrinsics
```c
// Use intrinsics if precision allows
float result = __sinf(x) * __cosf(y);
float power = __powf(base, exp);
float root = __fsqrt_rn(val);
// Compile with: nvcc -use_fast_math
```

## 4C. Warp-Level Primitives
```c
// Reduction within warp using shuffle
__inline__ __device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

## 4D. Optimize Transfers
```c
// Pinned Memory
cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);

// Async Streams
cudaStreamCreate(&stream);
cudaMemcpyAsync(..., stream);
kernel<<<..., stream>>>(...);

// Unified Memory
cudaMallocManaged(&data, size);
```

## 4E. Optimize Shared Memory (Pad Banks)
```c
// Avoid bank conflicts (stride 32)
__shared__ float shared[32][33]; // Pad column
```

## 4S. Structural rewrite before micro-tuning
```c
// If CUDA kernel time is fine but end-to-end runtime is poor,
// revisit kernel count, host synchronization, and decomposition first.
```

## 4G. Texture Memory (Read-Only)
```c
texture<float, 1, cudaReadModeElementType> texRef;
cudaBindTexture(0, texRef, d_data, size);
// Kernel: tex1Dfetch(texRef, idx);
```

## Micro-Optimizations
- `__restrict__` pointers
- Loop unrolling `#pragma unroll`
- Register caching

## Optimization Checklist
- [ ] Kernel decomposition is sane for end-to-end runtime
- [ ] Block size optimized (multiple of 32)
- [ ] Fast math intrinsics used
- [ ] Warp-level primitives used
- [ ] Memory transfers optimized (Pinned/Async/Unified)
- [ ] Bank conflicts eliminated
- [ ] Warp divergence minimized
- [ ] Memory coalescing verified
- [ ] `-use_fast_math -O3` flags used
