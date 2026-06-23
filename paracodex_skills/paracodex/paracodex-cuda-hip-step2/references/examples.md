# Q1: Block Size Tuning?
> **Rule:** AMD wavefront is architecture-dependent — CDNA2/CDNA3=64, RDNA3/RDNA4=32.
> **Portable advice:** Use 256 (multiple of both 64 and 32). Avoid 32 on CDNA (partial wavefront).
> **Do not use** `__AMDGCN_WAVEFRONT_SIZE` — removed in ROCm 7.x. Use `warpSize` at runtime.

```cpp
// Bad for CDNA (partial wavefronts — only fills half of a 64-wide wavefront)
dim3 block(32);

// Good — 256 is portable: 4 wavefronts on CDNA, 8 wavefronts on RDNA
dim3 block(256);

// Query at runtime if adaptive tuning is needed:
// int ws = __builtin_amdgcn_wavefrontsize();  // device intrinsic (ROCm 5.x+)
// int ws = warpSize;  // standard built-in — always preferred
```

# Q2: Variable Type Optimization?
> **SGPR vs VGPR:** HIP separates Scalar (SGPR) and Vector (VGPR) registers.
> **Advice:** Move uniform values (same for all threads in wave) to SGPRs (use `__uniform__` hint if available, or compiler will deduce).

# Q3: Shared Memory (LDS) Banks?
> AMD GCN LDS has 32 banks, 4 bytes wide.
> **Conflict:** Same bank accessed by different threads.
> **Fix:** Padding — add +1 column to 2D shared array.

```cpp
// Without padding: every 32 threads hit bank 0 → conflict
__shared__ float tile[64][32];

// With padding: stride is 33, each row maps to a different bank
__shared__ float tile[64][33];   // +1 pad eliminates conflicts
```

# Q3b: Warp-Level Reduction (portable, wavefront-size agnostic)
```cpp
// Correct approach: use warpSize — resolves to 64 on CDNA, 32 on RDNA at runtime.
// DO NOT use __AMDGCN_WAVEFRONT_SIZE — removed in ROCm 7.x.
// DO NOT use __HIP_ARCH_GFX90A__ guards — fragile and not portable.
__device__ float warpReduce(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);  // HIP: no mask argument on AMD
    return val;
}
```

# Q3c: Async Transfers + Streams for Optimization
```cpp
hipStream_t s0, s1;
hipStreamCreate(&s0); hipStreamCreate(&s1);

// Overlap compute on s0 with transfer on s1
kernel<<<grid, block, 0, s0>>>(d_A, d_B, N);
hipMemcpyAsync(d_next, h_next, size, hipMemcpyHostToDevice, s1);

hipStreamSynchronize(s0); hipStreamSynchronize(s1);
hipStreamDestroy(s0); hipStreamDestroy(s1);
```

# Q4: Inline ASM?
> If migrating `asm("...")` from PTX, rewrite for GCN:
```cpp
// GCN inline asm example
asm volatile ("v_add_f32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
```
**Preferred:** Use intrinsics!

# Q5: Profiling?
> Use `rocprof`.
> `rocprof --stats ./my_app`
