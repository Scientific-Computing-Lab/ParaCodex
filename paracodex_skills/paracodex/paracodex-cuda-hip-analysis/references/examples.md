# Analysis Decision Tree

## Q1: Is it portable?
- Check for `nvcc` specific flags.
- Check for `__CUDA_ARCH__`.

## Q2: Warp Size Assumptions?
- Grep for `32`, `31`, `0x1f`.
- Code using `threadIdx.x % 32` is dangerous on AMD (64 threads/wave).
- **Flag as:** Warp Assumption Issue.

## Q3: Inline PTX?
- Grep for `asm("...")`.
- **Flag as:** Manual Rewrite Needed.

## Q4: Libraries?
- `cuBLAS` -> `rocBLAS` (Available)
- `cuDNN` -> `MIOpen` (Available)
- `cuSparse` -> `rocSparse` (Available)
- Proprietary libs? (Flag as blocker).

## Q5: Cooperative Groups?
- Experimental support in HIP. Flag if used.

---

## Code Examples

### grep commands for HIP analysis
```bash
# Warp-size assumptions (dangerous on AMD: wavefront=64)
grep -n "\b32\b\|0x1f\|threadIdx\.x % 32\|warp.*32" *.cu *.cpp 2>/dev/null | head -30

# Inline PTX (needs rewrite to GCN ASM or intrinsics)
grep -n 'asm\s*(' *.cu *.cpp 2>/dev/null

# CUDA library calls that need rocm equivalents
grep -n "cublas\|cufft\|cusparse\|curand\|cudnn" *.cu *.cpp *.h 2>/dev/null -i

# Cooperative groups (experimental HIP support)
grep -n "cooperative_groups\|this_thread_block\|grid_group" *.cu 2>/dev/null
```

### Warp size fix pattern
```cpp
// BAD: hardcoded 32 — wrong on CDNA (wavefront=64)
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);

// GOOD: use warpSize — resolves at runtime
for (int offset = warpSize/2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);  // HIP: no mask argument on AMD
```

### Library migration quick reference
| CUDA | HIP/ROCm | Notes |
|------|----------|-------|
| `cuBLAS` | `rocBLAS` | Drop-in via hipBLAS |
| `cuFFT` | `rocFFT` | Drop-in via hipFFT |
| `cuSPARSE` | `rocSPARSE` | Drop-in via hipSPARSE |
| `cuRAND` | `rocRAND` | Drop-in via hipRAND |
| `cuDNN` | `MIOpen` | API differs — manual port |
