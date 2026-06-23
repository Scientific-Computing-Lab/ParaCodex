# HIP Migration Plan

## Phase 1: Source Conversion (Hipify)
### Tool Selection
- [ ] `hipify-perl` (Script-based, no dependencies) - Recommended for simple cases.
- [ ] `hipify-clang` (Clang-based, robust) - Recommended for complex C++.

### File Mapping
| Original CUDA | Target HIP | Status |
| :--- | :--- | :--- |
| `kernel.cu` | `kernel.hip.cpp` | [ ] |
| `main.cu` | `main.hip.cpp` | [ ] |

## Phase 2: API & Kernel Fixes
### Kernel Launch Syntax
- [ ] Convert `kernel<<<grid, block>>>` to `hipLaunchKernelGGL` (if needed by compiler) or keep `<<<>>>` if supported by hipcc.
- **Rule:** `hipcc` generally supports `<<<>>>`.

### Warp Size Handling
- [ ] Check for hardcoded `32` or `0x1f`.
- [ ] Replace with `warpSize` or `__AMDGCN_WAVEFRONT_SIZE`.

### API Conversion Checklist
- [ ] `cudaMalloc` → `hipMalloc`
- [ ] `cudaMemcpy` → `hipMemcpy`
- [ ] `__global__` → `__global__` (No change)
- [ ] `__shared__` → `__shared__` (No change)
- [ ] `atomicAdd` → `atomicAdd` (Check float support)

## Phase 3: Build System
- [ ] Update Makefile/CMake to use `hipcc` instead of `nvcc`.
- [ ] Link against ROCm libraries (`rocblas`, `rocfft`) if needed.

## Phase 4: Verification
- [ ] Compile with `hipcc`.
- [ ] Run on AMD GPU (or NVIDIA GPU via HIP).
- [ ] Compare output with baseline.
