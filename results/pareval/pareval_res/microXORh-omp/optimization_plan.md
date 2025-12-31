# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: ≈0.0024s (sum of kernel + memcpy times reported by `nsys`)
- Main kernel: `nvkernel__Z8cellsXORPiS_m_F1L24_2` (100% of GPU time, 1 launch in `cuda_gpu_kern_sum`)
- Memory transfer: 77% (≈1.815ms) Device→Host + 23% (≈0.541ms) Host→Device for a total of 8.388MB moved
- Kernel launches: 1 (listed in `cuda_api_sum`)

## Bottleneck Hypothesis (pick 1–2)
- [x] Transfers too high (growing transfer portion dwarfs the tiny compute kernel)
- [ ] Too many kernels / target regions (single kernel already in place)
- [ ] Missing collapse vs CUDA grid dimensionality (already collapse(2))
- [x] Hot kernel needs micro-opts (low compute density but still GPU-bound; small index math tweaks can help)

## Actions (1–3 max)
1. [MICRO-OPT]: Cache `row_base = i * N` and neighbors so the `j*N` math isn’t recomputed per neighbor – lowers instruction pressure and helps the compiler hoist repeated multiplications (expected ≈1–2% gain).
2. [MICRO-OPT]: Qualify the `input` pointer as `const int* __restrict` and `output` as `int* __restrict` in the target region so the compiler can assume no aliasing and better schedule loads/stores (expected ≈1% gain).
