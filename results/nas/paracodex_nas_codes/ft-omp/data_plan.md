# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| u_real | 257 doubles (~2.1 KB) | const (FFT twiddle real) | host+H→D | RO |
| u_imag | 257 doubles (~2.1 KB) | const (FFT twiddle imag) | host+H→D | RO |
| u0_real | 8,421,376 doubles (~64.3 MB) | working (field real) | device via init/evolve | R/W |
| u0_imag | 8,421,376 doubles (~64.3 MB) | working (field imag) | device via init/evolve | R/W |
| u1_real | 8,421,376 doubles (~64.3 MB) | working (field real) | host RNG + H→D | R/W |
| u1_imag | 8,421,376 doubles (~64.3 MB) | working (field imag) | host RNG + H→D | R/W |
| twiddle | 8,421,376 doubles (~64.3 MB) | const (spectral weights) | host compute_indexmap + H→D | RO |
| gty1_real | 16,777,216 doubles (~128.0 MB) | scratch (FFT buffer) | device | R/W |
| gty1_imag | 16,777,216 doubles (~128.0 MB) | scratch (FFT buffer) | device | R/W |
| gty2_real | 16,777,216 doubles (~128.0 MB) | scratch (FFT buffer) | device | R/W |
| gty2_imag | 16,777,216 doubles (~128.0 MB) | scratch (FFT buffer) | device | R/W |
| host_u1_real | 8,421,376 doubles (~64.3 MB) | host staging (RNG real) | host | W (host only) |
| host_u1_imag | 8,421,376 doubles (~64.3 MB) | host staging (RNG imag) | host | W (host only) |

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| compute_indexmap | twiddle | once per timed section (before iters) | host + copy to device |
| compute_initial_conditions | host_u1_*, u1_real/u1_imag | once per timed section | host + copy to device |
| fft_init | u_real/u_imag | once per timed section | device |
| fft(1): cffts1_pos/cffts2_pos/cffts3_pos | u1_real/u1_imag, gty1*, gty2*, u_real/u_imag | once per timed section | device |
| evolve | u0_real/u0_imag, u1_real/u1_imag, twiddle | per-iteration | device |
| fft(-1): cffts3_neg/cffts2_neg/cffts1_neg | u1_real/u1_imag, gty1*, gty2*, u_real/u_imag | per-iteration | device |
| checksum | u1_real/u1_imag | per-iteration | device (scalar reduction returned) |

## Data Movement Strategy

**Chosen Strategy:** C

**Device Allocations (once):**
```
d_u_real/d_u_imag: 257 doubles each via omp_target_alloc
d_u0_real/d_u0_imag: NTOTALP doubles each via omp_target_alloc
d_u1_real/d_u1_imag: NTOTALP doubles each via omp_target_alloc
d_twiddle: NTOTALP doubles via omp_target_alloc
d_gty1_real/d_gty1_imag: MAXDIM^3 doubles each via omp_target_alloc
d_gty2_real/d_gty2_imag: MAXDIM^3 doubles each via omp_target_alloc
```

**Host→Device Transfers:**
- When: after host setup kernels (compute_indexmap, compute_initial_conditions) before each timed section
- Arrays: twiddle (NTOTALP, ~64.3 MB) twice; u1_real/u1_imag (NTOTALP each, ~64.3 MB) twice
- Total H→D: ~386 MB across both pre-timed and timed setup phases

**Device→Host Transfers:**
- When: after checksum to consume scalar sums on host
- Arrays: reduction results temp1/temp2 to host scalars
- Total D→H: ~0 MB (array copies avoided)

**Transfers During Iterations:** NO
- All working/scratch arrays remain resident on device across all iterations.

## Critical Checks (for chosen strategy)

**Strategy C:**
- [x] ALL functions in iteration loop use is_device_ptr.
- [x] Scratch arrays allocated on device (not host).
- [ ] No map() clauses (only is_device_ptr).

**Common Mistakes:**
-  Some functions on device, others on host (causes copying)
-  Scratch as host arrays in Strategy C
-  Forgetting to offload ALL functions in loop

## Expected Transfer Volume
- Total: ~386 MB H→D across setup calls, ~0 MB arrays D→H (only scalar reductions).
- **Red flag:** If actual >2x expected → data management wrong

## Additional Parallelization Notes
- **RNG Replicable?** RNG recurrence sequential per k/j; host generation matches reference ordering.
- **Outer Saturation?** Use teams loop with collapse(2) on outer spatial dims; rely on RTX 4060 (Ada) occupancy.
- **SpMV NONZER?** N/A
- **Histogram Strategy?** N/A

**Summary:** 12 arrays (6 working, 4 scratch, 2 host staging), 7 functions, Strategy C with host-built setup arrays then device-resident iteration. Expected: ~386 MB H→D, ~0 MB D→H beyond scalar reductions.
