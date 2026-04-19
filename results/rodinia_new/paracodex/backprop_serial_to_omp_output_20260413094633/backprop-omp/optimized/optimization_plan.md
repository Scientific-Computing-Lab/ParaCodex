# Backprop OpenMP Optimization Plan

## Current Metrics
- Input size profiled: `262144`
- GPU kernels in timed path: `6` launches total
- CUDA memcpy count: `20` total copies
- CUDA memcpy volume:
  - Host to Device: `36.701 MB`
  - Device to Host: `35.652 MB`
- CUDA API hotspots:
  - `cuStreamSynchronize`: `107.301 ms` across `24` calls
  - `cuMemcpyDtoHAsync_v2`: `6.919 ms`
  - `cuMemcpyHtoDAsync_v2`: `3.724 ms`
- GPU kernel hotspots:
  - `nvkernel_bpnn_adjust_weights_device_*`: `100.106 ms` across `2` instances
  - `nvkernel_bpnn_layerforward_device_*`: `6.312 ms` across `2` instances
- End-to-end host wait:
  - `wait`: `788.271 ms`

## Structural Check
- Timed path already keeps the full training step on the GPU, so this is not a residency rewrite problem.
- The current structure is execution-structure limited:
  - The hot hidden-layer weight update is only parallel over `16` outer iterations.
  - The inner weight-update loop carries the real work but is still serial inside each GPU thread block.
  - Kernel launch count is reasonable, but the dominant kernel is under-parallelized.
- The first optimization should therefore widen the hot kernel’s parallelism before changing the data strategy.

## Strategy
1. Keep the step-1 resident-data approach intact.
2. Increase parallelism in `bpnn_adjust_weights_device()` by flattening the nested loops with `collapse(2)`.
3. Add aliasing and const qualifiers where safe so the compiler can generate cleaner device code.
4. Rebuild and run the correctness input.
5. Re-profile the larger input and compare combined kernel + memcpy + sync cost, not kernel time alone.

## Iteration Loop
1. Apply the kernel change.
2. Build with `make -f Makefile.nvc`.
3. Run `env OMP_TARGET_OFFLOAD=MANDATORY make -f Makefile.nvc run`.
4. If the output changes, fix correctness before profiling.
5. If the output matches, run `nsys profile` again and compare:
   - kernel time
   - memcpy time
   - stream synchronize time
   - total runtime

## SpMV Inner Loop Decision
- N/A. This kernel is backpropagation, not SpMV.

## Final Summary
- Baseline metrics at `262144` input elements:
  - GPU kernels: `106.421 ms`
  - CUDA memcpy: `10.643 ms`
  - CUDA stream synchronize: `107.301 ms`
  - Combined GPU kernel + memcpy + sync cost: `224.366 ms`
- Final metrics at `262144` input elements:
  - GPU kernels: `8.802 ms`
  - CUDA memcpy: `7.842 ms`
  - CUDA stream synchronize: `9.608 ms`
  - Combined GPU kernel + memcpy + sync cost: `26.252 ms`
- Speedup on the profiled default input: `8.55x` on combined GPU cost.
- Optimizations applied:
  - Flattened the hot hidden-layer weight-update kernel with `collapse(2)`.
  - Added `const` and `restrict` qualifiers to help the compiler optimize device memory access.
- Optimizations reverted:
  - None.
- Key insights:
  - The step-1 resident-data strategy was already acceptable; the bottleneck was under-parallelized execution, not excessive transfer volume.
  - The hidden-layer weight update was the critical kernel because it exposed only 16-way outer parallelism before the rewrite.
  - Larger-input profiling at `1000000` confirmed the same structural fix scales, and the kernel mix becomes more balanced as the workload grows.
- Step-1 offload/data strategy:
  - Not changed. The resident-device approach stayed intact because the issue was parallelism density, not data placement.
- Larger-input profiling impact:
  - It did not change the optimization decision; it confirmed the structural rewrite and showed the optimized path remains stable at scale.
- Final result classification:
  - execution-structure limited but stable
