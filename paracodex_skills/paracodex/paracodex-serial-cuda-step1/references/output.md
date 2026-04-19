# CUDA Migration Plan (Serial to Parallel)

## Phase 1: Preparation
- [ ] Identify Hotspots (from analysis.md).
- [ ] Choose CUDA kernel structure: direct kernel / fused kernel / staged kernels.
- [ ] Allocate Device Memory (`cudaMalloc`).
- [ ] Manage Data Transfer (`cudaMemcpy`).

## Structural Plan
- [ ] Identify the natural CUDA offload unit.
- [ ] If helper-by-helper translation would create tiny kernels, rewrite as a fused kernel.
- [ ] Record which data stays resident on device across the full hot path.
- [ ] Budget combined GPU kernel + memcpy + sync cost, not just kernel time.
- [ ] Record the default correctness size and one larger practical profiling size.
- [ ] Reject a structure that only wins on tiny inputs while scaling poorly.

## Phase 2: Implementation
### Kernel 1: [Name]
- [ ] Define block/grid dimensions.
- [ ] Write `__global__` kernel.
- [ ] Handle boundary checks (`idx < N`).
- [ ] Replace updates with Atomics if race conditions exist.
- [ ] Verify this kernel boundary is performance-justified and not just inherited from the serial code.

### Host Code
- [ ] Initialize device data.
- [ ] Launch kernel.
- [ ] Copy results back.
- [ ] Free memory.

## Phase 3: Verification
- [ ] Compile with `nvcc`.
- [ ] Compare output with serial execution.
- [ ] Confirm the step1 design is a good optimization base, not only a correct port.
- [ ] All generated placeholders resolved.
- [ ] Plain run target/command works without ad hoc argument overrides.
- [ ] `{nsys_profile_cmd} > {profile_log_path} 2>&1` produces GPU kernel information in the log.
- [ ] The chosen structure is still plausible at the larger practical profiling size.
