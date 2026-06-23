# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

|| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
||---------------|-----------------|------|------------------|
|| d_[name] | cudaMalloc | [bytes] | H→D once/D→H once/both |
|| [name] | host array | [bytes] | source/destination |

**CUDA Operations:**
- cudaMalloc calls: [list with sizes]
- cudaMemcpy H→D: [list with timing]
- cudaMemcpy D→H: [list with timing]
- Kernel launches: [list with frequency]

## Kernel Inventory
|| Kernel Name | Launch Config | Frequency | Arrays Used |
||-------------|---------------|-----------|-------------|
|| kernel_name<<<G,B>>> | grid=[X], block=[Y] | per-iteration/once | [list] |

## Structural Mapping
- **Natural OpenMP offload unit:** [single region / fused routine / helper-by-helper]
- **Preserve original CUDA kernel boundaries?** [YES/NO/PARTIALLY]
- **Reason:** [performance / synchronization / simplicity]
- **Fragmentation risk:** [none / low / medium / high]
- **Combined GPU+mem budget:** [kernel + memcpy + sync]
- **Default correctness size:** [value]
- **Larger practical profiling size:** [value or rule]
- **Why this size materially exercises the GPU:** [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- **Hardware basis from `system_info_summary.txt`:** [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]
- **Likely small-input sensitive?** [YES/NO]
- **Scaling risk if migrated this way:** [kernel count / transfer count / host sync / occupancy / none]

**Kernel Launch Patterns:**
- In outer loop? → Multiple target teams loop
- Sequential kernels? → Multiple target regions OR nowait+depend
- Conditional launch? → target if clause

## OMP Data Movement Strategy

**Chosen Strategy:** [A/B/C]

**Rationale:** [Map CUDA pattern to strategy]

**Device Allocations (OMP equivalent):**
[List allocations based on strategy]

**Host→Device Transfers (OMP equivalent):**
- When: [before iterations/once at start]
- Arrays: [list with sizes]
- Total H→D: ~[X] MB

**Device→Host Transfers (OMP equivalent):**
- When: [after iterations/once at end]
- Arrays: [list with sizes]
- Total D→H: ~[Y] MB

**Transfers During Iterations:** [YES/NO]
- If YES: [which arrays and why - may indicate wrong strategy]

## Kernel to OMP Mapping (short)
- Replace each CUDA kernel launch with a `#pragma omp target teams loop` over the same *logical* work domain.
- Replace `blockIdx/threadIdx` indexing with the loop induction variable.
- Keep bounds checks; keep inner device loops as normal C loops inside the offloaded loop body.

## Critical Migration Issues

**From analysis.md "OMP Migration Issues":**
- [ ] __syncthreads() usage: [locations and resolution strategy]
- [ ] Shared memory: [convert to private/firstprivate]
- [ ] Atomics: [verify OMP atomic equivalents]
- [ ] Dynamic indexing: [verify OMP handles correctly]

**__syncthreads() Resolution:**
- Within single kernel → May need to split into multiple target regions
- At kernel boundaries → Natural OMP barrier between target regions
- Strategy: [describe approach]

**Shared memory / barriers:**
- No direct equivalent for CUDA `__shared__` + `__syncthreads()`; refactor and document your approach.

## Expected Performance
- CUDA kernel time: [X] ms (from profiling if available)
- OMP expected: [Y] ms (may be slower due to __syncthreads elimination)
- Red flag: If >3x slower → wrong strategy or missing parallelism

**Summary:** [num] kernels, [num] device arrays, Strategy [A/B/C]. 
CUDA pattern: [describe]. OMP approach: [describe].
Expected: ~[X] MB H→D, ~[Y] MB D→H.

## Build / Run Readiness
- **Plain build command works?** [YES/NO]
- **Plain `make -f Makefile.nvc run` works?** [YES/NO]
- **Unresolved placeholders remaining?** [YES/NO]
- **`{nsys_profile_cmd} > {profile_log_path} 2>&1` shows GPU kernels?** [YES/NO]
- **Chosen structure still plausible at larger size?** [YES/NO]
