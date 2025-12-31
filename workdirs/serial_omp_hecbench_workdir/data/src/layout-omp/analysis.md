### Offload Candidate Analysis for main.cpp

**Primary Candidate:**
* **Function/Loop:** AoS accumulation loop inside `main` (nested `n`/`gid`/`i` loops), lines 70-78.
* **Justification:** Each outer iteration repeats a reduction over all `treeNumber = 4096` trees, and every tree walks `treeSize = 4096` apples, yielding ~16.7M additions per iteration trip count (`iterations` can scale this further). Iterations are fully independent per `gid`, enabling a straightforward mapping to GPU teams/threads with a reduction on `res`. The data is stored contiguously as `AppleTree` structs (array-of-structures), so each tree’s apples occupy a dense 16 KB region, enabling coalesced loads when threads cooperatively process adjacent trees. This triple-nested loop dominates runtime (reported as “Average kernel execution time (AoS)”) and is the top offload target.

**Secondary Candidate:**
* **Function/Loop:** SoA accumulation loop inside `main` (nested `n`/`gid`/`i` loops), lines 107-115.
* **Justification:** This loop performs the same arithmetic intensity as the AoS version (~16.7M additions per outer repetition) but reads the data in a structure-of-arrays layout, offering even better memory coalescing when mapping the inner `gid` dimension to GPU threads. Each iteration streams through large contiguous slices of `applesOnTrees[i].trees`, so offloading can exploit high bandwidth by assigning one thread per tree and keeping the `i` loop in registers/shared memory. The loop is embarrassingly parallel with only a private accumulator per `gid`, making it an ideal GPU kernel once the AoS version is ported.
