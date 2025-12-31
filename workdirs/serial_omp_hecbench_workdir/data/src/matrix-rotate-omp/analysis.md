### Offload Candidate Analysis for main.cpp

**Primary Candidate:**
* **Function/Loop:** `rotate_matrix_parallel` layered rotation loop, lines 9-25 in `main.cpp`.
* **Justification:** This triple-nested loop dominates runtime with an iteration space of roughly `repeat * (n/2) * n ≈ O(repeat * n^2)` swaps. Each inner iteration performs four dependent loads and four stores plus several index arithmetic operations, giving ~12 floating-point/integer ops per element quartet. Different `i` iterations within the same layer manipulate disjoint quartets of the matrix, so they are embarrassingly data-parallel once the target layer index is fixed. Accesses stride through contiguous rows (`matrix[first * n + i]`, `matrix[i * n + last]`, etc.), enabling coalesced global memory transactions on the GPU.

**Secondary Candidate:**
* **Function/Loop:** `rotate_matrix_serial` layered rotation loop, lines 34-50 in `main.cpp`.
* **Justification:** Although used for verification, it mirrors the exact workload of the parallel routine (≈`n^2` element rotations) and therefore represents the same compute and bandwidth characteristics. Offloading this loop would greatly accelerate the reference computation when large `repeat` counts are required for correctness validation or benchmarking. The iterations are identical and independent across `i`, with dense row-major accesses, making them GPU-friendly.
