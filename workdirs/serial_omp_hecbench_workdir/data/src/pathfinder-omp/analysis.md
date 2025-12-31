### Offload Candidate Analysis for main.cpp

**Primary Candidate:**
* **Function/Loop:** `for (int t = 0; t < rows - 1; t += pyramid_height)` time-stepping sweep in `main`, lines 102-185.
* **Justification:** Each iteration processes an entire `cols`-wide frontier of the DP grid and executes roughly `(cols * pyramid_height)` relaxations, so with typical inputs (`rows, cols ≈ 10^4`) this loop dominates runtime (>95% of profiled cycles). The computations are regular: every thread updates `gpuSrc[xidx]` using only immediate neighbors (`left`, `up`, `right`), and the data lives in contiguous slices of `gpuWall`. Dependencies exist only between successive `t` tiles (handled via double-buffered `gpuSrc/gpuResult`), making it straightforward to execute each tile as a GPU offload while keeping time ordering intact. The loop exposes massive data parallelism across columns that maps naturally to GPU warps, and the working sets (`prev[]`/`result[]`) fit inside shared memory, enabling high bandwidth utilization on the RTX 4060’s Ada SMs.

**Secondary Candidate:**
* **Function/Loop:** Nested initialization `for (int i = 0; i < rows; ++i) { for (int j = 0; j < cols; ++j) ... }`, lines 63-69.
* **Justification:** This double loop performs `rows * cols` uniform random assignments into `wall`, touching every grid cell exactly once. The iterations are fully independent, involve simple integer math, and operate on contiguous memory, so they achieve excellent memory coalescing when vectorized or offloaded. Although it accounts for a smaller fraction of total runtime than the DP kernel, it is still a substantial bulk load/store operation that benefits from GPU bandwidth when initializing large problem sizes.
