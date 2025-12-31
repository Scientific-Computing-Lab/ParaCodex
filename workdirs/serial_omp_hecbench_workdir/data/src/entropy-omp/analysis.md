### Offload Candidate Analysis for main.cpp

**Primary Candidate:**
* **Function/Loop:** `entropy()` nested sweep over `y`/`x`, lines 13-47.
* **Justification:** Each iteration touches one output pixel but performs ~25 neighbor reads plus 16-log evaluations, so the total work is O(height×width×41) floating-point/byte ops. The iterations are embarrassingly parallel—the per-pixel histogram and entropy accumulator live in registers/stack and do not share state—making it ideal for thousands of GPU threads. Memory accesses are dense (`d_val` / `d_entropy` are linearized 2D arrays), so we can achieve coalesced loads if we map adjacent threads to adjacent `x` values.

**Secondary Candidate:**
* **Function/Loop:** `entropy_opt()` tile-local histogram construction and entropy write-back, lines 75-95.
* **Justification:** Although currently written with CPU OpenMP queries, the logic still boils down to per-pixel histogramming over the same 5×5 neighborhood. Moving this computation to the GPU lets each CTA/team reuse shared tile data (`sd_count`) and amortize neighbor reads, boosting arithmetic intensity. The computation is data-parallel and works on contiguous tiles of `d_val` / `d_entropy`, so it can benefit from GPU shared memory and cooperative reductions.
