### Offload Candidate Analysis for main.cpp

**Primary Candidate:**
* **Function/Loop:** `reference` dense update, lines 18-27.
* **Justification:** Outer loop covers every batch entry (up to the runtime `batch_size`), middle loop spans the entire embedding dimension (`768–12288` columns), and the innermost loop walks each ragged slice of the embedding table (`offset[batch_idx+1]-offset[batch_idx]`, typically O(batch_size*ncols)). That gives O(batch_size * embedding_dim * range/embedding_dim) ≈ O(total_nonzeros) iterations with two floating-point ops (`+` and assignment) per iteration. All iterations are independent except for the final write to disjoint `output[...]`, so the work is perfectly data-parallel across the flattened (batch, idx, nested_idx) space. Accesses stride through contiguous segments of `input`, `output`, and `dense`, so an offloaded kernel can achieve high memory throughput with coalesced loads/stores.

**Secondary Candidate:**
* **Function/Loop:** Dense embedding kernel `k1` body, lines 96-110.
* **Justification:** This is the production version of the same computation. For each repeat, it iterates over the same large embedding dimension and ragged-slice loops, but now inside the timing harness. The loop nest performs `repeat * batch_size * embedding_dim * (range/embedding_dim)` updates with regular memory access patterns. Thread teams are currently implied via OpenMP but run on CPUs; mapping the `batch_idx` / `idx` space to GPU teams/threads will expose abundant parallelism, and the lack of cross-iteration dependencies makes it a strong offload target.

**Tertiary Candidate:**
* **Function/Loop:** Dense embedding kernel `k2` body, lines 120-133.
* **Justification:** This alternative kernel mirrors `k1` with identical arithmetic intensity and data reuse, differing only in index precomputation. It touches the same large contiguous buffers and has the same independence properties. Offloading it alongside `k1` provides a second implementation to validate correctness and explore different launch configurations (e.g., mapping `idx` to lanes and `batch_idx` to teams) while keeping data transfers amortized across repeats.
