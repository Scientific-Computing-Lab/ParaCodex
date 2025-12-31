### Offload Candidate Analysis for main.cpp

**Primary Candidate:**
* **Function/Loop:** Jacobi stencil update inside the `while` loop (`main`, lines 46-57).
* **Justification:** Iterates over the 2046×2046 interior grid (`>4.1M` points) each iteration of the solver, performing ~8 floating-point ops per point (4 loads, 1 average, 1 diff, 1 accumulation, 1 store). Each lattice update depends only on the previous iteration (`f_old`), so iterations are independent and map cleanly to a 2D GPU grid. Memory accesses follow contiguous rows in the flattened arrays, enabling coalesced global-memory transactions.

**Secondary Candidate:**
* **Function/Loop:** State refresh copy (`main`, lines 59-62).
* **Justification:** Sweeps the entire `N×N` domain every iteration to copy interior values from `f` to `f_old`. Although lighter arithmetically, it still touches ~4M elements with perfectly regular, independent iterations over contiguous arrays, making it amenable to GPU offload or fusion with the primary stencil kernel to eliminate the extra pass.

**Tertiary Candidate:**
* **Function/Loop:** Boundary initialization (`initialize_data`, lines 14-25).
* **Justification:** Runs only once but still covers all `N×N` points while evaluating `sinf` for boundary cells. The iterations have no dependencies and operate on contiguous memory regions, so they can be batched on the GPU if startup time matters or if we wish to keep data on the device before iterations begin.
