# Analysis Output

## Loop Nesting Structure
```
- compute_tran_temp::for i < num_iterations (line:170) Type E
  └── single_iteration::for chunk < num_chunk (line:58) Type G
      ├── boundary path: for r (line:67) -> for c (line:68) Type G
      └── interior path: for r (line:124) -> for c (line:125) Type G
- read_input::for i < grid_rows * grid_cols (line:225) Type A
- writeoutput::for i < grid_rows (line:202)
  └── for j < grid_cols (line:203) Type A
- main OUTPUT loop::for i < grid_rows * grid_cols (line:300) Type A
```

## Loop Details

### Loop: compute_tran_temp at hotspot_openmp.cpp:170
- **Iterations:** `num_iterations`
- **Type:** E - temporal recurrence across time steps; each step consumes the previous step's temperature field
- **Parent loop:** none
- **Contains:** `single_iteration(...)`, pointer swap between `r` and `t`
- **Dependencies:** stage / recurrence
- **Nested bounds:** variable
- **Private vars:** `i`, `r`, `t`, `tmp`
- **Arrays:** `result(RW)`, `temp(RW)`, `power(R)`, then alternating `r/t` aliases
- **Issues:** stage dependency; do not parallelize across iterations

### Loop: single_iteration at hotspot_openmp.cpp:58
- **Iterations:** `row * col / (BLOCK_SIZE_R * BLOCK_SIZE_C)`
- **Type:** G - stencil/tiled grid update with neighbor reads
- **Parent loop:** none
- **Contains:** boundary `r/c` loops at lines 67-68 and interior `r/c` loops at lines 124-125
- **Dependencies:** none within a timestep
- **Nested bounds:** variable at the chunk level, constant tile bounds inside
- **Private vars:** `chunk`, `r`, `c`, `r_start`, `c_start`, `r_end`, `c_end`, `delta`
- **Arrays:** `result(W)`, `temp(R)`, `power(R)`
- **Issues:** boundary divergence; branch-heavy edge handling

### Loop: single_iteration at hotspot_openmp.cpp:67
- **Iterations:** `BLOCK_SIZE_R`
- **Type:** G - boundary stencil rows
- **Parent loop:** line 58
- **Contains:** inner `c` loop at line 68
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `r`, `r_start`
- **Arrays:** `result(W)`, `temp(R)`, `power(R)`
- **Issues:** branch-heavy edge path; executes only for boundary chunks

### Loop: single_iteration at hotspot_openmp.cpp:68
- **Iterations:** `BLOCK_SIZE_C`
- **Type:** G - boundary stencil columns
- **Parent loop:** line 67
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `c`, `c_start`, `delta`
- **Arrays:** `result(W)`, `temp(R)`, `power(R)`
- **Issues:** boundary condition cascade; no loop-carried dependency

### Loop: single_iteration at hotspot_openmp.cpp:124
- **Iterations:** `BLOCK_SIZE_R`
- **Type:** G - interior stencil rows
- **Parent loop:** line 58
- **Contains:** inner `c` loop at line 125
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `r`, `r_start`
- **Arrays:** `result(W)`, `temp(R)`, `power(R)`
- **Issues:** none

### Loop: single_iteration at hotspot_openmp.cpp:125
- **Iterations:** `BLOCK_SIZE_C`
- **Type:** G - interior stencil columns
- **Parent loop:** line 124
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** constant
- **Private vars:** `c`, `c_start`
- **Arrays:** `result(W)`, `temp(R)`, `power(R)`
- **Issues:** none

### Loop: read_input at hotspot_openmp.cpp:225
- **Iterations:** `grid_rows * grid_cols`
- **Type:** A - linear fill from file input
- **Parent loop:** none
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`, `index`, `val`
- **Arrays:** `vect(W)`
- **Issues:** setup I/O; not part of timed compute region

### Loop: writeoutput at hotspot_openmp.cpp:202
- **Iterations:** `grid_rows`
- **Type:** A - row-wise output wrapper
- **Parent loop:** none
- **Contains:** inner `j` loop at line 203
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`, `j`, `index`
- **Arrays:** `vect(R)`, `fp(W)`
- **Issues:** output I/O; not part of timed compute region

### Loop: writeoutput at hotspot_openmp.cpp:203
- **Iterations:** `grid_cols`
- **Type:** A - linear per-row output
- **Parent loop:** line 202
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `j`, `index`
- **Arrays:** `vect(R)`, `fp(W)`
- **Issues:** output I/O

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| compute_tran_temp:170 | E | CRITICAL | none | `num_iterations` | stage/recurrence | cannot parallelize across timesteps |
| single_iteration:58 | G | CRITICAL | none | `row*col/256` | none | boundary divergence |
| single_iteration:67 | G | IMPORTANT | 58 | `BLOCK_SIZE_R` | none | boundary-only path |
| single_iteration:68 | G | IMPORTANT | 67 | `BLOCK_SIZE_C` | none | boundary-only path |
| single_iteration:124 | G | IMPORTANT | 58 | `BLOCK_SIZE_R` | none | none |
| single_iteration:125 | G | IMPORTANT | 124 | `BLOCK_SIZE_C` | none | none |
| read_input:225 | A | SECONDARY | none | `grid_rows*grid_cols` | none | file I/O |
| writeoutput:202/203 | A | SECONDARY | none | `grid_rows*grid_cols` | none | file I/O |
| main OUTPUT loop:300 | A | AVOID | none | `grid_rows*grid_cols` | none | debug output |

## Structural Recommendations
- **Natural offload unit:** fused routine
- **Why:** the real hotspot is one stencil update per timestep; keep the outer time-step recurrence on the host and offload the whole per-step `single_iteration` body as one GPU kernel-shaped region.
- **Fragmentation risk:** medium
- **Would preserving helper boundaries likely create tiny kernels?** YES

## Scalability Check
- **Default input likely too small to guide optimization?** NO
- **Small-input sensitive?** NO
- **Expected scaling risks:** transfer volume, host sync, memory footprint
- **Larger practical profile size recommendation:** `2048 x 2048` with `sim_time` around `10-20`
- **Why this size materially exercises the GPU:** it raises per-step work to a sustained stencil over millions of cells without stressing the 8 GB GPU memory budget

## Data Details
- **Dominant compute loop:** `compute_tran_temp` time-step loop calling `single_iteration`
- **Arrays swapped between functions?:** YES
- **Scratch arrays?:** YES
- **Mid-computation sync?:** NO
- **RNG in timed loop?:** NO
- **Preferred GPU pragma family:** `target teams distribute parallel for collapse(2)`
