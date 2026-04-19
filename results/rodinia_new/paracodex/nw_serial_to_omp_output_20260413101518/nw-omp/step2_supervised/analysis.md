# Analysis Output Template

## Loop Nesting Structure
```
- nw_optimized stage loop A (needle.cpp:97-145) Type E
  └── b_index_x tile loop (needle.cpp:99-143) Type B
      ├── copy reference nest (needle.cpp:106-112) Type A
      ├── copy input nest (needle.cpp:115-121) Type A
      ├── compute nest (needle.cpp:124-132) Type E
      └── copy-back nest (needle.cpp:135-141) Type A
- nw_optimized stage loop B (needle.cpp:148-195) Type E
  └── b_index_x tile loop (needle.cpp:150-194) Type B
      ├── copy reference nest (needle.cpp:158-164) Type A
      ├── copy input nest (needle.cpp:167-173) Type A
      ├── compute nest (needle.cpp:176-184) Type E
      └── copy-back nest (needle.cpp:187-193) Type A
```

## Loop Details
For each CRITICAL/IMPORTANT/SECONDARY loop:

### Loop: nw_optimized stage loops at needle.cpp:97-145 and needle.cpp:148-195
- **Iterations:** O(max_cols / 16) staged wavefront levels
- **Type:** E - wavefront stage dependency; later diagonals consume results from earlier diagonals
- **Parent loop:** none
- **Contains:** `b_index_x` tile loop, plus tile-local copy/compute/copy helper nests
- **Dependencies:** stage
- **Nested bounds:** variable
- **Private vars:** `blk`, `b_index_x`, `b_index_y`, `i`, `j`, `input_itemsets_l`, `reference_l`
- **Arrays:** `input_itemsets(RW)`, `referrence(R)`, `input_itemsets_l(RW scratch)`, `reference_l(R scratch)`
- **Issues:** stage dependency, fused offload needed, preserving helpers would fragment the hot path

### Loop: b_index_x tile loops at needle.cpp:99-143 and needle.cpp:150-194
- **Iterations:** triangular number per stage, total O((N / 16)^2) tiles
- **Type:** B - inner bound varies with the stage index, but each tile iteration is independent within a stage
- **Parent loop:** `blk`
- **Contains:** 4 fixed-size inner nests: copy reference, copy input, compute, copy back
- **Dependencies:** stage
- **Nested bounds:** variable
- **Private vars:** `b_index_x`, `b_index_y`, `i`, `j`
- **Arrays:** `input_itemsets(RW)`, `referrence(R)`
- **Issues:** varying bounds, repeated scratch copies, offload granularity is at tile level rather than helper level

### Loop: tile compute nests at needle.cpp:124-132 and needle.cpp:176-184
- **Iterations:** 16 x 16 per tile
- **Type:** E - recurrence; each cell depends on left, top, and diagonal neighbors within the same tile
- **Parent loop:** `b_index_x`
- **Contains:** none
- **Dependencies:** recurrence
- **Nested bounds:** constant
- **Private vars:** `i`, `j`
- **Arrays:** `input_itemsets_l(RW)`, `reference_l(R)`, `penalty(scalar)`
- **Issues:** cannot parallelize directly with plain `parallel for`; needs diagonalization or outer-tile parallelism

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| `nw_optimized` stage loops | E | CRITICAL | none | O(max_cols / 16) stages | stage | stage dependency, helper fragmentation risk |
| `b_index_x` tile loops | B | IMPORTANT | `blk` | O((N / 16)^2) tiles | stage | variable bounds, launch granularity |
| tile compute nests | E | CRITICAL | `b_index_x` | 256 updates per tile | recurrence | loop-carried dependency, sequential within tile |

## Structural Recommendations
- **Natural offload unit:** fused routine
- **Why:** the timed path is a staged wavefront with repeated tile-local copy/compute/copy helpers; keeping the helper boundaries would create many tiny offload regions and extra synchronization/data movement
- **Fragmentation risk:** high
- **Would preserving helper boundaries likely create tiny kernels?** YES

## Scalability Check
- **Default input likely too small to guide optimization?** YES
- **Small-input sensitive?** YES
- **Expected scaling risks:** kernel count, transfer count, transfer volume, host sync
- **Larger practical profile size recommendation:** use a square size that is a multiple of 16 and at least `4096`; `8192` is better for stable profiling on this machine if memory allows

## Data Details
- **Dominant compute loop:** the tile compute nests in `nw_optimized` inside the timed region
- **Arrays swapped between functions?:** NO
- **Scratch arrays?:** YES
- **Mid-computation sync?:** YES
- **RNG in timed loop?:** NO
- **Preferred GPU pragma family:** `target teams loop` on the tile loop, with the stage loop kept serial
