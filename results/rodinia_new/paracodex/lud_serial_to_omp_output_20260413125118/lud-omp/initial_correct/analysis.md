# Analysis Output Template

## Loop Nesting Structure
```
- lud_omp offset loop (omp/lud_omp.c:37) Type E
  - lud_diagonal_omp block helper (omp/lud_omp.c:11) Type E
  - perimeter chunk loop (omp/lud_omp.c:46) Type A
  - interior chunk loop (omp/lud_omp.c:94) Type A
```

## Loop Details

### Loop: lud_omp at omp/lud_omp.c:37
- **Iterations:** `size / BS`
- **Type:** E - stage recurrence; each offset depends on the previous block factorization.
- **Parent loop:** none
- **Contains:** diagonal block helper call, perimeter chunk loop, interior chunk loop
- **Dependencies:** stage / recurrence
- **Nested bounds:** variable
- **Private vars:** `offset`, `chunk_idx`, `size_inter`, `chunks_in_inter_row`, `chunks_per_inter`
- **Arrays:** `a(RW)`
- **Issues:** stage dependency, helper fragmentation risk, variable bounds

### Loop: lud_diagonal_omp at omp/lud_omp.c:11
- **Iterations:** `BS` (fixed at 16)
- **Type:** E - block-local recurrence; later rows in the block depend on earlier ones.
- **Parent loop:** line 37
- **Contains:** two fixed-size inner `j` loops and reduction-like `k` loops
- **Dependencies:** recurrence
- **Nested bounds:** constant
- **Private vars:** `i`, `j`, `k`, `temp`
- **Arrays:** `a(RW)`
- **Issues:** fixed small block, sequential helper, not a good independent offload unit

### Loop: lud_omp at omp/lud_omp.c:46
- **Iterations:** `chunks_in_inter_row`
- **Type:** A - each chunk updates a disjoint perimeter block.
- **Parent loop:** line 37
- **Contains:** scratch copy loop and two fixed-size update nests
- **Dependencies:** none at chunk level; inner `k` loops are local reductions
- **Nested bounds:** variable
- **Private vars:** `chunk_idx`, `i`, `j`, `k`, `i_global`, `j_global`, `i_here`, `j_here`, `sum`, `temp`
- **Arrays:** `a(RW)`, `temp(W, local scratch)`
- **Issues:** scratch temp, inner reductions, block-size constant

### Loop: lud_omp at omp/lud_omp.c:94
- **Iterations:** `chunks_per_inter`
- **Type:** A - each iteration updates a distinct interior block.
- **Parent loop:** line 37
- **Contains:** scratch copy loops and a fixed-size block multiply-update nest
- **Dependencies:** none at chunk level; inner `k` loop is a local accumulation
- **Nested bounds:** variable
- **Private vars:** `chunk_idx`, `i`, `j`, `k`, `i_global`, `j_global`, `sum`, `temp_top`, `temp_left`
- **Arrays:** `a(RW)`, `temp_top(W, local scratch)`, `temp_left(W, local scratch)`, `sum(W, local scratch)`
- **Issues:** scratch arrays, reduction-like inner accumulation, block-size constant

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| `lud_omp` offset loop | E | CRITICAL | none | `size / BS` | stage / recurrence | stage dependency, helper fragmentation risk |
| `lud_diagonal_omp` | E | SECONDARY | line 37 | `BS` | recurrence | fixed small block, sequential helper |
| `lud_omp` perimeter chunk loop | A | IMPORTANT | line 37 | `chunks_in_inter_row` | none at chunk level | scratch temp, inner reductions |
| `lud_omp` interior chunk loop | A | CRITICAL | line 37 | `chunks_per_inter` | none at chunk level | scratch arrays, inner accumulation |

## Structural Recommendations
- **Natural offload unit:** fused routine
- **Why:** `lud_omp` already has a staged block-LU shape; keeping `lud_diagonal_omp` and the chunk-update phases separate would create tiny device regions and extra host/device synchronization.
- **Fragmentation risk:** high
- **Would preserving helper boundaries likely create tiny kernels?** YES

## Scalability Check
- **Default input likely too small to guide optimization?** YES
- **Small-input sensitive?** YES
- **Expected scaling risks:** kernel count, host sync, transfer volume, launch overhead
- **Larger practical profile size recommendation:** `2048`
- **Why this size materially exercises the GPU:** it gives enough block count and cubic work to amortize launch overhead while keeping the matrix footprint modest and comfortably within memory limits

## Data Details
- **Dominant compute loop:** `lud_omp` offset loop
- **Arrays swapped between functions?:** NO
- **Scratch arrays?:** YES
- **Mid-computation sync?:** YES
- **RNG in timed loop?:** NO
- **Preferred GPU pragma family:** `target teams distribute parallel for`
