# Analysis Output Template

## Loop Nesting Structure
```
- bpnn_layerforward (backprop.c:243) Type A
  └── inner accumulation loop (backprop.c:247) Type F
- bpnn_output_error (backprop.c:262) Type F
- bpnn_hidden_error (backprop.c:286) Type A
  └── inner accumulation loop (backprop.c:289) Type F
- bpnn_adjust_weights (backprop.c:308) Type A
  └── inner weight update loop (backprop.c:309) Type A
- setup/load/randomization/free/save/read loops (setup and I/O only)
```

## Loop Details

### Loop: bpnn_layerforward at backprop.c:243
- **Iterations:** `n2` outer iterations, each with `n1 + 1` inner multiply-adds
- **Type:** A - dense outer loop over hidden/output units; each iteration is independent
- **Parent loop:** none
- **Contains:** inner reduction loop at backprop.c:247
- **Dependencies:** inner reduction on `sum`
- **Nested bounds:** variable
- **Private vars:** `j`, `k`, `sum`
- **Arrays:** `l1(R)`, `l2(W)`, `conn(R)`, `sum(RW)`
- **Issues:** reduction in inner loop; pointer-to-pointer matrix access

### Loop: bpnn_layerforward inner loop at backprop.c:247
- **Iterations:** `n1 + 1`
- **Type:** F - scalar reduction into `sum`
- **Parent loop:** backprop.c:243
- **Contains:** none
- **Dependencies:** reduction: `sum`
- **Nested bounds:** variable
- **Private vars:** `k`
- **Arrays:** `conn(R)`, `l1(R)`
- **Issues:** reduction clause needed if parallelized

### Loop: bpnn_output_error at backprop.c:262
- **Iterations:** `nj` outer iterations
- **Type:** F - reduction to scalar `errsum` with independent per-element `delta[j]`
- **Parent loop:** none
- **Contains:** none
- **Dependencies:** reduction: `errsum`
- **Nested bounds:** variable
- **Private vars:** `j`, `o`, `t`
- **Arrays:** `output(R)`, `target(R)`, `delta(W)`
- **Issues:** tiny loop in this benchmark because `out = 1`; reduction only

### Loop: bpnn_hidden_error at backprop.c:286
- **Iterations:** `nh` outer iterations, each with `no` inner multiplies
- **Type:** A - dense outer loop over hidden units; each hidden unit update is independent
- **Parent loop:** none
- **Contains:** inner reduction loop at backprop.c:289
- **Dependencies:** inner reduction on `sum`
- **Nested bounds:** variable
- **Private vars:** `j`, `k`, `h`, `sum`
- **Arrays:** `hidden(R)`, `delta_o(R)`, `who(R)`, `delta_h(W)`
- **Issues:** reduction in inner loop; pointer-to-pointer matrix access

### Loop: bpnn_hidden_error inner loop at backprop.c:289
- **Iterations:** `no`
- **Type:** F - scalar reduction into `sum`
- **Parent loop:** backprop.c:286
- **Contains:** none
- **Dependencies:** reduction: `sum`
- **Nested bounds:** variable
- **Private vars:** `k`
- **Arrays:** `delta_o(R)`, `who(R)`
- **Issues:** reduction clause needed if parallelized

### Loop: bpnn_adjust_weights at backprop.c:308
- **Iterations:** `ndelta` outer iterations, each with `nly + 1` updates
- **Type:** A - dense update loop over output/hidden units
- **Parent loop:** none
- **Contains:** inner weight-update loop at backprop.c:309
- **Dependencies:** none across iterations
- **Nested bounds:** variable
- **Private vars:** `j`, `k`, `new_dw`
- **Arrays:** `delta(R)`, `ly(R)`, `w(RW)`, `oldw(RW)`
- **Issues:** write to distinct `w[k][j]` and `oldw[k][j]` elements; pointer-to-pointer matrices

### Loop: bpnn_adjust_weights inner loop at backprop.c:309
- **Iterations:** `nly + 1`
- **Type:** A - each `(k, j)` update is independent
- **Parent loop:** backprop.c:308
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `k`, `new_dw`
- **Arrays:** `delta(R)`, `ly(R)`, `w(RW)`, `oldw(RW)`
- **Issues:** safe for `collapse(2)` or nested parallelization, but the fused routine is the better offload unit

### Loop: bpnn_randomize_weights at backprop.c:96
- **Iterations:** `(m + 1) * (n + 1)`
- **Type:** A - dense initialization
- **Parent loop:** none
- **Contains:** nested loop at backprop.c:97
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`, `j`
- **Arrays:** `w(W)`
- **Issues:** setup only; uses `rand()`

### Loop: bpnn_zero_weights at backprop.c:122
- **Iterations:** `(m + 1) * (n + 1)`
- **Type:** A - dense initialization
- **Parent loop:** none
- **Contains:** nested loop at backprop.c:123
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`, `j`
- **Arrays:** `w(W)`
- **Issues:** setup only

### Loop: alloc_2d_dbl at backprop.c:82
- **Iterations:** `m`
- **Type:** A - setup allocation
- **Parent loop:** none
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`
- **Arrays:** `new(W)`
- **Issues:** setup only; allocation overhead

### Loop: bpnn_free at backprop.c:185 and backprop.c:192
- **Iterations:** `n1 + 1` and `n2 + 1`
- **Type:** A - cleanup traversal
- **Parent loop:** none
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`
- **Arrays:** pointer arrays only
- **Issues:** teardown only

### Loop: bpnn_save at backprop.c:407 and backprop.c:421
- **Iterations:** weight-matrix sized traversals
- **Type:** A - serialization
- **Parent loop:** none
- **Contains:** inner loop for each matrix
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`, `j`, `dvalue`, `memcnt`
- **Arrays:** `w(R)`, `mem(W)`
- **Issues:** I/O path only; not part of timed training kernel

### Loop: bpnn_read at backprop.c:461 and backprop.c:474
- **Iterations:** weight-matrix sized traversals
- **Type:** A - deserialization
- **Parent loop:** none
- **Contains:** inner loop for each matrix
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`, `j`, `memcnt`
- **Arrays:** `mem(R)`, `new->input_weights(W)`, `new->hidden_weights(W)`
- **Issues:** I/O path only; not part of timed training kernel

### Loop: load at imagenet.c:20
- **Iterations:** `nr`
- **Type:** A - dense input initialization
- **Parent loop:** none
- **Contains:** none
- **Dependencies:** none
- **Nested bounds:** variable
- **Private vars:** `i`, `k`
- **Arrays:** `units(W)`
- **Issues:** setup only; uses `rand()`; `nc` is uninitialized in the dead `imgsize` expression

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| bpnn_layerforward | A | CRITICAL | none | `n2` x (`n1 + 1`) | inner reduction on `sum` | reduction, pointer-to-pointer |
| bpnn_layerforward inner | F | CRITICAL | backprop.c:243 | `n1 + 1` | reduction: `sum` | reduction clause needed |
| bpnn_output_error | F | AVOID | none | `nj` | reduction: `errsum` | tiny loop, out layer is size 1 |
| bpnn_hidden_error | A | IMPORTANT | none | `nh` x `no` | inner reduction on `sum` | reduction, pointer-to-pointer |
| bpnn_hidden_error inner | F | IMPORTANT | backprop.c:286 | `no` | reduction: `sum` | reduction clause needed |
| bpnn_adjust_weights | A | CRITICAL | none | `ndelta` x (`nly + 1`) | none | dense update, pointer-to-pointer |
| bpnn_adjust_weights inner | A | CRITICAL | backprop.c:308 | `nly + 1` | none | safe for collapse |
| bpnn_randomize_weights | A | SECONDARY | none | `(m + 1) * (n + 1)` | none | setup/RNG only |
| bpnn_zero_weights | A | SECONDARY | none | `(m + 1) * (n + 1)` | none | setup only |
| alloc_2d_dbl | A | SECONDARY | none | `m` | none | allocation only |
| bpnn_free | A | SECONDARY | none | `n1 + 1`, `n2 + 1` | none | teardown only |
| bpnn_save / bpnn_read | A | SECONDARY | none | matrix-sized | none | I/O only |
| load | A | SECONDARY | none | `nr` | none | setup RNG; `nc` unused |

## Structural Recommendations
- **Natural offload unit:** fused routine
- **Why:** `bpnn_train_kernel()` already sequences the full compute path, and the helper calls inside it have strict stage dependencies: layerforward -> output_error/hidden_error -> adjust_weights. Preserving helper boundaries would create multiple small device regions and likely extra host/device sync or redundant mappings. The best OpenMP target shape is one fused training kernel with the dense loop nests inside it.
- **Fragmentation risk:** high
- **Would preserving helper boundaries likely create tiny kernels?** YES

## Scalability Check
- **Default input likely too small to guide optimization?** YES
- **Small-input sensitive?** YES
- **Expected scaling risks:** kernel count, host sync, transfer count, transfer volume
- **Larger practical profile size recommendation:** use a `layer_size` large enough that `input_weights` and `input_prev_weights` are each tens of MB at minimum; a good starting point is `layer_size >= 262144`, with `layer_size ~ 1,000,000` preferred if memory allows

## Data Details
- **Dominant compute loop:** `bpnn_train_kernel()` in `backprop_kernel.c:42` drives the timed path; the dominant work is in `bpnn_layerforward()` and `bpnn_adjust_weights()`
- **Arrays swapped between functions?:** NO
- **Scratch arrays?:** YES
- **Mid-computation sync?:** YES if helper boundaries are preserved; NO if the full training path is fused into one target region
- **RNG in timed loop?:** NO
- **Preferred GPU pragma family:** `target teams distribute parallel for`
