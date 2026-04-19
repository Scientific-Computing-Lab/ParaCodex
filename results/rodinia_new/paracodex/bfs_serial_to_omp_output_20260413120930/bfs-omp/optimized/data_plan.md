# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| `h_graph_nodes` | `no_of_nodes * sizeof(Node)` | index | host, then mapped to device | R |
| `h_graph_edges` | `edge_list_size * sizeof(int)` | index | host, then mapped to device | R |
| `h_graph_mask` | `no_of_nodes * sizeof(int)` | working | host init, then mapped to device | R/W |
| `h_updating_graph_mask` | `no_of_nodes * sizeof(int)` | scratch | host init, then mapped to device | R/W |
| `h_graph_visited` | `no_of_nodes * sizeof(int)` | working | host init, then mapped to device | R/W |
| `h_cost` | `no_of_nodes * sizeof(int)` | working | host init, then mapped to device | R/W |

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| BFS frontier sweep loop | `h_graph_nodes`, `h_graph_edges`, `h_graph_mask`, `h_graph_visited`, `h_updating_graph_mask`, `h_cost` | per BFS level | device |
| BFS frontier materialization loop | `h_graph_mask`, `h_updating_graph_mask` | per BFS level | device |

## Offload Unit Decision
- **Chosen Offload Unit:** fused BFS level body inside the convergence loop
- **Why this unit:** the timed path is already split into two level-synchronous passes; keeping both passes on device avoids host round-trips between frontier expansion and frontier materialization while preserving the required BFS barrier between levels
- **Timed-region stage count if left unfused:** 2 device kernels per BFS level plus host sync on the loop condition
- **Structural Risk:** host-device sync if frontier state is staged back each level, and duplicate-discovery races if frontier expansion is parallelized without atomics
- **Required rewrite before pragmas?** YES
- **Combined GPU+mem budget:** one persistent `target data` region, no mid-iteration `target update`, and two per-level GPU kernels

## Data Movement Strategy

**Chosen Strategy:** A

**Device Allocations (once):**
```
Strategy A: h_graph_nodes, h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost in one target data region
```

**Host→Device Transfers:**
- When: once, before the BFS convergence loop
- Arrays: `h_graph_nodes`, `h_graph_edges`, `h_graph_mask`, `h_updating_graph_mask`, `h_graph_visited`, `h_cost`
- Total H→D: ~`(sizeof(Node) * no_of_nodes) + (5 * sizeof(int) * no_of_nodes) + (sizeof(int) * edge_list_size)` bytes

**Device→Host Transfers:**
- When: once, after the BFS convergence loop
- Arrays: `h_graph_mask`, `h_updating_graph_mask`, `h_graph_visited`, `h_cost`
- Total D→H: ~`(4 * sizeof(int) * no_of_nodes)` bytes

**Transfers During Iterations:** NO
- If NO: all BFS frontier state stays resident on device until the loop finishes

**Mid-computation sync in timed region:** NO
- If NO: no `target update` or host-side inspection is needed inside the BFS levels

## Critical Checks (for chosen strategy)

**Strategy A:**
- [x] Functions inside target data use `present,alloc` wrapper?
- [x] Scratch arrays use enter/exit data OR omp_target_alloc?
- [x] Chosen offload unit avoids avoidable tiny-kernel staging?

**Common Mistakes:**
- Some functions on device, others on host, which would reintroduce copying
- Leaving frontier discovery serial inside the GPU region
- Forgetting to keep the frontier masks resident across BFS levels
- Leaving generated placeholders such as `<RUN_ARGS>` or `<PROGRAM_NAME>` in `Makefile.nvc`

## Expected Transfer Volume
- Total: roughly one full graph load to the device and one full result copy back
- **Red flag:** if profiling shows repeated transfer volume per BFS level, the data region is wrong

## Additional Parallelization Notes
- **RNG Replicable?** NO
- **Outer Saturation?** one node sweep per BFS level
- **Histogram Strategy?** N/A
- **Kernel Granularity Check:** good if the two BFS passes stay fused at the level boundary
- **Preferred GPU pragma form:** `target teams distribute parallel for` - this compiler/runtime has a clearer mapping for the sparse node sweeps than `target teams loop` for this kernel

## Scalability Check
- **Default correctness size:** `../../../data/bfs/graph1MW_6.txt` as the benchmark input path
- **Larger practical profiling size:** a larger BFS graph with the same file format and a wider frontier profile, sized so the level-synchronous loops run long enough to dominate launch overhead
- **Why this size materially exercises the GPU:** the sparse frontier sweep touches all nodes each BFS level, so the graph must be large enough to produce sustained device work rather than a launch-dominated microbenchmark
- **Likely small-input sensitive?** YES
- **Scaling risk if this design is chosen:** kernel count per BFS level and host synchronization on the `stop` condition
- **Chosen structure still plausible at larger size?** YES

**Summary:** 6 arrays (1 scratch, 5 working/index), 2 functions in the timed path, Strategy A, offload unit fused convergence loop. Expected: one H→D transfer at start, one D→H transfer at end, 2 hot kernels per BFS level.

## Build / Run Readiness
- **Plain build command works?** YES after replacing placeholders
- **Plain `make -f Makefile.nvc run` works?** YES once the BFS input file is present at the configured path
- **Unresolved placeholders remaining?** NO
- **`{nsys_profile_cmd} > {profile_log_path} 2>&1` shows GPU kernels?** expected YES after the input file and build are in place
