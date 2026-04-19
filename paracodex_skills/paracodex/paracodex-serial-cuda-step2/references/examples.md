# Q1: Shared Memory Tiling?
> **Concept:** Load block of data into fast `__shared__` memory, synchronize, then process.
```cpp
__global__ void tiledKernel(float* A, float* B) {
    __shared__ float tile[TILE_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    tile[tid] = A[idx]; // Load
    __syncthreads();
    
    // Compute using tile
    B[idx] = tile[tid] * 2.0f;
}
```

# Q2: Coalesced Access?
> **Rule:** Threads in a warp (32) should access contiguous memory addresses.
> **Avoid:** Stride > 1.
> **Structure of Arrays (SoA)** is better than Array of Structures (AoS).

# Q3: Async Transfers?
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

# Q4: Warp Shuffle Reduction (and Cooperative Groups)?
> **Concept:** Replace shared-memory reductions with warp-shuffle — no `__shared__` needed within a warp.
```cpp
// Classic approach: manual shuffle loop
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;  // Valid in lane 0
}

// Modern preferred approach: Cooperative Groups (CUDA 9+)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void sumKernel(const float* in, float* out, int N) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < N) ? in[idx] : 0.f;
    // cg::reduce handles the shuffle internally — portable and clean
    float sum = cg::reduce(warp, val, cg::plus<float>());
    if (warp.thread_rank() == 0)
        atomicAdd(out, sum);
}
```

# Q5: Occupancy-Tuned Block Size?
```cpp
int minGridSize, bestBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize,
                                   myKernel, 0, 0);
int grid = (N + bestBlockSize - 1) / bestBlockSize;
myKernel<<<grid, bestBlockSize>>>(d_A, d_B, d_C, N);
```

# Q6: Shared Memory Padding (Avoid Bank Conflicts)?
```cpp
// 32 banks × 4 bytes. Stride-32 access causes conflicts.
// Add +1 padding to each row to break the stride pattern.
__shared__ float tile[TILE][TILE + 1];  // +1 eliminates bank conflicts
```

# Q7: CUDA Graphs (Eliminate Per-Launch Overhead)?
```cpp
// Capture a stream into a graph (CUDA 10+)
// Best for kernels launched repeatedly in a loop with fixed parameters
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStream_t stream;
cudaStreamCreate(&stream);

// --- Capture phase (run once) ---
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
myKernel<<<grid, block, 0, stream>>>(d_A, d_B, N);
cudaMemcpyAsync(h_B, d_B, size, cudaMemcpyDeviceToHost, stream);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// --- Replay phase (many iterations, near-zero overhead) ---
for (int iter = 0; iter < NUM_ITERS; iter++) {
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}

cudaGraphExecDestroy(graphExec);
cudaGraphDestroy(graph);
cudaStreamDestroy(stream);
```

# Q8: Structural rewrite when profile says the kernel split is wrong
```cpp
// If the profile shows many tiny kernels or large host overhead,
// rewrite the hot path before micro-tuning.
// Examples: fuse producer-consumer kernels, keep intermediates on device,
// or flatten pointer-heavy inputs into contiguous buffers.
```
