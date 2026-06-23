# Q1: Basic Vector Add?
```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Launch
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

# Q2: Error Handling?
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
}
```

# Q3: Memory Management (cudaMallocAsync + Stream)?
```cpp
// Preferred (CUDA 11.2+): stream-ordered async alloc — lower overhead than cudaMalloc
cudaStream_t stream;
cudaStreamCreate(&stream);

float *d_A, *d_B, *d_C;
cudaMallocAsync((void**)&d_A, size, stream);  // Allocate on stream
cudaMallocAsync((void**)&d_B, size, stream);
cudaMallocAsync((void**)&d_C, size, stream);

// Async H→D
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

// Launch on same stream (executes after transfers complete)
int block = 256;
int grid = (N + block - 1) / block;
vectorAdd<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);

// Async D→H
cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);

cudaFreeAsync(d_A, stream);  // Stream-ordered free — allows reuse
cudaFreeAsync(d_B, stream);
cudaFreeAsync(d_C, stream);
cudaStreamDestroy(stream);
```

# Q4: __launch_bounds__ and Grid Dimensions?
```cpp
#define BLOCK_SIZE 256

// Cap register usage; helps compiler schedule better
__launch_bounds__(BLOCK_SIZE)
__global__ void myKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// Grid dim: ceiling division
int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
myKernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_A, d_B, d_C, N);
```

# Q4b: Fused kernel beats helper-by-helper translation
```cpp
// If the serial hot path is:
stage1(...);
stage2(...);
stage3(...);

// And each stage is small, prefer:
__global__ void fusedKernel(...) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // stage1 logic
        // stage2 logic
        // stage3 logic
    }
}
```
Use this when it reduces launches, temporary writes, or host-device traffic.

# Q5: OpenACC Alternative?
```cpp
#pragma acc parallel loop copyin(A[0:N], B[0:N]) copyout(C[0:N])
for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
}
```
