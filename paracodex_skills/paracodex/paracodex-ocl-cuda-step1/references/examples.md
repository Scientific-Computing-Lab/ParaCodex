# CUDA Migration Examples

## Work-Item Indexing
```c
// OpenCL 2D
int gid_x = get_global_id(0);
int gid_y = get_global_id(1);

// CUDA 2D
int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
```

## Dynamic Shared Memory
```c
// OpenCL: clSetKernelArg(kernel, argIndex, sharedMemSize, NULL);

// CUDA: Third launch parameter
extern __shared__ float smem[]; // In kernel
kernel<<<grid, block, sharedMemSize>>>(...);

// Multiple dynamic arrays (offsets)
extern __shared__ char sharedMem[];
float *smem1 = (float*)sharedMem;
float *smem2 = (float*)&sharedMem[size1];
```

## Kernel Launch Mapping
```c
// OpenCL
size_t global[3] = {N*M, 1, 1};
size_t local[3] = {M, 1, 1};
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, ...);

// CUDA
dim3 block(M);
dim3 grid(N); // Note: Grid is Number of Blocks, not Global Size
kernel<<<grid, block>>>(...);
```

## Error Handling
```c
#define CUDA_CHECK(call) \
    { cudaError_t err = call; \
      if (err != cudaSuccess) { \
          fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
          exit(1); \
      } }

CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

## Image Objects to Textures
```c
// OpenCL
float4 pixel = read_imagef(img, sampler, coord);

// CUDA (Texture Object or Reference)
texture<float4, 2, cudaReadModeElementType> tex;
float4 pixel = tex2D(tex, x, y);
```

## Streams + Async Transfers (Best Practice)
```c
// Replace OpenCL event-based async with CUDA streams
cudaStream_t stream;
cudaStreamCreate(&stream);

// Async H→D on stream
cudaMalloc(&d_A, size);
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);

// Kernel on same stream (executes after transfer)
int block = 256;  // Multiple of 32 (warp size)
int grid  = (N + block - 1) / block;
__launch_bounds__(256)
myKernel<<<grid, block, 0, stream>>>(d_A, d_B, N);

// Async D→H
cudaMemcpyAsync(h_B, d_B, size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

## Replace weak OpenCL decomposition when CUDA gives a better structure
```c
// OpenCL often carries host-side enqueue structure that CUDA does not need.
// Prefer the CUDA kernel decomposition that minimizes launches and host overhead.
```

## Warp-Level Primitives (CUDA bonus vs OpenCL)
```c
// Replace local-memory reduction with warp shuffle
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

## Unified Memory (simplify migration — no explicit transfers)
```c
float *data;
cudaMallocManaged(&data, size);  // Accessible from both CPU and GPU
myKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();
// data is ready on host — no cudaMemcpy needed
```
