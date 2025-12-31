#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define BLOCK_SIZE 256

template <typename T>
void BlockRangeAtomicOnGlobalMem(T* data, int n)
{
    for ( unsigned int i = 0; i < n; i++) {
        data[i % BLOCK_SIZE]++;  

  }
}

template <typename T>
void WarpRangeAtomicOnGlobalMem(T* data, int n)
{
    for ( unsigned int i = 0; i < n; i++) {
        data[i & 0x1F]++; 

  }
}

template <typename T>
void SingleRangeAtomicOnGlobalMem(T* data, int offset, int n)
{
    for ( unsigned int i = 0; i < n; i++) {
        data[0]++;    

  }
}

template <typename T>
void BlockRangeAtomicOnSharedMem(T* data, int n)
{
    {
    T smem_data[BLOCK_SIZE];
        {
      unsigned int blockIdx_x = omp_get_team_num();
      unsigned int gridDim_x = omp_get_num_teams();
      unsigned int blockDim_x = omp_get_num_threads();
      unsigned int threadIdx_x = omp_get_thread_num();
      unsigned int tid = (blockIdx_x * blockDim_x) + threadIdx_x;
      for ( unsigned int i = tid; i < n; i += blockDim_x*gridDim_x){
        smem_data[threadIdx_x]++;
      }
      if (blockIdx_x == gridDim_x)
        data[threadIdx_x] = smem_data[threadIdx_x];
    }
  }
}

template <typename T>
void WarpRangeAtomicOnSharedMem(T* data, int n)
{
    {
    T smem_data[32];
        {
      unsigned int blockIdx_x = omp_get_team_num();
      unsigned int gridDim_x = omp_get_num_teams();
      unsigned int blockDim_x = omp_get_num_threads();
      unsigned int threadIdx_x = omp_get_thread_num();
      unsigned int tid = (blockIdx_x * blockDim_x) + threadIdx_x;
      for ( unsigned int i = tid; i < n; i += blockDim_x*gridDim_x){
        smem_data[i & 0x1F]++;
      }
      if (blockIdx_x == gridDim_x && threadIdx_x < 0x1F)
        data[threadIdx_x] = smem_data[threadIdx_x];
    }
  }
}

template <typename T>
void SingleRangeAtomicOnSharedMem(T* data, int offset, int n)
{
    {
    T smem_data[BLOCK_SIZE];
        {
      unsigned int blockIdx_x = omp_get_team_num();
      unsigned int gridDim_x = omp_get_num_teams();
      unsigned int blockDim_x = omp_get_num_threads();
      unsigned int threadIdx_x = omp_get_thread_num();
      unsigned int tid = (blockIdx_x * blockDim_x) + threadIdx_x;
      for ( unsigned int i = tid; i < n; i += blockDim_x*gridDim_x){
        smem_data[offset]++;
      }
      if (blockIdx_x == gridDim_x && threadIdx_x == 0)
        data[threadIdx_x] = smem_data[threadIdx_x];
    }
  }
}

template <typename T>
void atomicPerf (int n, int t, int repeat)
{
  size_t data_size = sizeof(T) * t;

  T* data = (T*) malloc (data_size);

  for(int i=0; i<t; i++) data[i] = i%1024+1;

    {
        auto start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      BlockRangeAtomicOnGlobalMem<T>(data, n);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);

    for(int i=0; i<t; i++) data[i] = i%1024+1;
        start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      WarpRangeAtomicOnGlobalMem<T>(data, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);

    for(int i=0; i<t; i++) data[i] = i%1024+1;
        start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      SingleRangeAtomicOnGlobalMem<T>(data, i % BLOCK_SIZE, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);

    for(int i=0; i<t; i++) data[i] = i%1024+1;
        start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      BlockRangeAtomicOnSharedMem<T>(data, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n",
            time * 1e-3f / repeat);

    for(int i=0; i<t; i++) data[i] = i%1024+1;
        start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      WarpRangeAtomicOnSharedMem<T>(data, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n",
            time * 1e-3f / repeat);

    for(int i=0; i<t; i++) data[i] = i%1024+1;
        start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      SingleRangeAtomicOnSharedMem<T>(data, i % BLOCK_SIZE, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n",
            time * 1e-3f / repeat);

  }
  free(data);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int n = 3*4*7*8*9*256; 

  const int len = 1024; 

  
  printf("\nFP64 atomic add\n");
  atomicPerf<double>(n, len, repeat); 

  printf("\nINT32 atomic add\n");
  atomicPerf<int>(n, len, repeat); 

  printf("\nFP32 atomic add\n");
  atomicPerf<float>(n, len, repeat); 

  return 0;
}