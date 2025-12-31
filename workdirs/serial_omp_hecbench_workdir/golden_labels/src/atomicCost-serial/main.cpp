#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#define BLOCK_SIZE 256

template <typename T>
void woAtomicOnGlobalMem(T* result, int size, int n)
{
    for (unsigned int tid = 0; tid < n; tid++) {
        for ( unsigned int i = tid * size; i < (tid + 1) * size; i++) {
            result[tid] += i % 2;
        }
    }
}

template <typename T>
void wiAtomicOnGlobalMem(T* result, int size, int n)
{
    for (unsigned int tid = 0; tid < n; tid++) {
        for ( unsigned int i = tid * size; i < (tid + 1) * size; i++) {
            result[tid] += i % 2;
        }
    }
}

template <typename T>
void atomicCost (int length, int size, int repeat)
{
  printf("\n\n");
  printf("Each thread sums up %d elements\n", size);

  int num_threads = length / size;
  assert(length % size == 0);
  assert(num_threads % BLOCK_SIZE == 0);

  size_t result_size = sizeof(T) * num_threads;

  T* result_wi = (T*) malloc (result_size);
  T* result_wo = (T*) malloc (result_size);
  memset(result_wi, 0, result_size);
  memset(result_wo, 0, result_size);

  {
    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      wiAtomicOnGlobalMem<T>(result_wi, size, num_threads);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);
    
    start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      woAtomicOnGlobalMem<T>(result_wo, size, num_threads);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);
    
    int diff = memcmp(result_wi, result_wo, result_size);
    printf("%s\n", diff ? "FAIL" : "PASS");
  }

  free(result_wi);
  free(result_wo);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: the number of elements to sum per thread (1 - 16)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int length = 922521600;
  assert(length % BLOCK_SIZE == 0);

  printf("\nFP64 atomic add\n");
  atomicCost<double>(length, nelems, repeat);

  printf("\nINT32 atomic add\n");
  atomicCost<int>(length, nelems, repeat);

  printf("\nFP32 atomic add\n");
  atomicCost<float>(length, nelems, repeat);

  return 0;
}
