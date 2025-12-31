#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <type_traits>
#include "../../../gate_sdk/gate.h"

constexpr int BLOCK_SIZE = 256;

template <typename T>
void wiAtomicOnGlobalMem(T* result, int size, int n, bool use_device) {
  const int total_elements = n * size;
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)      \
    map(present : result[0:n]) if (use_device)
  for (int gid = 0; gid < total_elements; ++gid) {
    const int tid = gid / size;
    const T contribution = static_cast<T>(gid & 1);
#pragma omp atomic update
    result[tid] += contribution;
  }
}

template <typename T>
void woAtomicOnGlobalMem(T* result, int size, int n, bool use_device) {
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)      \
    map(present : result[0:n]) if (use_device)
  for (int tid = 0; tid < n; ++tid) {
    const int base = tid * size;
    T local_sum = static_cast<T>(0);
#pragma omp simd reduction(+ : local_sum)
    for (int offset = 0; offset < size; ++offset) {
      const int idx = base + offset;
      local_sum += static_cast<T>(idx & 1);
    }
    result[tid] += local_sum;
  }
}

template <typename T>
void atomicCost(int length, int size, int repeat) {
  std::printf("\n\n");
  std::printf("Each thread sums up %d elements\n", size);

  const int num_threads = length / size;
  assert(length % size == 0);
  assert(num_threads % BLOCK_SIZE == 0);

  const std::size_t result_size = static_cast<std::size_t>(num_threads) * sizeof(T);

  T* result_wi = static_cast<T*>(std::malloc(result_size));
  T* result_wo = static_cast<T*>(std::malloc(result_size));
  if (!result_wi || !result_wo) {
    std::fprintf(stderr, "Unable to allocate result buffers\n");
    std::free(result_wi);
    std::free(result_wo);
    std::exit(1);
  }

  std::memset(result_wi, 0, result_size);
  std::memset(result_wo, 0, result_size);

  {
    auto start = std::chrono::steady_clock::now();
    const bool use_device = omp_get_num_devices() > 0;
#pragma omp target data map(tofrom : result_wi[0:num_threads]) if (use_device)
    {
      for (int i = 0; i < repeat; ++i) {
        wiAtomicOnGlobalMem(result_wi, size, num_threads, use_device);
      }
    }
    auto end = std::chrono::steady_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
                time * 1e-3f / repeat);

    start = std::chrono::steady_clock::now();
#pragma omp target data map(tofrom : result_wo[0:num_threads]) if (use_device)
    {
      for (int i = 0; i < repeat; ++i) {
        woAtomicOnGlobalMem(result_wo, size, num_threads, use_device);
      }
    }
    end = std::chrono::steady_clock::now();
    const auto time_wo = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
                time_wo * 1e-3f / repeat);

    const int diff = std::memcmp(result_wi, result_wo, result_size);
    std::printf("%s\n", diff ? "FAIL" : "PASS");

    const char* type_label = std::is_same<T, double>::value
                                 ? "fp64"
                                 : (std::is_same<T, float>::value ? "fp32"
                                                                  : "int32");
    char name_wi[32];
    char name_wo[32];
    std::snprintf(name_wi, sizeof(name_wi), "result_wi_%s", type_label);
    std::snprintf(name_wo, sizeof(name_wo), "result_wo_%s", type_label);

    GATE_CHECKSUM_BYTES(name_wi, result_wi, result_size);
    GATE_CHECKSUM_BYTES(name_wo, result_wo, result_size);
  }

  std::free(result_wi);
  std::free(result_wo);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::printf("Usage: %s <N> <repeat>\n", argv[0]);
    std::printf("N: the number of elements to sum per thread (1 - 16)\n");
    return 1;
  }

  const int nelems = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  const int length = 922521600;
  assert(length % BLOCK_SIZE == 0);

  std::printf("\nFP64 atomic add\n");
  atomicCost<double>(length, nelems, repeat);

  std::printf("\nINT32 atomic add\n");
  atomicCost<int>(length, nelems, repeat);

  std::printf("\nFP32 atomic add\n");
  atomicCost<float>(length, nelems, repeat);

  return 0;
}
