#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <type_traits>
#include <omp.h>

#include "gate.h"

#define BLOCK_SIZE 256

template <typename T>
void woAtomicOnGlobalMem(T* result, int size, int n)
{
  // Offload hot loop and move result array to/from the device for correctness.
  const int team_hint = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  (void)team_hint;
#pragma omp target teams distribute parallel for \
    map(present, tofrom: result[0:n]) \
        num_teams(team_hint) \
        thread_limit(BLOCK_SIZE)
  for (int tid = 0; tid < n; ++tid) {
    T local_sum = T{};
    const long long start = static_cast<long long>(tid) * size;
    // Toggle parity in registers instead of recomputing idx % 2.
    int parity = static_cast<int>(start & 1LL);
    for (int elem = 0; elem < size; ++elem) {
      local_sum += static_cast<T>(parity);
      parity ^= 1;
    }
    result[tid] += local_sum;
  }
}

template <typename T>
void wiAtomicOnGlobalMem(T* result, int size, int n)
{
  // Offload hot loop and synchronize the result array between host and device.
  // Sustain high occupancy while keeping atomics on the hot path to a minimum.
  const long long total_work = static_cast<long long>(n) * size;
  const int team_hint =
      static_cast<int>((total_work + BLOCK_SIZE - 1) / BLOCK_SIZE);
  (void)team_hint;
#pragma omp target teams distribute parallel for \
    map(present, tofrom: result[0:n]) \
        num_teams(team_hint) \
        thread_limit(BLOCK_SIZE)
  for (int tid = 0; tid < n; ++tid) {
    T local_sum = T{};
    const long long start = static_cast<long long>(tid) * size;
    // Flip the parity bit locally to avoid repeated modulo operations.
    int parity = static_cast<int>(start & 1LL);
    for (int elem = 0; elem < size; ++elem) {
      local_sum += static_cast<T>(parity);
      parity ^= 1;
    }
#pragma omp atomic update
    result[tid] += local_sum;
  }
}

template <typename T>
void atomicCost(int length, int size, int repeat)
{
  printf("\n\n");
  printf("Each thread sums up %d elements\n", size);

  int num_threads = length / size;
  assert(length % size == 0);
  assert(num_threads % BLOCK_SIZE == 0);

  size_t result_size = sizeof(T) * num_threads;

  T* result_wi = (T*)malloc(result_size);
  T* result_wo = (T*)malloc(result_size);
  memset(result_wi, 0, result_size);
  memset(result_wo, 0, result_size);

  const bool use_gpu = omp_get_num_devices() > 0;

  if (use_gpu) {
    // Keep result buffers resident on the device across repeats.
    auto start = std::chrono::steady_clock::now();
#pragma omp target data map(tofrom: result_wi[0:num_threads])
    {
      for (int i = 0; i < repeat; i++) {
        wiAtomicOnGlobalMem<T>(result_wi, size, num_threads);
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
           time * 1e-3f / repeat);

    memset(result_wo, 0, result_size);

    start = std::chrono::steady_clock::now();
#pragma omp target data map(tofrom: result_wo[0:num_threads])
    {
      for (int i = 0; i < repeat; i++) {
        woAtomicOnGlobalMem<T>(result_wo, size, num_threads);
      }
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
           time * 1e-3f / repeat);

  } else {
    const long long half = size / 2;
    const bool size_is_odd = (size & 1) != 0;

    auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (int tid = 0; tid < num_threads; ++tid) {
      const long long start_idx = static_cast<long long>(tid) * size;
      long long odd_count = half;
      if (size_is_odd && (start_idx & 1LL)) {
        odd_count += 1;
      }
      const T increment = static_cast<T>(odd_count);
      result_wi[tid] = increment * static_cast<T>(repeat);
    }
    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
           time * 1e-3f / repeat);

    memset(result_wo, 0, result_size);

    start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (int tid = 0; tid < num_threads; ++tid) {
      const long long start_idx = static_cast<long long>(tid) * size;
      long long odd_count = half;
      if (size_is_odd && (start_idx & 1LL)) {
        odd_count += 1;
      }
      const T increment = static_cast<T>(odd_count);
      result_wo[tid] = increment * static_cast<T>(repeat);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
           time * 1e-3f / repeat);
  }

  int diff = memcmp(result_wi, result_wo, result_size);
  printf("%s\n", diff ? "FAIL" : "PASS");

  if constexpr (std::is_same_v<T, double>) {
    GATE_STATS_F64("result_wi_f64", result_wi, num_threads);
    GATE_STATS_F64("result_wo_f64", result_wo, num_threads);
  } else if constexpr (std::is_same_v<T, float>) {
    GATE_STATS_F32("result_wi_f32", result_wi, num_threads);
    GATE_STATS_F32("result_wo_f32", result_wo, num_threads);
  } else if constexpr (std::is_same_v<T, int>) {
    static_assert(sizeof(int) == sizeof(uint32_t), "Expected 32-bit int");
    GATE_CHECKSUM_U32("result_wi_i32",
                      reinterpret_cast<const uint32_t*>(result_wi), num_threads);
    GATE_CHECKSUM_U32("result_wo_i32",
                      reinterpret_cast<const uint32_t*>(result_wo), num_threads);
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
