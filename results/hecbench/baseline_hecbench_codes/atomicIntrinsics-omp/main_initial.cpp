#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include "reference.h"
#include "gate.h"

template <typename T>
constexpr const char *gate_gpu_data_name();

template <>
constexpr const char *gate_gpu_data_name<int>() { return "gpu_data_int"; }

template <>
constexpr const char *gate_gpu_data_name<unsigned int>() { return "gpu_data_uint"; }

namespace
{
constexpr int kNumSlots = 9;

template <typename T>
void launch_gpu_stage(const T *data, T *gpu_data, const int len, const int repeat,
                      const bool copy_back, const bool use_gpu)
{
  if (!use_gpu) {
    for (int n = 0; n < repeat; ++n) {
      T add_res = 0;
      T sub_res = 0;
      T and_res = data[4];
      T or_res = data[5];
      T xor_res = data[6];
      T max_res = data[2];
      T min_res = data[3];

      for (int i = 0; i < len; ++i) {
        add_res += static_cast<T>(10);
        sub_res -= static_cast<T>(10);
        T and_val = static_cast<T>(2 * i + 7);
        and_res &= and_val;
        T or_val = static_cast<T>(1 << i);
        or_res |= or_val;
        T candidate = static_cast<T>(i);
        xor_res ^= candidate;
        if (candidate > max_res) {
          max_res = candidate;
        }
        if (candidate < min_res) {
          min_res = candidate;
        }
      }

      if (copy_back) {
        gpu_data[0] = data[0] + add_res;
        gpu_data[1] = data[1] + sub_res;
        gpu_data[2] = max_res;
        gpu_data[3] = min_res;
        gpu_data[4] = and_res;
        gpu_data[5] = or_res;
        gpu_data[6] = xor_res;
      }
    }
    return;
  }

  #pragma omp target data map(alloc: gpu_data[0:kNumSlots]) map(to: data[0:kNumSlots])
  {
    for (int n = 0; n < repeat; ++n) {
      T add_res = 0;
      T sub_res = 0;
      T and_res = data[4];
      T or_res = data[5];
      T xor_res = data[6];
      T max_res = data[2];
      T min_res = data[3];

      #pragma omp target teams distribute parallel for reduction(+:add_res) reduction(+:sub_res) \
        reduction(&:and_res) reduction(|:or_res) reduction(^:xor_res) reduction(max:max_res) \
        reduction(min:min_res)
      for (int i = 0; i < len; ++i) {
        add_res += static_cast<T>(10);
        sub_res -= static_cast<T>(10);
        T and_val = static_cast<T>(2 * i + 7);
        and_res &= and_val;
        T or_val = static_cast<T>(1 << i);
        or_res |= or_val;
        T candidate = static_cast<T>(i);
        xor_res ^= candidate;
        if (candidate > max_res) {
          max_res = candidate;
        }
        if (candidate < min_res) {
          min_res = candidate;
        }
      }

      T stage_results[7];
      stage_results[0] = data[0] + add_res;
      stage_results[1] = data[1] + sub_res;
      stage_results[2] = max_res;
      stage_results[3] = min_res;
      stage_results[4] = and_res;
      stage_results[5] = or_res;
      stage_results[6] = xor_res;

      #pragma omp target map(present: gpu_data[0:kNumSlots]) map(to: stage_results[0:7])
      for (int idx = 0; idx < 7; ++idx) {
        gpu_data[idx] = stage_results[idx];
      }
    }

    if (copy_back) {
      #pragma omp target update from(gpu_data[0:kNumSlots])
    }
  }
}
} // namespace

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  const T data[kNumSlots] = {0, 0, static_cast<T>(-256), static_cast<T>(256), 255, 0, 255, 0, 0};
  T gpu_data[kNumSlots];
  const bool use_gpu = omp_get_num_devices() > 0;

  launch_gpu_stage(data, gpu_data, len, repeat, true, use_gpu);

  const T inc_limit = static_cast<T>(17);
  const T dec_limit = static_cast<T>(137);
  gpu_data[7] = static_cast<T>(len % (inc_limit + 1));
  gpu_data[8] = static_cast<T>(dec_limit - ((len - 1) % (dec_limit + 1)));

  computeGold<T>(gpu_data, len);
  GATE_CHECKSUM_BYTES(gate_gpu_data_name<T>(), gpu_data, sizeof(gpu_data));

  auto start = std::chrono::steady_clock::now();
  launch_gpu_stage(data, gpu_data, len, repeat, false, use_gpu);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = std::atoi(argv[1]);
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  return 0;
}
