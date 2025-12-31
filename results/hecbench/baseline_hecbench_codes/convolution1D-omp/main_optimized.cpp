#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <omp.h>
#include "gate.h"

#define MAX_MASK_WIDTH 10
#define BLOCK_SIZE 256
#define TILE_SIZE BLOCK_SIZE

template <typename T>
inline void gate_record_output(const char *name, const T *data, size_t n) {
  GATE_CHECKSUM_BYTES(name, data, n * sizeof(T));
}

inline void gate_record_output(const char *name, const double *data, size_t n) {
  GATE_STATS_F64(name, data, n);
}

inline void gate_record_output(const char *name, const float *data, size_t n) {
  GATE_STATS_F32(name, data, n);
}

template <typename T>
struct gate_type_tag {
  static const char *name() { return "UNKNOWN"; }
};

template <>
struct gate_type_tag<double> {
  static const char *name() { return "FP64"; }
};

template <>
struct gate_type_tag<float> {
  static const char *name() { return "FP32"; }
};

template <>
struct gate_type_tag<int16_t> {
  static const char *name() { return "INT16"; }
};

template <typename T>
struct compute_type_selector {
  using type = typename std::conditional<
      std::is_integral<T>::value && (sizeof(T) < sizeof(int)), int, T>::type;
};

template <typename T>
inline void conv1d_serial_host(const T *__restrict__ mask,
                               const T *__restrict__ in,
                               T *__restrict__ out,
                               const int input_width,
                               const int mask_width)
{
  using compute_t = typename compute_type_selector<T>::type;
  const int radius = mask_width / 2;
  for (int i = 0; i < input_width; ++i) {
    compute_t sum = compute_t(0);
    const int start = i - radius;
    for (int j = 0; j < mask_width; ++j) {
      const int idx = start + j;
      if (idx >= 0 && idx < input_width) {
        sum += static_cast<compute_t>(in[idx]) *
               static_cast<compute_t>(mask[j]);
      }
    }
    out[i] = static_cast<T>(sum);
  }
}

template <typename T>
void conv1d_gpu(const T *__restrict__ mask,
                const T *__restrict__ in,
                      T *__restrict__ out,
                const int input_width,
                const int mask_width,
                const int repeat,
                const bool use_gpu)
{
  if (input_width == 0 || mask_width <= 0 || repeat <= 0) {
    return;
  }

  using compute_t = typename compute_type_selector<T>::type;
  const int radius = mask_width / 2;

  if (!use_gpu) {
    for (int iter = 0; iter < repeat; ++iter) {
      conv1d_serial_host(mask, in, out, input_width, mask_width);
    }
    return;
  }

  #pragma omp target data if(use_gpu) map(to: mask[0:mask_width], in[0:input_width]) \
                          map(from: out[0:input_width])
  {
    for (int iter = 0; iter < repeat; ++iter) {
      #pragma omp target teams distribute parallel for if(use_gpu) \
          num_teams(((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE)) \
          thread_limit(BLOCK_SIZE) schedule(static) \
          map(present: mask[0:mask_width], in[0:input_width], out[0:input_width])
      for (int i = 0; i < input_width; ++i) {
        compute_t sum = compute_t(0);
        const int start = i - radius;
        for (int j = 0; j < mask_width; ++j) {
          const int idx = start + j;
          if (idx >= 0 && idx < input_width) {
            sum += static_cast<compute_t>(in[idx]) *
                   static_cast<compute_t>(mask[j]);
          }
        }
        out[i] = static_cast<T>(sum);
      }
    }
  }
}

template <typename T>
void conv1d_tiled_gpu(const T *__restrict__ mask,
                      const T *__restrict__ in,
                            T *__restrict__ out,
                      const int input_width,
                      const int mask_width,
                      const int repeat,
                      const bool use_gpu)
{
  if (input_width == 0 || mask_width <= 0 || repeat <= 0) {
    return;
  }

  using compute_t = typename compute_type_selector<T>::type;
  const int team_size = TILE_SIZE;
  const int radius = mask_width / 2;
  const int shared_span = team_size + mask_width - 1;

  if (!use_gpu) {
    for (int iter = 0; iter < repeat; ++iter) {
      conv1d_serial_host(mask, in, out, input_width, mask_width);
    }
    return;
  }

  #pragma omp target data if(use_gpu) map(to: mask[0:mask_width], in[0:input_width]) \
                          map(from: out[0:input_width])
  {
    for (int iter = 0; iter < repeat; ++iter) {
      #pragma omp target teams if(use_gpu) \
          num_teams(((input_width + team_size - 1) / team_size)) \
          thread_limit(team_size) \
          map(present: mask[0:mask_width], in[0:input_width], out[0:input_width])
      {
        T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
        const int team = omp_get_team_num();
        const int block_start = team * team_size;

        if (block_start < input_width) {
          #pragma omp parallel
          {
            const int lid = omp_get_thread_num();
            for (int offset = lid; offset < shared_span; offset += team_size) {
              const int global_idx = block_start + offset - radius;
              T val = T(0);
              if (global_idx >= 0 && global_idx < input_width) {
                val = in[global_idx];
              }
              tile[offset] = val;
            }
            #pragma omp barrier

            for (int t = lid;
                 t < team_size && (block_start + t) < input_width;
                 t += team_size) {
              compute_t sum = compute_t(0);
              for (int j = 0; j < mask_width; ++j) {
                sum += static_cast<compute_t>(tile[t + j]) *
                       static_cast<compute_t>(mask[j]);
              }
              out[block_start + t] = static_cast<T>(sum);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void conv1d_tiled_caching_gpu(const T *__restrict__ mask,
                              const T *__restrict__ in,
                                    T *__restrict__ out,
                              const int input_width,
                              const int mask_width,
                              const int repeat,
                              const bool use_gpu)
{
  if (input_width == 0 || mask_width <= 0 || repeat <= 0) {
    return;
  }

  using compute_t = typename compute_type_selector<T>::type;
  const int team_size = TILE_SIZE;
  const int radius = mask_width / 2;

  if (!use_gpu) {
    for (int iter = 0; iter < repeat; ++iter) {
      conv1d_serial_host(mask, in, out, input_width, mask_width);
    }
    return;
  }

  #pragma omp target data if(use_gpu) map(to: mask[0:mask_width], in[0:input_width]) \
                          map(from: out[0:input_width])
  {
    for (int iter = 0; iter < repeat; ++iter) {
      #pragma omp target teams if(use_gpu) \
          num_teams(((input_width + team_size - 1) / team_size)) \
          thread_limit(team_size) \
          map(present: mask[0:mask_width], in[0:input_width], out[0:input_width])
      {
        T tile[TILE_SIZE];
        T mask_tile[MAX_MASK_WIDTH];

        const int team = omp_get_team_num();
        const int block_start = team * team_size;
        const int next_tile_start = block_start + team_size;

        if (block_start < input_width) {
          #pragma omp parallel
          {
            const int lid = omp_get_thread_num();
            for (int j = lid; j < mask_width; j += team_size) {
              mask_tile[j] = mask[j];
            }

            const int global_idx = block_start + lid;
            T val = T(0);
            if (global_idx < input_width) {
              val = in[global_idx];
            }
            if (lid < team_size) {
              tile[lid] = val;
            }
            #pragma omp barrier

            for (int t = lid;
                 t < team_size && (block_start + t) < input_width;
                 t += team_size) {
              const int start = block_start + t - radius;
              compute_t sum = compute_t(0);
              for (int j = 0; j < mask_width; ++j) {
                const int in_index = start + j;
                if (in_index >= 0 && in_index < input_width) {
                  if (in_index >= block_start && in_index < next_tile_start) {
                    const int local_idx = in_index - block_start;
                    sum += static_cast<compute_t>(tile[local_idx]) *
                           static_cast<compute_t>(mask_tile[j]);
                  } else {
                    sum += static_cast<compute_t>(in[in_index]) *
                           static_cast<compute_t>(mask_tile[j]);
                  }
                }
              }
              out[block_start + t] = static_cast<T>(sum);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void reference(const T *h_in,
               const T *d_out,
               const T *mask,
               const int input_width,
               const int mask_width,
               const char *gate_label)
{
  bool ok = true;
  for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += h_in[start + j] * mask[j];
      }
    }
    if (fabs(static_cast<double>(s) - static_cast<double>(d_out[i])) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  gate_record_output(gate_label, d_out, static_cast<size_t>(input_width));
}

template <typename KernelFunc, typename T>
void benchmark_kernel(const char *label,
                      const char *gate_label_base,
                      KernelFunc kernel,
                      const T *mask,
                      const T *input,
                      T *output,
                      const int input_width,
                      const int mask_width,
                      const int repeat)
{
  std::fill(output, output + input_width, T(0));
  auto start = std::chrono::steady_clock::now();
  const bool use_gpu = omp_get_num_devices() > 0;
  kernel(mask, input, output, input_width, mask_width, repeat, use_gpu);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count();
  printf("Average kernel execution time of %s: %f (us)\n",
         label,
         (time * 1e-3f) / repeat);
  char gate_label[128];
  snprintf(gate_label,
           sizeof(gate_label),
           "%s:mask%d::%s",
           gate_label_base,
           mask_width,
           gate_type_tag<T>::name());
  reference(input, output, mask, input_width, mask_width, gate_label);
}

template <typename T>
void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = static_cast<size_t>(input_width) * sizeof(T);

  T *a = static_cast<T *>(malloc(size_bytes));
  T *b = static_cast<T *>(malloc(size_bytes));

  if (!a || !b) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a);
    free(b);
    return;
  }

  T mask[MAX_MASK_WIDTH];
  for (int i = 0; i < MAX_MASK_WIDTH; i++) {
    mask[i] = static_cast<T>(1);
  }

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = static_cast<T>(rand() % 256);
  }

  benchmark_kernel("conv1d kernel",
                   "conv1d_kernel",
                   conv1d_gpu<T>,
                   mask,
                   a,
                   b,
                   input_width,
                   mask_width,
                   repeat);

  benchmark_kernel("conv1d-tiled kernel",
                   "conv1d-tiled_kernel",
                   conv1d_tiled_gpu<T>,
                   mask,
                   a,
                   b,
                   input_width,
                   mask_width,
                   repeat);

  benchmark_kernel("conv1d-tiled-caching kernel",
                   "conv1d-tiled-caching_kernel",
                   conv1d_tiled_caching_gpu<T>,
                   mask,
                   a,
                   b,
                   input_width,
                   mask_width,
                   repeat);

  free(a);
  free(b);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_width = atoi(argv[1]);
  input_width = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  const int repeat = atoi(argv[2]);

  for (int mask_width = 3; mask_width < MAX_MASK_WIDTH; mask_width += 2) {
    printf("\n---------------------\n");
    printf("Mask width: %d\n", mask_width);

    printf("1D convolution (FP64)\n");
    conv1D<double>(input_width, mask_width, repeat);

    printf("1D convolution (FP32)\n");
    conv1D<float>(input_width, mask_width, repeat);

    printf("1D convolution (INT16)\n");
    conv1D<int16_t>(input_width, mask_width, repeat);
  }

  return 0;
}
