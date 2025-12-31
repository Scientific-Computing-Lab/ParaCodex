#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#include "gate.h"

template <typename T>
struct GateRecorder;

template <>
struct GateRecorder<double> {
  static void record_conv1d(const double* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_fp64_out_mask%d", mask_width);
    GATE_STATS_F64(name, data, n);
  }
  static void record_conv1d_tiled(const double* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_tiled_fp64_out_mask%d", mask_width);
    GATE_STATS_F64(name, data, n);
  }
  static void record_conv1d_tiled_caching(const double* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_tiled_caching_fp64_out_mask%d", mask_width);
    GATE_STATS_F64(name, data, n);
  }
};

template <>
struct GateRecorder<float> {
  static void record_conv1d(const float* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_fp32_out_mask%d", mask_width);
    GATE_STATS_F32(name, data, n);
  }
  static void record_conv1d_tiled(const float* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_tiled_fp32_out_mask%d", mask_width);
    GATE_STATS_F32(name, data, n);
  }
  static void record_conv1d_tiled_caching(const float* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_tiled_caching_fp32_out_mask%d", mask_width);
    GATE_STATS_F32(name, data, n);
  }
};

template <>
struct GateRecorder<int16_t> {
  static void record_conv1d(const int16_t* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_int16_out_mask%d", mask_width);
    GATE_CHECKSUM_BYTES(name, data, n * sizeof(int16_t));
  }
  static void record_conv1d_tiled(const int16_t* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_tiled_int16_out_mask%d", mask_width);
    GATE_CHECKSUM_BYTES(name, data, n * sizeof(int16_t));
  }
  static void record_conv1d_tiled_caching(const int16_t* data, size_t n, int mask_width) {
    char name[64];
    snprintf(name, sizeof(name), "conv1d_tiled_caching_int16_out_mask%d", mask_width);
    GATE_CHECKSUM_BYTES(name, data, n * sizeof(int16_t));
  }
};

#define MAX_MASK_WIDTH 10
#define BLOCK_SIZE 256
#define TILE_SIZE BLOCK_SIZE

template<typename T>
void conv1d(const T * __restrict__ mask,
            const T * __restrict__ in,
                  T * __restrict__ out,
            const int input_width,
            const int mask_width)
{
    for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += in[start + j] * mask[j];
      }
    }
    out[i] = s;
  }
}

template<typename T>
void conv1d_tiled(const T *__restrict__ mask,
                  const T *__restrict__ in,
                        T *__restrict__ out,
                  const int input_width,
                  const int mask_width)
{
  // Serial tiling: fill full halo regions explicitly, then compute the stencil
  T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
  const int radius = mask_width / 2;

  for (int blockStart = 0; blockStart < input_width; blockStart += TILE_SIZE) {
    // Left halo
    for (int t = 0; t < radius; t++) {
      const int src = blockStart - radius + t;
      tile[t] = (src >= 0) ? in[src] : (T)0;
    }

    // Center tile
    for (int t = 0; t < TILE_SIZE; t++) {
      const int src = blockStart + t;
      tile[radius + t] = (src < input_width) ? in[src] : (T)0;
    }

    // Right halo
    for (int t = 0; t < radius; t++) {
      const int src = blockStart + TILE_SIZE + t;
      tile[radius + TILE_SIZE + t] = (src < input_width) ? in[src] : (T)0;
    }

    // Convolution for this block
    for (int t = 0; t < TILE_SIZE && (blockStart + t) < input_width; t++) {
      T s = 0;
      for (int j = 0; j < mask_width; j++) {
        s += tile[t + j] * mask[j];
      }
      out[blockStart + t] = s;
    }
  }
}

template<typename T>
void conv1d_tiled_caching(const T *__restrict__ mask,
                          const T *__restrict__ in,
                                T *__restrict__ out,
                          const int input_width,
                          const int mask_width)
{
    {
    T tile[TILE_SIZE];
        {
      int bid = omp_get_team_num();
      int lid = omp_get_thread_num();
      int dim = omp_get_num_threads();
      int i = bid * dim + lid;
      tile[lid] = in[i];
      
      int this_tile_start = bid * dim;
      int next_tile_start = (bid + 1) * dim;
      int start = i - (mask_width / 2);
      T s = 0;
      for (int j = 0; j < mask_width; j++) {
        int in_index = start + j;
        if (in_index >= 0 && in_index < input_width) {
          if (in_index >= this_tile_start && in_index < next_tile_start) {
            s += tile[lid + j - (mask_width / 2)] * mask[j];
          } else {
            s += in[in_index] * mask[j];
          }
        }
      }
      out[i] = s;
    }
  }
}

template <typename T>
void reference(const T *h_in,
               const T *d_out,
               const T *mask,
               const int input_width,
               const int mask_width)
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
    if (fabs(s - d_out[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

template <typename T>
void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); 

  b = (T *)malloc(size_bytes); 


  T mask[MAX_MASK_WIDTH];

  for (int i = 0; i < MAX_MASK_WIDTH; i++) mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

    {
    

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      conv1d(mask, a, b, input_width, mask_width);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
        reference(a, b, mask, input_width, mask_width);
        GateRecorder<T>::record_conv1d(b, input_width, mask_width);

    

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      conv1d_tiled(mask, a, b, input_width, mask_width);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d-tiled kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
        reference(a, b, mask, input_width, mask_width);
        GateRecorder<T>::record_conv1d_tiled(b, input_width, mask_width);

    

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      conv1d_tiled_caching(mask, a, b, input_width, mask_width);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d-tiled-caching kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
        reference(a, b, mask, input_width, mask_width);
        GateRecorder<T>::record_conv1d_tiled_caching(b, input_width, mask_width);
  }

  free(a);
  free(b);
}

int main(int argc, char* argv[]) {
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
