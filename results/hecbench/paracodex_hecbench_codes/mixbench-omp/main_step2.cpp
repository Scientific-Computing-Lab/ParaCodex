#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "gate.h"

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

void benchmark_func(float *cd, int grid_dim, int block_dim, int compute_iterations, long vector_size) { 
  const int stride = block_dim;
  const int big_stride = grid_dim * block_dim * granularity;
  // Offload the synthetic compute kernel; reuse the persistent device allocation to avoid remapping overhead.
#pragma omp target teams distribute parallel for collapse(2) \
    map(present: cd[0:vector_size]) \
    firstprivate(compute_iterations, stride, big_stride)
  for (int team = 0; team < grid_dim; team++) {
    for (int thread = 0; thread < block_dim; thread++) {
      const int idx = team * block_dim * granularity + thread;
      float tmps[granularity];
      for (int k = 0; k < fusion_degree; k++) {
        for (int j = 0; j < granularity; j++) {
          const long base_index = idx + j * stride + k * big_stride;
          tmps[j] = cd[base_index];
          for (int iter = 0; iter < compute_iterations; iter++) {
            tmps[j] = tmps[j] * tmps[j] + (float)seed;
          }
        }
        float sum = 0.0f;
        for (int j = 0; j < granularity; j += 2) {
          sum += tmps[j] * tmps[j + 1];
        }
        for (int j = 0; j < granularity; j++) {
          cd[idx + k * big_stride] = sum;
        }
      }
    }
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc (size*sizeof(float));

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

#pragma omp target data map(alloc: cd[0:size])
  {
    // Keep cd resident on the GPU across kernels; use present mapping to avoid implicit copies on the Ada GPU (SM 8.9).
#pragma omp target teams distribute parallel for map(present: cd[0:size])
    for (long i = 0; i < size; i++) {
      cd[i] = 0.0f;
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_func(cd, grid_dim, block_dim, i, size);
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      benchmark_func(cd, grid_dim, block_dim, i, size);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

#pragma omp target update from(cd[0:size])
  }

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd[i] != 0) {
      if (fabsf(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd[i]);
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  GATE_STATS_F32("cd", cd, size);

  free(cd);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE*sizeof(float);

  printf("Buffer size: %dMB\n", datasize/(1024*1024));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}
