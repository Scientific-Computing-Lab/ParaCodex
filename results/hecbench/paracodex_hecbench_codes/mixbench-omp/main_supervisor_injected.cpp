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

void benchmark_func(float *__restrict__ cd,
                    int grid_dim,
                    int block_dim,
                    int compute_iterations,
                    long vector_size,
                    bool use_gpu) {
  (void)grid_dim;
  const long stride = static_cast<long>(block_dim);
  const long big_stride = stride * granularity;
  const int iterations = compute_iterations;
  const float seed_local = seed;

#pragma omp target teams distribute parallel for if(use_gpu) \
    map(present: cd[0:vector_size]) \
    num_teams(fusion_degree) thread_limit(1) \
    firstprivate(iterations, stride, big_stride, seed_local)
  for (int k = 0; k < fusion_degree; ++k) {
    float tmps[granularity];
    for (int j = 0; j < granularity; ++j) {
      const long offset = static_cast<long>(k) * big_stride + static_cast<long>(j) * stride;
      float value = cd[offset];
      for (int iter = 0; iter < iterations; ++iter) {
        value = value * value + seed_local;
      }
      tmps[j] = value;
    }

    float sum = 0.0f;
    for (int j = 0; j < granularity; j += 2) {
      sum += tmps[j] * tmps[j + 1];
    }

    cd[static_cast<long>(k) * big_stride] = sum;
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc (size*sizeof(float));

  for (long i = 0; i < size; i++) {
    cd[i] = 0.0f;
  }

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;
  const bool use_gpu = omp_get_num_devices() > 0;

#pragma omp target data if(use_gpu) map(tofrom: cd[0:size])
  {
    if (use_gpu) {
#pragma omp target teams distribute parallel for \
    map(present: cd[0:size]) \
    num_teams((int)((size + block_dim - 1) / block_dim)) thread_limit(block_dim)
      for (long i = 0; i < size; i++) {
        cd[i] = 0.0f;
      }
    }

    for (int i = 0; i < repeat; i++) {
      benchmark_func(cd, grid_dim, block_dim, i, size, use_gpu);
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      benchmark_func(cd, grid_dim, block_dim, i, size, use_gpu);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

    if (use_gpu) {
#pragma omp target update from(cd[0:size])
    }
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
