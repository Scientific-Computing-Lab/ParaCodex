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

void benchmark_func(float *__restrict__ cd, int grid_dim, int block_dim, int compute_iterations, long vector_size) {
  const long stride = static_cast<long>(block_dim);
  const long big_stride = static_cast<long>(grid_dim) * block_dim * granularity;
  const int fused = grid_dim * fusion_degree;
  const int iterations = compute_iterations;
  const float seed_local = seed;

  // Offload synthetic compute work; flatten the outer loops so teams map cleanly to the fused launch space.
#pragma omp target teams distribute parallel for collapse(2) \
    map(present: cd[0:vector_size]) \
    num_teams(fused) thread_limit(block_dim) \
    firstprivate(iterations, stride, big_stride, seed_local)
  for (int fused_team = 0; fused_team < fused; ++fused_team) {
    for (int thread = 0; thread < block_dim; ++thread) {
      const int team = fused_team / fusion_degree;
      const int k = fused_team % fusion_degree;
      const long idx = static_cast<long>(team) * block_dim * granularity + thread;
      const long base_offset = idx + static_cast<long>(k) * big_stride;
      float *cd_ptr = cd + base_offset;

      float sum = 0.0f;

#pragma unroll
      for (int j = 0; j < granularity; j += 2) {
        const long offset0 = static_cast<long>(j) * stride;
        const long offset1 = offset0 + stride;

        float v0 = cd_ptr[offset0];
        float v1 = cd_ptr[offset1];

        for (int iter = 0; iter < iterations; ++iter) {
          v0 = v0 * v0 + seed_local;
          v1 = v1 * v1 + seed_local;
        }

        sum += v0 * v1;
      }

      cd_ptr[0] = sum;
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
    // Keep cd resident on the GPU across kernels; use present mapping and explicit launch geometry to avoid implicit copies on the Ada GPU (SM 8.9).
#pragma omp target teams distribute parallel for \
    map(present: cd[0:size]) \
    num_teams((int)((size + block_dim - 1) / block_dim)) thread_limit(block_dim)
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
