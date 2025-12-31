#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "gate.h"

constexpr long VECTOR_SIZE = 8 * 1024 * 1024;
constexpr int GRANULARITY = 8;
constexpr int FUSION_DEGREE = 4;
constexpr float SEED = 0.1f;

// Reproduces the original serial semantics when no accelerator is available.
static void benchmark_func_host_reference(float *cd,
                                          std::size_t total_elems,
                                          int block_dim,
                                          int compute_iterations) {
  const unsigned int blockSize = static_cast<unsigned int>(block_dim);
  const int stride = block_dim;
  const long team_num = omp_get_team_num();
  const long thread_num = omp_get_thread_num();
  const long idx = team_num * static_cast<long>(blockSize) * GRANULARITY +
                   thread_num;
  const long team_count = omp_get_num_teams();
  const long big_stride = (team_count > 0 ? team_count : 1) *
                          static_cast<long>(blockSize) * GRANULARITY;
  float tmps[GRANULARITY];

  for (int k = 0; k < FUSION_DEGREE; ++k) {
    for (int j = 0; j < GRANULARITY; ++j) {
      const long load_idx = idx + j * stride + k * big_stride;
      float val = 0.0f;
      if (load_idx >= 0 &&
          static_cast<std::size_t>(load_idx) < total_elems) {
        val = cd[load_idx];
        for (int iter = 0; iter < compute_iterations; ++iter) {
          val = val * val + SEED;
        }
      }
      tmps[j] = val;
    }

    float sum = 0.0f;
    for (int j = 0; j < GRANULARITY; j += 2) {
      sum += tmps[j] * tmps[j + 1];
    }

    const long store_idx = idx + k * big_stride;
    if (store_idx >= 0 &&
        static_cast<std::size_t>(store_idx) < total_elems) {
      cd[store_idx] = sum;
    }
  }
}

// Launches a block-strided compute kernel across the mapped device buffer.
static void benchmark_func(float *cd,
                           std::size_t total_elems,
                           int grid_dim,
                           int block_dim,
                           int compute_iterations,
                           bool use_device) {
  if (grid_dim <= 0 || block_dim <= 0 || total_elems == 0) {
    return;
  }

  if (!use_device) {
    benchmark_func_host_reference(cd, total_elems, block_dim,
                                  compute_iterations);
    return;
  }

  const std::size_t stride = static_cast<std::size_t>(block_dim);
  const std::size_t team_span = stride * static_cast<std::size_t>(GRANULARITY);

#pragma omp target teams num_teams(grid_dim) thread_limit(block_dim)               \
    map(present : cd [0:total_elems])
  {
    const std::size_t active_teams = static_cast<std::size_t>(omp_get_num_teams());
    const std::size_t big_stride = active_teams * team_span;

#pragma omp parallel
    {
      const std::size_t team_id = static_cast<std::size_t>(omp_get_team_num());
      const std::size_t thread_id =
          static_cast<std::size_t>(omp_get_thread_num());
      const std::size_t threads_in_team =
          static_cast<std::size_t>(omp_get_num_threads());

      for (std::size_t lane = thread_id; lane < stride; lane += threads_in_team) {
        const std::size_t base_idx = team_id * team_span + lane;
        if (base_idx >= total_elems) {
          continue;
        }

        float tmps[GRANULARITY];

        for (int k = 0; k < FUSION_DEGREE; ++k) {
          const std::size_t block_offset =
              base_idx + static_cast<std::size_t>(k) * big_stride;
          if (block_offset >= total_elems) {
            break;
          }

#pragma unroll
          for (int j = 0; j < GRANULARITY; ++j) {
            const std::size_t load_idx =
                block_offset + static_cast<std::size_t>(j) * stride;
            float val = 0.0f;
            if (load_idx < total_elems) {
              float tmp = cd[load_idx];
              for (int iter = 0; iter < compute_iterations; ++iter) {
                tmp = tmp * tmp + SEED;
              }
              val = tmp;
            }
            tmps[j] = val;
          }

          float sum = 0.0f;
#pragma unroll
          for (int j = 0; j < GRANULARITY; j += 2) {
            sum += tmps[j] * tmps[j + 1];
          }

          cd[block_offset] = sum;
        }
      }
    }
  }
}

static void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  std::printf("Trade-off type:%s\n", benchtype);

  float *cd = static_cast<float *>(std::malloc(size * sizeof(float)));
  if (!cd) {
    std::fprintf(stderr, "Failed to allocate host buffer\n");
    std::exit(EXIT_FAILURE);
  }

  for (long i = 0; i < size; ++i) {
    cd[i] = 0.0f;
  }

  const long reduced_grid_size = size / GRANULARITY / 128;
  const int block_dim = 256;
  const int grid_dim =
      static_cast<int>(reduced_grid_size / block_dim); // matches original

  const std::size_t total_elems = static_cast<std::size_t>(size);
  const bool use_device = omp_get_num_devices() > 0;

#pragma omp target enter data if (use_device) map(alloc : cd [0:total_elems])
#pragma omp target update if (use_device) to(cd [0:total_elems])

  for (int i = 0; i < repeat; ++i) {
    benchmark_func(cd, total_elems, grid_dim, block_dim, i, use_device);
  }

  const auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; ++i) {
    benchmark_func(cd, total_elems, grid_dim, block_dim, i, use_device);
  }

  const auto end = std::chrono::steady_clock::now();
  const double time_sec =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() *
  1e-9;
  std::printf("Total kernel execution time: %f (s)\n", time_sec);

#pragma omp target update if (use_device) from(cd [0:total_elems])
#pragma omp target exit data if (use_device) map(delete : cd [0:total_elems])

  bool ok = true;
  for (long i = 0; i < size; ++i) {
    if (cd[i] != 0.0f) {
      if (std::fabs(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        std::printf("Verification failed at index %ld: %f\n", i, cd[i]);
        break;
      }
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  GATE_STATS_F32("cd", cd, size);

  std::free(cd);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = std::atoi(argv[1]);
  const unsigned int datasize = VECTOR_SIZE * sizeof(float);
  std::printf("Buffer size: %uMB\n", datasize / (1024u * 1024u));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}
