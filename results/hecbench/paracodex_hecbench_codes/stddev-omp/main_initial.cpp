#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <climits>
#include <omp.h>
#include "reference.h"
#include "gate.h"

void stddev(float *std, const float *data, int D, int N, bool sample,
            bool use_gpu) {
  // Simple serial implementation that matches the reference algorithm
  int sample_size = sample ? N - 1 : N;
  const int threads_per_team = 256;  // Good balance for Ada SM width (8 warps)
  const int sm_count = 46;        // RTX 4070 has 46 SMs (system_info.txt)
  const int waves_per_sm = 4;     // Keep multiple resident teams per SM
  const int tile_cols = 64;       // Tile columns to limit reduction footprint
  long long total_elems = static_cast<long long>(D) * N;
  int requested_teams = sm_count * waves_per_sm;
  long long min_teams_for_work_ll =
      (total_elems + threads_per_team - 1) / threads_per_team;
  if (min_teams_for_work_ll > INT_MAX)
    min_teams_for_work_ll = INT_MAX;
  int min_teams_for_work = static_cast<int>(min_teams_for_work_ll);
  if (min_teams_for_work > requested_teams)
    requested_teams = min_teams_for_work;
  if (requested_teams < 1)
    requested_teams = 1;
  int tile_count = (D + tile_cols - 1) / tile_cols;
  int launch_teams = requested_teams;
  if (tile_count > 0 && launch_teams > tile_count)
    launch_teams = tile_count;
  if (launch_teams < 1)
    launch_teams = 1;

  // Tile the column dimension so each team works on a contiguous chunk.
  // This keeps memory accesses coalesced while shrinking each thread's
  // reduction workspace to the tile size.
#pragma omp target teams distribute \
    map(to : data[0:N * D], sample_size, N, D) map(tofrom : std[0:D]) \
        num_teams(launch_teams) thread_limit(threads_per_team) if (use_gpu)
  for (int tile = 0; tile < D; tile += tile_cols) {
    const int cols = (tile + tile_cols <= D) ? tile_cols : (D - tile);
    float tile_sums[tile_cols];
#pragma omp parallel
    {
#pragma omp for
      for (int c = 0; c < tile_cols; ++c)
        tile_sums[c] = 0.0f;

#pragma omp for reduction(+ : tile_sums[:tile_cols])
      for (int r = 0; r < N; ++r) {
        const float *row = data + r * D + tile;
#pragma omp simd
        for (int c = 0; c < cols; ++c)
          tile_sums[c] += row[c] * row[c];
      }

#pragma omp for
      for (int c = 0; c < cols; ++c)
        std[tile + c] = sqrtf(tile_sums[c] / sample_size);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    printf("D: number of columns of data (must be a multiple of 32)\n");
    printf("N: number of rows of data (at least one row)\n");
    return 1;
  }
  int D = atoi(argv[1]);

  int N = atoi(argv[2]);

  int repeat = atoi(argv[3]);

  bool sample = true;
  long inputSize = D * N;
  long inputSizeByte = inputSize * sizeof(float);
  float *data = (float *)malloc(inputSizeByte);

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++)
      data[i * D + j] = rand() / (float)RAND_MAX;

  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std = (float *)malloc(outputSizeByte);
  float *std_ref = (float *)malloc(outputSizeByte);

  bool use_gpu = omp_get_num_devices() > 0;

#pragma omp target data map(to : data[0:N * D]) map(tofrom : std[0:D]) if (use_gpu)
  {
    // Keep the working set resident on the GPU across repeated kernel launches.
    stddev(std, data, D, N, sample, use_gpu);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      stddev(std, data, D, N, sample, use_gpu);

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count();
    printf("Average execution time of stddev kernels: %f (s)\n",
           (time * 1e-9f) / repeat);
  }

  GATE_STATS_F32("stddev_out", std, D);

  stddev_ref(std_ref, data, D, N, sample);

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std_ref[i] - std[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  free(std_ref);
  free(std);
  free(data);
  return 0;
}
