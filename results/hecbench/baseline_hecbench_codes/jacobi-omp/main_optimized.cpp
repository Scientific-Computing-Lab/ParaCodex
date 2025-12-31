#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>

#include <omp.h>
#include "../../../gate_sdk/gate.h"

constexpr int N = 2048;
constexpr std::size_t GRID_POINTS =
    static_cast<std::size_t>(N) * static_cast<std::size_t>(N);

#define IDX(i, j) ((i) + (j) * N)

static void initialize_data(float *f) {
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N - 1) {
        f[IDX(i, j)] = sinf(j * 2 * M_PI / (N - 1));
      } else if (j == 0 || j == N - 1) {
        f[IDX(i, j)] = sinf(i * 2 * M_PI / (N - 1));
      } else {
        f[IDX(i, j)] = 0.0f;
      }
    }
  }
}

int main() {
  std::clock_t start_time = std::clock();

  float *f =
      static_cast<float *>(std::aligned_alloc(64, GRID_POINTS * sizeof(float)));
  float *f_old =
      static_cast<float *>(std::aligned_alloc(64, GRID_POINTS * sizeof(float)));

  if (!f || !f_old) {
    std::cerr << "Failed to allocate memory\n";
    std::free(f);
    std::free(f_old);
    return -1;
  }

  initialize_data(f);
  initialize_data(f_old);

  float error = std::numeric_limits<float>::max();
  constexpr float tolerance = 0.5e-3f;
  constexpr float inv_total_points = 1.0f / static_cast<float>(N * N);
  constexpr int max_iters = 10000;
  int num_iters = 0;
  const bool use_device = omp_get_num_devices() > 0;

  constexpr int threads_per_team = 256;
  const int interior_points = (N - 2) * (N - 2);
  const int team_count =
      (interior_points + threads_per_team - 1) / threads_per_team;

  auto loop_start = std::chrono::steady_clock::now();

#pragma omp target data if (use_device) map(tofrom : f[0:GRID_POINTS], f_old[0:GRID_POINTS])
  {
    while (error > tolerance && num_iters < max_iters) {
      float error_sum = 0.0f;

#pragma omp target teams distribute parallel for simd collapse(2)                    \
    if (use_device) map(present : f[0:GRID_POINTS], f_old[0:GRID_POINTS])            \
        reduction(+:error_sum)                                                       \
        thread_limit(threads_per_team) num_teams(team_count)
      for (int j = 1; j < N - 1; ++j) {
        for (int i = 1; i < N - 1; ++i) {
          const float west = f_old[IDX(i - 1, j)];
          const float east = f_old[IDX(i + 1, j)];
          const float south = f_old[IDX(i, j - 1)];
          const float north = f_old[IDX(i, j + 1)];
          const float new_val = 0.25f * (west + east + south + north);
          const float center = f_old[IDX(i, j)];
          const float diff = new_val - center;
          f[IDX(i, j)] = new_val;
          error_sum += diff * diff;
        }
      }

      error = sqrtf(error_sum * inv_total_points);

      if (num_iters % 1000 == 0) {
        std::cout << "Error after iteration " << num_iters << " = " << error
                  << std::endl;
      }

      if (use_device) {
#pragma omp target teams distribute parallel for simd collapse(2)                    \
    map(present : f[0:GRID_POINTS], f_old[0:GRID_POINTS])                            \
        thread_limit(threads_per_team) num_teams(team_count)
        for (int j = 1; j < N - 1; ++j) {
          for (int i = 1; i < N - 1; ++i) {
            f_old[IDX(i, j)] = f[IDX(i, j)];
          }
        }
      } else {
        for (int j = 1; j < N - 1; ++j) {
          for (int i = 1; i < N - 1; ++i) {
            f_old[IDX(i, j)] = f[IDX(i, j)];
          }
        }
      }

      ++num_iters;
    }
  }

  auto loop_end = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - loop_start)
          .count();

  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
    std::free(f);
    std::free(f_old);
    return -1;
  }

  std::cout << "Average execution time per iteration: "
            << (elapsed * 1e-9f) / static_cast<float>(num_iters) << " (s)\n";

  GATE_STATS_F32("f", f, N * N);
  GATE_STATS_F32("f_old", f_old, N * N);

  std::free(f);
  std::free(f_old);

  const double duration =
      (std::clock() - start_time) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration
            << " seconds" << std::endl;

  return 0;
}
