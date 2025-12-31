#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <utility>
#include <omp.h>

#include "../../../gate_sdk/gate.h"

#define N 2048

#define IDX(i, j) ((i) + (j) * N)

void initialize_data(float* f) {
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

  float* f = (float*)aligned_alloc(64, N * N * sizeof(float));
  float* f_old = (float*)aligned_alloc(64, N * N * sizeof(float));

  float error = {std::numeric_limits<float>::max()};
  const float tolerance = 0.5e-3f;

  initialize_data(f);
  initialize_data(f_old);

  const int max_iters = 10000;
  int num_iters = 0;
  bool write_to_f = true;  // toggle Jacobi ping-pong to avoid device-wide copies

  // RTX 4060 Laptop GPU (Ada, see system_info.txt) exposes 24 SMs / 3072 cores.
  // Keeping ~8 warps per team (256 threads) balances occupancy and register use.
  const int threads_per_team = 256;
  const int total_points = N * N;
  const int interior_points = (N - 2) * (N - 2);
  const int boundary_teams =
      (total_points + threads_per_team - 1) / threads_per_team;
  const int interior_teams =
      (interior_points + threads_per_team - 1) / threads_per_team;

  {
    auto start = std::chrono::steady_clock::now();

    // Keep the Jacobi state resident on the RTX 4060 (Ada, 8 GB) per system_info.txt.
    // Copy the boundary-conditioned initial state once (f_old) and only bring the converged
    // state (f) back to the host at the end to trim host-device transfers.
    #pragma omp target data map(from : f[0 : N * N]) \
                             map(tofrom : f_old[0 : N * N])
    {
      #pragma omp target teams distribute parallel for collapse(2) \
          num_teams(boundary_teams) thread_limit(threads_per_team)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
          f[IDX(i, j)] = f_old[IDX(i, j)];

      while (error > tolerance && num_iters < max_iters) {
        error = 0.f;

        if (write_to_f) {
          #pragma omp target teams distribute parallel for collapse(2) \
              num_teams(interior_teams) thread_limit(threads_per_team) \
              reduction(+ : error)
          for (int j = 1; j <= N - 2; ++j) {
            for (int i = 1; i <= N - 2; ++i) {
              const int idx = IDX(i, j);
              const float t =
                  0.25f * (f_old[IDX(i - 1, j)] + f_old[IDX(i + 1, j)] +
                           f_old[IDX(i, j - 1)] + f_old[IDX(i, j + 1)]);
              const float df = t - f_old[idx];
              f[idx] = t;
              error += df * df;
            }
          }
        } else {
          #pragma omp target teams distribute parallel for collapse(2) \
              num_teams(interior_teams) thread_limit(threads_per_team) \
              reduction(+ : error)
          for (int j = 1; j <= N - 2; ++j) {
            for (int i = 1; i <= N - 2; ++i) {
              const int idx = IDX(i, j);
              const float t =
                  0.25f * (f[IDX(i - 1, j)] + f[IDX(i + 1, j)] +
                           f[IDX(i, j - 1)] + f[IDX(i, j + 1)]);
              const float df = t - f[idx];
              f_old[idx] = t;
              error += df * df;
            }
          }
        }

        error = sqrtf(error / (N * N));
        write_to_f = !write_to_f;

        if (num_iters % 1000 == 0) {
          std::cout << "Error after iteration " << num_iters << " = " << error
                    << std::endl;
        }

        ++num_iters;
      }
    }

    if (write_to_f) {
      std::swap(f, f_old);
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << "Average execution time per iteration: "
              << (time * 1e-9f) / num_iters << " (s)\n";
  }

  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
    return -1;
  }

  GATE_CHECKSUM_U32("jacobi_f_checksum",
                    reinterpret_cast<const uint32_t*>(f), N * N);
  GATE_STATS_F32("jacobi_f_stats", f, N * N);
  GATE_CHECKSUM_U32("jacobi_f_old_checksum",
                    reinterpret_cast<const uint32_t*>(f_old), N * N);
  GATE_STATS_F32("jacobi_f_old_stats", f_old, N * N);

  free(f);
  free(f_old);

  double duration =
      (std::clock() - start_time) / (double)CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration
            << " seconds" << std::endl;

  return 0;
}
