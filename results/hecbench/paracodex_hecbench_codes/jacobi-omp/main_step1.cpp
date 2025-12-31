#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <omp.h>

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
  const float tolerance = 1.e-5f;

  initialize_data(f);
  initialize_data(f_old);

  const int max_iters = 10000;
  int num_iters = 0;

  {
    auto start = std::chrono::steady_clock::now();

    // Keep the Jacobi state resident on the RTX 4060 (Ada, 8 GB) per system_info.txt.
    #pragma omp target data map(tofrom : f[0 : N * N], f_old[0 : N * N])
    {
      while (error > tolerance && num_iters < max_iters) {
        error = 0.f;

        #pragma omp target teams distribute parallel for collapse(2) map(present : f[0 : N * N], f_old[0 : N * N]) reduction(+ : error)
        for (int i = 1; i <= N - 2; i++) {
          for (int j = 1; j <= N - 2; j++) {
            float t = 0.25f * (f_old[IDX(i - 1, j)] + f_old[IDX(i + 1, j)] +
                               f_old[IDX(i, j - 1)] + f_old[IDX(i, j + 1)]);
            float df = t - f_old[IDX(i, j)];
            f[IDX(i, j)] = t;
            error += df * df;
          }
        }

        #pragma omp target teams distribute parallel for collapse(2) map(present : f[0 : N * N], f_old[0 : N * N])
        for (int j = 0; j < N; j++)
          for (int i = 0; i < N; i++)
            if (j >= 1 && j <= N - 2 && i >= 1 && i <= N - 2)
              f_old[IDX(i, j)] = f[IDX(i, j)];

        error = sqrtf(error / (N * N));

        if (num_iters % 1000 == 0) {
          std::cout << "Error after iteration " << num_iters << " = " << error
                    << std::endl;
        }

        ++num_iters;
      }
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

  free(f);
  free(f_old);

  double duration =
      (std::clock() - start_time) / (double)CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration
            << " seconds" << std::endl;

  return 0;
}
