#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <omp.h>
#include <random>
#include <vector>
#include "gate.h"

#pragma omp declare target
inline float michalewicz_eval(const float *xValues, int dim) {
  float result = 0.0f;
  for (int i = 0; i < dim; ++i) {
    const float xi = xValues[i];
    const float sin_x = sinf(xi);
    const float angle =
        (static_cast<float>(i + 1) * xi * xi) / static_cast<float>(M_PI);
    const float sin_term = sinf(angle);
    const float pow_term = powf(sin_term, 20.0f);
    result += sin_x * pow_term;
  }
  return -result;
}
#pragma omp end declare target

void Error(float value, int dim) {
  std::printf("Global minima = %f\n", value);
  float trueMin = 0.0f;
  if (dim == 2) {
    trueMin = -1.8013f;
  } else if (dim == 5) {
    trueMin = -4.687658f;
  } else if (dim == 10) {
    trueMin = -9.66015f;
  }
  std::printf("Error = %f\n", std::fabs(trueMin - value));
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::printf("Usage: %s <number of vectors> <repeat>\n", argv[0]);
    return 1;
  }

  const std::size_t n = std::strtoull(argv[1], nullptr, 10);
  const int repeat = std::atoi(argv[2]);
  if (n == 0 || repeat <= 0) {
    std::printf("Both <number of vectors> and <repeat> must be positive.\n");
    return 1;
  }

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.0f, 4.0f);

  const std::array<int, 3> dims = {2, 5, 10};

  for (const int dim : dims) {
    const std::size_t size = n * static_cast<std::size_t>(dim);

    std::vector<float> values(size);
    for (float &val : values) {
      val = dis(gen);
    }

    float minValue = 0.0f;
    float *values_data = values.data();

    auto start = std::chrono::steady_clock::now();

    const int teamSize = 256;
    const std::size_t totalWork = static_cast<std::size_t>(repeat) * n;
    int numTeams = static_cast<int>(
        (totalWork + static_cast<std::size_t>(teamSize) - 1) /
        static_cast<std::size_t>(teamSize));
    if (numTeams < 1) {
      numTeams = 1;
    }
    if (numTeams > 4096) {
      numTeams = 4096;
    }

    const bool useGpu = omp_get_num_devices() > 0;

    if (useGpu) {
#pragma omp target teams distribute parallel for collapse(2)               \
    map(to : values_data[0:size]) reduction(min : minValue)                \
    num_teams(numTeams) thread_limit(teamSize)
      for (int rep = 0; rep < repeat; ++rep) {
        for (std::size_t j = 0; j < n; ++j) {
          const float *vec = values_data + j * dim;
          minValue = fminf(minValue, michalewicz_eval(vec, dim));
        }
      }
    } else {
      for (int rep = 0; rep < repeat; ++rep) {
        for (std::size_t j = 0; j < n; ++j) {
          const float *vec = values_data + j * dim;
          const float val = michalewicz_eval(vec, dim);
          if (val < minValue) {
            minValue = val;
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    const auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::printf("Average execution time of kernel (dim = %d): %f (us)\n", dim,
                (time * 1e-3f) / repeat);

    Error(minValue, dim);
    if (dim == 2) {
      GATE_STATS_F32("minValue_dim2", &minValue, 1);
    } else if (dim == 5) {
      GATE_STATS_F32("minValue_dim5", &minValue, 1);
    } else if (dim == 10) {
      GATE_STATS_F32("minValue_dim10", &minValue, 1);
    }
  }

  return 0;
}
