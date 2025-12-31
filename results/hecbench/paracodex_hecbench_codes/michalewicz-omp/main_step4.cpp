#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include <climits>
#include "gate.h"

#define min(a,b) (a) < (b) ? (a) : (b)

#pragma omp declare target
static constexpr float kInvPi = 1.0f / static_cast<float>(M_PI);

// Replace powf(x, 20) with explicit multiplications to reduce GPU math unit latency.
inline float pow20f(const float x) {
  const float x2 = x * x;          // x^2
  const float x4 = x2 * x2;        // x^4
  const float x8 = x4 * x4;        // x^8
  const float x16 = x8 * x8;       // x^16
  return x16 * x4;                 // x^20
}

template <int Dim>
inline float michalewicz_fixed_dim(const float *__restrict__ xValues) {
  float result = 0.0f;
#pragma unroll
  for (int i = 0; i < Dim; ++i) {
    const float xi = xValues[i];
    const float sin_xi = sinf(xi);
    const float xi_sq = xi * xi;
    const float angle = static_cast<float>(i + 1) * xi_sq * kInvPi;
    const float sin_term = sinf(angle);
    result += sin_xi * pow20f(sin_term);
  }
  return -result;
}

inline float michalewicz_fallback(const float *__restrict__ xValues, const int dim) {
  float result = 0.0f;
  for (int i = 0; i < dim; ++i) {
    const float xi = xValues[i];
    const float sin_xi = sinf(xi);
    const float xi_sq = xi * xi;
    const float angle = static_cast<float>(i + 1) * xi_sq * kInvPi;
    const float sin_term = sinf(angle);
    result += sin_xi * pow20f(sin_term);
  }
  return -result;
}

inline float michalewicz(const float *__restrict__ xValues, const int dim) {
  switch (dim) {
    case 2:
      return michalewicz_fixed_dim<2>(xValues);
    case 5:
      return michalewicz_fixed_dim<5>(xValues);
    case 10:
      return michalewicz_fixed_dim<10>(xValues);
    default:
      // Fallback to a generic path to preserve semantics for unexpected dims.
      return michalewicz_fallback(xValues, dim);
  }
}
#pragma omp end declare target



void Error(float value, int dim) {
  printf("Global minima = %f\n", value);
  float trueMin = 0.0;
  if (dim == 2)
    trueMin = -1.8013;
  else if (dim == 5)
    trueMin = -4.687658;
  else if (dim == 10)
    trueMin = -9.66015;
  printf("Error = %f\n", fabsf(trueMin - value));
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of vectors> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  const bool use_gpu = omp_get_num_devices() > 0;

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.0, 4.0);



  const int dims[] = {2, 5, 10};

  for (int d = 0; d < 3; d++) {

    const int dim = dims[d];

    const size_t size = n * dim;

    const size_t size_bytes = size * sizeof(float);

    float *values = (float*) malloc (size_bytes);

    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }

    float minValue = 0.0f;

    {
      auto start = std::chrono::steady_clock::now();
      if (use_gpu) {
        // Configure team geometry to better match the RTX 4060 Laptop GPU (SM 8.9).
        const int team_size = 256;
        const size_t total_work = static_cast<size_t>(repeat) * n;
        size_t raw_num_teams = (total_work + static_cast<size_t>(team_size) - 1) / static_cast<size_t>(team_size);
        if (raw_num_teams == 0) {
          raw_num_teams = 1;
        }
        const int configured_num_teams =
            raw_num_teams > static_cast<size_t>(INT_MAX) ? INT_MAX : static_cast<int>(raw_num_teams);
#pragma omp target data map(to: values[0:size]) map(tofrom: minValue)
        {
          // Keep values resident on the device while executing all GPU work.
#pragma omp target teams distribute parallel for collapse(2) firstprivate(dim) reduction(min:minValue) \
    num_teams(configured_num_teams) thread_limit(team_size) num_threads(team_size)
          for (int i = 0; i < repeat; i++) {
            for (size_t j = 0; j < n; j++) {
              const float current = michalewicz(values + j * dim, dim);
              minValue = min(minValue, current);
            }
          }
        }
      } else {
        for (int i = 0; i < repeat; i++) {
          for (size_t j = 0; j < n; j++) {
            const float current = michalewicz(values + j * dim, dim);
            minValue = min(minValue, current);
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of kernel (dim = %d): %f (us)\n",
             dim, (time * 1e-3f) / repeat);
    }
    Error(minValue, dim);
    char gate_name[32];
    snprintf(gate_name, sizeof(gate_name), "minValue_dim%d", dim);
    GATE_STATS_F32(gate_name, &minValue, 1);
    free(values);
  }

  return 0;
}
