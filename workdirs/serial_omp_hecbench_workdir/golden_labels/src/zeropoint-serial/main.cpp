#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <random>
#include "reference.h"

void zero_point (
    const float* x_min,
    const float* x_max,
    int32_t qmin,
    int32_t qmax,
    int size,
    bool preserve_sparsity,
    float* scale,
    int32_t* zero_point)
{
    for (int i = 0; i < size; i++) {
    float min_val = x_min[i];
    float max_val = x_max[i];

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      int symmetric_qmin = -((qmax - qmin) / 2 + 1);
      int symmetric_qmax = (qmax - qmin) / 2;
      double max_scale = fmax(
          fabs(min_val / symmetric_qmin), fabs(max_val / symmetric_qmax));
      min_val = max_scale * symmetric_qmin;
      max_val = max_scale * symmetric_qmax;
    }

    

    

    

    min_val = fminf(min_val, 0.f);
    max_val = fmaxf(max_val, 0.f);
    scale[i] = (static_cast<double>(max_val) - min_val) / (qmax - qmin);

    

    

    if (scale[i] == 0.0f || isinf(1.0f / scale[i])) {
      scale[i] = 0.1;
    }

    double zero_point_from_min = qmin - min_val / static_cast<double>(scale[i]);
    double zero_point_from_max = qmax - max_val / static_cast<double>(scale[i]);
    double zero_point_from_min_error = abs(qmin) + abs(min_val / static_cast<double>(scale[i]));
    double zero_point_from_max_error = abs(qmax) + abs(max_val / static_cast<double>(scale[i]));
    double initial_zero_point = zero_point_from_min_error < zero_point_from_max_error
                                ? zero_point_from_min
                                : zero_point_from_max;

    

    

    

    

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      initial_zero_point = static_cast<double>(qmin + qmax) / 2;
    }
    

    

    

    

    

    int32_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
      nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
      nudged_zero_point = qmax;
    } else {
      nudged_zero_point = nearbyint(initial_zero_point);
    }
    zero_point[i] = nudged_zero_point;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of min/max values> <repeat>\n", argv[0]);
    return 1;
  }
  const int size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int32_t qmin = -127;
  int32_t qmax =  127;
  bool preserve_sparsity = true;

  size_t size_bytes = sizeof(float) * size;

  float *scale = (float*) malloc (size_bytes);
  float *scale_ref = (float*) malloc (size_bytes);
  int32_t *zp = (int32_t*) malloc (size_bytes);
  int32_t *zp_ref = (int32_t*) malloc (size_bytes);
  float *min = (float*) malloc (size_bytes);
  float *max = (float*) malloc (size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-1.f, 1.f);
  for (int i = 0; i < size; i++) {
    min[i] = distr(g);
    max[i] = distr(g);
  }
  
  reference (min,
             max,
             qmin,
             qmax,
             size,
             preserve_sparsity,
             scale_ref,
             zp_ref);

    {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      zero_point(
          min,
          max,
          qmin,
          qmax,
          size,
          preserve_sparsity,
          scale,
          zp);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of zero-point kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
  }

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (zp[i] != zp_ref[i] || scale[i] - scale_ref[i] > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(zp);
  free(scale);
  free(zp_ref);
  free(scale_ref);
  free(min);
  free(max);

  return 0;
}