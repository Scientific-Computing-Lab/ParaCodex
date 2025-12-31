#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

void k0 (const float *__restrict a, float *__restrict o, const int n) {
    for (int t = 0; t < n; t++) {
    float x = a[t];
    o[t] = coshf(x)/sinhf(x) - 1.f/x;
  }
}

void k1 (const float *__restrict a, float *__restrict o, const int n) {
    for (int t = 0; t < n; t++) {
    float x = a[t];
    o[t] = 1.f / tanhf(x) - 1.f/x;
  }
}




void k2 (const float *__restrict a, float *__restrict o, const int n) {
    for (int t = 0; t < n; t++) {
    float x = a[t];
    float s, r;
    s = x * x;
    r =              7.70960469e-8f;
    r = fmaf (r, s, -1.65101926e-6f);
    r = fmaf (r, s,  2.03457112e-5f);
    r = fmaf (r, s, -2.10521728e-4f);
    r = fmaf (r, s,  2.11580913e-3f);
    r = fmaf (r, s, -2.22220998e-2f);
    r = fmaf (r, s,  8.33333284e-2f);
    r = fmaf (r, x,  0.25f * x);
    o[t] = r;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage %s <n> <repeat>\n", argv[0]);
    return 1;
  }

  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const size_t size = sizeof(float) * n;

  float *a, *o, *o0, *o1, *o2;

  a = (float*) malloc (size);
  o = (float*) malloc (size);
  

  for (int i = 0; i < n; i++) {
    a[i] = -1.8f + i * (1.79999f / n);
  }

  o0 = (float*) malloc (size);
  o1 = (float*) malloc (size);
  o2 = (float*) malloc (size);

    {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      k0(a, o0, n);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of k0: %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      k1(a, o1, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of k1: %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      k2(a, o2, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of k2: %f (s)\n", (time * 1e-9f) / repeat);
  }

  

  for (int i = 0; i < n; i++) {
    float x = a[i];
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;
    o[i] = x * (1.f/3.f - 1.f/45.f * x2 + 2.f/945.f * x4 - 1.f/4725.f * x6);
  }

  float e[3] = {0,0,0};

  for (int i = 0; i < n; i++) {
    e[0] += (o[i] - o0[i]) * (o[i] - o0[i]);
    e[1] += (o[i] - o1[i]) * (o[i] - o1[i]);
    e[2] += (o[i] - o2[i]) * (o[i] - o2[i]);
  }

  printf("\nError statistics for the kernels:\n");
  for (int i = 0; i < 3; i++) {
    printf("%f ", sqrt(e[i]));
  }
  printf("\n");

  free(a);
  free(o);
  free(o0);
  free(o1);
  free(o2);
  return 0;
}