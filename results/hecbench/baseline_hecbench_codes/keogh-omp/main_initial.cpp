#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

namespace {
constexpr float kMinStd = 1.0e-8f;
constexpr int kTeamSize = 256;  // 8 warps keeps Ada SMs well-occupied
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return -1;
  }

  const int M = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  if (M <= 0 || N <= 0 || repeat <= 0) {
    std::fprintf(stderr, "All input arguments must be positive.\n");
    return -1;
  }

  const int length = N - M + 1;
  if (length <= 0) {
    std::fprintf(stderr, "Subject length must be at least query length.\n");
    return -1;
  }

  std::printf("Query length = %d\n", M);
  std::printf("Subject length = %d\n", N);

  std::vector<float> subject(N);
  std::vector<float> lower_bound(M);
  std::vector<float> upper_bound(M);
  std::vector<float> lb(length, 0.0f);
  std::vector<float> lb_h(length, 0.0f);
  std::vector<float> avgs(length);
  std::vector<float> stds(length);
  std::vector<float> inv_stds(length);

  std::srand(123);
  for (int i = 0; i < N; ++i) {
    subject[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < length; ++i) {
    avgs[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    const float raw_std =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    const float sigma = std::max(raw_std, kMinStd);
    stds[i] = sigma;
    inv_stds[i] = 1.0f / sigma;
  }
  for (int i = 0; i < M; ++i) {
    upper_bound[i] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    lower_bound[i] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }

  const int num_teams = std::max(1, (length + kTeamSize - 1) / kTeamSize);

  float *__restrict subject_ptr = subject.data();
  float *__restrict avgs_ptr = avgs.data();
  float *__restrict stds_ptr = stds.data();
  float *__restrict inv_stds_ptr = inv_stds.data();
  float *__restrict lower_ptr = lower_bound.data();
  float *__restrict upper_ptr = upper_bound.data();
  float *__restrict lb_ptr = lb.data();

  auto start = std::chrono::steady_clock::now();

#pragma omp target enter data map(to : subject_ptr [0:N],                     \
                                  avgs_ptr [0:length],                        \
                                  inv_stds_ptr [0:length],                    \
                                  lower_ptr [0:M], upper_ptr [0:M])
#pragma omp target enter data map(alloc : lb_ptr [0:length])

  for (int iter = 0; iter < repeat; ++iter) {
#pragma omp target teams distribute parallel for simd                          \
    map(present : subject_ptr [0:N], avgs_ptr [0:length],                      \
        inv_stds_ptr [0:length], lower_ptr [0:M], upper_ptr [0:M],             \
        lb_ptr [0:length])                                                     \
    num_teams(num_teams) thread_limit(kTeamSize) schedule(static)
    for (int idx = 0; idx < length; ++idx) {
      const float avg = avgs_ptr[idx];
      const float inv_std = inv_stds_ptr[idx];
      const float shift = -avg * inv_std;

      float residues = 0.0f;
#pragma omp simd reduction(+ : residues)
      for (int i = 0; i < M; ++i) {
        const float value = subject_ptr[idx + i] * inv_std + shift;
        const float lower = value - lower_ptr[i];
        const float upper = value - upper_ptr[i];
        const float upper_term = upper > 0.0f ? upper * upper : 0.0f;
        const float lower_term = lower < 0.0f ? lower * lower : 0.0f;
        residues += upper_term + lower_term;
      }
      lb_ptr[idx] = residues;
    }
  }

#pragma omp target update from(lb_ptr [0:length])

#pragma omp target exit data map(delete : subject_ptr [0:N],                  \
                                 avgs_ptr [0:length],                         \
                                 inv_stds_ptr [0:length],                     \
                                 lower_ptr [0:M], upper_ptr [0:M])
#pragma omp target exit data map(delete : lb_ptr [0:length])

  auto end = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n",
              (elapsed * 1e-9f) / repeat);

  reference(subject_ptr, avgs_ptr, stds_ptr, lb_h.data(), lower_ptr, upper_ptr,
            M, N);

  bool ok = true;
  for (int i = 0; i < length; ++i) {
    if (std::fabs(lb[i] - lb_h[i]) > 1.0e-3f) {
      std::printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
