#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <dlfcn.h>
#include <unistd.h>

#include "reference.h"
#include "gate.h"

namespace {
constexpr int kTeamSize = 256;  // 8 warps keeps Ada SMs well-occupied

bool gpu_is_available() {
  if (access("/dev/nvidia0", R_OK) != 0 && access("/dev/nvidia1", R_OK) != 0 &&
      access("/dev/dxg", R_OK) != 0) {
    return false;
  }

  void *cuda_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (!cuda_handle) {
    return false;
  }
  using CuInitFn = int (*)(unsigned int);
  auto *cu_init = reinterpret_cast<CuInitFn>(dlsym(cuda_handle, "cuInit"));
  if (!cu_init) {
    dlclose(cuda_handle);
    return false;
  }
  const int status = cu_init(0u);
  dlclose(cuda_handle);
  return status == 0;
}
}  // namespace

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

  std::srand(123);
  for (int i = 0; i < N; ++i) {
    subject[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < length; ++i) {
    avgs[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < length; ++i) {
    stds[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < M; ++i) {
    upper_bound[i] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < M; ++i) {
    lower_bound[i] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }

  const int num_teams = std::max(1, (length + kTeamSize - 1) / kTeamSize);

  float *__restrict subject_ptr = subject.data();
  float *__restrict avgs_ptr = avgs.data();
  float *__restrict stds_ptr = stds.data();
  float *__restrict lower_ptr = lower_bound.data();
  float *__restrict upper_ptr = upper_bound.data();
  float *__restrict lb_ptr = lb.data();

  auto start = std::chrono::steady_clock::now();
  const bool use_device = gpu_is_available();

#pragma omp target enter data map(to : subject_ptr [0:N],                     \
                                  avgs_ptr [0:length],                        \
                                  stds_ptr [0:length],                        \
                                  lower_ptr [0:M], upper_ptr [0:M]) if (use_device)
#pragma omp target enter data map(alloc : lb_ptr [0:length]) if (use_device)

  for (int iter = 0; iter < repeat; ++iter) {
#pragma omp target teams distribute parallel for simd                          \
    map(present : subject_ptr [0:N], avgs_ptr [0:length],                      \
        stds_ptr [0:length], lower_ptr [0:M], upper_ptr [0:M],                 \
        lb_ptr [0:length])                                                     \
    num_teams(num_teams) thread_limit(kTeamSize) schedule(static) if (use_device)
    for (int idx = 0; idx < length; ++idx) {
      const float avg = avgs_ptr[idx];
      const float std = stds_ptr[idx];

      float residues = 0.0f;
#pragma omp simd reduction(+ : residues)
      for (int i = 0; i < M; ++i) {
        const float value = (subject_ptr[idx + i] - avg) / std;
        const float lower = value - lower_ptr[i];
        const float upper = value - upper_ptr[i];
        const float upper_term = upper > 0.0f ? upper * upper : 0.0f;
        const float lower_term = lower < 0.0f ? lower * lower : 0.0f;
        residues += upper_term + lower_term;
      }
      lb_ptr[idx] = residues;
    }
  }

#pragma omp target update from(lb_ptr [0:length]) if (use_device)

  GATE_STATS_F32("lb", lb.data(), length);

#pragma omp target exit data map(delete : subject_ptr [0:N],                  \
                                 avgs_ptr [0:length],                         \
                                 stds_ptr [0:length],                         \
                                 lower_ptr [0:M], upper_ptr [0:M]) if (use_device)
#pragma omp target exit data map(delete : lb_ptr [0:length]) if (use_device)

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
