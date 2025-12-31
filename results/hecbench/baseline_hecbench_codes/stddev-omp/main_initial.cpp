#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include "reference.h"
#include "gate.h"

namespace {
constexpr int kWarpSize = 32;
constexpr int kThreadsPerTeam = 8 * kWarpSize;          // 256 threads keeps RTX 4060 SMs busy
constexpr int kColsPerTeam = 4 * kWarpSize;              // 128 columns per team for coalesced row tiles

inline void stddev_cpu(float *std_out, const float *data, int D, int N, int sample_size) {
  const int valid_flag = sample_size > 0 ? 1 : 0;
  const float inv_sample = valid_flag ? 1.0f / static_cast<float>(sample_size) : 0.0f;

  for (int c = 0; c < D; ++c) {
    float sum = 0.0f;
    for (int r = 0; r < N; ++r) {
      const float val = data[r * D + c];
      sum += val * val;
    }
    std_out[c] = valid_flag ? std::sqrt(sum * inv_sample) : INFINITY;
  }
}

inline void stddev_gpu(float *std_out, const float *data, int D, int N, int sample_size) {
  const int valid_flag = sample_size > 0 ? 1 : 0;
  const float inv_sample = valid_flag ? 1.0f / static_cast<float>(sample_size) : 0.0f;
  const int tile_count = (D + kColsPerTeam - 1) / kColsPerTeam;

#pragma omp target teams distribute num_teams(tile_count) thread_limit(kThreadsPerTeam) \
    is_device_ptr(std_out, data)
  for (int tile = 0; tile < D; tile += kColsPerTeam) {
    const int cols = std::min(kColsPerTeam, D - tile);
    float block_sums[kColsPerTeam];

    for (int c = 0; c < kColsPerTeam; ++c) {
      block_sums[c] = 0.0f;
    }

#pragma omp parallel reduction(+ : block_sums[:kColsPerTeam])
    {
#pragma omp for schedule(static)
      for (int r = 0; r < N; ++r) {
        const std::size_t row_offset = static_cast<std::size_t>(r) * D + tile;
        const float *row_ptr = data + row_offset;
#pragma omp simd
        for (int c = 0; c < cols; ++c) {
          const float val = row_ptr[c];
          block_sums[c] = std::fmaf(val, val, block_sums[c]);
        }
      }
    }

    for (int c = 0; c < cols; ++c) {
      std_out[tile + c] = valid_flag ? std::sqrt(block_sums[c] * inv_sample) : INFINITY;
    }
  }
}
} // namespace

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    std::printf("D: number of columns of data (must be a multiple of 32)\n");
    std::printf("N: number of rows of data (at least one row)\n");
    return 1;
  }

  const int D = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  if (D <= 0 || N <= 0 || repeat <= 0) {
    std::fprintf(stderr, "Invalid input parameters.\n");
    return 1;
  }

  const bool sample = true;
  const int sample_size = sample ? (N - 1) : N;
  const std::size_t input_elems = static_cast<std::size_t>(D) * static_cast<std::size_t>(N);
  const std::size_t output_elems = static_cast<std::size_t>(D);

  float *data = static_cast<float *>(std::malloc(input_elems * sizeof(float)));
  if (!data) {
    std::fprintf(stderr, "Failed to allocate input buffer.\n");
    return 1;
  }

  std::srand(123);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      data[i * D + j] = std::rand() / static_cast<float>(RAND_MAX);
    }
  }

  float *std_out = static_cast<float *>(std::malloc(output_elems * sizeof(float)));
  float *std_ref = static_cast<float *>(std::malloc(output_elems * sizeof(float)));
  if (!std_out || !std_ref) {
    std::fprintf(stderr, "Failed to allocate output buffers.\n");
    std::free(std_ref);
    std::free(std_out);
    std::free(data);
    return 1;
  }

  double avg_seconds = 0.0;
  const bool has_device = omp_get_num_devices() > 0;

  if (has_device) {
#pragma omp target data map(to : data[0:input_elems]) map(alloc : std_out[0:output_elems])
    {
#pragma omp target data use_device_addr(data, std_out)
      {
        stddev_gpu(std_out, data, D, N, sample_size);

        auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < repeat; ++iter) {
          stddev_gpu(std_out, data, D, N, sample_size);
        }
        auto end = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        avg_seconds = (elapsed * 1e-9) / repeat;
      }

#pragma omp target update from(std_out[0:output_elems])
    }
  } else {
    stddev_cpu(std_out, data, D, N, sample_size);

    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < repeat; ++iter) {
      stddev_cpu(std_out, data, D, N, sample_size);
    }
    auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    avg_seconds = (elapsed * 1e-9) / repeat;
  }

  std::printf("Average execution time of stddev kernels: %f (s)\n", static_cast<float>(avg_seconds));

  GATE_STATS_F32("stddev_out", std_out, D);

  stddev_ref(std_ref, data, D, N, sample);

  bool ok = true;
  for (int c = 0; c < D; ++c) {
    if (std::fabs(std_ref[c] - std_out[c]) > 1e-3f) {
      ok = false;
      break;
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(std_ref);
  std::free(std_out);
  std::free(data);
  return ok ? 0 : 1;
}
