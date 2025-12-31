#include <math.h>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "gate.h"

#include "utils.h"
#include "constants.h"

namespace mean_shift::gpu {

namespace {

constexpr int kN = N;
constexpr int kD = D;
constexpr int kTileWidth = TILE_WIDTH;

bool detect_offload_preference() {
  if (const char* env = std::getenv("MEANSHIFT_FORCE_GPU")) {
    return std::atoi(env) != 0;
  }
  return false;
}

bool g_offload_enabled = detect_offload_preference();

inline int sanitize_launch_param(const int value, const int fallback) {
  return value > 0 ? value : fallback;
}

void mean_shift_host_impl(const float* data, float* data_next) {
  for (int tid = 0; tid < kN; ++tid) {
    const size_t row = static_cast<size_t>(tid) * kD;
    float new_position[kD] = {0.f};
    float tot_weight = 0.f;

    for (int i = 0; i < kN; ++i) {
      const size_t row_n = static_cast<size_t>(i) * kD;
      float sq_dist = 0.f;
      for (int j = 0; j < kD; ++j) {
        const float diff = data[row + j] - data[row_n + j];
        sq_dist += diff * diff;
      }
      if (sq_dist <= RADIUS) {
        const float weight = expf(-sq_dist / DBL_SIGMA_SQ);
        for (int j = 0; j < kD; ++j) {
          new_position[j] += weight * data[row_n + j];
        }
        tot_weight += weight;
      }
    }

    if (tot_weight <= 0.f) {
      for (int j = 0; j < kD; ++j) {
        data_next[row + j] = data[row + j];
      }
      continue;
    }

    const float inv_weight = 1.f / tot_weight;
    for (int j = 0; j < kD; ++j) {
      data_next[row + j] = new_position[j] * inv_weight;
    }
  }
}

void mean_shift_tiling_host_impl(const float* data, float* data_next) {
  float tile_data[kTileWidth * kD];

  for (int tid = 0; tid < kN; ++tid) {
    const size_t row = static_cast<size_t>(tid) * kD;
    float new_position[kD] = {0.f};
    float tot_weight = 0.f;

    for (int tile_base = 0; tile_base < kN; tile_base += kTileWidth) {
      const int remaining = kN - tile_base;
      const int tile_count = remaining > kTileWidth ? kTileWidth : remaining;

      for (int i = 0; i < tile_count; ++i) {
        const size_t src_row = static_cast<size_t>(tile_base + i) * kD;
        for (int j = 0; j < kD; ++j) {
          tile_data[i * kD + j] = data[src_row + j];
        }
      }

      for (int i = 0; i < tile_count; ++i) {
        const int base = i * kD;
        float sq_dist = 0.f;
        for (int j = 0; j < kD; ++j) {
          const float diff = data[row + j] - tile_data[base + j];
          sq_dist += diff * diff;
        }
        if (sq_dist <= RADIUS) {
          const float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < kD; ++j) {
            new_position[j] += weight * tile_data[base + j];
          }
          tot_weight += weight;
        }
      }
    }

    if (tot_weight <= 0.f) {
      for (int j = 0; j < kD; ++j) {
        data_next[row + j] = data[row + j];
      }
      continue;
    }

    const float inv_weight = 1.f / tot_weight;
    for (int j = 0; j < kD; ++j) {
      data_next[row + j] = new_position[j] * inv_weight;
    }
  }
}

}  // namespace

void set_offload_enabled(bool enabled) { g_offload_enabled = enabled; }

bool offload_enabled() { return g_offload_enabled; }

void mean_shift(const float *data, float *data_next,
                const int teams, const int threads) {
  if (!offload_enabled()) {
    mean_shift_host_impl(data, data_next);
    return;
  }

  const int launch_teams = sanitize_launch_param(teams, kN);
  const int launch_threads = sanitize_launch_param(threads, THREADS);

#pragma omp target teams distribute num_teams(launch_teams) thread_limit(launch_threads) is_device_ptr(data, data_next)
  for (int tid = 0; tid < kN; ++tid) {
    const size_t row = static_cast<size_t>(tid) * kD;
    float point[kD];
    for (int j = 0; j < kD; ++j) {
      point[j] = data[row + j];
    }

    float new_position[kD] = {0.f};
    float tot_weight = 0.f;

#pragma omp parallel for reduction(+:new_position[:kD], tot_weight) schedule(static)
    for (int i = 0; i < kN; ++i) {
      const size_t row_n = static_cast<size_t>(i) * kD;
      float sq_dist = 0.f;
      for (int j = 0; j < kD; ++j) {
        const float diff = point[j] - data[row_n + j];
        sq_dist += diff * diff;
      }
      if (sq_dist <= RADIUS) {
        const float weight = expf(-sq_dist / DBL_SIGMA_SQ);
        for (int j = 0; j < kD; ++j) {
          new_position[j] += weight * data[row_n + j];
        }
        tot_weight += weight;
      }
    }

    if (tot_weight <= 0.f) {
      for (int j = 0; j < kD; ++j) {
        data_next[row + j] = point[j];
      }
    } else {
      const float inv_weight = 1.f / tot_weight;
      for (int j = 0; j < kD; ++j) {
        data_next[row + j] = new_position[j] * inv_weight;
      }
    }
  }
}

void mean_shift_tiling(const float* data, float* data_next,
                       const int teams, const int threads) {
  if (!offload_enabled()) {
    mean_shift_tiling_host_impl(data, data_next);
    return;
  }

  const int launch_teams = sanitize_launch_param(teams, kN);
  const int launch_threads = sanitize_launch_param(threads, THREADS);

#pragma omp target teams distribute num_teams(launch_teams) thread_limit(launch_threads) is_device_ptr(data, data_next)
  for (int tid = 0; tid < kN; ++tid) {
    const size_t row = static_cast<size_t>(tid) * kD;
    float point[kD];
    for (int j = 0; j < kD; ++j) {
      point[j] = data[row + j];
    }

    float new_position[kD] = {0.f};
    float tot_weight = 0.f;
    float tile_data[kTileWidth * kD];

#pragma omp parallel reduction(+:new_position[:kD], tot_weight)
    {
      for (int tile_base = 0; tile_base < kN; tile_base += kTileWidth) {
        const int remaining = kN - tile_base;
        const int tile_count = remaining > kTileWidth ? kTileWidth : remaining;

#pragma omp for collapse(2) schedule(static)
        for (int li = 0; li < tile_count; ++li) {
          for (int j = 0; j < kD; ++j) {
            tile_data[li * kD + j] = data[(static_cast<size_t>(tile_base + li) * kD) + j];
          }
        }
#pragma omp barrier

#pragma omp for schedule(static)
        for (int li = 0; li < tile_count; ++li) {
          float sq_dist = 0.f;
          const int base = li * kD;
          for (int j = 0; j < kD; ++j) {
            const float diff = point[j] - tile_data[base + j];
            sq_dist += diff * diff;
          }
          if (sq_dist <= RADIUS) {
            const float weight = expf(-sq_dist / DBL_SIGMA_SQ);
            for (int j = 0; j < kD; ++j) {
              new_position[j] += weight * tile_data[base + j];
            }
            tot_weight += weight;
          }
        }
#pragma omp barrier
      }
    }

    if (tot_weight <= 0.f) {
      for (int j = 0; j < kD; ++j) {
        data_next[row + j] = point[j];
      }
    } else {
      const float inv_weight = 1.f / tot_weight;
      for (int j = 0; j < kD; ++j) {
        data_next[row + j] = new_position[j] * inv_weight;
      }
    }
  }
}

}  // namespace mean_shift::gpu

namespace {

using mean_shift::gpu::BLOCKS;
using mean_shift::gpu::THREADS;

template <typename Kernel>
void run_mean_shift_iterations(
    Kernel kernel,
    float* host_current,
    float* host_next,
    const size_t total_elems) {
  const bool use_offload = mean_shift::gpu::offload_enabled();

  if (!use_offload) {
    float* current = host_current;
    float* next = host_next;
    for (size_t iter = 0; iter < mean_shift::gpu::NUM_ITER; ++iter) {
      kernel(current, next, BLOCKS, THREADS);
      std::swap(current, next);
    }
    if (current != host_current) {
      std::copy(current, current + total_elems, host_current);
    }
    return;
  }

  const int device = omp_get_default_device();

#pragma omp target data if(use_offload) map(tofrom: host_current[0:total_elems]) map(alloc: host_next[0:total_elems])
  {
    float* device_current = static_cast<float*>(omp_get_mapped_ptr(host_current, device));
    float* device_next = static_cast<float*>(omp_get_mapped_ptr(host_next, device));
    float* device_result = device_current;

    for (size_t iter = 0; iter < mean_shift::gpu::NUM_ITER; ++iter) {
      kernel(device_current, device_next, BLOCKS, THREADS);
      std::swap(device_current, device_next);
    }

    if (device_current != device_result) {
#pragma omp target teams distribute parallel for if(use_offload) is_device_ptr(device_current, device_result) schedule(static)
      for (size_t idx = 0; idx < total_elems; ++idx) {
        device_result[idx] = device_current[idx];
      }
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to data> <path to centroids>" << std::endl;
    return 1;
  }
  const auto path_to_data = argv[1];
  const auto path_to_centroids = argv[2];

  constexpr auto N = mean_shift::gpu::N;
  constexpr auto D = mean_shift::gpu::D;
  constexpr auto M = mean_shift::gpu::M;
  constexpr auto TILE_WIDTH = mean_shift::gpu::TILE_WIDTH;
  constexpr auto DIST_TO_REAL = mean_shift::gpu::DIST_TO_REAL;

  mean_shift::gpu::utils::print_info(path_to_data, N, D, BLOCKS, THREADS, TILE_WIDTH);

  const std::array<float, M * D> real = mean_shift::gpu::utils::load_csv<M, D>(path_to_centroids, ',');
  std::array<float, N * D> data = mean_shift::gpu::utils::load_csv<N, D>(path_to_data, ',');
  std::array<float, N * D> result = data;

  const size_t total_elems = static_cast<size_t>(N) * D;
  std::vector<float> buffer(total_elems, 0.f);

  {
    auto start = std::chrono::steady_clock::now();
    run_mean_shift_iterations(mean_shift::gpu::mean_shift, result.data(), buffer.data(), total_elems);
    auto end = std::chrono::steady_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nAverage execution time of mean-shift (base) "
              << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

    auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
    const bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    if (centroids.size() == M && are_close)
       std::cout << "PASS\n";
    else
       std::cout << "FAIL\n";

    GATE_STATS_F32("result_base", result.data(), total_elems);
  }

  result = data;
  std::fill(buffer.begin(), buffer.end(), 0.f);

  {
    auto start = std::chrono::steady_clock::now();
    run_mean_shift_iterations(mean_shift::gpu::mean_shift_tiling, result.data(), buffer.data(), total_elems);
    auto end = std::chrono::steady_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nAverage execution time of mean-shift (opt) "
              << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

    auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
    const bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    if (centroids.size() == M && are_close)
       std::cout << "PASS\n";
    else
       std::cout << "FAIL\n";

    GATE_STATS_F32("result_opt", result.data(), total_elems);
  }

  return 0;
}
