#include <math.h>
#include <stdio.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "utils.h"
#include "constants.h"
#include "gate.h"

namespace mean_shift::gpu {
  void mean_shift(const float *data, float *data_next,
                  const int teams, const int threads) {
    (void)teams;
    (void)threads;

    for (size_t tid = 0; tid < N; tid++) {
      size_t row = tid * D;
      float new_position[D] = {0.f};
      float tot_weight = 0.f;
      for (size_t i = 0; i < N; ++i) {
        size_t row_n = i * D;
        float sq_dist = 0.f;
        for (size_t j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[row_n + j]) * (data[row + j] - data[row_n + j]);
        }
        if (sq_dist <= RADIUS) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (size_t j = 0; j < D; ++j) {
            new_position[j] += weight * data[row_n + j];
          }
          tot_weight += weight;
        }
      }
      if (tot_weight <= 0.f) {
        for (size_t j = 0; j < D; ++j) {
          data_next[row + j] = data[row + j];
        }
        continue;
      }

      const float inv_weight = 1.f / tot_weight;
      for (size_t j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] * inv_weight;
      }
    }
  }

  void mean_shift_tiling(const float* data, float* data_next,
                         const int teams, const int threads) {
    (void)teams;
    (void)threads;

    float tile_data[TILE_WIDTH * D];

    for (size_t tid = 0; tid < N; ++tid) {
      size_t row = tid * D;
      float new_position[D] = {0.f};
      float tot_weight = 0.f;

      for (size_t tile_base = 0; tile_base < N; tile_base += TILE_WIDTH) {
        size_t tile_count = std::min<size_t>(TILE_WIDTH, N - tile_base);
        for (size_t i = 0; i < tile_count; ++i) {
          size_t src_row = (tile_base + i) * D;
          for (size_t j = 0; j < D; ++j) {
            tile_data[i * D + j] = data[src_row + j];
          }
        }

        for (size_t i = 0; i < tile_count; ++i) {
          size_t local_row = i * D;
          float sq_dist = 0.f;
          for (size_t j = 0; j < D; ++j) {
            float diff = data[row + j] - tile_data[local_row + j];
            sq_dist += diff * diff;
          }
          if (sq_dist <= RADIUS) {
            float weight = expf(-sq_dist / DBL_SIGMA_SQ);
            for (size_t j = 0; j < D; ++j) {
              new_position[j] += weight * tile_data[local_row + j];
            }
            tot_weight += weight;
          }
        }
      }

      if (tot_weight <= 0.f) {
        for (size_t j = 0; j < D; ++j) {
          data_next[row + j] = data[row + j];
        }
        continue;
      }

      const float inv_weight = 1.f / tot_weight;
      for (size_t j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] * inv_weight;
      }
    }
  }
}

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
  constexpr auto THREADS = mean_shift::gpu::THREADS;
  constexpr auto BLOCKS = mean_shift::gpu::BLOCKS;
  constexpr auto TILE_WIDTH = mean_shift::gpu::TILE_WIDTH;
  constexpr auto DIST_TO_REAL = mean_shift::gpu::DIST_TO_REAL;

  mean_shift::gpu::utils::print_info(path_to_data, N, D, BLOCKS, THREADS, TILE_WIDTH);

  const std::array<float, M * D> real = mean_shift::gpu::utils::load_csv<M, D>(path_to_centroids, ',');
  std::array<float, N * D> data = mean_shift::gpu::utils::load_csv<N, D>(path_to_data, ',');
  std::array<float, N * D> result = data;

  const size_t total_elems = static_cast<size_t>(N) * D;
  std::vector<float> buffer(total_elems);
  float *d_data = result.data();
  float *d_data_next = buffer.data();

  {
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
      mean_shift::gpu::mean_shift(d_data, d_data_next, BLOCKS, THREADS);
      mean_shift::gpu::utils::swap(d_data, d_data_next);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nAverage execution time of mean-shift (base) "
              << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

    float *scratch = d_data;
    std::copy(scratch, scratch + total_elems, result.begin());
    d_data = result.data();
    d_data_next = buffer.data();
    GATE_STATS_F32("meanshift_base_result", result.data(), total_elems);

    auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
    bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    if (centroids.size() == M && are_close)
       std::cout << "PASS\n";
    else
       std::cout << "FAIL\n";
    GATE_STATS_F32("meanshift_base_centroids",
                   reinterpret_cast<const float*>(centroids.data()),
                   centroids.size() * D);

    result = data;
    d_data = result.data();
    d_data_next = buffer.data();

    start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
      mean_shift::gpu::mean_shift_tiling(d_data, d_data_next, BLOCKS, THREADS);
      mean_shift::gpu::utils::swap(d_data, d_data_next);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nAverage execution time of mean-shift (opt) "
              << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

    float *scratch_opt = d_data;
    std::copy(scratch_opt, scratch_opt + total_elems, result.begin());
    d_data = result.data();
    d_data_next = buffer.data();
    GATE_STATS_F32("meanshift_opt_result", result.data(), total_elems);

    centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
    are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    if (centroids.size() == M && are_close)
       std::cout << "PASS\n";
    else
       std::cout << "FAIL\n";
    GATE_STATS_F32("meanshift_opt_centroids",
                   reinterpret_cast<const float*>(centroids.data()),
                   centroids.size() * D);
  }

  return 0;
}
