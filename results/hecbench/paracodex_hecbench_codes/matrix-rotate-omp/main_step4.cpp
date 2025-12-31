#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include "gate.h"

namespace {
constexpr int kThreadsPerTeam = 256;
constexpr int kMaxTeamsHint = 4096;

int compute_num_teams_hint(const int layer_count, const int tile_cols) {
  if (layer_count <= 0 || tile_cols <= 0) {
    return 1;
  }

  const size_t total_work = static_cast<size_t>(layer_count) * static_cast<size_t>(tile_cols);
  size_t teams_from_work = (total_work + static_cast<size_t>(kThreadsPerTeam) - 1) /
                           static_cast<size_t>(kThreadsPerTeam);

  if (teams_from_work == 0) {
    teams_from_work = 1;
  }
  if (teams_from_work > static_cast<size_t>(kMaxTeamsHint)) {
    teams_from_work = kMaxTeamsHint;
  }

  return static_cast<int>(teams_from_work);
}
}  // namespace

void rotate_matrix_parallel(float *matrix, const int n, const int repeat) {
  auto start = std::chrono::steady_clock::now();

  const int total_elems = n * n;
  const int layer_count = n / 2;
  const int tile_cols = n > 0 ? n - 1 : 0;
  const int num_teams_hint = compute_num_teams_hint(layer_count, tile_cols);

  // Keep the matrix resident on the device across all repeats to avoid remapping.
  #pragma omp target data map(tofrom: matrix[0:total_elems])
  {
    for (int iter = 0; iter < repeat; ++iter) {
      // Hint at the grid geometry so the RTX 4060 Laptop GPU can launch enough work to saturate its 24 SMs.
      #pragma omp target teams distribute parallel for collapse(2) \
          map(present: matrix[0:total_elems]) num_teams(num_teams_hint) \
          thread_limit(kThreadsPerTeam)
      for (int layer = 0; layer < layer_count; ++layer) {
        const int first = layer;
        const int last = n - 1 - layer;
        const int span = last - first;
        if (span <= 0) {
          continue;
        }
        const int first_row = first * n;
        const int last_row = last * n;

        // Limit the parallel iteration space to the active edge to avoid threads that only hit the guard.
        for (int col = first; col < last; ++col) {
          const int offset = col - first;
          const int mirror_col = last - offset;
          const int mirror_row = mirror_col * n;
          const int col_row = col * n;

          const int top_idx = first_row + col;
          const int left_idx = mirror_row + first;
          const int bottom_idx = last_row + mirror_col;
          const int right_idx = col_row + last;

          float top = matrix[top_idx];
          matrix[top_idx] = matrix[left_idx];
          matrix[left_idx] = matrix[bottom_idx];
          matrix[bottom_idx] = matrix[right_idx];
          matrix[right_idx] = top;
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
}

void rotate_matrix_serial(float *matrix, const int n, const int repeat) {
  const int total_elems = n * n;
  const int layer_count = n / 2;
  const int tile_cols = n > 0 ? n - 1 : 0;
  const int num_teams_hint = compute_num_teams_hint(layer_count, tile_cols);

  // Mirror the data residency approach so the reference result avoids redundant transfers.
  #pragma omp target data map(tofrom: matrix[0:total_elems])
  {
    for (int iter = 0; iter < repeat; ++iter) {
      #pragma omp target teams distribute parallel for collapse(2) \
          map(present: matrix[0:total_elems]) num_teams(num_teams_hint) \
          thread_limit(kThreadsPerTeam)
      for (int layer = 0; layer < layer_count; ++layer) {
        const int first = layer;
        const int last = n - 1 - layer;
        const int span = last - first;
        if (span <= 0) {
          continue;
        }
        const int first_row = first * n;
        const int last_row = last * n;

        for (int col = first; col < last; ++col) {
          const int offset = col - first;
          const int mirror_col = last - offset;
          const int mirror_row = mirror_col * n;
          const int col_row = col * n;

          const int top_idx = first_row + col;
          const int left_idx = mirror_row + first;
          const int bottom_idx = last_row + mirror_col;
          const int right_idx = col_row + last;

          float top = matrix[top_idx];
          matrix[top_idx] = matrix[left_idx];
          matrix[left_idx] = matrix[bottom_idx];
          matrix[bottom_idx] = matrix[right_idx];
          matrix[right_idx] = top;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const size_t total_elems = static_cast<size_t>(n) * n;

  float *serial_res = (float *)aligned_alloc(1024, n * n * sizeof(float));
  float *parallel_res = (float *)aligned_alloc(1024, n * n * sizeof(float));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      serial_res[i * n + j] = parallel_res[i * n + j] = i * n + j;

  rotate_matrix_serial(serial_res, n, repeat);

  rotate_matrix_parallel(parallel_res, n, repeat);

  bool ok = true;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (serial_res[i * n + j] != parallel_res[i * n + j]) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      break;
    }
  }

  GATE_CHECKSUM_BYTES("serial_res", serial_res, total_elems * sizeof(float));
  GATE_CHECKSUM_BYTES("parallel_res", parallel_res, total_elems * sizeof(float));

  printf("%s\n", ok ? "PASS" : "FAIL");

  free(serial_res);
  free(parallel_res);
  return 0;
}
