#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "gate.h"

namespace {

constexpr int TILE_DIM = 32;

void rotate_matrix_serial(float *matrix, const int n);

void rotate_matrix_parallel(float *matrix, const int n, const int repeat) {
  if (repeat <= 0 || n <= 1) {
    printf("Average kernel execution time: 0.000000 (s)\n");
    return;
  }

  const size_t total_elems = static_cast<size_t>(n) * static_cast<size_t>(n);
  const size_t bytes = total_elems * sizeof(float);
  int device_id = omp_get_default_device();
  const int num_devices = omp_get_num_devices();

  if (num_devices <= 0) {
    for (int iter = 0; iter < repeat; ++iter) {
      rotate_matrix_serial(matrix, n);
    }
    printf("Average kernel execution time: 0.000000 (s)\n");
    return;
  }

  if (device_id < 0 || device_id >= num_devices) {
    device_id = 0;
  }

  float *tmp = static_cast<float *>(std::malloc(bytes));
  if (tmp == nullptr) {
    fprintf(stderr, "Failed to allocate workspace\n");
    std::exit(EXIT_FAILURE);
  }

  auto start = std::chrono::steady_clock::now();

  #pragma omp target data device(device_id) map(tofrom : matrix[0:total_elems]) map(alloc : tmp[0:total_elems])
  {
    const int tiles_i = (n + TILE_DIM - 1) / TILE_DIM;
    const int tiles_j = (n + TILE_DIM - 1) / TILE_DIM;

    for (int iter = 0; iter < repeat; ++iter) {
      #pragma omp target teams distribute collapse(2) thread_limit(TILE_DIM * TILE_DIM) \
          device(device_id) map(present: matrix[0:total_elems], tmp[0:total_elems])
      for (int tile_i = 0; tile_i < tiles_i; ++tile_i) {
        for (int tile_j = 0; tile_j < tiles_j; ++tile_j) {
          const int base_i = tile_i * TILE_DIM;
          const int base_j = tile_j * TILE_DIM;

          #pragma omp parallel for collapse(2) schedule(static)
          for (int ii = 0; ii < TILE_DIM; ++ii) {
            for (int jj = 0; jj < TILE_DIM; ++jj) {
              const int gi = base_i + ii;
              const int gj = base_j + jj;
              if (gi < n && gj < n) {
                const int dest_row = gj;
                const int dest_col = n - 1 - gi;
                tmp[static_cast<size_t>(dest_row) * n + dest_col] =
                    matrix[static_cast<size_t>(gi) * n + gj];
              }
            }
          }
        }
      }

      #pragma omp target teams distribute parallel for device(device_id) \
          map(present: matrix[0:total_elems], tmp[0:total_elems]) schedule(static)
      for (size_t idx = 0; idx < total_elems; ++idx) {
        matrix[idx] = tmp[idx];
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  std::free(tmp);
}

void rotate_matrix_serial(float *matrix, const int n) {
  for (int layer = 0; layer < n / 2; ++layer) {
    const int first = layer;
    const int last = n - 1 - layer;
    for (int i = first; i < last; ++i) {
      const int offset = i - first;
      const float top = matrix[first * n + i];

      matrix[first * n + i] = matrix[(last - offset) * n + first];
      matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];
      matrix[last * n + (last - offset)] = matrix[i * n + last];
      matrix[i * n + last] = top;
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  float *serial_res = static_cast<float *>(aligned_alloc(1024, static_cast<size_t>(n) * n * sizeof(float)));
  float *parallel_res = static_cast<float *>(aligned_alloc(1024, static_cast<size_t>(n) * n * sizeof(float)));

  if (!serial_res || !parallel_res) {
    fprintf(stderr, "Allocation failed\n");
    std::free(serial_res);
    std::free(parallel_res);
    return 1;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      serial_res[i * n + j] = parallel_res[i * n + j] = static_cast<float>(i * n + j);
    }
  }

  for (int i = 0; i < repeat; ++i) {
    rotate_matrix_serial(serial_res, n);
  }

  rotate_matrix_parallel(parallel_res, n, repeat);

  const size_t total_elems = static_cast<size_t>(n) * static_cast<size_t>(n);
  GATE_CHECKSUM_BYTES("serial_res", serial_res, total_elems * sizeof(float));
  GATE_CHECKSUM_BYTES("parallel_res", parallel_res, total_elems * sizeof(float));

  bool ok = true;
  for (int i = 0; i < n && ok; ++i) {
    for (int j = 0; j < n; ++j) {
      if (serial_res[static_cast<size_t>(i) * n + j] != parallel_res[static_cast<size_t>(i) * n + j]) {
        ok = false;
        break;
      }
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(serial_res);
  std::free(parallel_res);
  return ok ? 0 : 1;
}
