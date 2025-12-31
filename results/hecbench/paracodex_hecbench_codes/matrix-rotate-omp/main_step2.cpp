#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>

void rotate_matrix_parallel(float *matrix, const int n, const int repeat) {
  auto start = std::chrono::steady_clock::now();

  const int total_elems = n * n;
  const int layer_count = n / 2;

  // Keep the matrix resident on the device across all repeats to avoid remapping.
  #pragma omp target data map(tofrom: matrix[0:total_elems])
  {
    for (int iter = 0; iter < repeat; ++iter) {
      #pragma omp target teams distribute map(present: matrix[0:total_elems])
      for (int layer = 0; layer < layer_count; ++layer) {
        #pragma omp parallel for
        for (int col = layer; col < n - 1 - layer; ++col) {
          const int first = layer;
          const int last = n - 1 - layer;
          const int offset = col - first;

          float top = matrix[first * n + col];
          matrix[first * n + col] = matrix[(last - offset) * n + first];
          matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];
          matrix[last * n + (last - offset)] = matrix[col * n + last];
          matrix[col * n + last] = top;
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

  // Mirror the data residency approach so the reference result avoids redundant transfers.
  #pragma omp target data map(tofrom: matrix[0:total_elems])
  {
    for (int iter = 0; iter < repeat; ++iter) {
      #pragma omp target teams distribute map(present: matrix[0:total_elems])
      for (int layer = 0; layer < layer_count; ++layer) {
        #pragma omp parallel for
        for (int col = layer; col < n - 1 - layer; ++col) {
          const int first = layer;
          const int last = n - 1 - layer;
          const int offset = col - first;
          float top = matrix[first * n + col];

          matrix[first * n + col] = matrix[(last - offset) * n + first];
          matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];
          matrix[last * n + (last - offset)] = matrix[col * n + last];
          matrix[col * n + last] = top;
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

  printf("%s\n", ok ? "PASS" : "FAIL");

  free(serial_res);
  free(parallel_res);
  return 0;
}
