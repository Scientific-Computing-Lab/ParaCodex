#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "gate.h"

#define RADIUS 7
#define BLOCK_SIZE 256

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::printf("Usage: %s <length> <repeat>\n", argv[0]);
    std::printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }

  const int length = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  const int size = length;
  const int pad_size = length + RADIUS;

  if (size <= 0 || repeat <= 0) {
    std::printf("length and repeat must be strictly positive\n");
    return 1;
  }

  int* a = static_cast<int*>(std::malloc(pad_size * sizeof(int)));
  int* b = static_cast<int*>(std::malloc(size * sizeof(int)));

  if (!a || !b) {
    std::fprintf(stderr, "Allocation failed\n");
    std::free(a);
    std::free(b);
    return 1;
  }

  for (int i = 0; i < pad_size; ++i) {
    a[i] = i;
  }

  const int num_teams = std::max(1, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
  const int num_devices = omp_get_num_devices();
  const bool use_gpu = num_devices > 0;

  auto start = std::chrono::steady_clock::now();

#pragma omp target data map(to : a[0:pad_size]) map(from : b[0:size]) if (use_gpu)
  {
    for (int iter = 0; iter < repeat; ++iter) {
#pragma omp target teams distribute parallel for num_teams(num_teams) thread_limit(BLOCK_SIZE) schedule(static, 1) if (use_gpu)
      for (int idx = 0; idx < size; ++idx) {
        int lower = idx - RADIUS;
        if (lower < 0) {
          lower = 0;
        }

        int upper = idx + RADIUS;
        if (upper >= pad_size) {
          upper = pad_size - 1;
        }

        int result = 0;
#pragma omp simd reduction(+ : result)
        for (int src = lower; src <= upper; ++src) {
          result += a[src];
        }

        b[idx] = result;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  const auto wall_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (wall_time * 1e-9f) / repeat);
  GATE_CHECKSUM_BYTES("b", b, (size_t)size * sizeof(int));

  bool ok = true;
  for (int i = 0; i < 2 * RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i + 2 * RADIUS; j++) {
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    }
    if (s != b[i]) {
      std::printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2 * RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i - RADIUS; j <= i + RADIUS; j++) {
      s += a[j];
    }
    if (s != b[i]) {
      std::printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(a);
  std::free(b);

  return ok ? 0 : 1;
}
