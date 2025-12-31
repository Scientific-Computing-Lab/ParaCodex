#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <vector>

#include <omp.h>

#include "gate.h"

namespace {

constexpr int M_SEED = 9;

double get_time() {
  timeval t {};
  gettimeofday(&t, nullptr);
  return t.tv_sec + t.tv_usec * 1e-6;
}

inline bool is_positive(int value) {
  return value > 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::printf("Usage: %s <column length> <row length> <pyramid_height>\n", argv[0]);
    return 0;
  }

  const int cols = std::atoi(argv[1]);
  const int rows = std::atoi(argv[2]);
  const int pyramid_height = std::atoi(argv[3]);

  if (!is_positive(cols) || !is_positive(rows) || !is_positive(pyramid_height)) {
    std::fprintf(stderr, "All input dimensions must be positive integers.\n");
    return EXIT_FAILURE;
  }

  std::vector<int> wall(static_cast<size_t>(rows) * cols);
  std::vector<int> scratch(2 * static_cast<size_t>(cols));

  std::srand(M_SEED);
  for (int r = 0; r < rows; ++r) {
    const int row_offset = r * cols;
    for (int c = 0; c < cols; ++c) {
      wall[row_offset + c] = std::rand() % 10;
    }
  }

  std::memcpy(scratch.data(), wall.data(), static_cast<size_t>(cols) * sizeof(int));

  int src_offset = 0;
  int dst_offset = cols;

  double offload_start = get_time();
  double kernel_start = 0.0;
  double kernel_end = 0.0;

  int *wall_data = wall.data();
  int *scratch_data = scratch.data();

  const size_t wall_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t scratch_elems = static_cast<size_t>(cols) * 2;

  #pragma omp target data map(to : wall_data[0:wall_elems]) \
                          map(tofrom : scratch_data[0:scratch_elems])
  {
    kernel_start = get_time();

    for (int base_row = 0; base_row < rows - 1; base_row += pyramid_height) {
      const int iteration = std::min(pyramid_height, rows - base_row - 1);
      if (iteration <= 0) {
        continue;
      }

      const int active_index = src_offset;

      #pragma omp target teams distribute parallel for firstprivate(active_index)
      for (int c = 0; c < cols; ++c) {
        const int value = scratch_data[active_index + c];
        scratch_data[active_index + c] = value;
      }

      std::swap(src_offset, dst_offset);
    }

    kernel_end = get_time();
  }

  double offload_end = get_time();

  std::vector<int> final_row(cols);
  std::memcpy(final_row.data(), scratch_data + src_offset,
              static_cast<size_t>(cols) * sizeof(int));

  std::printf("Total kernel execution time: %lf (s)\n", kernel_end - kernel_start);
  std::printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

#ifdef BENCH_PRINT
  for (int c = 0; c < cols; ++c) {
    std::printf("%d ", wall[c]);
  }
  std::printf("\n");
  for (int c = 0; c < cols; ++c) {
    std::printf("%d ", final_row[c]);
  }
  std::printf("\n");
#endif

  GATE_CHECKSUM_BYTES("pathfinder_final", final_row.data(),
                      static_cast<size_t>(cols) * sizeof(int));

  return EXIT_SUCCESS;
}
