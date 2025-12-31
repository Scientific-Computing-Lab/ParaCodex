#include <iostream>
#include <random>
#include <cstdint>
#include <omp.h>
#include "gate.h"

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
   Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
   input and output are NxN grids of ints in row-major.
   Use OpenMP target offload to compute on the device.
   Example:

   input: [[0, 1, 1, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0]
   output: [[0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 1, 0]]
*/
void cellsXOR(const int * __restrict__ input, int * __restrict__ output, size_t N) {
  const int64_t n = static_cast<int64_t>(N);
  const int64_t n_minus_1 = n - 1;
  const int * __restrict__ input_ptr = input;
  int * __restrict__ output_ptr = output;

  // Cache row offsets to limit repeated multiplications inside the target loop.
  #pragma omp target teams loop collapse(2) is_device_ptr(input_ptr, output_ptr)
  for (int64_t i = 0; i < n; ++i) {
    const int64_t base_row = i * n;
    const int64_t top_row = base_row - n;
    const int64_t bottom_row = base_row + n;
    const bool has_top = (i > 0);
    const bool has_bottom = (i < n_minus_1);

    for (int64_t j = 0; j < n; ++j) {
      int count = 0;
      const int64_t idx = base_row + j;
      if (has_top && input_ptr[top_row + j] == 1) count++;
      if (has_bottom && input_ptr[bottom_row + j] == 1) count++;
      if (j > 0 && input_ptr[idx - 1] == 1) count++;
      if (j < n_minus_1 && input_ptr[idx + 1] == 1) count++;
      output_ptr[idx] = (count == 1) ? 1 : 0;
    }
  }
}

void cleanup(int *input, int *output) {
  delete[] input;
  delete[] output;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " N blockEdge" << std::endl;
    return 1;
  }

  size_t N = std::stoi(argv[1]);
  size_t blockEdge = std::stoi(argv[2]);

  if (N % blockEdge != 0) {
    std::cerr << "N must be divisible by blockEdge" << std::endl;
    return 1;
  }
  if (blockEdge < 2 || blockEdge > 32) {
    std::cerr << "blockEdge must be between 2 and 32" << std::endl;
    return 1;
  }
  if (N < 4) {
    std::cerr << "N must be at least 4" << std::endl;
    return 1;
  }

  size_t totalCells = N * N;
  int *input = new int[totalCells];
  int *output = new int[totalCells];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 1);
  for (size_t i = 0; i < N * N; i++) {
    input[i] = dis(gen);
  }

  #pragma omp target data map(to: input[0:totalCells]) map(from: output[0:totalCells])
  {
    cellsXOR(input, output, N);
  }

  // Validate the output
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      int count = 0;
      if (i > 0 && input[(i-1)*N + j] == 1) count++;
      if (i < N-1 && input[(i+1)*N + j] == 1) count++;
      if (j > 0 && input[i*N + (j-1)] == 1) count++;
      if (j < N-1 && input[i*N + (j+1)] == 1) count++;
      if (count == 1) {
        if (output[i*N + j] != 1) {
          std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
          cleanup(input, output);
          return 1;
        }
      } else {
        if (output[i*N + j] != 0) {
          std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
          cleanup(input, output);
          return 1;
        }
      }
    }
  }
  std::cout << "Validation passed." << std::endl;
  GATE_CHECKSUM_U32("output", reinterpret_cast<const uint32_t*>(output), N * N);
  cleanup(input, output);
  return 0;
}
