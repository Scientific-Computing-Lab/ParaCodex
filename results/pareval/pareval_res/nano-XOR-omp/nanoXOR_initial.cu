#include <iostream>
#include <random>
#include <cstdint>
#include <omp.h>
#include "gate.h"

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
   Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
   input and output are NxN grids of ints in row-major.
   The CUDA implementation launched a single kernel over the NxN grid; the OpenMP version
   mirrors that work by offloading a double loop over the same domain to the device. */
void cellsXOR(const int * __restrict__ input, int * __restrict__ output, size_t N) {
  const int * __restrict__ d_input = input;
  int * __restrict__ d_output = output;
  // Collapse the i/j loops so we keep the one-thread-per-cell logic that the CUDA grid used.
  // The available RTX 4060 Ada GPU feeds wide SIMD groups, so this simple layout keeps occupancy high.
  #pragma omp target teams loop collapse(2)
  for (size_t i = 0; i < N; ++i) {
    const size_t row_start = i * N;  // cache the row base so the inner loop reuses one multiply per iteration
    for (size_t j = 0; j < N; ++j) {
      const size_t idx = row_start + j;
      int count = 0;
      if (i > 0 && d_input[idx - N] == 1) count++;
      if (i < N-1 && d_input[idx + N] == 1) count++;
      if (j > 0 && d_input[idx - 1] == 1) count++;
      if (j < N-1 && d_input[idx + 1] == 1) count++;
      d_output[idx] = (count == 1) ? 1 : 0;
    }
  }
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

  int *input = new int[N * N];
  int *output = new int[N * N];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 1);
  for (size_t i = 0; i < N * N; i++) {
    input[i] = dis(gen);
  }

  // Map both arrays before the offload so the target can see the buffers directly (Strategy A).
  #pragma omp target data map(to: input[0:N*N]) map(from: output[0:N*N])
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
          delete[] input;
          delete[] output;
          return 1;
        }
      } else {
        if (output[i*N + j] != 0) {
          std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
          delete[] input;
          delete[] output;
          return 1;
        }
      }
    }
  }
  std::cout << "Validation passed." << std::endl;
  GATE_CHECKSUM_U32("output", reinterpret_cast<const uint32_t*>(output), N * N);
  delete[] input;
  delete[] output;
  return 0;
}
