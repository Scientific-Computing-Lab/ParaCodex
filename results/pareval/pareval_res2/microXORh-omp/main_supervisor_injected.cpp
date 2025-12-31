// microXORh driver

#include <iostream>
#include <random>
#include <omp.h>
#include "gate.h"

void cellsXOR(const int *__restrict__ input, int *__restrict__ output, size_t N) {
  // Cache row pointers so each iteration avoids recomputing `i*N`/`j` offsets.
#pragma omp target teams loop collapse(2)
  for (size_t i = 0; i < N; ++i) {
    const size_t rowStart = i * N;
    const bool hasUp = i > 0;
    const bool hasDown = i + 1 < N;
    const int *row = input + rowStart;
    int *rowOut = output + rowStart;
    const int *rowUp = hasUp ? row - N : nullptr;
    const int *rowDown = hasDown ? row + N : nullptr;

    for (size_t j = 0; j < N; ++j) {
      int count = 0;
      if (hasUp && rowUp[j] == 1) count++;
      if (hasDown && rowDown[j] == 1) count++;
      if (j > 0 && row[j - 1] == 1) count++;
      if (j + 1 < N && row[j + 1] == 1) count++;
      rowOut[j] = (count == 1) ? 1 : 0;
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

  size_t totalElements = N * N;
  int *input = new int[totalElements];
  int *output = new int[totalElements];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 1);
  for (size_t i = 0; i < totalElements; ++i) {
    input[i] = dis(gen);
  }

  // Keep the dense grids resident on the device for the compute stage.
#pragma omp target data map(to: input[0:totalElements]) map(from: output[0:totalElements])
  {
    cellsXOR(input, output, N);
  }
  GATE_CHECKSUM_U32("microXORh_output", reinterpret_cast<const uint32_t*>(output), N * N);

  // Validate the output on the host to keep the original logic.
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      int count = 0;
      if (i > 0 && input[(i - 1) * N + j] == 1) count++;
      if (i < N - 1 && input[(i + 1) * N + j] == 1) count++;
      if (j > 0 && input[i * N + (j - 1)] == 1) count++;
      if (j < N - 1 && input[i * N + (j + 1)] == 1) count++;
      if (count == 1) {
        if (output[i * N + j] != 1) {
          std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
          delete[] input;
          delete[] output;
          return 1;
        }
      } else {
        if (output[i * N + j] != 0) {
          std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
          delete[] input;
          delete[] output;
          return 1;
        }
      }
    }
  }

  std::cout << "Validation passed." << std::endl;
  delete[] input;
  delete[] output;
  return 0;
}
