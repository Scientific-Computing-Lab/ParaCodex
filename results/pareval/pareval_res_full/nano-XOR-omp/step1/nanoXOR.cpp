#include <iostream>
#include <random>
#include <cstdint>
#include "gate.h"

namespace {

void cellsXOR_device(const int *input, int *output, size_t N) {
  size_t total = N * N;
#pragma omp target teams loop
  for (size_t idx = 0; idx < total; ++idx) {
    size_t i = idx / N;
    size_t j = idx % N;
    int count = 0;
    if (i > 0 && input[(i - 1) * N + j] == 1) count++;
    if (i < N - 1 && input[(i + 1) * N + j] == 1) count++;
    if (j > 0 && input[i * N + (j - 1)] == 1) count++;
    if (j < N - 1 && input[i * N + (j + 1)] == 1) count++;
    output[i * N + j] = (count == 1) ? 1 : 0;
  }
}

}  // namespace

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

#pragma omp target data map(to: input[0:N*N]) map(from: output[0:N*N])
{
  cellsXOR_device(input, output, N);
}

  // Validate the output
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
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
  GATE_CHECKSUM_U32("output", reinterpret_cast<const uint32_t *>(output),
                    N * N);
  delete[] input;
  delete[] output;
  return 0;
}
