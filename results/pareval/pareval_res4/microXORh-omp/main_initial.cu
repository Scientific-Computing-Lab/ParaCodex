#include <iostream>
#include <random>
#include <string>
#include <omp.h>

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1.
   Otherwise set it to 0. The stencil never includes the center cell.
   The CUDA version used a single kernel launch over an NxN grid; here
   we offload the same 2D computation via OpenMP target teams/collapse(2).
*/
void cellsXOR(const int *__restrict__ input, int *__restrict__ output, size_t N) {
  size_t stride = N;
  size_t total = N * N;

  #pragma omp target data map(to: input[0:total]) map(from: output[0:total])
  {
    #pragma omp target teams distribute parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        int count = 0;
        if (i > 0 && input[(i - 1) * stride + j] == 1) count++;
        if (i < N - 1 && input[(i + 1) * stride + j] == 1) count++;
        if (j > 0 && input[i * stride + (j - 1)] == 1) count++;
        if (j < N - 1 && input[i * stride + (j + 1)] == 1) count++;
        output[i * stride + j] = (count == 1) ? 1 : 0;
      }
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

  size_t N = std::stoul(argv[1]);
  size_t blockEdge = std::stoul(argv[2]);

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

  size_t total = N * N;
  int *input = new int[total];
  int *output = new int[total];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 1);
  for (size_t i = 0; i < total; ++i) {
    input[i] = dis(gen);
  }

  cellsXOR(input, output, N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      int count = 0;
      if (i > 0 && input[(i - 1) * N + j] == 1) count++;
      if (i < N - 1 && input[(i + 1) * N + j] == 1) count++;
      if (j > 0 && input[i * N + (j - 1)] == 1) count++;
      if (j < N - 1 && input[i * N + (j + 1)] == 1) count++;
      int expected = (count == 1) ? 1 : 0;
      if (output[i * N + j] != expected) {
        std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
        cleanup(input, output);
        return 1;
      }
    }
  }

  std::cout << "Validation passed." << std::endl;
  cleanup(input, output);
  return 0;
}
