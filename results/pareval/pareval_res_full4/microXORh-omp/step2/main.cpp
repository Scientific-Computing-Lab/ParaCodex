// microXORh driver

#include <iostream>
#include <random>
#include <omp.h>

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
   Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
   input and output are NxN grids of ints in row-major.
   We now offload the stencil to the GPU through OpenMP target teams loops.
*/
void cellsXOR(const int * __restrict input, int * __restrict output, size_t N) {
  const int * __restrict d_input = input;
  int * __restrict d_output = output;
  const size_t stride = N;

  #pragma omp target teams loop collapse(2) is_device_ptr(d_input, d_output)
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      // Cache the flattened coordinate to reduce repeated multiplication.
      const size_t base = i * stride + j;
      int count = 0;
      if (i > 0 && d_input[base - stride] == 1) count++;
      if (i < N - 1 && d_input[base + stride] == 1) count++;
      if (j > 0 && d_input[base - 1] == 1) count++;
      if (j < N - 1 && d_input[base + 1] == 1) count++;
      d_output[base] = (count == 1) ? 1 : 0;
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
  for (size_t i = 0; i < totalCells; i++) {
    input[i] = dis(gen);
  }

  #pragma omp target data map(to: input[0:totalCells]) map(from: output[0:totalCells])
  {
    cellsXOR(input, output, N);
  }

  /*
  for (int i = 0; i < N*N; i++) {
    std::cout << output[i] << " ";
    if (i % N == N - 1) std::cout << std::endl;
  }
  */

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
          cleanup(input, output);
          return 1;
        }
      } else {
        if (output[i * N + j] != 0) {
          std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
          cleanup(input, output);
          return 1;
        }
      }
    }
  }
  std::cout << "Validation passed." << std::endl;
  cleanup(input, output);
  return 0;
}
