// microXORh driver

#include <algorithm>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
   Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
   input and output are NxN grids of ints in row-major.
   Use OpenMP target offload to compute the stencil in parallel. The target teams loop mirrors the
   original NxN kernel launch across the device.
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
void cleanup(int *input, int *output) {
  delete[] input;
  delete[] output;
}

void cellsXOR(const int *input, int *output, size_t N, size_t blockEdge) {
  const size_t numBlocks = (N + blockEdge - 1) / blockEdge;
  const size_t requestedTeams = numBlocks * numBlocks;
  const int availableTeams = static_cast<int>(
      std::min(requestedTeams, static_cast<size_t>(std::numeric_limits<int>::max())));
  const int threadsPerTeam = static_cast<int>(
      std::min(blockEdge * blockEdge, static_cast<size_t>(std::numeric_limits<int>::max())));

#pragma omp target data map(to: input[0:N * N]) map(from: output[0:N * N])
  {
#pragma omp target teams distribute parallel for collapse(2) num_teams(availableTeams) thread_limit(threadsPerTeam)
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        int count = 0;
        if (i > 0 && input[(i - 1) * N + j] == 1) count++;
        if (i < N - 1 && input[(i + 1) * N + j] == 1) count++;
        if (j > 0 && input[i * N + (j - 1)] == 1) count++;
        if (j < N - 1 && input[i * N + (j + 1)] == 1) count++;
        output[i * N + j] = (count == 1) ? 1 : 0;
      }
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

  cellsXOR(input, output, N, blockEdge);

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
  cleanup(input, output);
  return 0;
}
