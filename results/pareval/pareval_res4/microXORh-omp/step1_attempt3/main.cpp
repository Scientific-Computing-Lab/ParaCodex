// microXORh driver

#include <climits>
#include <iostream>
#include <random>
#include <omp.h>

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
   Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
   input and output are NxN grids of ints in row-major.
   Use OpenMP target offload to compute in parallel. The teams loop drives an NxN domain. */
void cleanup(int *input, int *output) {
  delete[] input;
  delete[] output;
}

void cellsXOR(const int *input, int *output, size_t N, size_t blockEdge) {
  const unsigned long long totalCellsULL = static_cast<unsigned long long>(N) * static_cast<unsigned long long>(N);
  const size_t threadCount = blockEdge * blockEdge;
  const int threadLimit = static_cast<int>(threadCount);
  unsigned long long teams = (totalCellsULL + threadLimit - 1) / threadLimit;
  if (teams == 0) teams = 1;
  if (teams > static_cast<unsigned long long>(INT_MAX)) teams = static_cast<unsigned long long>(INT_MAX);
  const int numTeams = static_cast<int>(teams);
  const size_t totalCells = N * N;
  const int *d_input = input;
  int *d_output = output;
  // Launch a single flattened loop so teams/thread assignments stay simple, honoring the RTX 4060’s occupancy hints.
#pragma omp target teams loop num_teams(numTeams) thread_limit(threadLimit) is_device_ptr(d_input, d_output)
  for (size_t idx = 0; idx < totalCells; ++idx) {
    const size_t i = idx / N;
    const size_t j = idx % N;
    int count = 0;
    if (i > 0 && d_input[(i-1)*N + j] == 1) count++;
    if (i < N-1 && d_input[(i+1)*N + j] == 1) count++;
    if (j > 0 && d_input[i*N + (j-1)] == 1) count++;
    if (j < N-1 && d_input[i*N + (j+1)] == 1) count++;
    d_output[i*N + j] = (count == 1) ? 1 : 0;
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

  const size_t totalCells = N * N;
  int *input = new int[totalCells];
  int *output = new int[totalCells];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 1);
  for (size_t i = 0; i < totalCells; i++) {
    input[i] = dis(gen);
  }

  // blockEdge parameter is retained for CLI compatibility; the OpenMP teams loop sweeps the full NxN domain.
#pragma omp target data map(to: input[0:totalCells]) map(from: output[0:totalCells])
  {
    cellsXOR(input, output, N, blockEdge);
  }

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
