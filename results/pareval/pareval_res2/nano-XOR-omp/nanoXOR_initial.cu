#include <iostream>
#include <random>
#include <cstdint>
#include <omp.h>
#include "gate.h"

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
   Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
   input and output are NxN grids of ints in row-major.
   OpenMP target offload now handles the parallel work using a flat teams/loop mapping.
   The grid is sized by blockEdge to keep the legacy interface and to hint at the original CUDA launch geometry. */
void cellsXOR(const int *__restrict__ input, int *__restrict__ output, size_t N, size_t blockEdge) {
  size_t workGroups = (N + blockEdge - 1) / blockEdge;
  size_t numTeams = workGroups * workGroups;
  size_t threadsPerTeam = blockEdge * blockEdge;
  const size_t stride = N;

  // Target the NVIDIA GeForce RTX 4060 (Ada Lovelace) reported in system_info.txt to keep the GPU busy.
  // Collapse the i/j loops to mirror the CUDA grid/block geometry and avoid div/mod.
  #pragma omp target teams distribute parallel for collapse(2) num_teams(numTeams) thread_limit(threadsPerTeam)
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * stride + j;
      int count = 0;

      if (i > 0 && input[idx - stride] == 1) count++;
      if (i + 1 < N && input[idx + stride] == 1) count++;
      if (j > 0 && input[idx - 1] == 1) count++;
      if (j + 1 < N && input[idx + 1] == 1) count++;

      output[idx] = (count == 1) ? 1 : 0;
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
    cellsXOR(input, output, N, blockEdge);
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
