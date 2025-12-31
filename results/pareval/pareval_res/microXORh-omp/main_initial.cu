// microXORh driver

#include <iostream>
#include <random>
#include <omp.h>

/* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
   Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
   input and output are NxN grids of ints in row-major.
   Use OpenMP target teams loop to compute in parallel. The compute domain is an NxN iteration space.
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
void cellsXOR(int *input, int *output, size_t N) {
  const int * __restrict input_dev = input;
  int * __restrict output_dev = output;
  const size_t last_index = N - 1;
  // Alias hints inform the target kernel that the buffers do not overlap.

#pragma omp target teams loop collapse(2) is_device_ptr(input_dev, output_dev)
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      const size_t row_base = i * N;
      const size_t cell_idx = row_base + j;
      // Cache the linear index so we reuse the row multiplier across the neighbor checks.
      int count = 0;
      if (i > 0 && input_dev[cell_idx - N] == 1) count++;
      if (i < last_index && input_dev[cell_idx + N] == 1) count++;
      if (j > 0 && input_dev[cell_idx - 1] == 1) count++;
      if (j < last_index && input_dev[cell_idx + 1] == 1) count++;
      output_dev[cell_idx] = (count == 1) ? 1 : 0;
    }
  }
}

void cleanup(int *input, int *output, int *d_input, int *d_output) {
  delete[] input;
  delete[] output;
  int device = omp_get_default_device();
  if (d_input) omp_target_free(d_input, device);
  if (d_output) omp_target_free(d_output, device);
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

  const size_t totalCells = N * N;
  const size_t bufferBytes = totalCells * sizeof(int);
  const int device = omp_get_default_device();
  const int host_device = omp_get_initial_device();

  int *d_input = static_cast<int *>(omp_target_alloc(bufferBytes, device));
  int *d_output = static_cast<int *>(omp_target_alloc(bufferBytes, device));
  if (!d_input || !d_output) {
    std::cerr << "Failed to allocate device buffers" << std::endl;
    cleanup(input, output, d_input, d_output);
    return 1;
  }

  omp_target_memcpy(d_input, input, bufferBytes, 0, 0, device, host_device);

  cellsXOR(d_input, d_output, N);

  omp_target_memcpy(output, d_output, bufferBytes, 0, 0, host_device, device);

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
          cleanup(input, output, d_input, d_output);
          return 1;
        }
      } else {
        if (output[i*N + j] != 0) {
          std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
          cleanup(input, output, d_input, d_output);
          return 1;
        }
      }
    }
  }
  std::cout << "Validation passed." << std::endl;
  cleanup(input, output, d_input, d_output);
  return 0;
}
