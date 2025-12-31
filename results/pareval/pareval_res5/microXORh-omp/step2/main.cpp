// microXORh driver

#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <omp.h>

// Offloaded 2D stencil that counts four-connected neighbors.
void cellsXOR_target(const int *__restrict__ device_input,
                     int *__restrict__ device_output, size_t N) {
  const size_t Nminus1 = N - 1;
  #pragma omp target teams loop collapse(2) is_device_ptr(device_input, device_output)
  for (size_t i = 0; i < N; ++i) {
    size_t row_base = i * N; // cache the row start to avoid repeated multiplies
    const bool has_top = (i > 0);
    const bool has_bottom = (i < Nminus1);
    for (size_t j = 0; j < N; ++j) {
      size_t idx = row_base + j;
      int count = 0;
      if (has_top && device_input[idx - N] == 1) count++;
      if (has_bottom && device_input[idx + N] == 1) count++;
      if (j > 0 && device_input[idx - 1] == 1) count++;
      if (j < Nminus1 && device_input[idx + 1] == 1) count++;
      device_output[idx] = (count == 1) ? 1 : 0;
    }
  }
}

// Release host and device allocations.
void cleanup(int *input, int *output, int *d_input, int *d_output, int device) {
  delete[] input;
  delete[] output;
  if (device >= 0) {
    if (d_input) omp_target_free(d_input, device);
    if (d_output) omp_target_free(d_output, device);
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
  size_t totalCells = N * N;
  for (size_t idx = 0; idx < totalCells; ++idx) {
    input[idx] = dis(gen);
  }

  int numDevices = omp_get_num_devices();
  if (numDevices <= 0) {
    std::cerr << "No OpenMP target device found; aborting." << std::endl;
    delete[] input;
    delete[] output;
    return 1;
  }

  int device = omp_get_default_device();
  int hostDevice = omp_get_initial_device();
  size_t byteCount = totalCells * sizeof(int);
  int *d_input =
      static_cast<int *>(omp_target_alloc(byteCount, device));
  int *d_output =
      static_cast<int *>(omp_target_alloc(byteCount, device));
  if (!d_input || !d_output) {
    std::cerr << "Device allocation failed." << std::endl;
    cleanup(input, output, d_input, d_output, device);
    return 1;
  }

  // Copy the host grid to the device buffer before offloading.
  omp_target_memcpy(d_input, input, byteCount, 0, 0, device, hostDevice);

  cellsXOR_target(d_input, d_output, N);

  // Copy the computed grid back to the host for verification.
  omp_target_memcpy(output, d_output, byteCount, 0, 0, hostDevice, device);

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
          cleanup(input, output, d_input, d_output, device);
          return 1;
        }
      } else {
        if (output[i*N + j] != 0) {
          std::cerr << "Validation failed at (" << i << ", " << j << ")" << std::endl;
          cleanup(input, output, d_input, d_output, device);
          return 1;
        }
      }
    }
  }
  std::cout << "Validation passed." << std::endl;
  cleanup(input, output, d_input, d_output, device);
  return 0;
}
