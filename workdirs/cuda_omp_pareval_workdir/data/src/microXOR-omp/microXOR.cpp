// microXOR stencil kernel translated to OpenMP offload

#include "microXOR.cuh"

/* Each output cell becomes 1 if exactly one of its four cardinal neighbors is 1. The
   CUDA version mapped threads across the NxN grid; the OpenMP version offloads the same
   iteration space with a teams/loop construct. */
void cellsXOR(const int *__restrict__ input, int *__restrict__ output, size_t N) {
  const size_t stride = N;
  const int *__restrict__ d_input = input;
  int *__restrict__ d_output = output;

#pragma omp target teams loop collapse(2) is_device_ptr(d_input, d_output)
  for (size_t i = 0; i < stride; ++i) {
    const size_t row_start = i * stride;
    const int *__restrict__ current_row = d_input + row_start;
    const int *__restrict__ prev_row = (i > 0) ? current_row - stride : nullptr;
    const int *__restrict__ next_row = (i + 1 < stride) ? current_row + stride : nullptr;
    // Check neighbors using cached row addresses to avoid redundant index multiplications.
    for (size_t j = 0; j < stride; ++j) {
      int count = 0;
      if (prev_row && prev_row[j] == 1) ++count;
      if (next_row && next_row[j] == 1) ++count;
      if (j > 0 && current_row[j - 1] == 1) ++count;
      if ((j + 1) < stride && current_row[j + 1] == 1) ++count;
      d_output[row_start + j] = (count == 1) ? 1 : 0;
    }
  }
}
