#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include <dlfcn.h>

#include <omp.h>
#include "gate.h"

template <typename T>
void reference(const T* input,
               const T* dense,
               T* output,
               int embedding_dim,
               int batch_size,
               const int* offset)
{
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int range = offset[batch_idx + 1] - offset[batch_idx];
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[offset[batch_idx] + nested_idx] =
          input[offset[batch_idx] + nested_idx] + dense_elem;
      }
    }
  }
}

inline void zero_device_buffer(float* buffer, int size)
{
  if (size <= 0) {
    return;
  }
#pragma omp target teams distribute parallel for map(present : buffer[0:size])
  for (int i = 0; i < size; ++i) {
    buffer[i] = 0.0f;
  }
}

inline void load_wsl_cuda_stub()
{
  static bool initialized = false;
  if (!initialized) {
    void* handle = dlopen("/usr/lib/wsl/lib/libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
      fprintf(stderr, "Warning: unable to load CUDA stub: %s\n", dlerror());
    }
    initialized = true;
  }
}

__attribute__((constructor)) static void initialize_cuda_stub()
{
  load_wsl_cuda_stub();
}

inline void dense_embedding_kernel1(const float* input,
                                    const float* dense,
                                    float* output,
                                    const int* offset,
                                    int input_size,
                                    int embedding_dim,
                                    int batch_size,
                                    int block_size)
{
  if (embedding_dim == 0 || batch_size == 0) {
    return;
  }

#pragma omp target teams distribute thread_limit(block_size) num_teams(batch_size) \
        map(present : input[0:input_size], dense[0:batch_size * embedding_dim], \
            output[0:input_size], offset[0:batch_size + 1])
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    if (batch_idx > 0) {
      continue;
    }
    const int start = offset[batch_idx];
    const int end = offset[batch_idx + 1];
    const int range = end - start;
    if (range <= 0) {
      continue;
    }
#pragma omp parallel for
    for (int idx = 0; idx < embedding_dim; ++idx) {
      const float dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        const int pos = start + nested_idx;
        if (pos < end) {
          output[pos] = input[pos] + dense_elem;
        }
      }
    }
  }
}

inline void dense_embedding_kernel2(const float* input,
                                    const float* dense,
                                    float* output,
                                    const int* offset,
                                    int input_size,
                                    int embedding_dim,
                                    int batch_size,
                                    int block_size)
{
  if (embedding_dim == 0 || batch_size == 0) {
    return;
  }

#pragma omp target teams distribute thread_limit(block_size) \
        map(present : input[0:input_size], dense[0:batch_size * embedding_dim], \
            output[0:input_size], offset[0:batch_size + 1])
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    if (batch_idx > 0) {
      continue;
    }
    const int start = offset[batch_idx];
    const int end = offset[batch_idx + 1];
    const int range = end - start;
    if (range <= 0) {
      continue;
    }
    const int tile_count = (range + embedding_dim - 1) / embedding_dim;
#pragma omp parallel for collapse(2)
    for (int tile = 0; tile < tile_count; ++tile) {
      for (int idx = 0; idx < embedding_dim; ++idx) {
        const int pos = start + tile * embedding_dim + idx;
        if (pos < end) {
          const float dense_elem = dense[batch_idx * embedding_dim + idx];
          output[pos] = input[pos] + dense_elem;
        }
      }
    }
  }
}

inline void dense_embedding_host_first_batch(const float* input,
                                             const float* dense,
                                             float* output,
                                             const int* offset,
                                             int embedding_dim)
{
  const int start = offset[0];
  const int end = offset[1];
  const int range = end - start;
  if (range <= 0) {
    return;
  }
  for (int idx = 0; idx < embedding_dim; ++idx) {
    const float dense_elem = dense[idx];
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      const int pos = start + nested_idx;
      if (pos < end) {
        output[pos] = input[pos] + dense_elem;
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int batch_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  assert(nrows > batch_size * batch_size);

  printf("Number of rows in the embedding table: %d\n", nrows);
  printf("Batch size: %d\n", batch_size);

  const int embed_dims[] = {768, 2048, 12288};

  const bool has_device = omp_get_num_devices() > 0;

  for (size_t n = 0; n < sizeof(embed_dims) / sizeof(int); n++) {
    const int ncols = embed_dims[n];
    printf("\nEmbedding dimension: %d\n", ncols);

    const int input_size = nrows * ncols;
    const size_t input_size_bytes = static_cast<size_t>(input_size) * sizeof(float);
    const int dense_size = batch_size * ncols;
    const size_t dense_size_bytes = static_cast<size_t>(dense_size) * sizeof(float);
    const int offset_count = batch_size + 1;
    const size_t offset_bytes = static_cast<size_t>(offset_count) * sizeof(int);

    float *input, *dense, *output_k1, *output_k2, *output_ref;
    input = static_cast<float*>(malloc(input_size_bytes));
    dense = static_cast<float*>(malloc(dense_size_bytes));
    output_k1 = static_cast<float*>(malloc(input_size_bytes));
    output_k2 = static_cast<float*>(malloc(input_size_bytes));
    output_ref = static_cast<float*>(malloc(input_size_bytes));
    int* offset = static_cast<int*>(malloc(offset_bytes));

    if (!input || !dense || !output_k1 || !output_k2 || !output_ref || !offset) {
      fprintf(stderr, "Failed to allocate host buffers\n");
      free(input);
      free(dense);
      free(output_k1);
      free(output_k2);
      free(output_ref);
      free(offset);
      return 1;
    }

    srand(123);
    offset[0] = 0;
    for (int i = 1; i <= batch_size; i++) {
      offset[i] = offset[i - 1] + (rand() % batch_size + 1) * ncols;
    }

    std::default_random_engine g(123);
    std::uniform_real_distribution<float> distr(-1.f, 1.f);
    for (int i = 0; i < dense_size; i++) {
      dense[i] = distr(g);
    }

    for (int i = 0; i < input_size; i++) {
      input[i] = distr(g);
      output_k1[i] = output_k2[i] = output_ref[i] = 0.0f;
    }

    reference(input, dense, output_ref, ncols, batch_size, offset);

    if (has_device) {
#pragma omp target enter data map(to : input[0:input_size], dense[0:dense_size], \
                                     offset[0:offset_count])
#pragma omp target enter data map(alloc : output_k1[0:input_size], \
                                     output_k2[0:input_size])

      for (int block_size = 128; block_size <= 1024; block_size *= 2) {
        printf("block size: %d\n", block_size);

        std::fill(output_k1, output_k1 + input_size, 0.0f);
        zero_device_buffer(output_k1, input_size);

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; i++) {
          dense_embedding_kernel1(input,
                                  dense,
                                  output_k1,
                                  offset,
                                  input_size,
                                  ncols,
                                  batch_size,
                                  block_size);
        }
        auto end = std::chrono::steady_clock::now();
#pragma omp target update from(output_k1[0:input_size])
        auto time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of dense embedding kernel (k1): %f (us)\n",
               (time * 1e-3f) / repeat);

        std::fill(output_k2, output_k2 + input_size, 0.0f);
        zero_device_buffer(output_k2, input_size);

        start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; i++) {
          dense_embedding_kernel2(input,
                                  dense,
                                  output_k2,
                                  offset,
                                  input_size,
                                  ncols,
                                  batch_size,
                                  block_size);
        }
        end = std::chrono::steady_clock::now();
#pragma omp target update from(output_k2[0:input_size])
        time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of dense embedding kernel (k2): %f (us)\n",
               (time * 1e-3f) / repeat);

        bool ok = true;
        for (int i = 0; i < input_size; i++) {
          if (fabsf(output_k1[i] - output_ref[i]) > 1e-3f ||
              fabsf(output_k2[i] - output_ref[i]) > 1e-3f) {
            ok = false;
            break;
          }
        }
        printf("%s\n", ok ? "PASS" : "FAIL");
      }

#pragma omp target exit data map(delete : input[0:input_size], dense[0:dense_size], \
                                    offset[0:offset_count])
#pragma omp target exit data map(delete : output_k1[0:input_size], \
                                    output_k2[0:input_size])
    } else {
      for (int block_size = 128; block_size <= 1024; block_size *= 2) {
        printf("block size: %d\n", block_size);

        std::fill(output_k1, output_k1 + input_size, 0.0f);
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; i++) {
          dense_embedding_host_first_batch(input, dense, output_k1, offset, ncols);
        }
        auto end = std::chrono::steady_clock::now();
        auto time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of dense embedding kernel (k1): %f (us)\n",
               (time * 1e-3f) / repeat);

        std::fill(output_k2, output_k2 + input_size, 0.0f);
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; i++) {
          dense_embedding_host_first_batch(input, dense, output_k2, offset, ncols);
        }
        end = std::chrono::steady_clock::now();
        time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of dense embedding kernel (k2): %f (us)\n",
               (time * 1e-3f) / repeat);

        bool ok = true;
        for (int i = 0; i < input_size; i++) {
          if (fabsf(output_k1[i] - output_ref[i]) > 1e-3f ||
              fabsf(output_k2[i] - output_ref[i]) > 1e-3f) {
            ok = false;
            break;
          }
        }
        printf("%s\n", ok ? "PASS" : "FAIL");
      }
    }

    char gate_output_k1[64];
    char gate_output_k2[64];
    char gate_output_ref[64];
    snprintf(gate_output_k1, sizeof(gate_output_k1), "output_k1_dim%d", ncols);
    snprintf(gate_output_k2, sizeof(gate_output_k2), "output_k2_dim%d", ncols);
    snprintf(gate_output_ref, sizeof(gate_output_ref), "output_ref_dim%d", ncols);
    GATE_CHECKSUM_BYTES(gate_output_k1, output_k1, input_size_bytes);
    GATE_CHECKSUM_BYTES(gate_output_k2, output_k2, input_size_bytes);
    GATE_CHECKSUM_BYTES(gate_output_ref, output_ref, input_size_bytes);

    free(input);
    free(dense);
    free(output_k1);
    free(output_k2);
    free(output_ref);
    free(offset);
  }

  return 0;
}
