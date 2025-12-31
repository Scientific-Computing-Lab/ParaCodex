#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cmath>
#include <omp.h>

template <typename T>
void reference(
    const T* input,
    const T* dense,
    T* output,
    int embedding_dim,
    int batch_size,
    const int* offset,
    int input_size)
{
  // Offload dense reference update; collapse batch/column dimensions for parallelism.
#pragma omp target teams distribute parallel for collapse(2) \
    map(to: input[0:input_size], dense[0:batch_size * embedding_dim], offset[0:batch_size + 1]) \
    map(tofrom: output[0:input_size])
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int idx = 0; idx < embedding_dim; idx++) {
      const int base = offset[batch_idx];
      const int range = offset[batch_idx + 1] - base;
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        const int write_idx = base + nested_idx;
        output[write_idx] = input[write_idx] + dense_elem;
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

  for (size_t n = 0; n < sizeof(embed_dims)/sizeof(int); n++) {
    int ncols = embed_dims[n];
    printf("\nEmbedding dimension: %d\n", ncols);

    int input_size = nrows * ncols;

    size_t input_size_bytes = input_size * sizeof(float);

    int dense_size = batch_size * ncols ;
    int dense_size_bytes = dense_size * sizeof(float);

    int batch_size_bytes = (batch_size + 1) * sizeof(float);

    float *input, *dense, *output_k1, *output_k2, *output_ref;
    input = (float*) malloc (input_size_bytes);

    dense = (float*) malloc (dense_size_bytes);

    output_k1 = (float*) malloc (input_size_bytes);

    output_k2 = (float*) malloc (input_size_bytes);

    output_ref = (float*) malloc (input_size_bytes);

    int *offset = (int*) malloc (batch_size_bytes);

    srand(123);
    offset[0] = 0;
    for (int i = 1; i <= batch_size; i++)
      offset[i] = offset[i-1] + (rand() % batch_size + 1) * ncols;

    std::default_random_engine g (123);
    std::uniform_real_distribution<float> distr (-1.f, 1.f);
    for (int i = 0; i < dense_size; i++) {
      dense[i] = distr(g);
    }

    for (int i = 0; i < input_size; i++) {
      input[i] = distr(g);
      output_k1[i] = output_k2[i] = output_ref[i] = 0;
    }

    reference(input, dense, output_ref, ncols, batch_size, offset, input_size);

    {
      for (int block_size = 128; block_size <= 1024; block_size = block_size * 2) {
        printf("block size: %d\n", block_size);

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          // GPU version of k1; map each (batch, column) to threads.
#pragma omp target teams distribute parallel for collapse(2) thread_limit(block_size) \
    map(to: input[0:input_size], dense[0:dense_size], offset[0:batch_size + 1]) \
    map(tofrom: output_k1[0:input_size]) num_teams(batch_size)
          for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (int idx = 0; idx < ncols; idx++) {
              const int start = offset[batch_idx];
              const int range = offset[batch_idx + 1] - start;
              const float dense_elem = dense[batch_idx * ncols + idx];
              for (int nested_idx = idx; nested_idx < range; nested_idx += ncols) {
                const int write_idx = start + nested_idx;
                output_k1[write_idx] = input[write_idx] + dense_elem;
              }
            }
          }
        }

        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of dense embedding kernel (k1): %f (us)\n", (time * 1e-3f) / repeat);

        start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          // GPU version of k2; mirror k1 but keeps explicit start offset.
#pragma omp target teams distribute parallel for collapse(2) thread_limit(block_size) \
    map(to: input[0:input_size], dense[0:dense_size], offset[0:batch_size + 1]) \
    map(tofrom: output_k2[0:input_size]) num_teams(batch_size)
          for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (int idx = 0; idx < ncols; idx++) {
              const int start = offset[batch_idx];
              const int range = offset[batch_idx + 1] - start;
              const float dense_elem = dense[batch_idx * ncols + idx];
              for (int nested_idx = idx; nested_idx < range; nested_idx += ncols) {
                const int write_idx = start + nested_idx;
                output_k2[write_idx] = input[write_idx] + dense_elem;
              }
            }
          }
        }

        end = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of dense embedding kernel (k2): %f (us)\n", (time * 1e-3f) / repeat);

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

    free(input);
    free(dense);
    free(output_k1);
    free(output_k2);
    free(output_ref);
    free(offset);
  }

  return 0;
}
