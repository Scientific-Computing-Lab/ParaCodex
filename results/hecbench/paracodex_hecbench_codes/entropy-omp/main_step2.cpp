#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

void entropy(
      float *__restrict d_entropy,
  const char*__restrict d_val,
  int height, int width)
{
  const size_t num_pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
  if (num_pixels == 0) {
    return;
  }

#pragma omp target teams distribute parallel for collapse(2) \
    map(present: d_val[0:num_pixels], d_entropy[0:num_pixels])
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {

      char count[16];
      for (int i = 0; i < 16; i++) count[i] = 0;

      char total = 0;

      for(int dy = -2; dy <= 2; dy++) {
        for(int dx = -2; dx <= 2; dx++) {
          int xx = x + dx;
          int yy = y + dy;
          if(xx >= 0 && yy >= 0 && yy < height && xx < width) {
            count[d_val[yy * width + xx]]++;
            total++;
          }
        }
      }

      float entropy = 0;
      if (total < 1) {
        total = 1;
      } else {
        for(int k = 0; k < 16; k++) {
          float p = (float)count[k] / (float)total;
          entropy -= p * log2f(p);
        }
      }

      d_entropy[y * width + x] = entropy;
    }
  }
}

template<int bsize_x, int bsize_y>
void entropy_opt(
       float *__restrict d_entropy,
  const  char*__restrict d_val,
  const float*__restrict d_logTable,
  int m, int n)
{
  const int teamX = (n+bsize_x-1)/bsize_x;
  const int teamY = (m+bsize_y-1)/bsize_y;
  const size_t num_pixels = static_cast<size_t>(m) * static_cast<size_t>(n);
  if (num_pixels == 0) {
    return;
  }

  // Launch teams over tiles and lanes over intra-tile threads for GPU execution.
#pragma omp target teams distribute parallel for collapse(4) \
    map(present: d_val[0:num_pixels], d_entropy[0:num_pixels]) \
    map(present: d_logTable[0:26])
  for (int teamIdx_y = 0; teamIdx_y < teamY; ++teamIdx_y) {
    for (int teamIdx_x = 0; teamIdx_x < teamX; ++teamIdx_x) {
      for (int threadIdx_y = 0; threadIdx_y < bsize_y; ++threadIdx_y) {
        for (int threadIdx_x = 0; threadIdx_x < bsize_x; ++threadIdx_x) {
          const int x = teamIdx_x * bsize_x + threadIdx_x;
          const int y = teamIdx_y * bsize_y + threadIdx_y;
          if (y >= m || x >= n) {
            continue;
          }

          int sd_count[16];
          for (int i = 0; i < 16; i++) sd_count[i] = 0;

          int total = 0;
          for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
              const int xx = x + dx;
              const int yy = y + dy;
              if (xx >= 0 && yy >= 0 && yy < m && xx < n) {
                sd_count[d_val[yy * n + xx]]++;
                total++;
              }
            }
          }

          float entropy = 0;
          for (int k = 0; k < 16; k++) {
            entropy -= d_logTable[sd_count[k]];
          }

          entropy = entropy / total + log2f(static_cast<float>(total));
          d_entropy[y * n + x] = entropy;
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_bytes = width * height * sizeof(char);
  const int output_bytes = width * height * sizeof(float);
  char* input = (char*) malloc (input_bytes);
  float* output = (float*) malloc (output_bytes);
  float* output_ref = (float*) malloc (output_bytes);

  float logTable[26];
  for (int i = 0; i <= 25; i++) logTable[i] = i <= 1 ? 0 : i*log2f(i);

  srand(123);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      input[i * width + j] = rand() % 16;

  const size_t num_pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
  auto start = std::chrono::steady_clock::now();
  auto end = start;
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // Map inputs once and keep the output buffer resident on the device.
#pragma omp target data map(to: input[0:num_pixels], logTable[0:26]) \
                        map(alloc: output[0:num_pixels])
  {
    // Keep the working sets resident on the GPU across all kernel launches.
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      entropy(output, input, height, width);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel (baseline) execution time %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      entropy_opt<16, 16>(output, input, logTable, height, width);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel (optimized) execution time %f (s)\n", (time * 1e-9f) / repeat);

    // Copy the final GPU result back to the host exactly once.
#pragma omp target update from(output[0:num_pixels])
  }



  reference(output_ref, input, height, width);

  bool ok = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (fabsf(output[i * width + j] - output_ref[i * width + j]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input);
  free(output);
  free(output_ref);
  return 0;
}
