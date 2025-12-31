#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "gate.h"

#define RADIUS 7
#define BLOCK_SIZE 256

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int size = length;
  int pad_size = (length + RADIUS);
  const int num_blocks = length / BLOCK_SIZE;
  const int launch_teams = num_blocks > 0 ? num_blocks : 1;
  const int threads_per_team = BLOCK_SIZE;

  int* a = (int *)malloc(pad_size*sizeof(int));
  int* b = (int *)malloc(size*sizeof(int));

  for (int i = 0; i < length+RADIUS; i++) a[i] = i;

  auto start = std::chrono::steady_clock::now();
  const int num_devices = omp_get_num_devices();
  const bool use_gpu = num_devices > 0;

  // Persist input/output buffers on the device to amortize transfers across repeats.
  #pragma omp target data map(to: a[0:pad_size]) map(from: b[0:size]) if (use_gpu)
  {
    for (int i = 0; i < repeat; i++) {
      // Offload the block-based stencil sweep to the GPU.
      // Hint the runtime with an occupancy-friendly launch configuration for the RTX 4060.
      #pragma omp target teams distribute num_teams(launch_teams) thread_limit(threads_per_team) \
          map(present: a[0:pad_size], b[0:size]) if (use_gpu)
      for (int block = 0; block < num_blocks; ++block) {
        const int block_base = block * BLOCK_SIZE;
        int temp[BLOCK_SIZE + 2 * RADIUS];
        #pragma omp parallel
        {
          #pragma omp for
          for (int j = 0; j < BLOCK_SIZE; j++) {
            const int gindex = block_base + j;
            temp[j + RADIUS] = a[gindex];
            if (j < RADIUS) {
              temp[j] = (gindex < RADIUS) ? 0 : a[gindex - RADIUS];
              temp[j + RADIUS + BLOCK_SIZE] = a[gindex + BLOCK_SIZE];
            }
          }

          #pragma omp for
          for (int j = 0; j < BLOCK_SIZE; j++) {
            int result = 0;
            #pragma omp simd reduction(+:result)
            for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
              result += temp[j + RADIUS + offset];
            b[block_base + j] = result;
          }
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  GATE_CHECKSUM_BYTES("a", a, pad_size * sizeof(int));
  GATE_CHECKSUM_BYTES("b", b, size * sizeof(int));

  bool ok = true;
  for (int i = 0; i < 2*RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i+2*RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2*RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i-RADIUS; j <= i+RADIUS; j++)
      s += a[j];
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(a);
  free(b);
  return 0;
}
