#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "gate.h"

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

void benchmark_func(float *cd, int grid_dim, int block_dim, int compute_iterations) { 
      
    { 
        {
      const unsigned int blockSize = block_dim;
      const int stride = blockSize;
      int idx = omp_get_team_num()*blockSize*granularity + omp_get_thread_num();
      const int big_stride = omp_get_num_teams()*blockSize*granularity;
      float tmps[granularity];
      for(int k=0; k<fusion_degree; k++) {
                for(int j=0; j<granularity; j++) {
          

          tmps[j] = cd[idx+j*stride+k*big_stride];

          

          for(int i=0; i<compute_iterations; i++)
            tmps[j] = tmps[j]*tmps[j]+(float)seed;
        }

        

        float sum = 0;
                for(int j=0; j<granularity; j+=2)
          sum += tmps[j]*tmps[j+1];

                for(int j=0; j<granularity; j++)
          cd[idx+k*big_stride] = sum;
      }
    }
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc (size*sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

    {
    

    for (int i = 0; i < repeat; i++) {
      benchmark_func(cd, grid_dim, block_dim, i);
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      benchmark_func(cd, grid_dim, block_dim, i);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %f (s)\n", time * 1e-9f);
  }

  

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd[i] != 0) {
      if (fabsf(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd[i]);
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  GATE_STATS_F32("cd", cd, size);

  free(cd);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE*sizeof(float);

  printf("Buffer size: %dMB\n", datasize/(1024*1024));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}
