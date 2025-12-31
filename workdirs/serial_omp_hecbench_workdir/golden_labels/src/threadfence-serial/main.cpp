#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

void sum (
    const int teams,
    const int blocks,
    const float*__restrict array,
    const int N,
    unsigned int *__restrict count,
    volatile float*__restrict result)
{
    {
    bool isLastBlockDone;
    float partialSum;
        {
      

      unsigned int bid = omp_get_team_num();
      unsigned int num_blocks = teams;
      unsigned int block_size = blocks;
      unsigned int lid = omp_get_thread_num();
      unsigned int gid = bid * block_size + lid;

      if (lid == 0) partialSum = 0;
      
      if (gid < N) {
                partialSum += array[gid];
      }

      
      if (lid == 0) {

        

        

        

        

        

        

        

        result[bid] = partialSum;

        

        unsigned int value;
                value = (*count)++;

        

        

        isLastBlockDone = (value == (num_blocks - 1));
      }

      

      

      
      if (isLastBlockDone) {

        

        

        if (lid == 0) partialSum = 0;
        
        for (int i = lid; i < num_blocks; i += block_size) {
                    partialSum += result[i];
        }

        
        if (lid == 0) {

          

          

          

          

          result[0] = partialSum;
          *count = 0;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int N = atoi(argv[2]);

  const int blocks = 256;
  const int grids = (N + blocks - 1) / blocks;

  float* h_array = (float*) malloc (N * sizeof(float));

  float* h_result = (float*) malloc (grids * sizeof(float));

  unsigned int* h_count = (unsigned int*) malloc (sizeof(unsigned int));
  h_count[0] = 0;

  bool ok = true;
  double time = 0.0;

  for (int i = 0; i < N; i++) h_array[i] = -1.f;
  
    {
    for (int n = 0; n < repeat; n++) {
  
      

  
      auto start = std::chrono::steady_clock::now();

      sum (grids, blocks, h_array, N, h_count, h_result);

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
        
      if (h_result[0] != -1.f * N) {
        ok = false;
        break;
      }
    }
  }

  if (ok) printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(h_array);
  free(h_count);
  free(h_result);

  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}