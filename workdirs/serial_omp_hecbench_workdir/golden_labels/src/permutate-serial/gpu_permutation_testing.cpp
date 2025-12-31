


#include "header.h"
#include "kernel_functions.hpp"



bool gpu_permutation_testing(double *gpu_runtime, uint32_t *counts, double *results,
                             double mean, double median, uint8_t *data, uint32_t size,
                             uint32_t len, uint32_t N, uint32_t num_block, uint32_t num_thread)
{
  uint32_t i;
  uint8_t num_runtest = 0;
  uint32_t loop = 10000 / N;
  if ((10000 % N) != 0)  loop++;
  uint32_t blen = 0;
  if (size == 1) {
    blen = len / 8;
    if ((len % 8) != 0)  blen++;
  }
  size_t Nlen = (size_t)N * len;
  size_t Nblen = (size_t)N * blen;

  uint8_t *Ndata = (uint8_t *) malloc (Nlen);

  

  uint8_t *bNdata = (uint8_t*) malloc (Nblen);

  

    {
    

    auto start = std::chrono::steady_clock::now();

    

    for (i = 0; i < loop; i++) {
      if (size == 1) {
        binary_shuffling_kernel(Ndata, bNdata, data, len, blen, N, num_block, num_thread);

        binary_statistical_tests_kernel(counts, results, mean, median, Ndata,
                                        bNdata, size, len, blen, N, num_block, num_thread);

        

        
        num_runtest = 0;
        for (int t = 0; t < 18; t++) {
          if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
            num_runtest++;
        }
        if (num_runtest == 18)
          break;
      }
      else {
        shuffling_kernel(Ndata, data, len, N, num_block, num_thread);

        statistical_tests_kernel(counts, results, mean, median, Ndata,
                                 size, len, N, num_block, num_thread);

        

        
        num_runtest = 0;
        for (int t = 0; t < 18; t++) {
          if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
            num_runtest++;
        }
        if (num_runtest == 18)
          break;
      }
    }

    

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    

    *gpu_runtime = (double)time * 1e-9;

  } 


  free(Ndata);
  free(bNdata);

  if (num_runtest == 18) 

    return true;
  else 

    return false;
}