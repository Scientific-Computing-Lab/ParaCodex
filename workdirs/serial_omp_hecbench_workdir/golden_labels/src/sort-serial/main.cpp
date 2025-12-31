#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

typedef unsigned int T;

template<typename T>
struct vec4 {
  T x;
  T y;
  T z;
  T w;
};

void verifySort(const T *keys, const size_t size)
{
  bool passed = true;
  for (size_t i = 0; i < size - 1; i++)
  {
    if (keys[i] > keys[i + 1])
    {
      passed = false;
#ifdef VERBOSE_OUTPUT
      std::cout << "Idx: " << i;
      std::cout << " Key: " << keys[i] << "\n";
#endif
      break;
    }
  }
  if (passed)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
}

int main(int argc, char** argv) 
{
  if (argc != 3) 
  {
    printf("Usage: %s <problem size> <number of passes>\n.", argv[0]);
    return -1;
  }

  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  

  int probSizes[4] = { 1, 8, 32, 64 };
  size_t size = probSizes[select];

  

  size = (size * 1024 * 1024) / sizeof(T);

  

  unsigned int bytes = size * sizeof(T);

  T* idata = (T*) malloc (bytes); 
  T* odata = (T*) malloc (bytes); 

  

  std::cout << "Initializing host memory." << std::endl;
  for (int i = 0; i < size; i++)
  {
    idata[i] = i % 16; 

    odata[i] = size - i;
  }

  std::cout << "Running benchmark with input array length " << size << std::endl;

  

  const size_t local_wsize  = 256;
  

  const size_t global_wsize = 16384; 
  

  const size_t num_work_groups = global_wsize / local_wsize;

  

  const int radix_width = 4; 

  

  const int num_digits = 16;

  T* isums = (T*) malloc (sizeof(T) * num_work_groups * num_digits);

    {
    double time = 0.0;

    for (int k = 0; k < passes; k++)
    {
      auto start = std::chrono::steady_clock::now();

      

      for (unsigned int shift = 0; shift < sizeof(T)*8; shift += radix_width)
      {
        


        

        

        


        

        

        bool even = ((shift / radix_width) % 2 == 0) ? true : false;

        T *in = even ? idata : odata;
        T *out = even ? odata : idata;

                {
          T lmem[local_wsize];
                    {
            #include "sort_reduce.h"
          }
        }

#ifdef DEBUG
        for (int i = 0; i < num_work_groups * num_digits; i++)
          printf("reduce: %d: %d\n", shift, isums[i]);
#endif

                {
          T lmem[local_wsize*2];
          T s_seed;
                    {
            #include "sort_top_scan.h"
          }
        }

#ifdef DEBUG
        for (int i = 0; i < num_work_groups * num_digits; i++)
          printf("top-scan: %d: %d\n", shift, isums[i]);
#endif

                {
          T lmem[local_wsize*2];
          T l_scanned_seeds[16];
          T l_block_counts[16];
                    {
            #include "sort_bottom_scan.h"
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }  


    printf("Average elapsed time per pass %lf (s)\n", time * 1e-9 / passes);
  }

  verifySort(odata, size);

  free(idata);
  free(isums);
  free(odata);
  return 0;
}