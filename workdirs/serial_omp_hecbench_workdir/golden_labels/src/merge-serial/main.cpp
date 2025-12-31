#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "kernels.h"

#define CSV 0
#if(CSV)
#define PS(X, S) std::cout << X << ", " << S << ", "; fflush(stdout);
#define PV(X) std::cout << X << ", "; fflush(stdout);
#else
#define PS(X, S) std::cout << X << " " << S <<" :\n"; fflush(stdout);
#define PV(X) std::cout << "\t" << #X << " \t: " << X << "\n"; fflush(stdout);
#endif



template<typename vec_t>
vec_t rand64() {
  vec_t rtn;
  do {
    uint32_t * rtn32 = (uint32_t *)&rtn;
    rtn32[0] = rand();
    if(sizeof(vec_t) > 4) rtn32[1] = rand();
  } while(!(rtn < getPositiveInfinity<vec_t>() &&
        rtn > getNegativeInfinity<vec_t>()));
  return rtn;
}



#define PADDING 1024
template<typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
void mergeType(const uint64_t size, const uint32_t runs) {
  

  std::vector<vec_t> hA (size + PADDING);
  std::vector<vec_t> hB (size + PADDING);
  std::vector<vec_t> hC (2*size + PADDING);
  std::vector<uint32_t> hD (2*(blocks+1));

  uint32_t errors = 0;

  vec_t *dA = hA.data(); 
  vec_t *dB = hB.data(); 
  vec_t *dC = hC.data(); 
  uint32_t *dpi = hD.data(); 


    {
    double total_time = 0.0;

    for(uint32_t r = 0; r < runs; r++) {

      

      for (uint64_t n = 0; n < size; n++) {
         hA[n] = rand64<vec_t>();
         hB[n] = rand64<vec_t>();
      }

      for (uint64_t n = size; n < size + PADDING; n++) {
        hA[n] = getPositiveInfinity<vec_t>();
        hB[n] = getPositiveInfinity<vec_t>();
      }

      std::sort(hA.begin(), hA.end());
      std::sort(hB.begin(), hB.end());

            
      auto start = std::chrono::steady_clock::now();

      

      workloadDiagonals<vec_t>(blocks, dA, size, dB, size, dpi);

      

      mergeSinglePath<vec_t,false,false> (blocks, threads,
          dA, size, dB, size, dpi, dC, size * 2);

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      total_time += time;

      

            for(uint32_t i = 1; i < size; i++) {
        errors += hC[i] < hC[i-1];
      }
    }

    PV(errors); 

    printf("%s. ", errors ? "FAIL" : "PASS");

    if (timing)
      printf("Average kernel execution time: %f (us).\n", (total_time * 1e-3f) / runs);
    else
      printf("Warmup run\n");
  }
}



template<uint32_t blocks, uint32_t threads>
void mergeAllTypes(const uint64_t size, const uint32_t runs) {
  PS("uint32_t", size)  mergeType<uint32_t, blocks, threads, false>(size, runs); printf("\n");
  PS("uint32_t", size)  mergeType<uint32_t, blocks, threads, true>(size, runs); printf("\n");

  PS("float",    size)  mergeType<float,    blocks, threads, false>(size, runs); printf("\n");
  PS("float",    size)  mergeType<float,    blocks, threads, true>(size, runs); printf("\n");

  PS("uint64_t", size)  mergeType<uint64_t, blocks, threads, false>(size, runs); printf("\n");
  PS("uint64_t", size)  mergeType<uint64_t, blocks, threads, true>(size, runs); printf("\n");

  PS("double", size)    mergeType<double,   blocks, threads, false>(size, runs); printf("\n");
  PS("double", size)    mergeType<double,   blocks, threads, true>(size, runs); printf("\n");
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length of the arrays> <runs>\n", argv[0]);
    return 1;
  }
  

  

  const uint64_t length = atol(argv[1]);

  const uint32_t runs = atoi(argv[2]);

  const int blocks = 112;
  const int threads = 128;  


  mergeAllTypes<blocks, threads>(length, runs);

  return 0;
}