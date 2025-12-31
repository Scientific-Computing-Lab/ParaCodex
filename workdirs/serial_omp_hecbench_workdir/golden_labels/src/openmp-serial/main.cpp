


#include <stdio.h>
#include <stdlib.h>
#include "reference.h"

int main(int argc, char *argv[]) {

  printf("%s Starting...\n\n", argv[0]);
  const int repeat = atoi(argv[1]);

  int num_gpus = 1; 


  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("number of devices:\t%d\n", num_gpus);

  

  unsigned int nwords = num_gpus * 33554432;
  unsigned int nbytes = nwords * sizeof(int);
  int b = 3;   

  int *a = (int *)malloc(nbytes); 


  if (NULL == a) {
    printf("couldn't allocate CPU memory\n");
    return 1;
  }
  double overhead; 


  

  

  for (int i = 0; i < 2; i++) {
    for (int f = 1; f <= 32; f = f*2) {
      double start = omp_get_wtime();
      omp_set_num_threads(f * num_gpus); 
            {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        

        unsigned int nwords_per_kernel = nwords / num_cpu_threads;
        int *sub_a = a + cpu_thread_id * nwords_per_kernel;

        for (unsigned int n = 0; n < nwords_per_kernel; n++)
          sub_a[n] = n + cpu_thread_id * nwords_per_kernel;

                {
                    for (int idx = 0; idx < nwords_per_kernel; idx++) {
            for (int i = 0; i < repeat; i++)
              sub_a[idx] += i % b;
          }
        }
      }
      double end = omp_get_wtime();
      printf("Work took %f seconds with %d CPU threads\n", end - start, f*num_gpus);
      
      if (f == 1) {
        if (i == 0) 
          overhead = end - start;
        else
          overhead = overhead - (end - start);
      }
      

      bool bResult = correctResult(a, nwords, b, repeat);
      printf("%s\n", bResult ? "PASS" : "FAIL");
    }
  }
  printf("Runtime overhead of first run is %f seconds\n", overhead);

  free(a);
  return 0;
}