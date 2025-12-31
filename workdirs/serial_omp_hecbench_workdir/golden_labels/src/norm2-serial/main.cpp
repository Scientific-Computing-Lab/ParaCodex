#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>

#define max(a, b) (a < b ? b : a)

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  

  const int repeat = max(1, atoi(argv[1]));

  bool ok = true;

  

  float* h_result = (float*) malloc (repeat * sizeof(float));
  if (h_result == nullptr) {
    printf ("output on host allocation failed");
    return 1;
  }

  for (int n = 512*1024; n <= 1024*1024*512; n = n * 2) {
    int i, j;
    size_t size = n * sizeof(float);
    float *a = (float *) malloc (size);
    if (a == nullptr) {
      printf ("input on host allocation failed");
      break;
    }

    

    double gold = 0.0;  

    for (i = 0; i < n; i++) {
      a[i] = (float)((i+1) % 7);
      gold += a[i]*a[i];
    }
    gold = sqrt(gold);

        {
      auto kstart = std::chrono::steady_clock::now();

      for (j = 0; j < repeat; j++) {
        double sum = 0.0;
                for (i = 0; i < n; i++) {
          float t = a[i]*a[i];
          sum += t;
        }
        h_result[j] = sqrt(sum);  

      }

      auto kend = std::chrono::steady_clock::now();
      auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
      printf("#elements = %.2f M: average omp nrm2 execution time = %f (us), performance = %f (Gop/s)\n",
             n / (1024.f*1024.f), (ktime * 1e-3f) / repeat, 1.f * (2*n+1) * repeat / ktime);
    }

    

    for (j = 0; j < repeat; j++) 
     if (fabsf((float)gold - h_result[j]) > 1e-3f) {
       printf("FAIL at iteration %d: gold=%f actual=%f for %d elements\n",
              j, (float)gold, h_result[j], i);
       ok = false;
       break;
     }

    free(a);
  }

  free(h_result);

  if (ok) printf("PASS\n");
  return 0;
}