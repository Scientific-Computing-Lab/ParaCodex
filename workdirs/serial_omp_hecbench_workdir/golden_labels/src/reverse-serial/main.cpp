#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: ./%s <iterations>\n", argv[0]);
    return 1;
  }

  

  const int iteration = atoi(argv[1]);

  

  const int len = 256;
  const int elem_size = len * sizeof(int);

  

  int test[len];

  

  int error = 0;
  int gold_odd[len];
  int gold_even[len];

  for (int i = 0; i < len; i++) {
    gold_odd[i] = len-i-1;
    gold_even[i] = i;
  }

  std::default_random_engine generator (123);
  

  std::uniform_int_distribution<int> distribution(100, 9999);

  long time = 0;

    {
    for (int i = 0; i < iteration; i++) {
      const int count = distribution(generator);

      memcpy(test, gold_even, elem_size);
      
      auto start = std::chrono::steady_clock::now();

      for (int j = 0; j < count; j++) {
                {
          int s[len];
                    {
            int t = omp_get_thread_num();
            s[t] = test[t];
                        test[t] = s[len-t-1];
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      
      if (count % 2 == 0)
        error = memcmp(test, gold_even, elem_size);
      else
        error = memcmp(test, gold_odd, elem_size);
      
      if (error) break;
    }
  }

  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);
  printf("%s\n", error ? "FAIL" : "PASS");

  return 0;
}