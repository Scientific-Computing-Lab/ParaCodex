#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_elems = atoi(argv[1]);
  const int block_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
    
  std::vector<int> input (num_elems);
  std::vector<int> output (num_elems);

  

  for (int i = 0; i < num_elems; i++) {
    input[i] = i - num_elems / 2;
  }

  std::mt19937 g;
  g.seed(19937);
  std::shuffle(input.begin(), input.end(), g);

  int *data_to_filter = input.data();
  int *filtered_data = output.data();
  int nres[1];

    {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      nres[0] = 0;
      

            {
        int l_n;
                {
          int i = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num() ;
          if (omp_get_thread_num() == 0)
            l_n = 0;
                    int d, pos;
        
          if(i < num_elems) {
            d = data_to_filter[i];
            if(d > 0) {
                            pos = l_n++;
            }
          }
            
          

          if (omp_get_thread_num() == 0) {
            

             int old;
                          {
                old = nres[0];
                nres[0] += l_n; 
             }
             l_n = old;
          }
                  
          

          if(i < num_elems && d > 0) {
            pos += l_n; 

            filtered_data[pos] = d;
          }
                  }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %lf (ms)\n", (time * 1e-6) / repeat);
  }

  std::vector<int> h_output (num_elems);

  

  int h_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (input[i] > 0) {
      h_output[h_flt_count++] = input[i];
    }
  }

  

  std::sort(h_output.begin(), h_output.begin() + h_flt_count);
  std::sort(output.begin(), output.begin() + nres[0]);

  bool equal = (h_flt_count == nres[0]) && 
               std::equal(h_output.begin(),
                          h_output.begin() + h_flt_count, output.begin());

  printf("\nFilter using shared memory %s \n",
         equal ? "PASS" : "FAIL");

  return 0;
}