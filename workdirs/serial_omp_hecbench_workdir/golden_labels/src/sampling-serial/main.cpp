


#include <chrono>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

struct Dataset {
  int nrows_exact;
  int nrows_sampled;
  int ncols;
  int nrows_background;
  int max_samples;
  uint64_t seed;
};

typedef float T;

double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; 

  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}  

template <typename DataT, typename IdxT>
void sampled_rows_kernel(const IdxT* nsamples, float* X, const IdxT nrows_X,
                         const IdxT ncols, DataT* background,
                         const IdxT nrows_background, DataT* dataset,
                         const DataT* observation, uint64_t seed) {
  

  

  int bid = omp_get_team_num();
  int tid = omp_get_thread_num();

  int k_blk = nsamples[bid];

  

  if (tid < k_blk) {
    int rand_idx = (int)(LCG_random_double(&seed) * ncols);

    

    while (1) {
      float x;
            {
        x = X[2 * bid * ncols + rand_idx];
        X[2 * bid * ncols + rand_idx] = (float)1;
      }
      if (x == 0) break;
      rand_idx = (int)(LCG_random_double(&seed) * ncols);
    }; 
  }
  
  

  int col_idx = tid;
  while (col_idx < ncols) {
    

    int curr_X = (int)X[2 * bid * ncols + col_idx];
    X[(2 * bid + 1) * ncols + col_idx] = 1 - curr_X;

    for (int bg_row_idx = 2 * bid * nrows_background;
         bg_row_idx < 2 * bid * nrows_background + nrows_background;
         bg_row_idx += 1) {
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] =
          background[(bg_row_idx % nrows_background) * ncols + col_idx];
      } else {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      }
    }

    for (int bg_row_idx = (2 * bid + 1) * nrows_background;
         bg_row_idx <
         (2 * bid + 1) * nrows_background + nrows_background;
         bg_row_idx += 1) {
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      } else {
        

        dataset[bg_row_idx * ncols + col_idx] =
          background[(bg_row_idx) % nrows_background * ncols + col_idx];
      }
    }
    col_idx += omp_get_num_threads();
  }
}

int main( int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int i, j, k;

  

  const std::vector<Dataset> inputs = {
    {1000, 0, 2000, 10, 11, 1234ULL},
    {0, 1000, 2000, 10, 11, 1234ULL},
    {1000, 1000, 2000, 10, 11, 1234ULL}
  };

  for (auto params : inputs) {

    double time = 0.0;

    for (int r = 0; r < repeat; r++) {

      

      T *background = (T*) malloc (sizeof(T) * params.nrows_background * params.ncols);
      

      T *observation = (T*) malloc (sizeof(T) * params.ncols);
      

      int *nsamples = (int*) malloc (sizeof(int) * params.nrows_sampled/2);
      
      int nrows_X = params.nrows_exact + params.nrows_sampled;
      float *X = (float*) malloc (sizeof(float) * nrows_X * params.ncols);
      T* dataset = (T*) malloc (sizeof(T) * nrows_X * params.nrows_background * params.ncols);

      

      T sent_value = nrows_X * params.nrows_background * params.ncols * 100;
      for (i = 0; i < params.ncols; i++) {
        observation[i] = sent_value;
      }

      

      

      for (i = 0; i < params.nrows_background; i++) {
        for (j = 0; j < params.ncols; j++) {
          background[i * params.ncols + j] = (i * 2) + 1;
        }
      }

      

      for (i = 0; i <  nrows_X * params.ncols; i++) X[i] = (float)0.0;
      for (i = 0; i < params.nrows_exact; i++) {
        for (j = i; j < i + 2; j++) {
          X[i * params.ncols + j] = (float)1.0;
        }
      }

      

      

      for (i = 0; i < params.nrows_sampled / 2; i++) {
        nsamples[i] = params.max_samples - i % 2;
      }

      const int ncols = params.ncols;
      const int nrows_background = params.nrows_background;
      const int nrows_sampled = params.nrows_sampled;
      uint64_t seed = params.seed;

            {
        int nthreads = std::min(256, ncols);
        int nblks = nrows_X - nrows_sampled;
        

      
        auto start = std::chrono::steady_clock::now();

        if (nblks > 0) {
                    {
                        {
              int gid = omp_get_team_num(); 
              int col = omp_get_thread_num();
              int row = gid * ncols;
      
              while (col < ncols) {
                

                int curr_X = (int)X[row + col];
      
                

                for (int row_idx = gid * nrows_background;
                     row_idx < gid * nrows_background + nrows_background;
                     row_idx += 1) {
                  if (curr_X == 0) {
                    dataset[row_idx * ncols + col] =
                      background[(row_idx % nrows_background) * ncols + col];
                  } else {
                    dataset[row_idx * ncols + col] = observation[col];
                  }
                }
                

                col += omp_get_num_threads();
              }
            }
          }
        }
      
        if (nrows_sampled > 0) {
          nblks = nrows_sampled / 2;
                    {
                        {
               sampled_rows_kernel (
                  nsamples, &X[(nrows_X - nrows_sampled) * ncols], nrows_sampled, ncols,
                  background, nrows_background,
                  &dataset[(nrows_X - nrows_sampled) * nrows_background * ncols], observation,
                  seed);
            }
          }
        }

        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }

      

      

      bool test_sampled_X = true;
      j = 0;
      int counter;

      for (i = params.nrows_exact * params.ncols; i < nrows_X * params.ncols / 2;
          i += 2 * params.ncols) {
        

        counter = 0;
        for (k = i; k < i+params.ncols; k++)
          if (X[k] == 1) counter++;
        test_sampled_X = (test_sampled_X && (counter == nsamples[j]));

        

        

        counter = 0;
        for (k = i+params.ncols; k < i+2*params.ncols; k++)
          if (X[k] == 1) counter++;
        test_sampled_X = (test_sampled_X && (counter == (params.ncols - nsamples[j])));
        j++;
      }

      

      bool test_scatter_exact = true;
      for (i = 0; i < params.nrows_exact; i++) {
        for (j = i * params.nrows_background * params.ncols;
            j < (i + 1) * params.nrows_background * params.ncols;
            j += params.ncols) {
          counter = 0;
          for (k = j; k < j+params.ncols; k++)
            if (dataset[k] == sent_value) counter++; 

          

          test_scatter_exact = test_scatter_exact && (counter == 2);
          if (!test_scatter_exact) {
            printf("test_scatter_exact counter failed with: %d", counter);
            printf(", expected value was 2.\n");
            break;
          }
        }
        if (!test_scatter_exact) {
          break;
        }
      }

      

      bool test_scatter_sampled = true;

      

      

      int compliment_ctr = 0;
      for (i = params.nrows_exact;
          i < params.nrows_exact + params.nrows_sampled / 2; i++) {
        

        for (j = (i + compliment_ctr) * params.nrows_background * params.ncols;
            j <
            (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
            j += params.ncols) {

          counter = 0;
          for (k = j; k < j+params.ncols; k++)
            if (dataset[k] == sent_value) counter++; 

          test_scatter_sampled = test_scatter_sampled && (counter == nsamples[i - params.nrows_exact]);
          if (!test_scatter_sampled) {
            printf("test_scatter_sampled counter failed with: %d", counter);
            printf(", expected value was %d.\n",  nsamples[i - params.nrows_exact]);
            break;
          }
        }

        

        compliment_ctr++;
        for (j = (i + compliment_ctr) * params.nrows_background * params.ncols;
            j <
            (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
            j += params.ncols) {
          

          counter = 0;
          for (k = j; k < j+params.ncols; k++)
            if (dataset[k] == sent_value) counter++; 
          test_scatter_sampled = test_scatter_sampled &&
            (counter == params.ncols - nsamples[i - params.nrows_exact]);
          if (!test_scatter_sampled) {
            printf("test_scatter_sampled counter failed with: %d", counter);
            printf(", expected value was %d.\n", params.ncols - nsamples[i - params.nrows_exact]);
            break;
          }
        }
      }

      free(observation);
      free(background);
      free(X);
      free(nsamples);
      free(dataset);
    }

    printf("Average execution time of kernels: %f (us)\n", (time * 1e-3) / repeat);
  }

  return 0;
}