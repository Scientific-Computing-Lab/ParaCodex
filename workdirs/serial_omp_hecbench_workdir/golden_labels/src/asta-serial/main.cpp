


#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>  

#include <chrono>

#include "support/common.h"
#include "support/verify.h"




struct Params {

  int device;
  int n_gpu_threads;
  int n_gpu_blocks;
  int n_warmup;
  int n_reps;
  int m;
  int n;
  int s;

  Params(int argc, char **argv) {
    device        = 0;
    n_gpu_threads  = 64;
    n_gpu_blocks = 16;
    n_warmup      = 10;
    n_reps        = 100;
    m             = 197;
    n             = 35588;
    s             = 32;
    int opt;
    while((opt = getopt(argc, argv, "hd:i:g:w:r:m:n:s:")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'i': n_gpu_threads  = atoi(optarg); break;
        case 'g': n_gpu_blocks = atoi(optarg); break;
        case 'w': n_warmup      = atoi(optarg); break;
        case 'r': n_reps        = atoi(optarg); break;
        case 'm': m             = atoi(optarg); break;
        case 'n': n             = atoi(optarg); break;
        case 's': s             = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./main [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -i <I>    # of device threads per block (default=64)"
        "\n    -g <G>    # of device blocks (default=16)"
        "\n    -w <W>    # of warmup iterations (default=10)"
        "\n    -r <R>    # of repetition iterations (default=100)"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -m <M>    matrix height (default=197)"
        "\n    -n <N>    matrix width (default=35588)"
        "\n    -s <M>    super-element size (default=32)"
        "\n");
  }
};



void read_input(FP *x_vector, const Params &p) {
  int tiled_n = divceil(p.n, p.s);
  int in_size = p.m * tiled_n * p.s;
  srand(5432);
  for(int i = 0; i < in_size; i++) {
    x_vector[i] = ((FP)(rand() % 100) / 100);
  }
}



int main(int argc, char **argv) {

  const Params p(argc, argv);
  int blocks = p.n_gpu_blocks;
  int threads = p.n_gpu_threads;
  const int max_gpu_threads = 256;
  assert(threads <= max_gpu_threads && 
          "The thread block size is greater than the maximum thread block size that can be used on this device");


  

  int tiled_n       = divceil(p.n, p.s);
  int in_size       = p.m * tiled_n * p.s;
  int finished_size = p.m * tiled_n;

  size_t in_size_bytes = in_size * sizeof(FP);
  size_t finished_size_bytes = finished_size * sizeof(int);

  FP *h_in_out = (FP *)malloc(in_size_bytes);
  int *h_finished = (int *)malloc(finished_size_bytes);
  int *h_head = (int *)malloc(sizeof(int));
  FP *h_in_backup = (FP *)malloc(in_size_bytes);

  

  read_input(h_in_out, p);
  memcpy(h_in_backup, h_in_out, in_size_bytes); 


  const int A = p.m;
  const int B = tiled_n;
  const int b = p.s;

  {
    double time = 0;

    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

      memcpy(h_in_out, h_in_backup, in_size_bytes);
      memset((void *)h_finished, 0, finished_size_bytes);
      h_head[0] = 0;





     auto start = std::chrono::steady_clock::now();

      {
        int lmem[2];
        {
          const int tid = omp_get_thread_num();
          int       m   = A * B - 1;

          if(tid == 0) {

            lmem[1] = h_head[0]++;
          }

          while(lmem[1] < m) {
            int next_in_cycle = (lmem[1] * A) - m * (lmem[1] / B);
            if(next_in_cycle == lmem[1]) {
              if(tid == 0) {

                lmem[1] = h_head[0]++;
              }
              continue;
            }
            FP   data1, data2, data3, data4;
            int i = tid;
            if(i < b)
              data1 = h_in_out[lmem[1] * b + i];
            i += omp_get_num_threads();
            if(i < b)
              data2 = h_in_out[lmem[1] * b + i];
            i += omp_get_num_threads();
            if(i < b)
              data3 = h_in_out[lmem[1] * b + i];
            i += omp_get_num_threads();
            if(i < b)
              data4 = omp_get_num_threads();

            if(tid == 0) {
              lmem[0] = h_finished[lmem[1]];
            }

            for(; lmem[0] == 0; next_in_cycle = (next_in_cycle * A) - m * (next_in_cycle / B)) {
              FP backup1, backup2, backup3, backup4;
              i = tid;
              if(i < b)
                backup1 = h_in_out[next_in_cycle * b + i];
              i += omp_get_num_threads();
              if(i < b)
                backup2 = h_in_out[next_in_cycle * b + i];
              i += omp_get_num_threads();
              if(i < b)
                backup3 = h_in_out[next_in_cycle * b + i];
              i += omp_get_num_threads();
              if(i < b)
                backup4 = h_in_out[next_in_cycle * b + i];

              if(tid == 0) {
                {
                  lmem[0] = h_finished[next_in_cycle];
                  h_finished[next_in_cycle] = (int)1;
                }
              }

              if(!lmem[0]) {
                i = tid;
                if(i < b)
                  h_in_out[next_in_cycle * b + i] = data1;
                i += omp_get_num_threads();
                if(i < b)
                  h_in_out[next_in_cycle * b + i] = data2;
                i += omp_get_num_threads();
                if(i < b)
                  h_in_out[next_in_cycle * b + i] = data3;
                i += omp_get_num_threads();
                if(i < b)
                  h_in_out[next_in_cycle * b + i] = data4;
              }
              i = tid;
              if(i < b)
                data1 = backup1;
              i += omp_get_num_threads();
              if(i < b)
                data2 = backup2;
              i += omp_get_num_threads();
              if(i < b)
                data3 = backup3;
              i += omp_get_num_threads();
              if(i < b)
                data4 = backup4;
            }

            if(tid == 0) { 

              lmem[1] = h_head[0]++;
            }
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      if (rep >= p.n_warmup) 
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    


    }

    printf("Average kernel execution time %lf (s)\n", (time * 1e-9) / p.n_reps);
  }

  

  int status = verify(h_in_out, h_in_backup, tiled_n * p.s, p.m, p.s);
  printf("%s\n", (status == 0) ? "PASS" : "FAIL");

  

  free(h_in_out);
  free(h_finished);
  free(h_head);
  free(h_in_backup);

  return 0;
}