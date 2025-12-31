#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

using namespace std;

#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define IN_RANGE(x, min, max)  ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

int main(int argc, char** argv)
{
  int   rows, cols;
  int*  data;
  int** wall;
  int*  result;
  int   pyramid_height;

  if (argc == 4)
  {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  }
  else
  {
    printf("Usage: %s <column length> <row length> <pyramid_height>\n", argv[0]);

    exit(0);
  }

  data = new int[rows * cols];
  wall = new int*[rows];
  for (int n = 0; n < rows; n++)
  {
    wall[n] = data + cols * n;
  }
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  const int totalCells = rows * cols;
  int* randomValues = (int*)malloc(sizeof(int) * totalCells);
  for (int idx = 0; idx < totalCells; ++idx)
  {
    randomValues[idx] = rand() % 10;
  }

#pragma omp target teams distribute parallel for collapse(2) map(to: randomValues[0:totalCells]) map(from: data[0:totalCells])
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      data[i * cols + j] = randomValues[i * cols + j];
    }
  }
  free(randomValues);
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%d ", wall[i][j]);
    }
    printf("\n");
  }
#endif

  const int borderCols = (pyramid_height) * HALO;

  const int size = rows * cols;
  const size_t wallSpan = (size_t)(rows - 1) * cols;

  const int lws = 250;
  const int gws = size/lws;

  int theHalo = HALO;
  int* outputBuffer = (int*)calloc(16384, sizeof(int));

  double offload_start = get_time();

  int* gpuWall = data+cols;

  int* gpuSrc = (int*) malloc (sizeof(int)*cols);
  int* gpuResult = (int*) malloc (sizeof(int)*cols);
  memcpy(gpuSrc, data, cols*sizeof(int));

#pragma omp target data map(to: gpuWall[0:wallSpan]) \
    map(tofrom: gpuSrc[0:cols]) map(alloc: gpuResult[0:cols]) \
    map(tofrom: outputBuffer[0:16384])
  {
    // Keep long-lived arrays resident on the GPU to avoid per-iteration transfers.
    double kstart = 0.0;

    for (int t = 0; t < rows - 1; t += pyramid_height)
    {
      if (t == pyramid_height) {
        kstart = get_time();
      }

      int iteration = MIN(pyramid_height, rows-t-1);

      int small_block_cols = lws - (iteration*theHalo*2);
      if (small_block_cols < 1)
      {
        small_block_cols = 1;
      }
      int numBlocks = (cols + small_block_cols - 1) / small_block_cols;

#pragma omp target teams distribute map(present: gpuSrc[0:cols], gpuResult[0:cols], \
    gpuWall[0:wallSpan], outputBuffer[0:16384]) \
    thread_limit(lws) num_teams(numBlocks) firstprivate(iteration, borderCols, cols, theHalo, t)
      for (int bx = 0; bx < numBlocks; ++bx)
      {
        int prev[lws];
        int result[lws];
#pragma omp parallel num_threads(lws) shared(prev, result)
        {
          int BLOCK_SIZE = omp_get_num_threads();
          int tx = omp_get_thread_num();

          int small_block_cols_local = BLOCK_SIZE - (iteration*theHalo*2);
          if (small_block_cols_local < 1)
          {
            small_block_cols_local = 1;
          }

          int blkX = (small_block_cols_local*bx) - borderCols;
          int blkXmax = blkX+BLOCK_SIZE-1;

          int xidx = blkX+tx;

          int validXmin = (blkX < 0) ? -blkX : 0;
          int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

          int W = tx-1;
          int E = tx+1;

          W = (W < validXmin) ? validXmin : W;
          E = (E > validXmax) ? validXmax : E;

          bool isValid = IN_RANGE(tx, validXmin, validXmax);

          if(IN_RANGE(xidx, 0, cols-1))
          {
            prev[tx] = gpuSrc[xidx];
          }

          bool computed;
          for (int i = 0; i < iteration; i++)
          {
            computed = false;

            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid )
            {
              computed = true;
              int left = prev[W];
              int up = prev[tx];
              int right = prev[E];
              int shortest = MIN(left, up);
              shortest = MIN(shortest, right);

              int index = cols*(t+i)+xidx;
              result[tx] = shortest + gpuWall[index];

              if (tx==11 && i==0)
              {
                int bufIndex = gpuSrc[xidx];
                outputBuffer[bufIndex] = 1;
              }
            }

            if(i==iteration-1)
            {
              break;
            }

            if(computed)
            {
              prev[tx] = result[tx];
            }
          }

          if (computed)
          {
            gpuResult[xidx] = result[tx];
          }
        }
      }
      int *temp = gpuResult;
      gpuResult = gpuSrc;
      gpuSrc = temp;
    }

#pragma omp target update from(gpuSrc[0:cols])
    double kend = get_time();
    printf("Total kernel execution time: %lf (s)\n", kend - kstart);

  }

  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

  outputBuffer[16383] = '\0';

#ifdef BENCH_PRINT
  for (int i = 0; i < cols; i++)
    printf("%d ", data[i]);
  printf("\n");
  for (int i = 0; i < cols; i++)
    printf("%d ", gpuSrc[i]);
  printf("\n");
#endif

  delete[] data;
  delete[] wall;
  delete[] result;
  free(outputBuffer);
  free(gpuSrc);
  free(gpuResult);

  return EXIT_SUCCESS;
}
