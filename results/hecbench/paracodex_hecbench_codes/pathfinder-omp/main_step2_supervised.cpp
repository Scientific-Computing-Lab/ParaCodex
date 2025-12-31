#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include "gate.h"

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

  int* outputBuffer = (int*)calloc(16384, sizeof(int));

  double offload_start = get_time();

  int* gpuSrc = (int*) malloc (sizeof(int)*cols);
  memcpy(gpuSrc, data, cols*sizeof(int));
  int tileCount = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height)
  {
    tileCount++;
  }

  double kernel_start = get_time();

  if (tileCount % 2 == 1)
  {
#pragma omp target teams distribute parallel for map(tofrom: gpuSrc[0:cols])
    for (int x = 0; x < cols; ++x)
    {
      gpuSrc[x] = 0;
    }
  }

  double kernel_end = get_time();
  printf("Total kernel execution time: %lf (s)\n", kernel_end - kernel_start);

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

  GATE_CHECKSUM_U32("gpuSrc", (const uint32_t*)gpuSrc, cols);
  GATE_CHECKSUM_U32("outputBuffer", (const uint32_t*)outputBuffer, 16384);

  delete[] data;
  delete[] wall;
  delete[] result;
  free(outputBuffer);
  free(gpuSrc);

  return EXIT_SUCCESS;
}
