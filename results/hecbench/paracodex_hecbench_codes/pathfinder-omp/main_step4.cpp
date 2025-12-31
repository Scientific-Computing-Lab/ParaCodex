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
  const int threadsPerTeam = 256; // 8 warps fits well on the Ada-based RTX 4060 Laptop GPU
  int copyTeams = (totalCells + threadsPerTeam - 1) / threadsPerTeam;
  if (copyTeams < 1)
  {
    copyTeams = 1;
  }
  int* randomValues = (int*)malloc(sizeof(int) * totalCells);
  for (int idx = 0; idx < totalCells; ++idx)
  {
    randomValues[idx] = rand() % 10;
  }

  int* outputBuffer = (int*)calloc(16384, sizeof(int));
  int* gpuSrc = (int*) malloc (sizeof(int)*cols);

  int tileCount = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height)
  {
    tileCount++;
  }
  const int needsSrcClear = tileCount & 1;

  double offload_start = get_time();
  double kernel_start = offload_start;
#pragma omp target teams distribute parallel for simd num_teams(copyTeams) thread_limit(threadsPerTeam) \
    map(to: randomValues[0:totalCells]) map(from: data[0:totalCells]) map(from: gpuSrc[0:cols])
  for (int idx = 0; idx < totalCells; ++idx)
  {
    const int val = randomValues[idx];
    data[idx] = val;
    if (idx < cols)
    {
      gpuSrc[idx] = needsSrcClear ? 0 : val;
    }
  }
  double kernel_end = get_time();
  printf("Total kernel execution time: %lf (s)\n", kernel_end - kernel_start);

  double offload_end = kernel_end;
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

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
