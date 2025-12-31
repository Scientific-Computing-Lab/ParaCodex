






#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <chrono>







typedef unsigned char uchar_misaligned;

typedef unsigned short int ushort_misaligned;

typedef struct
{
  unsigned char r, g, b, a;
} uchar4_misaligned;

typedef struct
{
  unsigned int l, a;
} uint2_misaligned;

typedef struct
{
  unsigned int r, g, b;
} uint3_misaligned;

typedef struct
{
  unsigned int r, g, b, a;
} uint4_misaligned;

typedef struct
{
  uint4_misaligned c1, c2;
} uint8_misaligned;








typedef struct __attribute__((__aligned__(4)))
{
  unsigned char r, g, b, a;
}
uchar4_aligned;

typedef unsigned int uint_aligned;

typedef struct __attribute__((__aligned__(8)))
{
  unsigned int l, a;
}
uint2_aligned;

typedef struct __attribute__((__aligned__(16)))
{
  unsigned int r, g, b;
}
uint3_aligned;

typedef struct __attribute__((__aligned__(16)))
{
  unsigned int r, g, b, a;
}
uint4_aligned;




















typedef struct __attribute__((__aligned__(16)))
{
  uint4_aligned c1, c2;
}
uint8_aligned;











int iDivUp(int a, int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}



int iDivDown(int a, int b)
{
  return a / b;
}



int iAlignUp(int a, int b)
{
  return (a % b != 0) ? (a - a % b + b) : a;
}



int iAlignDown(int a, int b)
{
  return a - a % b;
}


















template<class TData> int testCPU(
    TData *h_odata,
    TData *h_idata,
    int numElements,
    int packedElementSize
    )
{
  for (int pos = 0; pos < numElements; pos++)
  {
    TData src = h_idata[pos];
    TData dst = h_odata[pos];

    for (int i = 0; i < packedElementSize; i++)
      if (((char *)&src)[i] != ((char *)&dst)[i])
      {
        return 0;
      }
  }
  return 1;
}











const int       MEM_SIZE = 50000000;
const int NUM_ITERATIONS = 1000;



unsigned char *h_idataCPU;

template<class TData> int runTest(
  unsigned char *d_idata,
  unsigned char *d_odata,
  int packedElementSize,
  int memory_size)
{
  const int totalMemSizeAligned = iAlignDown(memory_size, sizeof(TData));
  const int         numElements = iDivDown(memory_size, sizeof(TData));

  

    for (int i = 0; i < memory_size; i++) 
    d_odata[i] = 0;
  

  

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
        for (int pos = 0; pos < numElements; pos++)
    {
      reinterpret_cast<TData*>(d_odata)[pos] = 
        reinterpret_cast<TData*>(d_idata)[pos];
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double gpuTime = (double)elapsed_seconds.count() / NUM_ITERATIONS;

  printf(
      "Avg. time: %f ms / Copy throughput: %f GB/s.\n", gpuTime * 1000,
      (double)totalMemSizeAligned / (gpuTime * 1073741824.0)
        );

  

  
  int flag = testCPU(
      (TData *)d_odata,
      (TData *)h_idataCPU,
      numElements,
      packedElementSize
      );

  printf(flag ? "\tTEST OK\n" : "\tTEST FAILURE\n");

  return !flag;
}

int main(int argc, char **argv)
{
  int i, nTotalFailures = 0;

  printf("[%s] - Starting...\n", argv[0]);

  printf("Allocating memory...\n");
  int   MemorySize = (int)(MEM_SIZE) & 0xffffff00; 

  h_idataCPU = (unsigned char *)malloc(MemorySize);
  unsigned char *d_odata = (unsigned char *)malloc(MemorySize);

  printf("Generating host input data array...\n");

  for (i = 0; i < MemorySize; i++)
  {
    h_idataCPU[i] = (i & 0xFF) + 1;
  }

  printf("Uploading input data to GPU memory...\n");
  unsigned char *d_idata = h_idataCPU;

{

  printf("Testing misaligned types...\n");
  printf("uchar_misaligned...\n");
  nTotalFailures += runTest<uchar_misaligned>(d_idata, d_odata, 1, MemorySize);

  printf("uchar4_misaligned...\n");
  nTotalFailures += runTest<uchar4_misaligned>(d_idata, d_odata, 4, MemorySize);

  printf("uchar4_aligned...\n");
  nTotalFailures += runTest<uchar4_aligned>(d_idata, d_odata, 4, MemorySize);

  printf("ushort_misaligned...\n");
  nTotalFailures += runTest<ushort_misaligned>(d_idata, d_odata, 2, MemorySize);

  printf("uint_aligned...\n");
  nTotalFailures += runTest<uint_aligned>(d_idata, d_odata, 4, MemorySize);

  printf("uint2_misaligned...\n");
  nTotalFailures += runTest<uint2_misaligned>(d_idata, d_odata, 8, MemorySize);

  printf("uint2_aligned...\n");
  nTotalFailures += runTest<uint2_aligned>(d_idata, d_odata, 8, MemorySize);

  printf("uint3_misaligned...\n");
  nTotalFailures += runTest<uint3_misaligned>(d_idata, d_odata, 12, MemorySize);

  printf("uint3_aligned...\n");
  nTotalFailures += runTest<uint3_aligned>(d_idata, d_odata, 12, MemorySize);

  printf("uint4_misaligned...\n");
  nTotalFailures += runTest<uint4_misaligned>(d_idata, d_odata, 16, MemorySize);

  printf("uint4_aligned...\n");
  nTotalFailures += runTest<uint4_aligned>(d_idata, d_odata, 16, MemorySize);

  printf("uint8_misaligned...\n");
  nTotalFailures += runTest<uint8_misaligned>(d_idata, d_odata, 32, MemorySize);

  printf("uint8_aligned...\n");
  nTotalFailures += runTest<uint8_aligned>(d_idata, d_odata, 32, MemorySize);

  printf("\n[alignedTypes] -> Test Results: %d Failures\n", nTotalFailures);

  printf("Shutting down...\n");
}
  free(d_odata);
  free(h_idataCPU);

  if (nTotalFailures != 0)
  {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}