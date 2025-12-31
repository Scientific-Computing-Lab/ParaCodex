


#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>

using namespace std::chrono;

#define warp_size 32
#define NBytes 8

inline uint lsr(uint x, int sa) {
  if(sa > 0 && sa < 32) return (x >> sa);
  return x;
}

inline uint lsl(uint x, int sa) {
  if (sa > 0 && sa < 32) return (x << sa);
  return x;
}

inline uint set_bit(uint &data, int y) {
  data |= lsl(1, y);
  return data;
}



uint popcnt( uint x )
{
  x -= ((x >> 1) & 0x55555555);
  x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
  x = (((x >> 4) + x) & 0x0f0f0f0f);
  x += (x >> 8);
  x += (x >> 16);
  return x & 0x0000003f;
}

inline int __clz( int x )
{
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return 32 - popcnt(x);
}

#include "kernel.h"
#include "reference.h"

int main(int argc, const char * const argv[])
{
  if (argc != 5) {
    printf("Usage: ./%s [ReadLength] [ReadandRefFile] [#reads] [repeat]\n", argv[0]);
    exit(-1);
  }

  int ReadLength = atoi(argv[1]);

  int NumReads = atoi(argv[3]); 

  int repeat = atoi(argv[4]);
  int Size_of_uint_in_Bit = 32; 


  FILE * fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;
  char *p;


  int Number_of_warps_inside_each_block = 8; 
  int Concurrent_threads_In_Block = warp_size * Number_of_warps_inside_each_block;
  int Number_of_blocks_inside_each_kernel = (NumReads + Concurrent_threads_In_Block - 1) / 
                                            Concurrent_threads_In_Block;

  int F_ErrorThreshold =0;

  uint* ReadSeq = (uint *) calloc(NumReads * 8, sizeof(uint));
  uint* RefSeq = (uint *) calloc(NumReads * 8, sizeof(uint));
  int* DFinal_Results = (int *) calloc(NumReads, sizeof(int));
  int* HFinal_Results = (int *) calloc(NumReads, sizeof(int));

  int tokenIndex=1;
  fp = fopen(argv[2], "r");
  if (!fp){
    printf("The file %s does not exist or you do not have access permission\n", argv[2]);
    return 0;
  }
  for(int this_read = 0; this_read < NumReads; this_read++) {
    read = getline(&line, &len, fp);
    tokenIndex=1;
    for (p = strtok(line, "\t"); p != NULL; p = strtok(NULL, "\t"))
    {
      if (tokenIndex==1)
      {
        for (int j = 0; j < ReadLength; j++)
        {
          if(p[j] == 'A')
          {
            

          }
          else if (p[j] == 'C')
          {
            ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = set_bit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
          }
          else if (p[j] == 'G')
          {
            ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = set_bit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2));
          }
          else if (p[j] == 'T')
          {
            ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = set_bit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2));

            ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = set_bit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
          }
        }
      }
      else if(tokenIndex==2)
      {
        for (int j = 0; j < ReadLength; j++)
        {
          if(p[j] == 'A')
          {
            

          }
          else if (p[j] == 'C')
          {
            RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = set_bit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
          }
          else if (p[j] == 'G')
          {
            RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = set_bit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2));
          }
          else if (p[j] == 'T')
          {
            RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = set_bit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2));

            RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = set_bit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
          }
        }
      }
      tokenIndex=tokenIndex+1;
    }
  }
  fclose(fp);

    {

  bool error = false;
  for (int loopPar = 0; loopPar <= 25; loopPar++) {

    F_ErrorThreshold = (loopPar*ReadLength)/100;

    auto t1 = high_resolution_clock::now();

    for (int n = 0; n < repeat; n++) {
      sneaky_snake(Number_of_blocks_inside_each_kernel, Concurrent_threads_In_Block,
                   ReadSeq, RefSeq, DFinal_Results, NumReads, F_ErrorThreshold);
    }

    auto t2 = high_resolution_clock::now();
    double elapsed_time = duration_cast<microseconds>(t2 - t1).count();

    
    

    sneaky_snake_ref(ReadSeq, RefSeq, HFinal_Results, NumReads, F_ErrorThreshold);
    error = memcmp(DFinal_Results, HFinal_Results, NumReads * sizeof(int));
    if (error) break;

    

    int D_accepted = 0;
    for(int i = 0; i < NumReads; i++) if(DFinal_Results[i] == 1) D_accepted++;

    printf("Error threshold: %2d | Average kernel time (us): %5.4f | Accepted: %10d | Rejected: %10d\n", 
          F_ErrorThreshold, elapsed_time / repeat, D_accepted, NumReads - D_accepted);
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  }

  free(ReadSeq);
  free(RefSeq);
  free(DFinal_Results);
  free(HFinal_Results);
  return 0;
}