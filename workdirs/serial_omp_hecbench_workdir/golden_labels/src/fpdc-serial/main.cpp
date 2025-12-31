



#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>

#define ull unsigned long long
#define MAX (64*1024*1024)
#define WARPSIZE 32
#define min(a,b) (a) < (b) ? (a) : (b)




inline int clzll(ull num) {
  int count = 0;
  while(!(num & 0x1000000000000000ULL)) {
    count++;
    num <<= 1;
  }
  return count;
}




void CompressionKernel(
  const int nTeams,
  const int nThreads,
  const int dimensionalityd,
  const ull *__restrict cbufd,
  char *__restrict dbufd,
  const int *__restrict cutd,
  int *__restrict offd)
{
    {
    int ibufs[32 * (3 * WARPSIZE / 2)]; 

        {
      int offset, code, bcount, tmp, off, beg, end, lane, warp, iindex, lastidx, start, term;
      ull diff, prev;
      int lid = omp_get_thread_num();

      

      lane = lid & 31;
      

      iindex = lid / WARPSIZE * (3 * WARPSIZE / 2) + lane;
      ibufs[iindex] = 0;
      iindex += WARPSIZE / 2;
      lastidx = (lid / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
      

      warp = (lid + omp_get_team_num() * nThreads) / WARPSIZE;
      

      offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

      

      start = 0;
      if (warp > 0) start = cutd[warp-1];
      term = cutd[warp];
      off = ((start+1)/2*17);

      prev = 0;
      for (int i = start + lane; i < term; i += WARPSIZE) {
        

        

        diff = cbufd[i] - prev;
        code = (diff >> 60) & 8;
        if (code != 0) {
          diff = -diff;
        }

        

        bcount = 8 - (clzll(diff) >> 3);
        if (bcount == 2) bcount = 3; 


        

        ibufs[iindex] = bcount;
                ibufs[iindex] += ibufs[iindex-1];
                ibufs[iindex] += ibufs[iindex-2];
                ibufs[iindex] += ibufs[iindex-4];
                ibufs[iindex] += ibufs[iindex-8];
                ibufs[iindex] += ibufs[iindex-16];
        
        

        beg = off + (WARPSIZE/2) + ibufs[iindex-1];
        end = beg + bcount;
        for (; beg < end; beg++) {
          dbufd[beg] = diff;
          diff >>= 8;
        }

        if (bcount >= 3) bcount--; 

        tmp = ibufs[lastidx];
        code |= bcount;
        ibufs[iindex] = code;
        
        

        

        if ((lane & 1) != 0) {
          dbufd[off + (lane >> 1)] = ibufs[iindex-1] | (code << 4);
        }
        off += tmp + (WARPSIZE/2);

        

        

        prev = cbufd[i + offset];
      }

      

      if (lane == 31) offd[warp] = off;
    }
  }
}




static void Compress(int blocks, int warpsperblock, int repeat, int dimensionality)
{
  

  FILE *fp = fopen("input.bin", "wb");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open input file input.bin for write.\n");
  }
  for (int i = 0; i < MAX; i++) {
    double t = i;
    fwrite(&t, 8, 1, fp);
  }
  fclose(fp);

  fp = fopen("input.bin", "rb");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open input file input.bin for read.\n");
  }

  

  ull *cbuf = (ull *)malloc(sizeof(ull) * MAX); 

  if (cbuf == NULL) {
    fprintf(stderr, "cannot allocate cbuf\n");
  }

  int doubles = fread(cbuf, 8, MAX, fp);
  if (doubles != MAX) {
    fprintf(stderr, "Error in reading input.bin. Exit\n");
    if (cbuf != NULL) free(cbuf);
    fclose(fp);
    return ;
  }
  fclose(fp);

  const int num_warps = blocks * warpsperblock;

  char *dbuf = (char *)malloc(sizeof(char) * ((MAX+1)/2*17)); 

  if (dbuf == NULL) {
    fprintf(stderr, "cannot allocate dbuf\n");
  }
  int *cut = (int *)malloc(sizeof(int) * num_warps); 

  if (cut == NULL) {
    fprintf(stderr, "cannot allocate cut\n");
  }
  int *off = (int *)malloc(sizeof(int) * num_warps); 

  if (off == NULL) {
    fprintf(stderr, "cannot allocate off\n");
  }

  

  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  

  int per = (doubles + num_warps - 1) / (num_warps);
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0, before = 0, d = 0;
  for (int i = 0; i < num_warps; i++) {
    curr += per;
    cut[i] = min(curr, doubles);
    if (cut[i] - before > 0) {
      d = cut[i] - before;
    }
    before = cut[i];
  }

  

  if (d <= WARPSIZE) {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = 0;
    }
  } else {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = cbuf[(i & -WARPSIZE) - (dimensionality - i % dimensionality)];
    }
  }

    {
    auto start = std::chrono::steady_clock::now();
 
    for (int i = 0; i < repeat; i++)
      CompressionKernel(blocks, WARPSIZE*warpsperblock,
        dimensionality, cbuf, dbuf, cut, off);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    fprintf(stderr, "Average compression kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

    

    
    

    fp = fopen("output.bin", "wb");
    if (fp == NULL) {
      fprintf(stderr, "Failed to open output file output.bin.\n");
    }

    int num;
    int doublecnt = doubles-padding;
    num = fwrite(&blocks, 1, 1, fp);
    assert(1 == num);
    num = fwrite(&warpsperblock, 1, 1, fp);
    assert(1 == num);
    num = fwrite(&dimensionality, 1, 1, fp);
    assert(1 == num);
    num = fwrite(&doublecnt, 4, 1, fp);
    assert(1 == num);
    

    for(int i = 0; i < num_warps; i++) {
      int start = 0;
      if(i > 0) start = cut[i-1];
      off[i] -= ((start+1)/2*17);
      num = fwrite(&off[i], 4, 1, fp); 

      assert(1 == num);
    }
    

    for(int i = 0; i < num_warps; i++) {
      int offset, start = 0;
      if(i > 0) start = cut[i-1];
      offset = ((start+1)/2*17);
      

            num = fwrite(&dbuf[offset], 1, off[i], fp);
      assert(off[i] == num);
    }
    fclose(fp);
    
    

    fp = fopen("input.bin", "rb");
    fseek (fp, 0, SEEK_END);
    long input_size = ftell (fp);

    fp = fopen("output.bin", "rb");
    fseek (fp, 0, SEEK_END);
    long output_size = ftell (fp);

    fprintf(stderr, "Compression ratio = %lf\n", 1.0 * input_size / output_size);

    free(cbuf);
    free(dbuf);
    free(cut);
    free(off);
  }
}




static void VerifySystemParameters()
{
  assert(1 == sizeof(char));
  assert(4 == sizeof(int));
  assert(8 == sizeof(ull));

  int val = 1;
  assert(1 == *((char *)&val));
   
  if ((WARPSIZE <= 0) || ((WARPSIZE & (WARPSIZE-1)) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }
}




int main(int argc, char *argv[])
{
  fprintf(stderr, "GPU FP Compressor v2.2\n");
  fprintf(stderr, "Copyright 2011-2020 Texas State University\n");

  VerifySystemParameters();

  int blocks, warpsperblock, dimensionality;
  int repeat;

  if((4 == argc) || (5 == argc)) { 

    blocks = atoi(argv[1]);
    assert((0 < blocks) && (blocks < 256));

    warpsperblock = atoi(argv[2]);
    assert((0 < warpsperblock) && (warpsperblock < 256));

    repeat = atoi(argv[3]);

    if(4 == argc) {
      dimensionality = 1;
    } else {
      dimensionality = atoi(argv[4]);
    }
    assert((0 < dimensionality) && (dimensionality <= WARPSIZE));

    Compress(blocks, warpsperblock, repeat, dimensionality);
  }
  else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "compress: %s <blocks> <warps/block> <repeat> <dimensionality>\n", argv[0]);
    fprintf(stderr, "\ninput.bin is generated by the program and the compressed output file is output.bin.\n");
  }

  return 0;
}