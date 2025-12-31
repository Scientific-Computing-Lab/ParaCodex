#include <stdio.h>
#include <stdlib.h>
#include <chrono>

typedef unsigned long long int u64Int;
typedef long long int s64Int;



#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

#define NUPDATE (4 * TableSize)

u64Int
HPCC_starts(s64Int n)
{
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;

    for (i=0; i<64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
  }

  for (i=62; i>=0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
        for (j=0; j<64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
  }

  return ran;
}


int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int failure;
  u64Int i;
  u64Int temp;
  double totalMem;
  u64Int *Table = NULL;
  u64Int logTableSize, TableSize;

  

  totalMem = 1024*1024*512;
  totalMem /= sizeof(u64Int);

  

  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ; 


   printf("Table size = %llu\n",  TableSize);

   posix_memalign((void**)&Table, 1024, TableSize * sizeof(u64Int));

  if (! Table ) {
    fprintf( stderr, "Failed to allocate memory for the update table %llu\n", TableSize);
    return 1;
  }

  

  fprintf(stdout, "Main table size   = 2^%llu = %llu words\n", logTableSize,TableSize);
  fprintf(stdout, "Number of updates = %llu\n", NUPDATE);

  u64Int ran[128];

{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    

        for (i=0; i<TableSize; i++) {
      Table[i] = i;
    }
        for (int j=0; j<128; j++)
      ran[j] = HPCC_starts ((NUPDATE/128) * j);

        for (int j=0; j<128; j++) {
      for (u64Int i=0; i<NUPDATE/128; i++) {
        ran[j] = (ran[j] << 1) ^ ((s64Int) ran[j] < 0 ? POLY : 0);
                Table[ran[j] & (TableSize-1)] ^= ran[j];
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

}

  

  temp = 0x1;
  for (i=0; i<NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int) temp < 0) ? POLY : 0);
    Table[temp & (TableSize-1)] ^= temp;
  }
  
  temp = 0;
  for (i=0; i<TableSize; i++)
    if (Table[i] != i) {
      temp++;
    }

  fprintf( stdout, "Found %llu errors in %llu locations (%s).\n",
           temp, TableSize, (temp <= 0.01*TableSize) ? "passed" : "failed");
  if (temp <= 0.01*TableSize) failure = 0;
  else failure = 1;

  free( Table );
  return failure;

}
