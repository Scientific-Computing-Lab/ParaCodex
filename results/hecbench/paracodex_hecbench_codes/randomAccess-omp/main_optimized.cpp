#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include "gate.h"

typedef unsigned long long int u64Int;
typedef long long int s64Int;

#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

#define NUPDATE (4 * TableSize)

constexpr int kRandomStreams = 16384;  // expose enough RNG streams to keep the GPU busy
constexpr int kThreadsPerTeam = 256;
constexpr int kMaxTeamCount = 4096;  // ~24 SMs * 32 teams keeps the RTX 4060 busy
constexpr int kLfsrUnroll = 4;

u64Int HPCC_starts(s64Int n) {
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;

  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
  }

  for (i = 62; i >= 0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j = 0; j < 64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int)ran < 0 ? POLY : 0);
  }

  return ran;
}

int main(int argc, char **argv) {
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

  totalMem = 1024 * 1024 * 512;
  totalMem /= sizeof(u64Int);

  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1; totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ;

  printf("Table size = %llu\n", TableSize);

  posix_memalign((void **)&Table, 1024, TableSize * sizeof(u64Int));

  if (!Table) {
    fprintf(stderr, "Failed to allocate memory for the update table %llu\n", TableSize);
    return 1;
  }

  fprintf(stdout, "Main table size   = 2^%llu = %llu words\n", logTableSize,
          TableSize);
  fprintf(stdout, "Number of updates = %llu\n", NUPDATE);

  u64Int ran[kRandomStreams];
  const u64Int updatesPerStream = NUPDATE / kRandomStreams;
  const u64Int tableMask = TableSize - 1;
  const int numTeamsInit = static_cast<int>(std::min<u64Int>(
      static_cast<u64Int>(kMaxTeamCount),
      (TableSize + static_cast<u64Int>(kThreadsPerTeam) - 1) /
          static_cast<u64Int>(kThreadsPerTeam)));
  const int numTeamsReduce = numTeamsInit;
  const int numTeamsRandom =
      std::min(kRandomStreams, kMaxTeamCount);  // hardware-friendly cap

  {
    auto start = std::chrono::steady_clock::now();

    // Keep the RandomAccess table resident on the GPU across kernels to avoid
    // paying the map(tofrom:) penalty for every target region.
    #pragma omp target data map(alloc : Table[0:TableSize])
    {
      for (int rep = 0; rep < repeat; rep++) {
        // Initialize the table on the GPU; one thread per element.
        #pragma omp target teams distribute parallel for map(present : Table[0:TableSize]) num_teams(numTeamsInit) thread_limit(kThreadsPerTeam)
        for (u64Int idx = 0; idx < TableSize; idx++) {
          Table[idx] = idx;
        }
        for (int j = 0; j < kRandomStreams; j++)
          ran[j] = HPCC_starts(updatesPerStream * j);

        // Random updates: distribute independent RNG streams across GPU teams.
        #pragma omp target teams distribute parallel for map(present : Table[0:TableSize]) map(to : ran[0:kRandomStreams]) num_teams(numTeamsRandom) thread_limit(kThreadsPerTeam)
        for (int j = 0; j < kRandomStreams; j++) {
          u64Int local_ran = ran[j];
          u64Int update = 0;
          for (; update + (kLfsrUnroll - 1) < updatesPerStream;
               update += kLfsrUnroll) {
#pragma unroll
            for (int step = 0; step < kLfsrUnroll; ++step) {
              local_ran =
                  (local_ran << 1) ^ ((s64Int)local_ran < 0 ? POLY : 0);
              u64Int index = local_ran & tableMask;
#pragma omp atomic update
              Table[index] ^= local_ran;
            }
          }
          for (; update < updatesPerStream; ++update) {
            local_ran =
                (local_ran << 1) ^ ((s64Int)local_ran < 0 ? POLY : 0);
            u64Int index = local_ran & tableMask;
#pragma omp atomic update
            Table[index] ^= local_ran;
          }
        }
      }

      // Host needs the final table contents for the serial verification pass.
      #pragma omp target update from(Table[0:TableSize])
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

  temp = 0x1;
  for (i = 0; i < NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int)temp < 0) ? POLY : 0);
    Table[temp & (TableSize - 1)] ^= temp;
  }

  temp = 0;
#pragma omp target teams distribute parallel for map(to : Table[0:TableSize]) reduction(+ : temp) num_teams(numTeamsReduce) thread_limit(kThreadsPerTeam)
  for (u64Int idx = 0; idx < TableSize; idx++)
    if (Table[idx] != idx) {
      temp++;
    }

  fprintf(stdout, "Found %llu errors in %llu locations (%s).\n", temp, TableSize,
          (temp <= 0.01 * TableSize) ? "passed" : "failed");
  GATE_CHECKSUM_BYTES("randomAccess.Table", Table, TableSize * sizeof(u64Int));
  if (temp <= 0.01 * TableSize)
    failure = 0;
  else
    failure = 1;

  free(Table);
  return failure;
}
