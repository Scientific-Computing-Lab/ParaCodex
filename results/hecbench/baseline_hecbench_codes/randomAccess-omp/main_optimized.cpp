#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gate.h"

using u64Int = unsigned long long int;
using s64Int = long long int;

#define POLY 0x0000000000000007ULL
#define PERIOD 1317624576693539401LL

static inline u64Int HPCC_starts(s64Int n) {
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1ULL;

  temp = 0x1ULL;

  for (i = 0; i < 64; ++i) {
    m2[i] = temp;
    temp = (temp << 1) ^ (((s64Int)temp < 0) ? POLY : 0);
    temp = (temp << 1) ^ (((s64Int)temp < 0) ? POLY : 0);
  }

  for (i = 62; i >= 0; --i) {
    if ((n >> i) & 1) {
      break;
    }
  }

  ran = 0x2ULL;
  while (i > 0) {
    temp = 0;
    for (j = 0; j < 64; ++j) {
      if ((ran >> j) & 1) temp ^= m2[j];
    }
    ran = temp;
    --i;
    if ((n >> i) & 1) {
      ran = (ran << 1) ^ (((s64Int)ran < 0) ? POLY : 0);
    }
  }

  return ran;
}

static inline u64Int compute_start_index(u64Int stream_id,
                                         u64Int base_chunk,
                                         u64Int remainder) {
  if (stream_id < remainder) {
    return stream_id * (base_chunk + 1);
  }
  return remainder * (base_chunk + 1) + (stream_id - remainder) * base_chunk;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::max(1, std::atoi(argv[1]));

  int failure;
  u64Int temp;
  double totalMem;
  u64Int* Table = nullptr;
  u64Int logTableSize, TableSize;

  totalMem = 1024.0 * 1024.0 * 512.0;
  totalMem /= sizeof(u64Int);

  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1; totalMem >= 1.0;
       totalMem *= 0.5, ++logTableSize, TableSize <<= 1) {
    // Intentionally empty; determines largest power-of-two table size.
  }

  std::printf("Table size = %llu\n", TableSize);

  if (posix_memalign(reinterpret_cast<void**>(&Table), 1024,
                     TableSize * sizeof(u64Int)) != 0) {
    std::fprintf(stderr,
                 "Failed to allocate memory for the update table %llu\n",
                 TableSize);
    return 1;
  }

  std::fprintf(stdout, "Main table size   = 2^%llu = %llu words\n",
               logTableSize, TableSize);

  const u64Int totalUpdates = 4ULL * TableSize;
  std::fprintf(stdout, "Number of updates = %llu\n", totalUpdates);

  const u64Int maxStreams = 16384ULL;
  const u64Int numStreams =
      std::min(std::max<u64Int>(1ULL, totalUpdates), maxStreams);
  const u64Int baseChunk = totalUpdates / numStreams;
  const u64Int remainderChunk = totalUpdates % numStreams;
  const u64Int tableMask = TableSize - 1;

  std::vector<u64Int> ranInit(numStreams);
  std::vector<u64Int> ranHost(numStreams);

  for (u64Int stream = 0; stream < numStreams; ++stream) {
    const u64Int startIndex =
        compute_start_index(stream, baseChunk, remainderChunk);
    ranInit[stream] = HPCC_starts(static_cast<s64Int>(startIndex));
  }

  std::copy(ranInit.begin(), ranInit.end(), ranHost.begin());

  u64Int* tablePtr = Table;
  u64Int* ranPtr = ranHost.data();

  const int initThreadLimit = 256;
  const int initNumTeams = static_cast<int>(
      std::max<u64Int>(1ULL,
                       (TableSize + initThreadLimit - 1) / initThreadLimit));
  const int updateThreadLimit = 256;
  const int updateNumTeams = static_cast<int>(
      std::max<u64Int>(1ULL,
                       (numStreams + updateThreadLimit - 1) / updateThreadLimit));

  auto start = std::chrono::steady_clock::now();

#pragma omp target data map(alloc: tablePtr[0:TableSize]) \
    map(alloc: ranPtr[0:numStreams])
  {
    for (int iter = 0; iter < repeat; ++iter) {
      std::copy(ranInit.begin(), ranInit.end(), ranHost.begin());

#pragma omp target update to(ranPtr[0:numStreams])

#pragma omp target teams distribute parallel for                  \
    map(present: tablePtr[0:TableSize]) num_teams(initNumTeams)    \
        thread_limit(initThreadLimit) schedule(static, 1)
      for (u64Int idx = 0; idx < TableSize; ++idx) {
        tablePtr[idx] = idx;
      }

#pragma omp target teams distribute parallel for                      \
    map(present: tablePtr[0:TableSize]) map(present: ranPtr[0:numStreams]) \
        num_teams(updateNumTeams) thread_limit(updateThreadLimit)      \
        schedule(static, 1)
      for (u64Int stream = 0; stream < numStreams; ++stream) {
        u64Int state = ranPtr[stream];
        const u64Int updatesThisStream =
            baseChunk + (stream < remainderChunk ? 1ULL : 0ULL);
        for (u64Int update = 0; update < updatesThisStream; ++update) {
          state = (state << 1) ^ (((s64Int)state < 0) ? POLY : 0);
          const u64Int loc = state & tableMask;
#pragma omp atomic update
          tablePtr[loc] ^= state;
        }
        ranPtr[stream] = state;
      }
    }

#pragma omp target update from(tablePtr[0:TableSize])
  }

  auto end = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n",
              (elapsed * 1e-9) / repeat);

  temp = 0x1ULL;
  for (u64Int i = 0; i < totalUpdates; ++i) {
    temp = (temp << 1) ^ (((s64Int)temp < 0) ? POLY : 0);
    const u64Int loc = temp & tableMask;
    Table[loc] ^= temp;
  }

  temp = 0;
  for (u64Int i = 0; i < TableSize; ++i) {
    if (Table[i] != i) {
      ++temp;
    }
  }

  std::fprintf(stdout, "Found %llu errors in %llu locations (%s).\n", temp,
               TableSize, (temp <= 0.01 * TableSize) ? "passed" : "failed");
  failure = (temp <= 0.01 * TableSize) ? 0 : 1;

  GATE_CHECKSUM_BYTES("Table", Table, TableSize * sizeof(u64Int));

  std::free(Table);
  return failure;
}
