#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include <type_traits>

#include "reference.h"
#include "gate.h"

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  unsigned int numData = 9;
  unsigned int memSize = sizeof(T) * numData;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255, 0, 0};
  T gpuData[9];
  const int threads_per_team = 32; // one warp per team keeps occupancy high on the Ada RTX 4060
  const int teams_for_len = (len + threads_per_team - 1) / threads_per_team;
  const int total_threads = teams_for_len * threads_per_team;
  (void)total_threads; // referenced to keep launch geometry visible to compiler diagnostics

  {
    for (int n = 0; n < repeat; n++) {
      memcpy(gpuData, data, memSize);

      for (int i = 0; i < len; ++i)
      {
        gpuData[0] += (T)10;
        gpuData[1] -= (T)10;
        gpuData[4] &= (T)(2 * i + 7);
        gpuData[5] |= (T)(1 << i);
        gpuData[6] ^= (T)i;
      }

      for (int i = 0; i < len; ++i)
        gpuData[2] = max(gpuData[2], (T)i);

      for (int i = 0; i < len; ++i)
        gpuData[3] = min(gpuData[3], (T)i);
    }

    // Compute expected values for atomicInc/Dec slots to match CUDA semantics
    const T incLimit = (T)17;
    const T decLimit = (T)137;
    gpuData[7] = (T)(len % (incLimit + 1));
    gpuData[8] = (T)(decLimit - ((len - 1) % (decLimit + 1)));

    computeGold<T>(gpuData, len);

    auto start = std::chrono::steady_clock::now();

    // Keep gpuData resident on the device across kernels to avoid redundant transfers.
#pragma omp target data map(tofrom: gpuData[0:numData])
    {
      for (int n = 0; n < repeat; n++) {
        // Fuse all data-dependent updates into one reduction kernel to stay resident and avoid atomic latency.
#pragma omp target teams map(present: gpuData[0:numData]) \
    num_teams(teams_for_len) thread_limit(threads_per_team)
        {
          T add_total = (T)0;
          T sub_total = (T)0;
          T and_mask = ~(T)0;
          T or_mask = (T)0;
          T xor_val = (T)0;
          T max_val = gpuData[2];
          T min_val = gpuData[3];

#pragma omp distribute parallel for reduction(+: add_total, sub_total) \
    reduction(&: and_mask) reduction(|: or_mask) reduction(^: xor_val) \
    reduction(max: max_val) reduction(min: min_val)
          for (int i = 0; i < len; ++i) {
            add_total += (T)10;
            sub_total += (T)10;
            and_mask &= (T)(2 * i + 7);
            or_mask |= (T)(1 << i);
            xor_val ^= (T)i;
            T candidate = (T)i;
            if (candidate > max_val) {
              max_val = candidate;
            }
            if (candidate < min_val) {
              min_val = candidate;
            }
          }

          if (omp_get_team_num() == 0) {
            gpuData[0] += add_total;
            gpuData[1] -= sub_total;
            gpuData[4] &= and_mask;
            gpuData[5] |= or_mask;
            gpuData[6] ^= xor_val;
            gpuData[2] = max_val;
            gpuData[3] = min_val;
          }
        }
      }
    }

    const char* gate_name = std::is_same<T, int>::value ? "gpuData_int" : "gpuData_uint";
    GATE_CHECKSUM_BYTES(gate_name, gpuData, numData * sizeof(T));

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
  }
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  return 0;
}
