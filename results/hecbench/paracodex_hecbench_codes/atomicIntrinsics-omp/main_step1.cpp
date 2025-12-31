#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <type_traits>

#include "reference.h"
#include "gate.h"

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  unsigned int numThreads = 256;
  unsigned int numData = 9;
  unsigned int memSize = sizeof(T) * numData;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255, 0, 0};
  T gpuData[9];

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

    for (int n = 0; n < repeat; n++) {

      // Offload the per-element updates while preserving atomic semantics.
#pragma omp target teams distribute parallel for map(tofrom: gpuData[0:numData])
      for (int i = 0; i < len; ++i)
      {
#pragma omp atomic update
        gpuData[0] += (T)10;
#pragma omp atomic update
        gpuData[1] -= (T)10;
#pragma omp atomic update
        gpuData[4] &= (T)(2 * i + 7);
#pragma omp atomic update
        gpuData[5] |= (T)(1 << i);
#pragma omp atomic update
        gpuData[6] ^= (T)i;
      }

      T max_val = gpuData[2];
#pragma omp target teams distribute parallel for map(tofrom: max_val) reduction(max: max_val)
      for (int i = 0; i < len; ++i) {
        T candidate = (T)i;
        if (candidate > max_val) {
          max_val = candidate;
        }
      }
      gpuData[2] = max_val;

      T min_val = gpuData[3];
#pragma omp target teams distribute parallel for map(tofrom: min_val) reduction(min: min_val)
      for (int i = 0; i < len; ++i) {
        T candidate = (T)i;
        if (candidate < min_val) {
          min_val = candidate;
        }
      }
      gpuData[3] = min_val;
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
