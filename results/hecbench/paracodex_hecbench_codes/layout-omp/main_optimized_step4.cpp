#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "gate.h"

#define TREE_NUM 4096
#define TREE_SIZE 4096
#define GROUP_SIZE 256

using uint = unsigned int;

struct AppleTree {
  int apples[TREE_SIZE];
};

struct ApplesOnTrees {
  int trees[TREE_NUM];
};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int iterations = atoi(argv[1]);

  const int treeSize = TREE_SIZE;
  const int treeNumber = TREE_NUM;
  bool fail = false;
  const int tunedThreadLimit = GROUP_SIZE;
  int tunedNumTeams =
      ((treeNumber + tunedThreadLimit - 1) / tunedThreadLimit) * 8;
  if (tunedNumTeams < 128)
    tunedNumTeams = 128; // Ensure enough thread blocks to keep 24-SM RTX 4060 busy.

  if (iterations < 1) {
    std::cout << "Iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }

  if (treeNumber < GROUP_SIZE) {
    std::cout << "treeNumber should be larger than the work group size"
              << std::endl;
    return -1;
  }
  if (treeNumber % 256 != 0) {
    std::cout << "treeNumber should be a multiple of 256" << std::endl;
    return -1;
  }

  const int elements = treeSize * treeNumber;
  size_t inputSize = elements * sizeof(int);
  size_t outputSize = treeNumber * sizeof(int);

  int *data = (int *)malloc(inputSize);
  int *output = (int *)malloc(outputSize);
  int *reference = (int *)malloc(outputSize);
  memset(reference, 0, outputSize);
  for (int i = 0; i < treeNumber; i++)
    for (int j = 0; j < treeSize; j++)
      reference[i] += i * treeSize + j;

  {
    for (int i = 0; i < treeNumber; i++)
      for (int j = 0; j < treeSize; j++)
        data[j + i * treeSize] = j + i * treeSize;

    AppleTree *trees = (AppleTree *)data;

    auto start = std::chrono::steady_clock::now();

    // Keep AoS data and output resident on the GPU across all iterations.
#pragma omp target data map(to: trees[0:treeNumber])                         \
    map(from: output[0:treeNumber])
    {
      for (int n = 0; n < iterations; n++) {
        // Offload AoS accumulation per tree to the GPU.  Each team now
        // cooperatively reduces one tree so all 256 threads issue loads,
        // improving latency hiding versus the previous single-thread sum.
#pragma omp target teams distribute                                         \
    num_teams(tunedNumTeams) thread_limit(tunedThreadLimit)                  \
        map(present, to: trees[0:treeNumber])                                \
            map(present, tofrom: output[0:treeNumber])
        for (uint gid = 0; gid < treeNumber; gid++) {
          uint res = 0;
          const int *treePtr = trees[gid].apples;
#pragma omp parallel for simd num_threads(tunedThreadLimit)                 \
    reduction(+ : res) schedule(static)
          for (int i = 0; i < treeSize; i++) {
            res += treePtr[i];
          }
          output[gid] = res;
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << "Average kernel execution time (AoS): "
              << (time * 1e-3f) / iterations << " (us)\n";

    for (int i = 0; i < treeNumber; i++) {
      if (output[i] != reference[i]) {
        fail = true;
        break;
      }
    }

    if (fail)
      std::cout << "FAIL\n";
    else
      std::cout << "PASS\n";

    GATE_CHECKSUM_U32("layout_aos_output",
                      reinterpret_cast<const uint32_t *>(output), treeNumber);

    for (int i = 0; i < treeNumber; i++)
      for (int j = 0; j < treeSize; j++)
        data[i + j * treeNumber] = j + i * treeSize;

    ApplesOnTrees *applesOnTrees = (ApplesOnTrees *)data;

    start = std::chrono::steady_clock::now();

    // Keep the SoA layout and outputs on the device while iterating.
#pragma omp target data map(to: applesOnTrees[0:treeSize])                   \
    map(from: output[0:treeNumber])
    {
      for (int n = 0; n < iterations; n++) {
        // Offload SoA accumulation per tree to the GPU.  Cooperative
        // reductions again keep every lane active while streaming the
        // column-major data with a consistent stride.
#pragma omp target teams distribute                                         \
    num_teams(tunedNumTeams) thread_limit(tunedThreadLimit)                  \
        map(present, to: applesOnTrees[0:treeSize])                          \
            map(present, tofrom: output[0:treeNumber])
        for (uint gid = 0; gid < treeNumber; gid++) {
          uint res = 0;
          const int *columnPtr = applesOnTrees[0].trees + gid;
#pragma omp parallel for simd num_threads(tunedThreadLimit)                 \
    reduction(+ : res) schedule(static)
          for (int i = 0; i < treeSize; i++) {
            res += columnPtr[i * treeNumber];
          }
          output[gid] = res;
        }
      }
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    std::cout << "Average kernel execution time (SoA): "
              << (time * 1e-3f) / iterations << " (us)\n";

    for (int i = 0; i < treeNumber; i++) {
      if (output[i] != reference[i]) {
        fail = true;
        break;
      }
    }

    if (fail)
      std::cout << "FAIL\n";
    else
      std::cout << "PASS\n";

    GATE_CHECKSUM_U32("layout_soa_output",
                      reinterpret_cast<const uint32_t *>(output), treeNumber);
  }

  free(output);
  free(reference);
  free(data);
  return 0;
}
