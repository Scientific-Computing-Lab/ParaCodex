#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

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

    for (int n = 0; n < iterations; n++) {
      // Offload AoS accumulation per tree to the GPU.
#pragma omp target teams distribute parallel for                            \
    map(to: trees[0:treeNumber]) map(from: output[0:treeNumber])
      for (uint gid = 0; gid < treeNumber; gid++) {
        uint res = 0;
        for (int i = 0; i < treeSize; i++) {
          res += trees[gid].apples[i];
        }
        output[gid] = res;
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

    for (int i = 0; i < treeNumber; i++)
      for (int j = 0; j < treeSize; j++)
        data[i + j * treeNumber] = j + i * treeSize;

    ApplesOnTrees *applesOnTrees = (ApplesOnTrees *)data;

    start = std::chrono::steady_clock::now();

    for (int n = 0; n < iterations; n++) {
      // Offload SoA accumulation per tree to the GPU.
#pragma omp target teams distribute parallel for                            \
    map(to: applesOnTrees[0:treeSize]) map(from: output[0:treeNumber])
      for (uint gid = 0; gid < treeNumber; gid++) {
        uint res = 0;
        for (int i = 0; i < treeSize; i++) {
          res += applesOnTrees[i].trees[gid];
        }
        output[gid] = res;
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
  }

  free(output);
  free(reference);
  free(data);
  return 0;
}
