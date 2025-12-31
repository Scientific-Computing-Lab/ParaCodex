#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cstdint>

#include <omp.h>
#include "gate.h"

#define TREE_NUM 4096
#define TREE_SIZE 4096
#define GROUP_SIZE 256

struct AppleTree {
  int apples[TREE_SIZE];
};

struct ApplesOnTrees {
  int trees[TREE_NUM];
};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int iterations = std::atoi(argv[1]);

  const int treeSize = TREE_SIZE;
  const int treeNumber = TREE_NUM;
  bool overall_fail = false;

  if (iterations < 1) {
    std::cout << "Iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }

  if (treeNumber < GROUP_SIZE) {
    std::cout << "treeNumber should be larger than the work group size" << std::endl;
    return -1;
  }
  if (treeNumber % GROUP_SIZE != 0) {
    std::cout << "treeNumber should be a multiple of " << GROUP_SIZE << std::endl;
    return -1;
  }

  const int elements = treeSize * treeNumber;
  const size_t inputSize = static_cast<size_t>(elements) * sizeof(int);
  const size_t outputSize = static_cast<size_t>(treeNumber) * sizeof(int);

  int *data = static_cast<int *>(std::malloc(inputSize));
  int *output = static_cast<int *>(std::malloc(outputSize));
  int *reference = static_cast<int *>(std::malloc(outputSize));

  if (!data || !output || !reference) {
    std::cerr << "Memory allocation failed\n";
    std::free(data);
    std::free(output);
    std::free(reference);
    return -1;
  }

  std::memset(reference, 0, outputSize);
  for (int i = 0; i < treeNumber; i++) {
    const int base = i * treeSize;
    for (int j = 0; j < treeSize; j++) {
      reference[i] += base + j;
    }
  }

  for (int i = 0; i < treeNumber; i++) {
    const int base = i * treeSize;
    for (int j = 0; j < treeSize; j++) {
      data[j + base] = base + j;
    }
  }

  AppleTree *trees = reinterpret_cast<AppleTree *>(data);

  auto start = std::chrono::steady_clock::now();

  const bool has_target_device = omp_get_num_devices() > 0;

  if (has_target_device) {
    #pragma omp target data map(to : trees[0:treeNumber]) map(from : output[0:treeNumber]) \
        use_device_addr(trees, output)
    {
      for (int n = 0; n < iterations; n++) {
        #pragma omp target teams distribute is_device_ptr(trees, output) num_teams(treeNumber) thread_limit(GROUP_SIZE)
        for (int gid = 0; gid < treeNumber; gid++) {
          int local_sum = 0;
          #pragma omp parallel for simd reduction(+ : local_sum)
          for (int idx = 0; idx < treeSize; idx++) {
            local_sum += trees[gid].apples[idx];
          }
          output[gid] = local_sum;
        }
      }
    }
  } else {
    for (int n = 0; n < iterations; n++) {
      for (int gid = 0; gid < treeNumber; gid++) {
        int local_sum = 0;
        for (int idx = 0; idx < treeSize; idx++) {
          local_sum += trees[gid].apples[idx];
        }
        output[gid] = local_sum;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (AoS): "
            << (time * 1e-3f) / iterations << " (us)\n";

  GATE_CHECKSUM_U32("AoS_output", reinterpret_cast<const uint32_t *>(output), treeNumber);

  bool layout_fail = false;
  for (int i = 0; i < treeNumber; i++) {
    if (output[i] != reference[i]) {
      layout_fail = true;
      break;
    }
  }
  overall_fail |= layout_fail;

  if (layout_fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  for (int i = 0; i < treeNumber; i++) {
    for (int j = 0; j < treeSize; j++) {
      data[i + j * treeNumber] = j + i * treeSize;
    }
  }

  ApplesOnTrees *applesOnTrees = reinterpret_cast<ApplesOnTrees *>(data);

  start = std::chrono::steady_clock::now();

  if (has_target_device) {
    #pragma omp target data map(to : applesOnTrees[0:treeSize]) map(from : output[0:treeNumber]) \
        use_device_addr(applesOnTrees, output)
    {
      for (int n = 0; n < iterations; n++) {
        #pragma omp target teams distribute is_device_ptr(applesOnTrees, output) num_teams(treeNumber) thread_limit(GROUP_SIZE)
        for (int gid = 0; gid < treeNumber; gid++) {
          int local_sum = 0;
          #pragma omp parallel for simd reduction(+ : local_sum)
          for (int idx = 0; idx < treeSize; idx++) {
            local_sum += applesOnTrees[idx].trees[gid];
          }
          output[gid] = local_sum;
        }
      }
    }
  } else {
    for (int n = 0; n < iterations; n++) {
      for (int gid = 0; gid < treeNumber; gid++) {
        int local_sum = 0;
        for (int idx = 0; idx < treeSize; idx++) {
          local_sum += applesOnTrees[idx].trees[gid];
        }
        output[gid] = local_sum;
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (SoA): "
            << (time * 1e-3f) / iterations << " (us)\n";

  GATE_CHECKSUM_U32("SoA_output", reinterpret_cast<const uint32_t *>(output), treeNumber);

  layout_fail = false;
  for (int i = 0; i < treeNumber; i++) {
    if (output[i] != reference[i]) {
      layout_fail = true;
      break;
    }
  }
  overall_fail |= layout_fail;

  if (layout_fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  std::free(output);
  std::free(reference);
  std::free(data);
  return overall_fail ? 1 : 0;
}
