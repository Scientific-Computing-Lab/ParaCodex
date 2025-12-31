#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <omp.h>

#include "reference.h"
#include "gate.h"

namespace {

constexpr int kNumSymbols = 16;
constexpr int kWindowRadius = 2;
constexpr int kWindowDiameter = 2 * kWindowRadius + 1;
constexpr int kWindowArea = kWindowDiameter * kWindowDiameter;  // 25 samples
constexpr int kTableSize = kWindowArea + 1;                     // counts up to 25
constexpr int kThreadLimit = 256;

void entropy_gpu(float *entropy,
                 const char *values,
                 const float *logCountTable,
                 const float *invTotalTable,
                 const float *logTotalTable,
                 int height,
                 int width) {
  const int totalElements = height * width;
  if (totalElements == 0) {
    return;
  }

#pragma omp target teams distribute parallel for collapse(2) \
    thread_limit(kThreadLimit) schedule(static)                \
    map(present : entropy [0:totalElements], values [0:totalElements], \
        logCountTable [0:kTableSize], invTotalTable [0:kTableSize],      \
        logTotalTable [0:kTableSize])
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      unsigned char counts[kNumSymbols];
      for (int i = 0; i < kNumSymbols; ++i) {
        counts[i] = 0;
      }

      int y0 = y - kWindowRadius;
      if (y0 < 0) y0 = 0;
      int y1 = y + kWindowRadius;
      if (y1 >= height) y1 = height - 1;
      int x0 = x - kWindowRadius;
      if (x0 < 0) x0 = 0;
      int x1 = x + kWindowRadius;
      if (x1 >= width) x1 = width - 1;

      int total = 0;
      for (int yy = y0; yy <= y1; ++yy) {
        const int rowOffset = yy * width;
        for (int xx = x0; xx <= x1; ++xx) {
          const unsigned char symbol =
              static_cast<unsigned char>(values[rowOffset + xx]);
          counts[symbol]++;
          total++;
        }
      }

      float sum = 0.0f;
      for (int k = 0; k < kNumSymbols; ++k) {
        sum += logCountTable[counts[k]];
      }

      const float entropyValue =
          logTotalTable[total] - sum * invTotalTable[total];
      entropy[y * width + x] = entropyValue;
    }
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = std::atoi(argv[1]);
  const int height = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  if (width <= 0 || height <= 0 || repeat <= 0) {
    std::fprintf(stderr, "All input dimensions and repeat count must be > 0.\n");
    return 1;
  }

  const std::size_t totalElements =
      static_cast<std::size_t>(height) * static_cast<std::size_t>(width);

  std::vector<char> input(totalElements);
  std::vector<float> output(totalElements, 0.0f);
  std::vector<float> outputRef(totalElements, 0.0f);

  std::vector<float> logCountTable(kTableSize, 0.0f);
  std::vector<float> invTotalTable(kTableSize, 0.0f);
  std::vector<float> logTotalTable(kTableSize, 0.0f);

  for (int i = 0; i < kTableSize; ++i) {
    if (i > 1) {
      logCountTable[i] =
          static_cast<float>(i) * log2f(static_cast<float>(i));
    }
    invTotalTable[i] = (i > 0) ? 1.0f / static_cast<float>(i) : 0.0f;
    logTotalTable[i] = (i > 0) ? log2f(static_cast<float>(i)) : 0.0f;
  }

  std::srand(123);
  for (std::size_t idx = 0; idx < totalElements; ++idx) {
    input[idx] = static_cast<char>(std::rand() % kNumSymbols);
  }

  const bool hasGpu = omp_get_num_devices() > 0;

  if (hasGpu) {
    const float *logCountPtr = logCountTable.data();
    const float *invTotalPtr = invTotalTable.data();
    const float *logTotalPtr = logTotalTable.data();
    float *outputPtr = output.data();
    const char *inputPtr = input.data();

#pragma omp target data map(to : inputPtr [0:totalElements]) \
    map(to : logCountPtr [0:kTableSize], invTotalPtr [0:kTableSize], \
        logTotalPtr [0:kTableSize])                             \
        map(alloc : outputPtr [0:totalElements])
    {
      // Warm-up launch to amortize first-use overheads.
      entropy_gpu(outputPtr, inputPtr, logCountPtr, invTotalPtr, logTotalPtr,
                  height, width);

      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i) {
        entropy_gpu(outputPtr, inputPtr, logCountPtr, invTotalPtr,
                    logTotalPtr, height, width);
      }
      auto end = std::chrono::steady_clock::now();
      const double seconds =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() *
          1e-9;
      std::printf("Average kernel execution time %f (s)\n",
                  seconds / repeat);

#pragma omp target update from(outputPtr [0:totalElements])
    }
  } else {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      reference(output.data(), input.data(), height, width);
    }
    auto end = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() *
        1e-9;
    std::printf("Average kernel execution time %f (s) [CPU fallback]\n",
                seconds / repeat);
  }

  reference(outputRef.data(), input.data(), height, width);

  bool ok = true;
  for (std::size_t idx = 0; idx < totalElements; ++idx) {
    if (std::fabs(output[idx] - outputRef[idx]) > 1e-3f) {
      ok = false;
      break;
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  GATE_CHECKSUM_BYTES("entropy.output_ref", outputRef.data(),
                      totalElements * sizeof(float));
  return ok ? 0 : 1;
}
