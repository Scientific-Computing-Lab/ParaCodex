#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <omp.h>
#include "gate.h"

namespace {

constexpr int RY = 15;
constexpr int YG = 6;
constexpr int GC = 4;
constexpr int CB = 11;
constexpr int BM = 13;
constexpr int MR = 6;
constexpr int MAXCOLS = RY + YG + GC + CB + BM + MR;
constexpr float PI_F = 3.14159265358979323846f;

using uchar = unsigned char;

#pragma omp declare target
extern float colorwheel[MAXCOLS][3];
#pragma omp end declare target

float colorwheel[MAXCOLS][3];

inline void setcols(float cw[MAXCOLS][3], int r, int g, int b, int k) {
  cw[k][0] = static_cast<float>(r) / 255.0f;
  cw[k][1] = static_cast<float>(g) / 255.0f;
  cw[k][2] = static_cast<float>(b) / 255.0f;
}

void initColorwheel() {
  int k = 0;
  for (int i = 0; i < RY; ++i) setcols(colorwheel, 255, 255 * i / RY, 0, k++);
  for (int i = 0; i < YG; ++i) setcols(colorwheel, 255 - 255 * i / YG, 255, 0, k++);
  for (int i = 0; i < GC; ++i) setcols(colorwheel, 0, 255, 255 * i / GC, k++);
  for (int i = 0; i < CB; ++i) setcols(colorwheel, 0, 255 - 255 * i / CB, 255, k++);
  for (int i = 0; i < BM; ++i) setcols(colorwheel, 255 * i / BM, 0, 255, k++);
  for (int i = 0; i < MR; ++i) setcols(colorwheel, 255, 0, 255 - 255 * i / MR, k++);
}

#pragma omp declare target
static inline void computeColor(float fx, float fy, uchar *pix) {
  const float rad = sqrtf(fx * fx + fy * fy);
  const float a = atan2f(-fy, -fx) / PI_F;
  const float fk = (a + 1.0f) * 0.5f * static_cast<float>(MAXCOLS - 1);

  int k0 = static_cast<int>(fk);
  if (k0 < 0) k0 = 0;
  if (k0 >= MAXCOLS) k0 = MAXCOLS - 1;
  const int k1 = (k0 + 1) % MAXCOLS;
  const float f = fk - static_cast<float>(k0);

  for (int b = 0; b < 3; ++b) {
    float col = (1.0f - f) * colorwheel[k0][b] + f * colorwheel[k1][b];
    if (rad <= 1.0f) {
      col = 1.0f - rad * (1.0f - col);
    } else {
      col *= 0.75f;
    }
    int ival = static_cast<int>(255.0f * col);
    if (ival < 0) ival = 0;
    if (ival > 255) ival = 255;
    pix[2 - b] = static_cast<uchar>(ival);
  }
}
#pragma omp end declare target

}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::printf("Usage: %s <range> <size> <repeat>\n", argv[0]);
    std::exit(1);
  }

  const float truerange = std::atof(argv[1]);
  const int size = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  const float range = 1.04f * truerange;
  const int half_size = size / 2;
  const float inv_true_range = 1.0f / truerange;
  const float scale = half_size > 0 ? range / static_cast<float>(half_size) : 0.0f;

  const std::size_t imgSize = static_cast<std::size_t>(size) * size * 3;
  uchar *pix = static_cast<uchar *>(std::malloc(imgSize));
  uchar *d_pix = static_cast<uchar *>(std::malloc(imgSize));
  uchar *res = static_cast<uchar *>(std::malloc(imgSize));

  if (!pix || !d_pix || !res) {
    std::fprintf(stderr, "Memory allocation failed\n");
    std::free(pix);
    std::free(d_pix);
    std::free(res);
    return 1;
  }

  std::memset(pix, 0, imgSize);
  std::memset(d_pix, 0, imgSize);
  std::memset(res, 0, imgSize);

  initColorwheel();

  // Reference CPU execution for validation
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      if (x == half_size || y == half_size) continue;
      const float fx = static_cast<float>(x) * scale - range;
      const float fy = static_cast<float>(y) * scale - range;
      const std::size_t idx = (static_cast<std::size_t>(y) * size + x) * 3;
      computeColor(fx * inv_true_range, fy * inv_true_range, pix + idx);
    }
  }

  std::printf("Start execution on a device\n");

  double avg_kernel_ms = 0.0;
  const int num_devices = omp_get_num_devices();
  if (num_devices > 0) {
    #pragma omp target data map(to: colorwheel[0:MAXCOLS][0:3]) \
                            map(tofrom: d_pix[0:imgSize])
    {
      const auto start = std::chrono::steady_clock::now();
      for (int iter = 0; iter < repeat; ++iter) {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int y = 0; y < size; ++y) {
          for (int x = 0; x < size; ++x) {
            if (x == half_size || y == half_size) {
              continue;
            }

            const float fx = static_cast<float>(x) * scale - range;
            const float fy = static_cast<float>(y) * scale - range;
            const std::size_t idx = (static_cast<std::size_t>(y) * size + x) * 3;
            computeColor(fx * inv_true_range, fy * inv_true_range, d_pix + idx);
          }
        }
      }
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      if (repeat > 0) {
        avg_kernel_ms = (elapsed * 1e-6) / repeat;
      }
    }
  } else {
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < repeat; ++iter) {
      for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
          if (x == half_size || y == half_size) continue;
          const float fx = static_cast<float>(x) * scale - range;
          const float fy = static_cast<float>(y) * scale - range;
          const std::size_t idx = (static_cast<std::size_t>(y) * size + x) * 3;
          computeColor(fx * inv_true_range, fy * inv_true_range, d_pix + idx);
        }
      }
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (repeat > 0) {
      avg_kernel_ms = (elapsed * 1e-6) / repeat;
    }
  }

  std::printf("Average kernel execution time : %f (ms)\n", avg_kernel_ms);

  GATE_CHECKSUM_U8("pix", pix, imgSize);
  GATE_CHECKSUM_U8("d_pix", d_pix, imgSize);

  int fail = std::memcmp(pix, d_pix, imgSize);
  if (fail) {
    int max_error = 0;
    for (std::size_t i = 0; i < imgSize; ++i) {
      int e = std::abs(static_cast<int>(d_pix[i]) - static_cast<int>(pix[i]));
      if (e > max_error) max_error = e;
    }
    std::printf("Maximum error between host and device results: %d\n", max_error);
  } else {
    std::printf("%s\n", "PASS");
  }

  std::free(d_pix);
  std::free(pix);
  std::free(res);
  return 0;
}
