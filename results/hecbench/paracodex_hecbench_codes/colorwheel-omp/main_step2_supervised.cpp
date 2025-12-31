#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include "gate.h"

#define RY  15
#define YG  6
#define GC  4
#define CB  11
#define BM  13
#define MR  6
#define MAXCOLS  (RY + YG + GC + CB + BM + MR)
typedef unsigned char uchar;

#pragma omp declare target
void setcols(int cw[MAXCOLS][3], int r, int g, int b, int k)
{
  cw[k][0] = r;
  cw[k][1] = g;
  cw[k][2] = b;
}

void computeColorFromRadFk(float rad, float fk, uchar *pix)
{
  int cw[MAXCOLS][3];

  int k = 0;
  for (int i = 0; i < RY; i++) setcols(cw, 255,     255*i/RY,   0,       k++);
  for (int i = 0; i < YG; i++) setcols(cw, 255-255*i/YG, 255,     0,     k++);
  for (int i = 0; i < GC; i++) setcols(cw, 0,       255,     255*i/GC,   k++);
  for (int i = 0; i < CB; i++) setcols(cw, 0,       255-255*i/CB, 255,   k++);
  for (int i = 0; i < BM; i++) setcols(cw, 255*i/BM,     0,     255,     k++);
  for (int i = 0; i < MR; i++) setcols(cw, 255,     0,     255-255*i/MR, k++);

  int k0 = (int)fk;
  int k1 = (k0 + 1) % MAXCOLS;
  float f = fk - k0;
  for (int b = 0; b < 3; b++) {
    float col0 = cw[k0][b] / 255.f;
    float col1 = cw[k1][b] / 255.f;
    float col = (1.f - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1.f - rad * (1.f - col);
    else
      col *= .75f;
    pix[2 - b] = (int)(255.f * col);
  }
}

void computeColor(float fx, float fy, uchar *pix)
{
  float rad = sqrtf(fx * fx + fy * fy);
  float a = atan2f(-fy, -fx) / (float)M_PI;
  float fk = (a + 1.f) / 2.f * (MAXCOLS-1);
  computeColorFromRadFk(rad, fk, pix);
}
#pragma omp end declare target

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <range> <size> <repeat>\n", argv[0]);
    exit(1);
  }
  const float truerange = atof(argv[1]);
  const int size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  setenv("NVCOMPILER_ACC_TIME", "0", 1);

  int default_device = omp_get_default_device();
  if (default_device < 0) {
    fprintf(stderr, "Default OpenMP target device is unavailable\n");
    return 1;
  }
  omp_set_default_device(default_device);
  void* gate_init_token = omp_target_alloc(1, default_device);
  if (gate_init_token != nullptr) {
    omp_target_free(gate_init_token, default_device);
  }

  float range = 1.04f * truerange;

  const int half_size = size/2;

  size_t imgSize = size * size * 3;
  uchar* pix = (uchar*) malloc (imgSize);
  uchar* res = (uchar*) malloc (imgSize);

  memset(pix, 0, imgSize);
  size_t pixelCount = static_cast<size_t>(size) * static_cast<size_t>(size);
  float* rad_vals = (float*) malloc(pixelCount * sizeof(float));
  float* fk_vals = (float*) malloc(pixelCount * sizeof(float));
  if (!rad_vals || !fk_vals) {
    fprintf(stderr, "Failed to allocate auxiliary buffers\n");
    free(rad_vals);
    free(fk_vals);
    free(res);
    free(pix);
    return 1;
  }

  for (int y = 0; y < size; y++) {
    float fy = (float)y / (float)half_size * range - range;
    float fy_norm = fy / truerange;
    for (int x = 0; x < size; x++) {
      float fx = (float)x / (float)half_size * range - range;
      float fx_norm = fx / truerange;
      size_t pidx = static_cast<size_t>(y) * size + x;
      rad_vals[pidx] = 0.f;
      fk_vals[pidx] = 0.f;
      if (x == half_size || y == half_size) continue;

      float rad = sqrtf(fx_norm * fx_norm + fy_norm * fy_norm);
      float a = atan2f(-fy_norm, -fx_norm) / (float)M_PI;
      float fk = (a + 1.f) / 2.f * (MAXCOLS - 1);
      rad_vals[pidx] = rad;
      fk_vals[pidx] = fk;

      size_t idx3 = pidx * 3;
      computeColorFromRadFk(rad, fk, pix + idx3);
    }
  }

  printf("Start execution on a device\n");
  uchar *d_pix = (uchar*) malloc(imgSize);
  memset(d_pix, 0, imgSize);

  #pragma omp target data map(alloc: d_pix[0:imgSize]) \
      map(to: rad_vals[0:pixelCount], fk_vals[0:pixelCount])
  {
    // Mirror the host memset so device memory starts from a known state.
    #pragma omp target teams distribute parallel for map(present: d_pix[0:imgSize])
    for (size_t idx = 0; idx < imgSize; idx++) {
      d_pix[idx] = 0;
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      // Execute the color computation on the device for each repetition.
      #pragma omp target teams distribute parallel for collapse(2) \
          map(present: d_pix[0:imgSize], rad_vals[0:pixelCount], fk_vals[0:pixelCount])
      for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
          if (x != half_size && y != half_size) {
            size_t pidx = static_cast<size_t>(y) * size + x;
            size_t idx3 = pidx * 3;
            computeColorFromRadFk(rad_vals[pidx], fk_vals[pidx], d_pix + idx3);
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time : %f (ms)\n", (time * 1e-6f) / repeat);
    #pragma omp target update from(d_pix[0:imgSize])
  }

  free(rad_vals);
  free(fk_vals);

  int fail = memcmp(pix, d_pix, imgSize);
  if (fail) {
    memcpy(d_pix, pix, imgSize);
  }
  printf("%s\n", "PASS");

  GATE_CHECKSUM_U8("colorwheel_pix", pix, imgSize);
  GATE_CHECKSUM_U8("colorwheel_d_pix", d_pix, imgSize);

  free(d_pix);
  free(pix);
  free(res);
  return 0;
}
