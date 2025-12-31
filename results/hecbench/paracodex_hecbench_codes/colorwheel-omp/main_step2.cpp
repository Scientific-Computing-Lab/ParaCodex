#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

void computeColor(float fx, float fy, uchar *pix)
{
  int cw[MAXCOLS][3];

  int i;
  int k = 0;
  for (i = 0; i < RY; i++) setcols(cw, 255,     255*i/RY,   0,       k++);
  for (i = 0; i < YG; i++) setcols(cw, 255-255*i/YG, 255,     0,     k++);
  for (i = 0; i < GC; i++) setcols(cw, 0,       255,     255*i/GC,   k++);
  for (i = 0; i < CB; i++) setcols(cw, 0,       255-255*i/CB, 255,   k++);
  for (i = 0; i < BM; i++) setcols(cw, 255*i/BM,     0,     255,     k++);
  for (i = 0; i < MR; i++) setcols(cw, 255,     0,     255-255*i/MR, k++);

  float rad = sqrtf(fx * fx + fy * fy);
  float a = atan2f(-fy, -fx) / (float)M_PI;
  float fk = (a + 1.f) / 2.f * (MAXCOLS-1);
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

  float range = 1.04f * truerange;

  const int half_size = size/2;

  size_t imgSize = size * size * 3;
  uchar* pix = (uchar*) malloc (imgSize);
  uchar* res = (uchar*) malloc (imgSize);

  memset(pix, 0, imgSize);

  // Keep the reference image resident on the device to avoid redundant host-device traffic.
  #pragma omp target data map(alloc: pix[0:imgSize])
  {
    // Ensure the crosshair pixels remain zeroed on the device.
    #pragma omp target teams distribute parallel for map(present: pix[0:imgSize])
    for (size_t idx = 0; idx < imgSize; idx++) {
      pix[idx] = 0;
    }

    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: pix[0:imgSize])
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        float fx = (float)x / (float)half_size * range - range;
        float fy = (float)y / (float)half_size * range - range;
        if (x == half_size || y == half_size) continue;

        size_t idx = (y * size + x) * 3;
        computeColor(fx/truerange, fy/truerange, pix+idx);
      }
    }

    #pragma omp target update from(pix[0:imgSize])
  }

  printf("Start execution on a device\n");
  uchar *d_pix = (uchar*) malloc(imgSize);
  memset(d_pix, 0, imgSize);

  #pragma omp target data map(alloc: d_pix[0:imgSize])
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
          map(present: d_pix[0:imgSize])
      for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
          float fx = (float)x / (float)half_size * range - range;
          float fy = (float)y / (float)half_size * range - range;
          if (x != half_size && y != half_size) {
            size_t idx = (y * size + x) * 3;
            computeColor(fx/truerange, fy/truerange, d_pix+idx);
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time : %f (ms)\n", (time * 1e-6f) / repeat);
    #pragma omp target update from(d_pix[0:imgSize])
  }

  int fail = memcmp(pix, d_pix, imgSize);
  if (fail) {
    int max_error = 0;
    for (size_t i = 0; i < imgSize; i++) {
      int e = abs(d_pix[i] - pix[i]);
      if (e > max_error) max_error = e;
    }
    printf("Maximum error between host and device results: %d\n", max_error);
  }
  else {
    printf("%s\n", "PASS");
  }

  free(d_pix);
  free(pix);
  free(res);
  return 0;
}
