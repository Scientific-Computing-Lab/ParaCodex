#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>

typedef float IMAGE_T;
typedef int INT_T;




inline int _clip(const int x, const int low, const int high)
{
  if (x > high)
    return high;
  else if (x < low)
    return low;
  else
    return x;
}





inline IMAGE_T _integ(const IMAGE_T * img,
                      const INT_T img_rows,
                      const INT_T img_cols,
                      int r,
                      int c,
                      const int rl,
                      const int cl)
{
  r = _clip(r, 0, img_rows - 1);
  c = _clip(c, 0, img_cols - 1);

  const int r2 = _clip(r + rl, 0, img_rows - 1);
  const int c2 = _clip(c + cl, 0, img_cols - 1);

  IMAGE_T ans = img[r * img_cols + c] + img[r2 * img_cols + c2] -
                img[r * img_cols + c2] - img[r2 * img_cols + c];

  return fmax((IMAGE_T)0, ans);
}





void hessian_matrix_det(const IMAGE_T* img,
                        const INT_T img_rows,
                        const INT_T img_cols,
                        const IMAGE_T sigma,
                        IMAGE_T* out)
{
    for (int tid = 0; tid < img_rows*img_cols; tid++) {

    const int r = tid / img_cols;
    const int c = tid % img_cols;

    int size = (int)((IMAGE_T)3.0 * sigma);

    const int b = (size - 1) / 2 + 1;
    const int l = size / 3;
    const int w = size;

    const IMAGE_T w_i = (IMAGE_T)1.0 / (size * size);

    const IMAGE_T tl = _integ(img, img_rows, img_cols, r - l, c - l, l, l); 

    const IMAGE_T br = _integ(img, img_rows, img_cols, r + 1, c + 1, l, l); 

    const IMAGE_T bl = _integ(img, img_rows, img_cols, r - l, c + 1, l, l); 

    const IMAGE_T tr = _integ(img, img_rows, img_cols, r + 1, c - l, l, l); 


    IMAGE_T dxy = bl + tr - tl - br;
    dxy = -dxy * w_i;

    IMAGE_T mid = _integ(img, img_rows, img_cols, r - l + 1, c - l, 2 * l - 1, w);  

    IMAGE_T side = _integ(img, img_rows, img_cols, r - l + 1, c - l / 2, 2 * l - 1, l);  


    IMAGE_T dxx = mid - (IMAGE_T)3 * side;
    dxx = -dxx * w_i;

    mid = _integ(img, img_rows, img_cols, r - l, c - b + 1, w, 2 * b - 1);
    side = _integ(img, img_rows, img_cols, r - b / 2, c - b + 1, b, 2 * b - 1);

    IMAGE_T dyy = mid - (IMAGE_T)3 * side;
    dyy = -dyy * w_i;

    out[tid] = (dxx * dyy - (IMAGE_T)0.81 * (dxy * dxy));
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  int h = atoi(argv[1]);
  int w = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  int img_size = h * w;
  size_t img_size_bytes = sizeof(float) * img_size;

  float *input_img = (float*) malloc (img_size_bytes);
  float *integral_img = (float*) malloc (img_size_bytes);
  float *output_img = (float*) malloc (img_size_bytes);

  std::default_random_engine rng (123);
  std::normal_distribution<float> norm_dist(0.f, 1.f);

  for (int i = 0; i < img_size; i++) {
    input_img[i] = norm_dist(rng);
  }

  printf("Integrating the input image may take a while...\n"); 
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      IMAGE_T s = 0;
      for (int y = 0; y <= i; y++)
        for (int x = 0; x <= j; x++)
          s += input_img[y * w + x];
      integral_img[i * w + j] = s;
    }
  }

  long time;

    {
    const IMAGE_T sigma = 4.0;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hessian_matrix_det(integral_img, h, w, sigma, output_img);
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  double checksum = 0;
  for (int i = 0; i < img_size; i++) {
    checksum += output_img[i];
  }

  free(input_img);
  free(integral_img);
  free(output_img);

  printf("Average kernel execution time : %f (us)\n", time * 1e-3 / repeat);
  printf("Kernel checksum: %lf\n", checksum);

  return 0;
}