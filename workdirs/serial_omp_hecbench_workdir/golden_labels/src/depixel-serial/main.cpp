#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include "kernels.h"

int main(int argc, char** argv) {

  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]);
    return 1;
  }

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  int size = width * height;
  size_t size_output_bytes = size * sizeof(uint);
  size_t size_image_bytes = size * sizeof(float3);

  std::mt19937 gen(19937);
  

  std::uniform_real_distribution<float> dis(0.f, 0.4f); 

  float3 *img = (float3*) malloc(size_image_bytes);

  uint *out = (uint*) malloc(size_output_bytes);
  uint *tmp = (uint*) malloc(size_output_bytes);

  float sum = 0;
  float total_time = 0;

    {
    for (int n = 0; n < repeat; n++) {

      for (int i = 0; i < size; i++) {
        img[i].x = dis(gen);
        img[i].y = dis(gen);
        img[i].z = dis(gen);
      }

      
      auto start = std::chrono::steady_clock::now();

      check_connect(img, tmp, width, height);
      eliminate_crosses(tmp, out, width, height);

      auto end = std::chrono::steady_clock::now();

      
      std::chrono::duration<float> time = end - start;
      total_time += time.count();

      float lsum = 0;
      for (int i = 0; i < size; i++)
        lsum += (out[i] & 0xff) / 256.f + 
               ((out[i] >> 8) & 0xff) / 256.f + 
               ((out[i] >> 16) & 0xff) / 256.f + 
               ((out[i] >> 24) & 0xff) / 256.f;

      sum += lsum / size;
    }
  }

  printf("Image size: %d (width) x %d (height)\ncheckSum: %f\n",
         width, height, sum);
  printf("Average kernel time over %d iterations: %f (s)\n",
         repeat, total_time / repeat);

  free(out);
  free(img);
  free(tmp);

  return 0;
}