

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include "reference.h"

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    printf("Usage: %s <input image> <output image> <iterations>\n", argv[0]) ;
    return -1 ;
  }

  unsigned short    input_image[Y_SIZE*X_SIZE] __attribute__((aligned(1024)));
  unsigned short    output_image[Y_SIZE*X_SIZE] __attribute__((aligned(1024)));
  unsigned short    output_image_ref[Y_SIZE*X_SIZE] __attribute__((aligned(1024)));

  

  std::cout << "Reading input image...\n";

  

  const char *inputImageFilename = argv[1];
  FILE *input_file = fopen(inputImageFilename, "rb");
  if (!input_file)
  {
    printf("Error: Unable to open input image file %s!\n", inputImageFilename);
    return 1;
  }

  printf("\n");
  printf("   Reading RAW Image\n");
  size_t items_read = fread(input_image, sizeof(input_image), 1, input_file);
  printf("   Bytes read = %d\n\n", (int)(items_read * sizeof(input_image)));
  fclose(input_file);

  const int iterations = atoi(argv[3]);

    {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; i++) {
            for (int y = 0; y < Y_SIZE; y++)
        for (int x = 0; x < X_SIZE; x++) {

          const float lx_rot   = 30.0f;
          const float ly_rot   = 0.0f; 
          const float lx_expan = 0.5f;
          const float ly_expan = 0.5f; 
          int   lx_move  = 0;
          int   ly_move  = 0;
          float affine[2][2];   

          float i_affine[2][2];
          float beta[2];
          float i_beta[2];
          float det;
          float x_new, y_new;
          float x_frac, y_frac;
          float gray_new;
          int   m, n;
          unsigned short output_buffer;

          

          affine[0][0] = lx_expan * cosf(lx_rot*PI/180.0f);
          affine[0][1] = ly_expan * sinf(ly_rot*PI/180.0f);
          affine[1][0] = lx_expan * sinf(lx_rot*PI/180.0f);
          affine[1][1] = ly_expan * cosf(ly_rot*PI/180.0f);
          beta[0]      = lx_move;
          beta[1]      = ly_move;

          

          det = (affine[0][0] * affine[1][1]) - (affine[0][1] * affine[1][0]);
          if (det == 0.0f)
          {
            i_affine[0][0] = 1.0f;
            i_affine[0][1] = 0.0f;
            i_affine[1][0] = 0.0f;
            i_affine[1][1] = 1.0f;
            i_beta[0]      = -beta[0];
            i_beta[1]      = -beta[1];
          } 
          else 
          {
            i_affine[0][0] =  affine[1][1]/det;
            i_affine[0][1] = -affine[0][1]/det;
            i_affine[1][0] = -affine[1][0]/det;
            i_affine[1][1] =  affine[0][0]/det;
            i_beta[0]      = -i_affine[0][0]*beta[0]-i_affine[0][1]*beta[1];
            i_beta[1]      = -i_affine[1][0]*beta[0]-i_affine[1][1]*beta[1];
          }

          


          x_new  = i_beta[0] + i_affine[0][0]*(x-X_SIZE/2.0f) + i_affine[0][1]*(y-Y_SIZE/2.0f) + X_SIZE/2.0f;
          y_new  = i_beta[1] + i_affine[1][0]*(x-X_SIZE/2.0f) + i_affine[1][1]*(y-Y_SIZE/2.0f) + Y_SIZE/2.0f;

          m      = (int)floorf(x_new);
          n      = (int)floorf(y_new);

          x_frac = x_new - m;
          y_frac = y_new - n;

          if ((m >= 0) && (m + 1 < X_SIZE) && (n >= 0) && (n+1 < Y_SIZE))
          {
            gray_new = (1.0f - y_frac) * ((1.0f - x_frac) * (input_image[(n * X_SIZE) + m])  + 
                x_frac * (input_image[(n * X_SIZE) + m + 1])) + 
              y_frac  * ((1.0f - x_frac) * (input_image[((n + 1) * X_SIZE) + m]) + 
                  x_frac * (input_image[((n + 1) * X_SIZE) + m + 1]));

            output_buffer = (unsigned short)gray_new;
          } 
          else if (((m + 1 == X_SIZE) && (n >= 0) && (n < Y_SIZE)) || ((n + 1 == Y_SIZE) && (m >= 0) && (m < X_SIZE))) 
          {
            output_buffer = input_image[(n * X_SIZE) + m];
          } 
          else 
          {
            output_buffer = WHITE;
          }

          output_image[(y * X_SIZE)+x] = output_buffer;
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "   Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";
  }

  

  affine_reference(input_image, output_image_ref);
  int max_error = 0;
  for (int y = 0; y < Y_SIZE; y++) {
    for (int x = 0; x < X_SIZE; x++) {
      max_error = std::max(max_error, std::abs(output_image[y*X_SIZE+x] - output_image_ref[y*X_SIZE+x]));
    }
  }
  printf("   Max output error is %d\n\n", max_error);

  printf("   Writing RAW Image\n");
  const char *outputImageFilename = argv[2];
  FILE *output_file = fopen(outputImageFilename, "wb");
  if (!output_file)
  {
    printf("Error: Unable to write  image file %s!\n", outputImageFilename);
    return 1;
  }
  size_t items_written = fwrite(output_image, sizeof(output_image), 1, output_file);
  printf("   Bytes written = %d\n\n", (int)(items_written * sizeof(output_image)));
  fclose(output_file);

  return 0 ;
}