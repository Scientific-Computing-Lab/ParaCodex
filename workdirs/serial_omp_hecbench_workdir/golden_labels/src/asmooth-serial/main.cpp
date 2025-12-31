#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

#include "reference.cpp"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("./%s <image dimension> <threshold> <max box size> <iterations>\n", argv[0]);
    exit(1);
  }

  

  const int Lx = atoi(argv[1]);
  const int Ly = Lx;
  const int size = Lx * Ly;

  const int Threshold = atoi(argv[2]);
  const int MaxRad = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const size_t size_bytes = size * sizeof(float);
  const size_t box_bytes = size * sizeof(int);

  

  float *img = (float*) malloc (size_bytes);

  

  float *norm = (float*) malloc (size_bytes);
  float *h_norm = (float*) malloc (size_bytes);

  int *box = (int*) malloc (box_bytes);
  int *h_box = (int*) malloc (box_bytes);

  float *out = (float*) malloc (size_bytes);
  float *h_out = (float*) malloc (size_bytes);

  srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = rand() % 256;
    norm[i] = box[i] = out[i] = 0;
  }

  double time = 0;

    {
    for (int i = 0; i < repeat; i++) {
      

            

      
      auto start = std::chrono::steady_clock::now();

      

            for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
          float sum = 0.f;
          int s = 1;
          int q = 1;
          int ksum = 0;

          while (sum < Threshold && q < MaxRad) {
            s = q;
            sum = 0.f;
            ksum = 0;

            for (int i = -s; i < s+1; i++)
              for (int j = -s; j < s+1; j++)
                if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
                  sum += img[(x+i)*Ly+y+j];
                  ksum++;
                }
            q++;
          }

          box[x*Ly+y] = s;  


          for (int i = -s; i < s+1; i++)
            for (int j = -s; j < s+1; j++)
              if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly)
                if (ksum != 0) {
                                    norm[(x+i)*Ly+y+j] += 1.f / (float)ksum;
                }
        }
      }

      

            for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++) 
          if (norm[x*Ly+y] != 0) img[x*Ly+y] /= norm[x*Ly+y];

      

            for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
          int s = box[x*Ly+y];
          float sum = 0.f;
          int ksum = 0;

          

          for (int i = -s; i < s+1; i++)
            for (int j = -s; j < s+1; j++) {
              if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
                sum += img[(x+i)*Ly+y+j];
                ksum++;
              }
            }
          if (ksum != 0) out[x*Ly+y] = sum / (float)ksum;
        }
      }
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    printf("Average filtering time %lf (s)\n", (time * 1e-9) / repeat);

              }

  

  reference (Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);
  verify(size, MaxRad, norm, h_norm, out, h_out, box, h_box);

  free(img);
  free(norm);
  free(h_norm);
  free(box);
  free(h_box);
  free(out);
  free(h_out);
  return 0;
}