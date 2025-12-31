#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define CHUNK_S 4096

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <#samples> <#voxels> <verify>\n", argv[0]);
    exit(1);
  }
  const int samples = atoi(argv[1]); 

  const int voxels = atoi(argv[2]);  

  const int verify = atoi(argv[3]);
  const int sampleSize = samples * sizeof(float);
  const int voxelSize = voxels * sizeof(float);

  float *h_rmu = (float*) malloc (voxelSize);
  float *h_imu = (float*) malloc (voxelSize);
  float *h_kx = (float*) malloc (voxelSize);
  float *h_ky = (float*) malloc (voxelSize);
  float *h_kz = (float*) malloc (voxelSize);

  float *h_rfhd = (float*) malloc (sampleSize);
  float *h_ifhd = (float*) malloc (sampleSize);
  float *h_x = (float*) malloc (sampleSize);
  float *h_y = (float*) malloc (sampleSize);
  float *h_z = (float*) malloc (sampleSize);

  

  float *rfhd = (float*) malloc (sampleSize);
  float *ifhd = (float*) malloc (sampleSize);

  srand(2);
  for (int i = 0; i < samples; i++) {
    rfhd[i] = h_rfhd[i] = (float)i/samples;
    ifhd[i] = h_ifhd[i] = (float)i/samples;
    h_x[i] = 0.3f + (rand()%2 ? 0.1 : -0.1);
    h_y[i] = 0.2f + (rand()%2 ? 0.1 : -0.1);
    h_z[i] = 0.1f + (rand()%2 ? 0.1 : -0.1);
  }

  for (int i = 0; i < voxels; i++) {
    h_rmu[i] = (float)i/voxels;
    h_imu[i] = (float)i/voxels;
    h_kx[i] = 0.1f + (rand()%2 ? 0.1 : -0.1);
    h_ky[i] = 0.2f + (rand()%2 ? 0.1 : -0.1);
    h_kz[i] = 0.3f + (rand()%2 ? 0.1 : -0.1);
  }

  printf("Run FHd on a device\n");

    {
    auto start = std::chrono::steady_clock::now();

        for (int n = 0; n < samples; n++) {
      float r = rfhd[n];
      float i = ifhd[n];
      float xn = h_x[n];
      float yn = h_y[n];
      float zn = h_z[n];
            for (int m = 0; m < voxels; m++) {
        float e = 2.f * (float)M_PI * 
                  (h_kx[m] * xn + h_ky[m] * yn + h_kz[m] * zn);
        float c = cosf(e);
        float s = sinf(e);
        r += h_rmu[m] * c - h_imu[m] * s;
        i += h_imu[m] * c + h_rmu[m] * s;
      }
      rfhd[n] = r;
      ifhd[n] = i;   
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Device execution time %f (s)\n", time * 1e-9f);
  }
  
  if (verify) {
    printf("Computing root mean square error between host and device results.\n");
    printf("This will take a while..\n");

        for (int n = 0; n < samples; n++) {
      float r = h_rfhd[n];
      float i = h_ifhd[n];
            for (int m = 0; m < voxels; m++) {
        float e = 2.f * (float)M_PI * 
                  (h_kx[m] * h_x[n] + h_ky[m] * h_y[n] + h_kz[m] * h_z[n]);
        float c = cosf(e);
        float s = sinf(e);
        r += h_rmu[m] * c - h_imu[m] * s;
        i += h_imu[m] * c + h_rmu[m] * s;
      }
      h_rfhd[n] = r;
      h_ifhd[n] = i;   
    }

    float err = 0.f;
    for (int i = 0; i < samples; i++) {
      err += (h_rfhd[i] - rfhd[i]) * (h_rfhd[i] - rfhd[i]) +
             (h_ifhd[i] - ifhd[i]) * (h_ifhd[i] - ifhd[i]) ;
    }
    printf("RMSE = %f\n", sqrtf(err / (2*samples)));
  }
 
  free(h_rmu);
  free(h_imu);
  free(h_kx);
  free(h_ky);
  free(h_kz);
  free(h_rfhd);
  free(h_ifhd);
  free(rfhd);
  free(ifhd);
  free(h_x);
  free(h_y);
  free(h_z);

  return 0;
}