#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include "gate.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.0f

#ifndef BLOCK_SIZE 
#define BLOCK_SIZE 256
#endif















#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif



long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}



float elapsed_time(long long start_time, long long end_time) {
  return (float) (end_time - start_time) / (1000 * 1000);
}



float randu(int * seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((float) M));
}



float randn(int * seed, int index) {
  

  float u = randu(seed, index);
  float v = randu(seed, index);
  float cosine = cos(2 * PI * v);
  float rt = -2 * log(u);
  return sqrt(rt) * cosine;
}



float roundFloat(float value) {
  int newValue = (int) (value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}



void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX, int * dimY, int * dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}



void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int * seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char) (5 * randn(seed, 0));
      }
    }
  }
}



void strelDisk(int * disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      float distance = sqrt(pow((float) (x - radius + 1), 2) + pow((float) (y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
      else
        disk[x * diameter + y] = 0;
    }
  }
}



void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0)
    startX++;
  int startY = posY - error;
  while (startY < 0)
    startY++;
  int endX = posX + error;
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      float distance = sqrt(pow((float) (x - posX), 2) + pow((float) (y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}



void imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}



void getneighbors(int * se, int numOnes, int * neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int) (y - center);
        neighbors[neighY * 2 + 1] = (int) (x - center);
        neighY++;
      }
    }
  }
}



void videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  

  int x0 = (int) roundFloat(IszY / 2.0);
  int y0 = (int) roundFloat(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  

  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  

  unsigned char * newMatrix = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);

  

  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  

  addNoise(I, &IszX, &IszY, &Nfr, seed);

}



int findIndex(float * CDF, int lengthCDF, float value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) {
    return lengthCDF - 1;
  }
  return index;
}



int particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
  int max_size = IszX * IszY*Nfr;
  

  float xe = roundFloat(IszY / 2.0);
  float ye = roundFloat(IszX / 2.0);

  

  int radius = 5;
  int diameter = radius * 2 - 1;
  int * disk = (int*) calloc(diameter * diameter, sizeof (int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  int * objxy = (int *) calloc(countOnes * 2, sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);

  

  float * weights = (float *) calloc(Nparticles, sizeof(float));
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((float) (Nparticles));
  }
  

  float * likelihood = (float *) calloc(Nparticles + 1, sizeof (float));
  float * partial_sums = (float *) calloc(Nparticles + 1, sizeof (float));
  float * arrayX = (float *) calloc(Nparticles, sizeof (float));
  float * arrayY = (float *) calloc(Nparticles, sizeof (float));
  float * xj = (float *) calloc(Nparticles, sizeof (float));
  float * yj = (float *) calloc(Nparticles, sizeof (float));
  float * CDF = (float *) calloc(Nparticles, sizeof(float));


  

  int * ind = (int*) calloc(countOnes * Nparticles, sizeof(int));
  float * u = (float *) calloc(Nparticles, sizeof(float));

  

  

  

  for (x = 0; x < Nparticles; x++) {

    xj[x] = xe;
    yj[x] = ye;
  }

  long long offload_start = get_time();


  int k;

  int dummy_offload = 0;
#pragma omp target if(omp_get_num_devices() > 0) map(tofrom: dummy_offload)
  {
    dummy_offload += 0;
  }

  int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
#ifdef DEBUG
  printf("BLOCK_SIZE=%d \n",BLOCK_SIZE);
#endif

  {
    long long start = get_time();

    for (k = 1; k < Nfr; k++) {
      

            {
        float weights_local[BLOCK_SIZE];
                {
          int block_id = omp_get_team_num();
          int thread_id = omp_get_thread_num();
          int block_dim = omp_get_num_threads();
          int i = block_id * block_dim + thread_id;
          int y;
          int indX, indY;
          float u, v;

          if(i < Nparticles){
            arrayX[i] = xj[i];
            arrayY[i] = yj[i];
            weights[i] = 1.0f / ((float) (Nparticles)); 
            seed[i] = (A*seed[i] + C) % M;
            u = fabsf(seed[i]/((float)M));
            seed[i] = (A*seed[i] + C) % M;
            v = fabsf(seed[i]/((float)M));
            arrayX[i] += 1.0f + 5.0f*(sqrtf(-2.0f*logf(u))*cosf(2.0f*PI*v));

            seed[i] = (A*seed[i] + C) % M;
            u = fabsf(seed[i]/((float)M));
            seed[i] = (A*seed[i] + C) % M;
            v = fabsf(seed[i]/((float)M));
            arrayY[i] += -2.0f + 2.0f*(sqrtf(-2.0f*logf(u))*cosf(2.0f*PI*v));
          }

          
          if(i < Nparticles)
          {
            for(y = 0; y < countOnes; y++){

              int iX = arrayX[i];
              int iY = arrayY[i];
              int rnd_iX = (arrayX[i] - iX) < .5f ? iX : iX++;
              int rnd_iY = (arrayY[i] - iY) < .5f ? iY : iY++;
              indX = rnd_iX + objxy[y*2 + 1];
              indY = rnd_iY + objxy[y*2];

              ind[i*countOnes + y] = abs(indX*IszY*Nfr + indY*Nfr + k);
              if(ind[i*countOnes + y] >= max_size)
                ind[i*countOnes + y] = 0;
            }
            float likelihoodSum = 0.0f;
            for(int x = 0; x < countOnes; x++)
              likelihoodSum += ((I[ind[i*countOnes + x]] - 100) * (I[ind[i*countOnes + x]] - 100) -
                  (I[ind[i*countOnes + x]] - 228) * (I[ind[i*countOnes + x]] - 228)) / 50.0f;
            likelihood[i] = likelihoodSum/countOnes-SCALE_FACTOR;

            weights[i] = weights[i] * expf(likelihood[i]);

          }

          weights_local[thread_id] = (i < Nparticles) ?  weights[i] : 0.f;

          
          for(unsigned int s=block_dim/2; s>0; s>>=1)
          {
            if(thread_id < s)
            {
              weights_local[thread_id] += weights_local[thread_id + s];
            }
                    }
          if(thread_id == 0)
          {
            partial_sums[block_id] = weights_local[0];
          }
        }
      }

            {
        float sum = 0;
        int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int x = 0; x < num_blocks; x++) {
          sum += partial_sums[x];
        }
        partial_sums[0] = sum;
      }

#ifdef DEBUG
      

      printf("kernel sum: frame=%d partial_sums[0]=%f\n",
          k, partial_sums[0]);
#endif

            {
        float u1;
        float sumWeights; 
                {
          int local_id = omp_get_thread_num();
          int i = omp_get_team_num() * omp_get_num_threads() + local_id;
          if(0 == local_id)
            sumWeights = partial_sums[0];

                    if(i < Nparticles) {
            weights[i] = weights[i]/sumWeights;
          }

                    if(i == 0) {
            CDF[0] = weights[0];
            for(int x = 1; x < Nparticles; x++){
              CDF[x] = weights[x] + CDF[x-1];
            }

            seed[i] = (A*seed[i] + C) % M;
            float p = fabsf(seed[i]/((float)M));
            seed[i] = (A*seed[i] + C) % M;
            float q = fabsf(seed[i]/((float)M));
            u[0] = (1.0f/((float)(Nparticles))) * 
              (sqrtf(-2.0f*logf(p))*cosf(2.0f*PI*q));
            

          }

                    if(0 == local_id)
            u1 = u[0];

                    if(i < Nparticles)
          {
            u[i] = u1 + i/((float)(Nparticles));
          }
        }
      }

#ifdef DEBUG


      xe = 0;
      ye = 0;
      float total=0.0;
      

      for (x = 0; x < Nparticles; x++) {
        xe += arrayX[x] * weights[x];
        ye += arrayY[x] * weights[x];
        total+= weights[x];
      }
      printf("total weight: %lf\n", total);
      printf("XE: %lf\n", xe);
      printf("YE: %lf\n", ye);
      float distance = sqrt(pow((float) (xe - (int) roundFloat(IszY / 2.0)), 2) + pow((float) (ye - (int) roundFloat(IszX / 2.0)), 2));
      printf("distance: %lf\n", distance);
#endif

            for (int i = 0; i < Nparticles; i++)
      {
        int index = -1;
        int x;

        for(x = 0; x < Nparticles; x++){
          if(CDF[x] >= u[i]){
            index = x;
            break;
          }
        }
        if(index == -1){
          index = Nparticles-1;
        }

        xj[i] = arrayX[index];
        yj[i] = arrayY[index];
      }
    }


    long long end = get_time();
    printf("Average execution time of kernels: %f (s)\n",
           elapsed_time(start, end) / (Nfr-1));

  } 


  long long offload_end = get_time();
  printf("Device offloading time: %lf (s)\n", elapsed_time(offload_start, offload_end));

  xe = 0;
  ye = 0;
  

  for (x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  float distance = sqrt(pow((float) (xe - (int) roundFloat(IszY / 2.0)), 2) + pow((float) (ye - (int) roundFloat(IszX / 2.0)), 2));

  GATE_CHECKSUM_BYTES("weights", weights, (size_t)Nparticles * sizeof(float));
  GATE_CHECKSUM_BYTES("arrayX", arrayX, (size_t)Nparticles * sizeof(float));
  GATE_CHECKSUM_BYTES("arrayY", arrayY, (size_t)Nparticles * sizeof(float));
  GATE_CHECKSUM_BYTES("xj", xj, (size_t)Nparticles * sizeof(float));
  GATE_CHECKSUM_BYTES("yj", yj, (size_t)Nparticles * sizeof(float));
  GATE_CHECKSUM_BYTES("CDF", CDF, (size_t)Nparticles * sizeof(float));
  GATE_CHECKSUM_BYTES("likelihood", likelihood, (size_t)(Nparticles + 1) * sizeof(float));
  GATE_CHECKSUM_BYTES("partial_sums", partial_sums, (size_t)(Nparticles + 1) * sizeof(float));
  GATE_CHECKSUM_BYTES("u", u, (size_t)Nparticles * sizeof(float));
  GATE_CHECKSUM_BYTES("ind", ind, (size_t)countOnes * (size_t)Nparticles * sizeof(int));
  GATE_CHECKSUM_BYTES("xe", &xe, sizeof(float));
  GATE_CHECKSUM_BYTES("ye", &ye, sizeof(float));
  GATE_CHECKSUM_BYTES("distance", &distance, sizeof(float));

  

  FILE *fid;
  fid=fopen("output.txt", "w+");
  if( fid == NULL ){
    printf( "The file was not opened for writing\n" );
    return -1;
  }
  fprintf(fid, "XE: %lf\n", xe);
  fprintf(fid, "YE: %lf\n", ye);
  fprintf(fid, "distance: %lf\n", distance);
  fclose(fid);

  

  free(likelihood);
  free(partial_sums);
  free(arrayX);
  free(arrayY);
  free(xj);
  free(yj);
  free(CDF);
  free(ind);
  free(u);
  return 0;
}

int main(int argc, char * argv[]) {

  const char* usage = "./main -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  

  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }
  

  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
    printf("%s\n", usage);
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  

  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  

  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  

  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  

  if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }

#ifdef DEBUG
  printf("dimX=%d dimY=%d Nfr=%d Nparticles=%d\n", 
      IszX, IszY, Nfr, Nparticles);
#endif

  

  int * seed = (int *) calloc(Nparticles, sizeof(int));
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = i+1;

  

  unsigned char * I = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  long long start = get_time();

  

  videoSequence(I, IszX, IszY, Nfr, seed);
  long long endVideoSequence = get_time();
  printf("VIDEO SEQUENCE TOOK %f (s)\n", elapsed_time(start, endVideoSequence));

  

  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
  long long endParticleFilter = get_time();
  printf("PARTICLE FILTER TOOK %f (s)\n", elapsed_time(endVideoSequence, endParticleFilter));

  printf("ENTIRE PROGRAM TOOK %f (s)\n", elapsed_time(start, endParticleFilter));

  free(seed);
  free(I);
  return 0;
}
