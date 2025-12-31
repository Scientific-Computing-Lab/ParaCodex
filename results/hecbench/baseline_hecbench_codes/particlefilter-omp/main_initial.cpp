#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <climits>
#include <sys/time.h>
#include <omp.h>

#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.0f

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#pragma omp declare target

static inline float clamp_min(float v, float min_v) {
  return (v < min_v) ? min_v : v;
}

static inline float randu(int *seed, int index) {
  long long num = static_cast<long long>(A) * static_cast<long long>(seed[index]) + C;
  seed[index] = static_cast<int>(num % M);
  float value = fabsf(seed[index] / static_cast<float>(M));
  return value > 1.0f ? 1.0f : value;
}

static inline float randn(int *seed, int index) {
  float u = clamp_min(randu(seed, index), 1.0e-12f);
  float v = randu(seed, index);
  float cosine = cosf(2.0f * PI * v);
  float rt = -2.0f * logf(u);
  return sqrtf(rt) * cosine;
}

static inline float roundFloat(float value) {
  int newValue = static_cast<int>(value);
  if (value - static_cast<float>(newValue) < 0.5f) {
    return static_cast<float>(newValue);
  }
  return static_cast<float>(newValue++);
}

static inline int findIndexDevice(const float *CDF, int lengthCDF, float value) {
  if (value <= CDF[0]) {
    return 0;
  }
  float last = CDF[lengthCDF - 1];
  if (value >= last) {
    return lengthCDF - 1;
  }
  int low = 0;
  int high = lengthCDF - 1;
  while (low < high) {
    int mid = (low + high) >> 1;
    if (CDF[mid] >= value) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return low;
}

static inline int linear_index(int indX, int indY, int frame, int yzStride, int frameStride, int max_size) {
  long long idx = static_cast<long long>(indX) * yzStride +
                  static_cast<long long>(indY) * frameStride + frame;
  if (idx < 0) {
    idx = -idx;
  }
  if (idx >= max_size) {
    idx = 0;
  }
  return static_cast<int>(idx);
}

#pragma omp end declare target

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000LL) + tv.tv_usec;
}

float elapsed_time(long long start_time, long long end_time) {
  return static_cast<float>(end_time - start_time) / (1000.0f * 1000.0f);
}

void setIf(int testValue, int newValue, unsigned char *array3D, int *dimXPtr, int *dimYPtr, int *dimZPtr) {
  int dimX = *dimXPtr;
  int dimY = *dimYPtr;
  int dimZ = *dimZPtr;
  if (dimX <= 0 || dimY <= 0 || dimZ <= 0) {
    return;
  }
  for (int x = 0; x < dimX; ++x) {
    for (int y = 0; y < dimY; ++y) {
      for (int z = 0; z < dimZ; ++z) {
        size_t idx = static_cast<size_t>(x) * dimY * dimZ +
                     static_cast<size_t>(y) * dimZ + z;
        if (array3D[idx] == static_cast<unsigned char>(testValue)) {
          array3D[idx] = static_cast<unsigned char>(newValue);
        }
      }
    }
  }
}

void addNoise(unsigned char *array3D, int *dimXPtr, int *dimYPtr, int *dimZPtr, int *seed) {
  int dimX = *dimXPtr;
  int dimY = *dimYPtr;
  int dimZ = *dimZPtr;
  if (dimX <= 0 || dimY <= 0 || dimZ <= 0) {
    return;
  }
  for (int x = 0; x < dimX; ++x) {
    for (int y = 0; y < dimY; ++y) {
      for (int z = 0; z < dimZ; ++z) {
        size_t idx = static_cast<size_t>(x) * dimY * dimZ +
                     static_cast<size_t>(y) * dimZ + z;
        array3D[idx] = static_cast<unsigned char>(
            array3D[idx] + static_cast<unsigned char>(5.0f * randn(seed, 0)));
      }
    }
  }
}

void strelDisk(int *disk, int radius) {
  int diameter = radius * 2 - 1;
  for (int x = 0; x < diameter; ++x) {
    for (int y = 0; y < diameter; ++y) {
      float distance = sqrtf(powf(static_cast<float>(x - radius + 1), 2.0f) +
                             powf(static_cast<float>(y - radius + 1), 2.0f));
      disk[x * diameter + y] = (distance < radius) ? 1 : 0;
    }
  }
}

void dilate_matrix(unsigned char *matrix, int posX, int posY, int posZ,
                   int dimX, int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0) {
    startX++;
  }
  int startY = posY - error;
  while (startY < 0) {
    startY++;
  }
  int endX = posX + error;
  while (endX > dimX) {
    endX--;
  }
  int endY = posY + error;
  while (endY > dimY) {
    endY--;
  }
  for (int x = startX; x < endX; ++x) {
    for (int y = startY; y < endY; ++y) {
      float distance = sqrtf(powf(static_cast<float>(x - posX), 2.0f) +
                             powf(static_cast<float>(y - posY), 2.0f));
      if (distance < error) {
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
      }
    }
  }
}

void imdilate_disk(unsigned char *matrix, int dimX, int dimY, int dimZ, int error,
                   unsigned char *newMatrix) {
  size_t total = static_cast<size_t>(dimX) * dimY * dimZ;
  std::fill(newMatrix, newMatrix + total, 0);
  for (int z = 0; z < dimZ; ++z) {
    for (int x = 0; x < dimX; ++x) {
      for (int y = 0; y < dimY; ++y) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}

void getneighbors(int *se, int numOnes, int *neighbors, int radius) {
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (int x = 0; x < diameter; ++x) {
    for (int y = 0; y < diameter; ++y) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = y - center;
        neighbors[neighY * 2 + 1] = x - center;
        neighY++;
      }
    }
  }
}

void videoSequence(unsigned char *I, int IszX, int IszY, int Nfr, int *seed) {
  int max_size = IszX * IszY * Nfr;
  if (max_size == 0) {
    return;
  }

  int x0 = static_cast<int>(roundFloat(IszY / 2.0f));
  int y0 = static_cast<int>(roundFloat(IszX / 2.0f));
  I[x0 * IszY * Nfr + y0 * Nfr] = 1;

  for (int k = 1; k < Nfr; ++k) {
    int xk = std::abs(x0 + (k - 1));
    int yk = std::abs(y0 - 2 * (k - 1));
    int pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size) {
      pos = 0;
    }
    I[pos] = 1;
  }

  std::vector<unsigned char> newMatrix(max_size, 0);
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix.data());
  for (int x = 0; x < IszX; ++x) {
    for (int y = 0; y < IszY; ++y) {
      for (int k = 0; k < Nfr; ++k) {
        I[x * IszY * Nfr + y * Nfr + k] =
            newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }

  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  addNoise(I, &IszX, &IszY, &Nfr, seed);
}

int particleFilter(unsigned char *I, int IszX, int IszY, int Nfr,
                   int *seed, int Nparticles) {
  if (Nparticles <= 0 || IszX <= 0 || IszY <= 0 || Nfr <= 0) {
    return -1;
  }
  int max_size = IszX * IszY * Nfr;
  if (max_size == 0) {
    return -1;
  }

  float xe_est = roundFloat(IszY / 2.0f);
  float ye_est = roundFloat(IszX / 2.0f);

  int radius = 5;
  int diameter = radius * 2 - 1;
  std::vector<int> disk(diameter * diameter, 0);
  strelDisk(disk.data(), radius);
  int countOnes = std::count(disk.begin(), disk.end(), 1);
  if (countOnes == 0) {
    return -1;
  }
  std::vector<int> objxy(countOnes * 2);
  getneighbors(disk.data(), countOnes, objxy.data(), radius);

  std::vector<float> weights(Nparticles, 1.0f / static_cast<float>(Nparticles));
  std::vector<float> arrayX(Nparticles, xe_est);
  std::vector<float> arrayY(Nparticles, ye_est);
  std::vector<float> xj(Nparticles, xe_est);
  std::vector<float> yj(Nparticles, ye_est);
  std::vector<float> CDF(Nparticles, 0.0f);

  unsigned char *I_ptr = I;
  int *objxy_ptr = objxy.data();
  float *weights_ptr = weights.data();
  float *arrayX_ptr = arrayX.data();
  float *arrayY_ptr = arrayY.data();
  float *xj_ptr = xj.data();
  float *yj_ptr = yj.data();
  float *CDF_ptr = CDF.data();

  const int yzStride = IszY * Nfr;
  const int frameStride = Nfr;
  const float invParticles = 1.0f / static_cast<float>(Nparticles);
  const float invCountOnes = 1.0f / static_cast<float>(countOnes);
  int numBlocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (numBlocks <= 0) {
    numBlocks = 1;
  }
  std::vector<float> blockAggregates(numBlocks, 0.0f);

  long long offload_start = get_time();
  long long kernels_start = get_time();

#pragma omp target data                                                                            \
    map(to : I_ptr[0:max_size], objxy_ptr[0:countOnes * 2])                                       \
    map(tofrom                                                                                     \
        : arrayX_ptr[0:Nparticles], arrayY_ptr[0:Nparticles], xj_ptr[0:Nparticles],                \
          yj_ptr[0:Nparticles], weights_ptr[0:Nparticles], seed[0:Nparticles])                     \
    map(alloc : CDF_ptr[0:Nparticles]) map(tofrom : blockAggregates[0:numBlocks])
  {
    for (int k = 1; k < Nfr; ++k) {
      float weightSum = 0.0f;
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)                          \
          map(present                                                                              \
              : I_ptr[0:max_size], objxy_ptr[0:countOnes * 2], arrayX_ptr[0:Nparticles],           \
                arrayY_ptr[0:Nparticles], xj_ptr[0:Nparticles], yj_ptr[0:Nparticles],              \
                weights_ptr[0:Nparticles], seed[0:Nparticles])                                     \
          reduction(+ : weightSum) firstprivate(k, yzStride, frameStride, max_size, countOnes,     \
                                                invParticles, invCountOnes)
      for (int i = 0; i < Nparticles; ++i) {
        float x_val = xj_ptr[i];
        float y_val = yj_ptr[i];
        weights_ptr[i] = invParticles;

        float noiseX = 1.0f + 5.0f * randn(seed, i);
        float noiseY = -2.0f + 2.0f * randn(seed, i);

        x_val += noiseX;
        y_val += noiseY;

        arrayX_ptr[i] = x_val;
        arrayY_ptr[i] = y_val;

        int baseX = static_cast<int>(x_val);
        int baseY = static_cast<int>(y_val);

        float likelihoodSum = 0.0f;
        for (int j = 0; j < countOnes; ++j) {
          int indX = baseX + objxy_ptr[j * 2 + 1];
          int indY = baseY + objxy_ptr[j * 2];
          int index = linear_index(indX, indY, k, yzStride, frameStride, max_size);
          float pixel = static_cast<float>(I_ptr[index]);
          float diff1 = pixel - 100.0f;
          float diff2 = pixel - 228.0f;
          likelihoodSum += (diff1 * diff1 - diff2 * diff2) * 0.02f;
        }

        float likelihoodVal = likelihoodSum * invCountOnes - SCALE_FACTOR;
        float weight = weights_ptr[i] * expf(likelihoodVal);
        weights_ptr[i] = weight;
        weightSum += weight;
      }

      float invSum = (weightSum > 0.0f && std::isfinite(weightSum)) ? (1.0f / weightSum) : 1.0f;
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)                          \
          map(present : weights_ptr[0:Nparticles])
      for (int i = 0; i < Nparticles; ++i) {
        weights_ptr[i] *= invSum;
      }

      std::fill(blockAggregates.begin(), blockAggregates.end(), 0.0f);
#pragma omp target update to(blockAggregates[0:numBlocks])

#pragma omp target teams num_teams(numBlocks) thread_limit(BLOCK_SIZE)                             \
          map(present                                                                              \
              : weights_ptr[0:Nparticles], CDF_ptr[0:Nparticles], blockAggregates[0:numBlocks])
      {
        int block = omp_get_team_num();
        int block_start = block * BLOCK_SIZE;
        int remaining = Nparticles - block_start;
        int block_count = remaining > BLOCK_SIZE
                               ? BLOCK_SIZE
                               : (remaining > 0 ? remaining : 0);
        float scan_local[BLOCK_SIZE];

#pragma omp parallel num_threads(BLOCK_SIZE)
        {
          int tid = omp_get_thread_num();
          float val = 0.0f;
          if (tid < block_count) {
            val = weights_ptr[block_start + tid];
          }
          scan_local[tid] = val;
#pragma omp barrier
          for (int offset = 1; offset < block_count; offset <<= 1) {
            float temp = 0.0f;
            if (tid >= offset && tid < block_count) {
              temp = scan_local[tid - offset];
            }
#pragma omp barrier
            if (tid >= offset && tid < block_count) {
              scan_local[tid] += temp;
            }
#pragma omp barrier
          }
          if (tid < block_count) {
            CDF_ptr[block_start + tid] = scan_local[tid];
            if (tid == block_count - 1) {
              blockAggregates[block] = scan_local[tid];
            }
          } else if (tid == 0 && block_count == 0) {
            blockAggregates[block] = 0.0f;
          }
        }
      }

#pragma omp target update from(blockAggregates[0:numBlocks])
      float running = 0.0f;
      for (int b = 0; b < numBlocks; ++b) {
        float block_total = blockAggregates[b];
        blockAggregates[b] = running;
        running += block_total;
      }
#pragma omp target update to(blockAggregates[0:numBlocks])

#pragma omp target teams num_teams(numBlocks) thread_limit(BLOCK_SIZE)                             \
          map(present : CDF_ptr[0:Nparticles], blockAggregates[0:numBlocks])
      {
        int block = omp_get_team_num();
        int block_start = block * BLOCK_SIZE;
        int remaining = Nparticles - block_start;
        int block_count = remaining > BLOCK_SIZE
                               ? BLOCK_SIZE
                               : (remaining > 0 ? remaining : 0);
        float offset = blockAggregates[block];
#pragma omp parallel for num_threads(BLOCK_SIZE)
        for (int tid = 0; tid < block_count; ++tid) {
          int idx = block_start + tid;
          CDF_ptr[idx] += offset;
        }
      }

      float u0 = 0.0f;
#pragma omp target map(present, tofrom : seed[0:Nparticles]) map(from : u0)
      {
        float p = clamp_min(randu(seed, 0), 1.0e-12f);
        float q = randu(seed, 0);
        u0 = invParticles * sqrtf(-2.0f * logf(p)) * cosf(2.0f * PI * q);
      }

#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)                          \
          map(present                                                                              \
              : arrayX_ptr[0:Nparticles], arrayY_ptr[0:Nparticles], CDF_ptr[0:Nparticles],         \
                xj_ptr[0:Nparticles], yj_ptr[0:Nparticles])                                         \
          firstprivate(u0, invParticles)
      for (int i = 0; i < Nparticles; ++i) {
        float value = u0 + static_cast<float>(i) * invParticles;
        if (value < 0.0f) {
          value = 0.0f;
        }
        float last = CDF_ptr[Nparticles - 1];
        if (value > last) {
          value = last;
        }
        int index = findIndexDevice(CDF_ptr, Nparticles, value);
        xj_ptr[i] = arrayX_ptr[index];
        yj_ptr[i] = arrayY_ptr[index];
      }
    }
  }

  long long kernels_end = get_time();
  long long offload_end = get_time();
  if (Nfr > 1) {
    printf("Average execution time of kernels: %f (s)\n",
           elapsed_time(kernels_start, kernels_end) /
               static_cast<float>(Nfr - 1));
  } else {
    printf("Average execution time of kernels: 0.000000 (s)\n");
  }
  printf("Device offloading time: %lf (s)\n",
         elapsed_time(offload_start, offload_end));

  float xe = 0.0f;
  float ye = 0.0f;
  for (int i = 0; i < Nparticles; ++i) {
    xe += arrayX[i] * weights[i];
    ye += arrayY[i] * weights[i];
  }

  float distance = sqrtf(
      powf(xe - roundFloat(IszY / 2.0f), 2.0f) +
      powf(ye - roundFloat(IszX / 2.0f), 2.0f));

  FILE *fid = fopen("output.txt", "w+");
  if (fid == NULL) {
    printf("The file was not opened for writing\n");
    return -1;
  }
  fprintf(fid, "XE: %lf\n", static_cast<double>(xe));
  fprintf(fid, "YE: %lf\n", static_cast<double>(ye));
  fprintf(fid, "distance: %lf\n", static_cast<double>(distance));
  fclose(fid);

  return 0;
}

int main(int argc, char *argv[]) {
  const char *usage = "./main -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";

  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }

  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") ||
      strcmp(argv[7], "-np")) {
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

  std::vector<int> seedVec(Nparticles);
  for (int i = 0; i < Nparticles; ++i) {
    seedVec[i] = i + 1;
  }
  int *seed = seedVec.data();

  std::vector<unsigned char> I(static_cast<size_t>(IszX) * IszY * Nfr, 0);
  long long start = get_time();

  videoSequence(I.data(), IszX, IszY, Nfr, seed);
  long long endVideoSequence = get_time();
  printf("VIDEO SEQUENCE TOOK %f (s)\n",
         elapsed_time(start, endVideoSequence));

  particleFilter(I.data(), IszX, IszY, Nfr, seed, Nparticles);
  long long endParticleFilter = get_time();
  printf("PARTICLE FILTER TOOK %f (s)\n",
         elapsed_time(endVideoSequence, endParticleFilter));

  printf("ENTIRE PROGRAM TOOK %f (s)\n",
         elapsed_time(start, endParticleFilter));
  return 0;
}
