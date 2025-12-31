#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "gate.h"

struct alignas(16) float4 {
  float x;
  float y;
  float z;
  float w;
};

#pragma omp declare target
inline float vincenty_distance(float latitude_1,
                               float longitude_1,
                               float latitude_2,
                               float longitude_2) {
  float dist;
  float rad_latitude_1;
  float rad_latitude_2;
  float rad_longitude_1;
  float rad_longitude_2;

  float BAZ, C, C2A, CU1, CU2, CX, CY, CZ, D, E, FAZ, SA, SU1, SX, SY, TU1,
      TU2, X, Y;

  const float GDC_DEG_TO_RAD = 3.141592654 / 180.0;
  const float GDC_FLATTENING = 1.0 - (6356752.31424518 / 6378137.0);
  const float GDC_ECCENTRICITY = (6356752.31424518 / 6378137.0);
  const float GDC_ELLIPSOIDAL =
      1.0 / (6356752.31414 / 6378137.0) / (6356752.31414 / 6378137.0) - 1.0;
  const float GDC_SEMI_MINOR = 6356752.31424518f;
  const float EPS = 0.5e-5f;

  rad_longitude_1 = longitude_1 * GDC_DEG_TO_RAD;
  rad_latitude_1 = latitude_1 * GDC_DEG_TO_RAD;
  rad_longitude_2 = longitude_2 * GDC_DEG_TO_RAD;
  rad_latitude_2 = latitude_2 * GDC_DEG_TO_RAD;

  TU1 = GDC_ECCENTRICITY * sinf(rad_latitude_1) / cosf(rad_latitude_1);
  TU2 = GDC_ECCENTRICITY * sinf(rad_latitude_2) / cosf(rad_latitude_2);

  CU1 = 1.0f / sqrtf(TU1 * TU1 + 1.0f);
  SU1 = CU1 * TU1;
  CU2 = 1.0f / sqrtf(TU2 * TU2 + 1.0f);
  dist = CU1 * CU2;
  BAZ = dist * TU2;
  FAZ = BAZ * TU1;
  X = rad_longitude_2 - rad_longitude_1;

  do {
    SX = sinf(X);
    CX = cosf(X);
    TU1 = CU2 * SX;
    TU2 = BAZ - SU1 * CU2 * CX;
    SY = sqrtf(TU1 * TU1 + TU2 * TU2);
    CY = dist * CX + FAZ;
    Y = atan2f(SY, CY);
    SA = dist * SX / SY;
    C2A = -SA * SA + 1.0f;
    CZ = FAZ + FAZ;
    if (C2A > 0.0f)
      CZ = -CZ / C2A + CY;
    E = CZ * CZ * 2.0f - 1.0f;
    C = ((-3.0f * C2A + 4.0f) * GDC_FLATTENING + 4.0f) * C2A * GDC_FLATTENING /
        16.0f;
    D = X;
    X = ((E * CY * C + CZ) * SY * C + Y) * SA;
    X = (1.0f - C) * X * GDC_FLATTENING + rad_longitude_2 - rad_longitude_1;
  } while (fabsf(D - X) > EPS);

  X = sqrtf(GDC_ELLIPSOIDAL * C2A + 1.0f) + 1.0f;
  X = (X - 2.0f) / X;
  C = 1.0f - X;
  C = (X * X / 4.0f + 1.0f) / C;
  D = (0.375f * X * X - 1.0f) * X;
  X = E * CY;
  dist = 1.0f - E - E;
  dist =
      ((((SY * SY * 4.0f - 3.0f) * dist * CZ * D / 6.0f - X) * D / 4.0f + CZ) *
           SY * D +
       Y) *
      C * GDC_SEMI_MINOR;
  return dist;
}
#pragma omp end declare target

float distance_host(float latitude_1,
                    float longitude_1,
                    float latitude_2,
                    float longitude_2) {
  return vincenty_distance(latitude_1, longitude_1, latitude_2, longitude_2);
}

void distance_device(const float4 *__restrict__ VA,
                     float *__restrict__ VC,
                     const size_t N,
                     const int iteration) {
  const int total = static_cast<int>(N);
  const bool use_device = omp_get_num_devices() > 0;
  auto start = std::chrono::steady_clock::now();

#pragma omp target data map(to : VA[0:total]) map(from : VC[0:total]) if (use_device)
  {
    for (int n = 0; n < iteration; n++) {
#pragma omp target teams distribute parallel for schedule(static) thread_limit(256) if (use_device)
      for (int wiID = 0; wiID < total; wiID++) {
        const float4 loc = VA[wiID];
        VC[wiID] = vincenty_distance(loc.x, loc.y, loc.z, loc.w);
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n",
              (time * 1e-9f) / iteration);
}

void verify(int size, const float *output, const float *expected_output) {
  float error_rate = 0;
  for (int i = 0; i < size; i++) {
    error_rate = std::max(error_rate,
                          std::fabs(output[i] - expected_output[i]));
  }
  std::printf("The maximum error in distance for single precision is %f\n",
              error_rate);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::printf("Usage %s <repeat>\n", argv[0]);
    return 1;
  }
  int iteration = std::atoi(argv[1]);

  const int num_cities = 2097152;
  const int num_ref_cities = 6;
  const int index_map[] = {436483, 1952407, 627919, 377884, 442703, 1863423};
  const int N = num_cities * num_ref_cities;
  int city = 0;
  float lat, lon;

  const char *filename = "locations.txt";
  std::printf("Reading city locations from file %s...\n", filename);
  FILE *fp = std::fopen(filename, "r");
  if (fp == nullptr) {
    std::perror("Error opening the file");
    return -1;
  }

  float4 *input =
      static_cast<float4 *>(aligned_alloc(4096, N * sizeof(float4)));
  float *output =
      static_cast<float *>(aligned_alloc(4096, N * sizeof(float)));
  float *expected_output = static_cast<float *>(std::malloc(N * sizeof(float)));

  if (!input || !output || !expected_output) {
    std::fprintf(stderr, "Memory allocation failed\n");
    std::fclose(fp);
    std::free(input);
    std::free(output);
    std::free(expected_output);
    return -1;
  }

  while (std::fscanf(fp, "%f %f\n", &lat, &lon) != EOF) {
    input[city].x = lat;
    input[city].y = lon;
    city++;
    if (city == num_cities)
      break;
  }
  std::fclose(fp);

  for (int c = 1; c < num_ref_cities; c++) {
    std::copy(input, input + num_cities, input + c * num_cities);
  }

  for (int c = 0; c < num_ref_cities; c++) {
    const int index = index_map[c] - 1;
    for (int j = c * num_cities; j < (c + 1) * num_cities; ++j) {
      input[j].z = input[index].x;
      input[j].w = input[index].y;
    }
  }

  for (int i = 0; i < N; i++) {
    const float4 loc = input[i];
    expected_output[i] =
        distance_host(loc.x, loc.y, loc.z, loc.w);
  }

  distance_device(input, output, N, iteration);

  verify(N, output, expected_output);

  GATE_CHECKSUM_BYTES("geodesic_output", expected_output, N * sizeof(float));

  std::free(input);
  std::free(output);
  std::free(expected_output);
  return 0;
}
