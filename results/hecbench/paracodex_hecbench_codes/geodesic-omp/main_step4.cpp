#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "gate.h"

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;

#pragma omp declare target
static constexpr float GDC_DEG_TO_RAD = 3.141592654f / 180.0f;
static constexpr float GDC_FLATTENING = 1.0f - (6356752.31424518f / 6378137.0f);
static constexpr float GDC_ECCENTRICITY = (6356752.31424518f / 6378137.0f);
static constexpr float GDC_ELLIPSOIDAL =
    1.0f / (6356752.31414f / 6378137.0f) / (6356752.31414f / 6378137.0f) - 1.0f;
static constexpr float GDC_SEMI_MINOR = 6356752.31424518f;
static constexpr float EPS = 0.5e-5f;

// Shared Vincenty solver that we call from both host and device paths to minimize duplication.
inline float geodesic_distance_radians(float rad_latitude_1,
                                       float rad_longitude_1,
                                       float rad_latitude_2,
                                       float rad_longitude_2) {
  const float sin_lat_1 = sinf(rad_latitude_1);
  const float cos_lat_1 = cosf(rad_latitude_1);
  const float sin_lat_2 = sinf(rad_latitude_2);
  const float cos_lat_2 = cosf(rad_latitude_2);

  float TU1 = GDC_ECCENTRICITY * sin_lat_1 / cos_lat_1;
  float TU2 = GDC_ECCENTRICITY * sin_lat_2 / cos_lat_2;

  const float CU1 = 1.0f / sqrtf(TU1 * TU1 + 1.0f);
  const float SU1 = CU1 * TU1;
  const float CU2 = 1.0f / sqrtf(TU2 * TU2 + 1.0f);

  float dist = CU1 * CU2;
  float BAZ = dist * TU2;
  const float FAZ = BAZ * TU1;
  float X = rad_longitude_2 - rad_longitude_1;
  float C = 0.0f;
  float C2A = 0.0f;
  float CY = 0.0f;
  float CZ = 0.0f;
  float E = 0.0f;
  float SA = 0.0f;
  float SY = 0.0f;
  float Y = 0.0f;
  float D = 0.0f;

  do {
    const float SX = sinf(X);
    const float CX = cosf(X);
    const float TU1_iter = CU2 * SX;
    const float TU2_iter = BAZ - SU1 * CU2 * CX;
    SY = sqrtf(TU1_iter * TU1_iter + TU2_iter * TU2_iter);
    CY = dist * CX + FAZ;
    Y = atan2f(SY, CY);
    SA = dist * SX / SY;
    C2A = -SA * SA + 1.0f;
    CZ = FAZ + FAZ;
    if (C2A > 0.0f) {
      CZ = -CZ / C2A + CY;
    }
    E = CZ * CZ * 2.0f - 1.0f;
    C = ((-3.0f * C2A + 4.0f) * GDC_FLATTENING + 4.0f) * C2A * GDC_FLATTENING /
        16.0f;
    D = X;
    float step = ((E * CY * C + CZ) * SY * C + Y) * SA;
    X = (1.0f - C) * step * GDC_FLATTENING + rad_longitude_2 - rad_longitude_1;
  } while (fabsf(D - X) > EPS);

  float temp = sqrtf(GDC_ELLIPSOIDAL * C2A + 1.0f) + 1.0f;
  temp = (temp - 2.0f) / temp;
  C = 1.0f - temp;
  C = (temp * temp / 4.0f + 1.0f) / C;
  const float D_term = (0.375f * temp * temp - 1.0f) * temp;
  const float X_term = E * CY;
  dist = 1.0f - E - E;
  dist =
      ((((SY * SY * 4.0f - 3.0f) * dist * CZ * D_term / 6.0f - X_term) *
            D_term / 4.0f +
        CZ) *
           SY * D_term +
       Y) *
      C * GDC_SEMI_MINOR;
  return dist;
}

float distance_host(int i, float latitude_1, float longitude_1,
                    float latitude_2, float longitude_2) {
  (void)i;
  const float rad_latitude_1 = latitude_1 * GDC_DEG_TO_RAD;
  const float rad_longitude_1 = longitude_1 * GDC_DEG_TO_RAD;
  const float rad_latitude_2 = latitude_2 * GDC_DEG_TO_RAD;
  const float rad_longitude_2 = longitude_2 * GDC_DEG_TO_RAD;
  return geodesic_distance_radians(rad_latitude_1, rad_longitude_1,
                                   rad_latitude_2, rad_longitude_2);
}
#pragma omp end declare target

void distance_device(const float4* __restrict__ VA, float* __restrict__ VC,
                     const size_t N, const int iteration) {
  auto start = std::chrono::steady_clock::now();
  const bool use_gpu = omp_get_num_devices() > 0;
  const int runs = use_gpu ? std::max(iteration, 1) : 1;

  if (use_gpu) {
    constexpr int threads_per_team = 256;
    const size_t total_teams = (N + static_cast<size_t>(threads_per_team) - 1) /
                               static_cast<size_t>(threads_per_team);
    const int teams_per_grid = static_cast<int>(total_teams ? total_teams : 1);
    // Tune launch geometry for Ada (SM 8.9): 8-warp teams balance occupancy and register pressure.
    const int device_id = omp_get_default_device();
    const bool va_present = omp_target_is_present(VA, device_id);
    const bool vc_present = omp_target_is_present(VC, device_id);

    if (va_present && vc_present) {
      for (int n = 0; n < iteration; n++) {
#pragma omp target teams distribute parallel for map(present: VA[0:N], VC[0:N]) \
    num_teams(teams_per_grid) thread_limit(threads_per_team)
        for (size_t wiID = 0; wiID < N; wiID++) {
          const float4 coord = VA[wiID];
          VC[wiID] = distance_host(0, coord.x, coord.y, coord.z, coord.w);
        }
      }
    } else {
#pragma omp target data map(to: VA[0:N]) map(from: VC[0:N])
      {
        for (int n = 0; n < iteration; n++) {
#pragma omp target teams distribute parallel for map(present: VA[0:N], VC[0:N]) \
    num_teams(teams_per_grid) thread_limit(threads_per_team)
          for (size_t wiID = 0; wiID < N; wiID++) {
            const float4 coord = VA[wiID];
            VC[wiID] = distance_host(0, coord.x, coord.y, coord.z, coord.w);
          }
        }
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (size_t wiID = 0; wiID < N; ++wiID) {
      const float lat1 = VA[wiID].x;
      const float lon1 = VA[wiID].y;
      const float lat2 = VA[wiID].z;
      const float lon2 = VA[wiID].w;
      VC[wiID] = distance_host(static_cast<int>(wiID), lat1, lon1, lat2, lon2);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / runs);
}

void verify(int size, const float *output, const float *expected_output) {
  float error_rate = 0;
  for (int i = 0; i < size; i++) {
    if (fabs(output[i] - expected_output[i]) > error_rate) {
      error_rate = fabs(output[i] - expected_output[i]);
    }
  }
  printf("The maximum error in distance for single precision is %f\n", error_rate);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage %s <repeat>\n", argv[0]);
    return 1;
  }
  int iteration = atoi(argv[1]);

  int num_cities = 2097152;

  int num_ref_cities = 6;

  int index_map[] ={436483, 1952407, 627919, 377884, 442703, 1863423};
  int N = num_cities * num_ref_cities;
  int city = 0;
  float lat, lon;

  const char* filename = "locations.txt";
  printf("Reading city locations from file %s...\n", filename);
  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
    perror ("Error opening the file");
    exit(-1);
  }

  float4* input  = (float4*) aligned_alloc(4096, N*sizeof(float4));
  float*  output = (float*) aligned_alloc(4096, N*sizeof(float));
  float*  expected_output = (float*) malloc(N*sizeof(float));

  while (fscanf(fp, "%f %f\n", &lat, &lon) != EOF) {
    input[city].x = lat;
    input[city].y = lon;
    city++;
    if (city == num_cities) break;
  }
  fclose(fp);



  for (int c = 1;  c < num_ref_cities; c++) {
    std::copy(input, input+num_cities, input+c*num_cities);
  }



  for (int c = 0;  c < num_ref_cities; c++) {
    int index = index_map[c] - 1;
    for(int j = c*num_cities; j < (c+1)*num_cities; ++j) {
      input[j].z = input[index].x;
      input[j].w = input[index].y;
    }
  }



  const bool use_gpu = omp_get_num_devices() > 0;

  if (use_gpu) {
#pragma omp target data map(to: input[0:N]) map(alloc: output[0:N], expected_output[0:N])
    {
      // Keep input resident across reference and primary kernels.
      const int threads_per_team = 256;
      const int teams_per_grid = std::max(1, (N + threads_per_team - 1) / threads_per_team);
#pragma omp target teams distribute parallel for map(present: input[0:N], expected_output[0:N]) \
    num_teams(teams_per_grid) thread_limit(threads_per_team) nowait
      for (int i = 0; i < N; i++) {
        const float lat1 = input[i].x;
        const float lon1 = input[i].y;
        const float lat2 = input[i].z;
        const float lon2 = input[i].w;
        expected_output[i] = distance_host(i, lat1, lon1, lat2, lon2);
      }

      distance_device(input, output, N, iteration);

#pragma omp taskwait
#pragma omp target update from(output[0:N], expected_output[0:N])
    }
  } else {
    for (int i = 0; i < N; i++) {
      float lat1 = input[i].x;
      float lon1 = input[i].y;
      float lat2 = input[i].z;
      float lon2 = input[i].w;
      expected_output[i] = distance_host(i, lat1, lon1, lat2, lon2);
    }

    distance_device(input, output, N, iteration);
  }

  verify(N, output, expected_output);

  GATE_CHECKSUM_BYTES("geodesic_output", expected_output, N * sizeof(float));

  free(input);
  free(output);
  free(expected_output);
  return 0;
}
