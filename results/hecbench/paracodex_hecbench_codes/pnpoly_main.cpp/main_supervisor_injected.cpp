#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>
#include "gate.h"

#define VERTICES 100
#define BLOCK_SIZE_X 256

typedef struct __attribute__((__aligned__(8))) {
  float x, y;
} float2;

typedef struct __attribute__((__aligned__(16))) {
  float vj_x;
  float vj_y;
  float vk_y;
  float slope;
} edge_coeff;

/*
 * This file contains the implementation of a kernel for the
 * point-in-polygon problem using the crossing number algorithm
 *
 * The kernel pnpoly_base is used for correctness checking.
 *
 * The algorithm used here is adapted from:
 *     'Inclusion of a Point in a Polygon', Dan Sunday, 2001
 *     (http://geomalgorithms.com/a03-_inclusion.html)
 *
 * Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */

#pragma omp declare target
/*
 * The is_between method returns a boolean that is True when the a is between c and b.
 */
inline int is_between(float a, float b, float c) {
  return (b > a) != (c > a);
}
#pragma omp end declare target

/*
 * The Point-in-Polygon kernel
 */
template <int tile_size>
void pnpoly_opt(int *__restrict bitmap,
                const float2 *__restrict point,
                const edge_coeff *__restrict edge,
                int n,
                bool use_gpu) {
  // Offload the point classification work; ensure all inputs/outputs are mapped correctly.
  const int chunk = BLOCK_SIZE_X;
  int total_tiles = (n + chunk - 1) / chunk;
  if (total_tiles == 0)
    total_tiles = 1;

  // Hint the runtime to keep 256-thread teams resident across the Ada SMs.
#pragma omp target teams distribute parallel for collapse(2) if (use_gpu) \
    map(present : point[0:n], edge[0:VERTICES], bitmap[0:n]) \
    num_teams(total_tiles) thread_limit(chunk)
  for (int tile = 0; tile < total_tiles; ++tile) {
    for (int lane = 0; lane < chunk; ++lane) {
      int i = tile * chunk + lane;
      if (i >= n)
        continue;

      int c[tile_size];
      float2 lpoint[tile_size];
#pragma unroll
      for (int ti = 0; ti < tile_size; ti++) {
        c[ti] = 0;
        if (i + chunk * ti < n) {
          lpoint[ti] = point[i + chunk * ti];
        }
      }

      for (int j = 0; j < VERTICES; ++j) { // edge from vj to vk
        const edge_coeff coeff = edge[j];
        const float vj_y = coeff.vj_y;
        const float vk_y = coeff.vk_y;
        const float slope = coeff.slope;
        const float vj_x = coeff.vj_x;

#pragma unroll
        for (int ti = 0; ti < tile_size; ti++) {
          int point_idx = i + chunk * ti;
          if (point_idx >= n)
            continue;

          float2 p = lpoint[ti];

          const float py = p.y;
          const float px = p.x;

          if (is_between(py, vj_y, vk_y)) {
            float crossing = fmaf(slope, py - vj_y, vj_x);
            if (px < crossing) {
              c[ti] = !c[ti];
            }
          }
        }
      }

#pragma unroll
      for (int ti = 0; ti < tile_size; ti++) {
        int point_idx = i + chunk * ti;
        if (point_idx < n)
          bitmap[point_idx] = c[ti];
      }
    }
  }
}

/*
 * The naive implementation is used for verifying correctness of the optimized implementation
 */
void pnpoly_base(int *__restrict bitmap,
                 const float2 *__restrict point,
                 const edge_coeff *__restrict edge,
                 int n,
                 bool use_gpu) {
  // GPU offload for baseline kernel to maintain identical verification path.
  const int chunk = BLOCK_SIZE_X;
  int total_tiles = (n + chunk - 1) / chunk;
  if (total_tiles == 0)
    total_tiles = 1;

  // Reuse the same launch geometry hints for the baseline verifier.
#pragma omp target teams distribute parallel for collapse(2) if (use_gpu) \
    map(present : point[0:n], edge[0:VERTICES], bitmap[0:n]) \
    num_teams(total_tiles) thread_limit(chunk)
  for (int tile = 0; tile < total_tiles; ++tile) {
    for (int lane = 0; lane < chunk; ++lane) {
      int i = tile * chunk + lane;
      if (i >= n)
        continue;

      int c = 0;
      float2 p = point[i];
      const float px = p.x;
      const float py = p.y;

      for (int j = 0; j < VERTICES; ++j) { // edge from v to vp
        const edge_coeff coeff = edge[j];
        const float vj_y = coeff.vj_y;
        const float vk_y = coeff.vk_y;
        const float slope = coeff.slope;
        const float vj_x = coeff.vj_x;

        if (((vj_y > py) != (vk_y > py)) &&
            (px < fmaf(slope, (py - vj_y), vj_x))) {
          c = !c;
        }
      }

      bitmap[i] = c; // 0 if even (out), and 1 if odd (in)
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 1e6;
  const int vertices = VERTICES;

  std::default_random_engine rng(123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2 *)malloc(sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2 *vertex = (float2 *)malloc(vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  edge_coeff *edge = (edge_coeff *)malloc(vertices * sizeof(edge_coeff));
  // Precompute edge coefficients once on the host to eliminate per-point divisions on the GPU.
  for (int j = 0; j < vertices; ++j) {
    int k = (j == 0) ? (vertices - 1) : (j - 1);
    float vj_y = vertex[j].y;
    float vk_y = vertex[k].y;
    float denom = vk_y - vj_y;
    float slope = 0.f;
    if (denom != 0.f) {
      slope = (vertex[k].x - vertex[j].x) / denom;
    }
    edge[j].vj_x = vertex[j].x;
    edge[j].vj_y = vj_y;
    edge[j].vk_y = vk_y;
    edge[j].slope = slope;
  }

  int *bitmap_ref = (int *)malloc(nPoints * sizeof(int));
  int *bitmap_opt = (int *)malloc(nPoints * sizeof(int));

  const int device_count = omp_get_num_devices();
  const bool use_gpu = device_count > 0;

  // Persist problem data on the device to avoid redundant host-device transfers across kernel sweeps.
#pragma omp target data if (use_gpu) map(to : point[0:nPoints], edge[0:vertices]) \
    map(from : bitmap_ref[0:nPoints], bitmap_opt[0:nPoints])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_base(bitmap_ref, point, edge, nPoints, use_gpu);

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("Average kernel execution time (pnpoly_base): %f (s)\n",
           (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<1>(bitmap_opt, point, edge, nPoints, use_gpu);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n",
           (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<2>(bitmap_opt, point, edge, nPoints, use_gpu);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n",
           (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<4>(bitmap_opt, point, edge, nPoints, use_gpu);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n",
           (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<8>(bitmap_opt, point, edge, nPoints, use_gpu);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n",
           (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<16>(bitmap_opt, point, edge, nPoints, use_gpu);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n",
           (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<32>(bitmap_opt, point, edge, nPoints, use_gpu);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n",
           (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<64>(bitmap_opt, point, edge, nPoints, use_gpu);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count();
    printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n",
           (time * 1e-9f) / repeat);
  }

  GATE_CHECKSUM_U32("bitmap_ref", reinterpret_cast<const uint32_t *>(bitmap_ref), nPoints);
  GATE_CHECKSUM_U32("bitmap_opt", reinterpret_cast<const uint32_t *>(bitmap_opt), nPoints);

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints * sizeof(int));

  int checksum = 0;
  for (int i = 0; i < nPoints; i++)
    checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(vertex);
  free(edge);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}
