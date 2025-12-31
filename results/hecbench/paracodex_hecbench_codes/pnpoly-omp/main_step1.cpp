#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>

#define VERTICES 600
#define BLOCK_SIZE_X 256

typedef struct __attribute__((__aligned__(8)))
{
  float x, y;
} float2;

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

/*
 * The is_between method returns a boolean that is True when the a is between c and b.
 */
#pragma omp begin declare target
inline int is_between(float a, float b, float c) {
  return (b > a) != (c > a);
}
#pragma omp end declare target

/*
 * The Point-in-Polygon kernel
 */
template <int tile_size>
void pnpoly_opt(
    int*__restrict bitmap,
    const float2*__restrict point,
    const float2*__restrict vertex,
    int n) 
{
  // Offload outer point loop to the GPU; map polygon data explicitly.
  #pragma omp target teams distribute parallel for map(to: point[0:n], vertex[0:VERTICES]) map(from: bitmap[0:n])
  for (int i = 0; i < n; i++) {
    int c[tile_size];
    float2 lpoint[tile_size];
    #pragma unroll
    for (int ti=0; ti<tile_size; ti++) {
      c[ti] = 0;
      if (i+BLOCK_SIZE_X*ti < n) {
        lpoint[ti] = point[i+BLOCK_SIZE_X*ti];
      }
    }

    int k = VERTICES-1;

    for (int j=0; j<VERTICES; k = j++) {    // edge from vj to vk
      float2 vj = vertex[j]; 
      float2 vk = vertex[k]; 

      float slope = (vk.x-vj.x) / (vk.y-vj.y);

      #pragma unroll
      for (int ti=0; ti<tile_size; ti++) {

        float2 p = lpoint[ti];

        if (is_between(p.y, vj.y, vk.y) &&         //if p is between vj and vk vertically
            (p.x < slope * (p.y-vj.y) + vj.x)
           ) {  //if p.x crosses the line vj-vk when moved in positive x-direction
          c[ti] = !c[ti];
        }
      }
    }

    #pragma unroll
    for (int ti=0; ti<tile_size; ti++) {
      //could do an if statement here if 1s are expected to be rare
      if (i+BLOCK_SIZE_X*ti < n) {
        #pragma omp atomic write
        bitmap[i+BLOCK_SIZE_X*ti] = c[ti];
      }
    }
  }
}


/*
 * The naive implementation is used for verifying correctness of the optimized implementation
 */
void pnpoly_base(
    int*__restrict bitmap,
    const float2*__restrict point,
    const float2*__restrict vertex,
    int n) 
{
  // Reuse the same offload strategy for the reference implementation.
  #pragma omp target teams distribute parallel for map(to: point[0:n], vertex[0:VERTICES]) map(from: bitmap[0:n])
  for (int i = 0; i < n; i++) {
    int c = 0;
    float2 p = point[i];

    int k = VERTICES-1;

    for (int j=0; j<VERTICES; k = j++) {    // edge from v to vp
      float2 vj = vertex[j]; 
      float2 vk = vertex[k]; 

      float slope = (vk.x-vj.x) / (vk.y-vj.y);

      if (((vj.y>p.y) != (vk.y>p.y)) &&            //if p is between vj and vk vertically
          (p.x < slope * (p.y-vj.y) + vj.x)) {   //if p.x crosses the line vj-vk when moved in positive x-direction
        c = !c;
      }
    }

    bitmap[i] = c; // 0 if even (out), and 1 if odd (in)
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 2e7;
  const int vertices = VERTICES;

  std::default_random_engine rng (123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2*) malloc (sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2 *vertex = (float2*) malloc (vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  int *bitmap_ref = (int*) malloc (nPoints * sizeof(int));
  int *bitmap_opt = (int*) malloc (nPoints * sizeof(int));

  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_base(bitmap_ref, point, vertex, nPoints);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<1>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<2>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<4>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<8>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<16>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<32>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      pnpoly_opt<64>(bitmap_opt, point, vertex, nPoints);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);
  }

  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 
  
  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}
