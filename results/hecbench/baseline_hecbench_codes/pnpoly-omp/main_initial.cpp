#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

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

#pragma omp declare target
inline int is_between(float a, float b, float c) {
  return (b > a) != (c > a);
}
#pragma omp end declare target

template <int tile_size>
void pnpoly_opt(int *__restrict bitmap,
                const float2 *__restrict point,
                const edge_coeff *__restrict edge,
                int n) {
  if (n <= 0)
    return;

  constexpr int chunk = BLOCK_SIZE_X;
  const int total_tiles = (n + chunk - 1) / chunk;

#pragma omp target teams distribute parallel for collapse(2)                 \
    map(present : point[0:n], edge[0:VERTICES], bitmap[0:n])                 \
    num_teams(total_tiles) thread_limit(chunk)
  for (int tile = 0; tile < total_tiles; ++tile) {
    for (int lane = 0; lane < chunk; ++lane) {
      const int base = tile * chunk + lane;
      if (base >= n)
        continue;

      int parity[tile_size];
      float2 local_point[tile_size];

#pragma unroll
      for (int ti = 0; ti < tile_size; ++ti) {
        parity[ti] = 0;
        const int point_idx = base + chunk * ti;
        if (point_idx < n) {
          local_point[ti] = point[point_idx];
        }
      }

#pragma unroll
      for (int j = 0; j < VERTICES; ++j) {
        const edge_coeff coeff = edge[j];
        const float vj_y = coeff.vj_y;
        const float vk_y = coeff.vk_y;
        const float slope = coeff.slope;
        const float vj_x = coeff.vj_x;

#pragma unroll
        for (int ti = 0; ti < tile_size; ++ti) {
          const int point_idx = base + chunk * ti;
          if (point_idx >= n)
            continue;

          const float2 p = local_point[ti];
          const float py = p.y;
          if (is_between(py, vj_y, vk_y)) {
            const float crossing = std::fmaf(slope, py - vj_y, vj_x);
            if (p.x < crossing) {
              parity[ti] = !parity[ti];
            }
          }
        }
      }

#pragma unroll
      for (int ti = 0; ti < tile_size; ++ti) {
        const int point_idx = base + chunk * ti;
        if (point_idx < n) {
          bitmap[point_idx] = parity[ti];
        }
      }
    }
  }
}

void pnpoly_base(int *__restrict bitmap,
                 const float2 *__restrict point,
                 const edge_coeff *__restrict edge,
                 int n) {
  if (n <= 0)
    return;

  constexpr int chunk = BLOCK_SIZE_X;
  const int total_tiles = (n + chunk - 1) / chunk;

#pragma omp target teams distribute parallel for collapse(2)                 \
    map(present : point[0:n], edge[0:VERTICES], bitmap[0:n])                 \
    num_teams(total_tiles) thread_limit(chunk)
  for (int tile = 0; tile < total_tiles; ++tile) {
    for (int lane = 0; lane < chunk; ++lane) {
      const int idx = tile * chunk + lane;
      if (idx >= n)
        continue;

      int parity = 0;
      const float2 p = point[idx];
      const float px = p.x;
      const float py = p.y;

#pragma unroll
      for (int j = 0; j < VERTICES; ++j) {
        const edge_coeff coeff = edge[j];
        const float vj_y = coeff.vj_y;
        const float vk_y = coeff.vk_y;

        if ((vj_y > py) != (vk_y > py)) {
          const float crossing = std::fmaf(coeff.slope, py - vj_y, coeff.vj_x);
          if (px < crossing) {
            parity = !parity;
          }
        }
      }

      bitmap[idx] = parity;
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 1'000'000;
  const int vertices = VERTICES;

  std::default_random_engine rng(123);
  std::normal_distribution<float> distribution(0.0f, 1.0f);

  float2 *point =
      static_cast<float2 *>(std::malloc(sizeof(float2) * static_cast<size_t>(nPoints)));
  float2 *vertex =
      static_cast<float2 *>(std::malloc(sizeof(float2) * static_cast<size_t>(vertices)));
  edge_coeff *edge = static_cast<edge_coeff *>(
      std::malloc(sizeof(edge_coeff) * static_cast<size_t>(vertices)));
  int *bitmap_ref =
      static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(nPoints)));
  int *bitmap_opt =
      static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(nPoints)));

  if (!point || !vertex || !edge || !bitmap_ref || !bitmap_opt) {
    std::fprintf(stderr, "Allocation failed\n");
    std::free(point);
    std::free(vertex);
    std::free(edge);
    std::free(bitmap_ref);
    std::free(bitmap_opt);
    return 1;
  }

  for (int i = 0; i < nPoints; ++i) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  for (int i = 0; i < vertices; ++i) {
    const float t = distribution(rng) * 2.0f * static_cast<float>(M_PI);
    vertex[i].x = static_cast<float>(std::cos(t));
    vertex[i].y = static_cast<float>(std::sin(t));
  }

  for (int j = 0; j < vertices; ++j) {
    const int k = (j == 0) ? (vertices - 1) : (j - 1);
    const float vj_x = vertex[j].x;
    const float vj_y = vertex[j].y;
    const float vk_x = vertex[k].x;
    const float vk_y = vertex[k].y;
    const float denom = vk_y - vj_y;
    const float slope = (vk_x - vj_x) / denom;
    edge[j].vj_x = vj_x;
    edge[j].vj_y = vj_y;
    edge[j].vk_y = vk_y;
    edge[j].slope = slope;
  }

#pragma omp target data map(to : point[0:nPoints], edge[0:vertices])        \
    map(from : bitmap_ref[0:nPoints], bitmap_opt[0:nPoints])
  {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      pnpoly_base(bitmap_ref, point, edge, nPoints);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time (pnpoly_base): %f (s)\n",
                (elapsed * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      pnpoly_opt<1>(bitmap_opt, point, edge, nPoints);
    }
    end = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n",
                (elapsed * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      pnpoly_opt<2>(bitmap_opt, point, edge, nPoints);
    }
    end = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n",
                (elapsed * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      pnpoly_opt<4>(bitmap_opt, point, edge, nPoints);
    }
    end = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n",
                (elapsed * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      pnpoly_opt<8>(bitmap_opt, point, edge, nPoints);
    }
    end = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n",
                (elapsed * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      pnpoly_opt<16>(bitmap_opt, point, edge, nPoints);
    }
    end = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n",
                (elapsed * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      pnpoly_opt<32>(bitmap_opt, point, edge, nPoints);
    }
    end = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n",
                (elapsed * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      pnpoly_opt<64>(bitmap_opt, point, edge, nPoints);
    }
    end = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n",
                (elapsed * 1e-9f) / repeat);
  }

  GATE_CHECKSUM_U32("bitmap_ref", reinterpret_cast<const uint32_t *>(bitmap_ref), nPoints);
  GATE_CHECKSUM_U32("bitmap_opt", reinterpret_cast<const uint32_t *>(bitmap_opt), nPoints);

  const int error =
      std::memcmp(bitmap_opt, bitmap_ref, sizeof(int) * static_cast<size_t>(nPoints));

  int checksum = 0;
  for (int i = 0; i < nPoints; ++i) {
    checksum += bitmap_opt[i];
  }
  std::printf("Checksum: %d\n", checksum);
  std::printf("%s\n", error ? "FAIL" : "PASS");

  std::free(vertex);
  std::free(edge);
  std::free(point);
  std::free(bitmap_ref);
  std::free(bitmap_opt);

  return error != 0;
}
