#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <omp.h>
#include <cstring>
#include "../../../gate_sdk/gate.h"

constexpr const char kLine[] = "--------------------\n";
constexpr int kThreadsPerTeam = 256;

#pragma omp declare target
inline double solution(double t,
                       double x,
                       double y,
                       double alpha,
                       double length) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double inv_len = 1.0 / length;
  const double scale = pi * inv_len;
  const double spatial = std::sin(scale * x) * std::sin(scale * y);
  const double decay =
      std::exp(-2.0 * alpha * pi * pi * t * inv_len * inv_len);
  return decay * spatial;
}
#pragma omp end declare target

int main(int argc, char *argv[]) {
  double start = omp_get_wtime();

  int n = 1000;
  int nsteps = 10;

  if (argc == 3) {
    n = std::atoi(argv[1]);
    if (n < 0) {
      std::fprintf(stderr, "Error: n must be positive\n");
      std::exit(EXIT_FAILURE);
    }

    nsteps = std::atoi(argv[2]);
    if (nsteps < 0) {
      std::fprintf(stderr, "Error: nsteps must be positive\n");
      std::exit(EXIT_FAILURE);
    }
  }

  const double alpha = 0.1;
  const double length = 1000.0;
  const double dx = length / (static_cast<double>(n) + 1.0);
  const double dt = 0.5 / static_cast<double>(nsteps);
  const double r = alpha * dt / (dx * dx);

  std::printf("\n");
  std::printf(" MMS heat equation\n\n");
  std::printf("%s", kLine);
  std::printf("Problem input\n\n");
  std::printf(" Grid size: %d x %d\n", n, n);
  std::printf(" Cell width: %E\n", dx);
  std::printf(" Grid length: %lf x %lf\n", length, length);
  std::printf("\n");
  std::printf(" Alpha: %E\n", alpha);
  std::printf("\n");
  std::printf(" Steps: %d\n", nsteps);
  std::printf(" Total time: %E\n", dt * static_cast<double>(nsteps));
  std::printf(" Time step: %E\n", dt);
  std::printf("%s", kLine);

  std::printf("Stability\n\n");
  std::printf(" r value: %lf\n", r);
  if (r > 0.5) {
    std::printf(" Warning: unstable\n");
  }
  std::printf("%s", kLine);

  const std::size_t total_cells =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  double *u =
      static_cast<double *>(std::malloc(sizeof(double) * total_cells));
  double *u_tmp =
      static_cast<double *>(std::malloc(sizeof(double) * total_cells));

  if (!u || !u_tmp) {
    std::fprintf(stderr, "Error: allocation failed\n");
    std::free(u);
    std::free(u_tmp);
    return EXIT_FAILURE;
  }

  const double r2 = 1.0 - 4.0 * r;
  double tic = 0.0;
  double toc = 0.0;
  double norm = 0.0;
  const double pi = 3.141592653589793238462643383279502884;
  const double scale = pi / length;

#pragma omp target data map(alloc : u [0:total_cells], u_tmp [0:total_cells])
  {
#pragma omp target teams distribute parallel for collapse(2)                  \
    map(present : u [0:total_cells])                                          \
        firstprivate(dx, scale) thread_limit(kThreadsPerTeam)
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t row_offset =
            static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
        const std::size_t idx = static_cast<std::size_t>(i) + row_offset;
        const double x = (static_cast<double>(i) + 1.0) * dx;
        const double y = (static_cast<double>(j) + 1.0) * dx;
        u[idx] = std::sin(scale * x) * std::sin(scale * y);
      }
    }

#pragma omp target teams distribute parallel for map(present : u_tmp [0:total_cells]) \
    thread_limit(kThreadsPerTeam)
    for (std::size_t idx = 0; idx < total_cells; ++idx) {
      u_tmp[idx] = 0.0;
    }

    tic = omp_get_wtime();
    for (int t = 0; t < nsteps; ++t) {
      if ((t & 1) == 0) {
#pragma omp target teams distribute parallel for collapse(2)                \
    map(present : u [0:total_cells], u_tmp [0:total_cells])                 \
        firstprivate(r, r2, n) thread_limit(kThreadsPerTeam)
        for (int j = 0; j < n; ++j) {
          for (int i = 0; i < n; ++i) {
            const std::size_t row_offset =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
            const std::size_t idx =
                static_cast<std::size_t>(i) + row_offset;
            const double center = u[idx];
            const double east =
                (i < n - 1) ? u[idx + 1] : 0.0;
            const double west =
                (i > 0) ? u[idx - 1] : 0.0;
            const double north =
                (j < n - 1) ? u[idx + n] : 0.0;
            const double south =
                (j > 0) ? u[idx - n] : 0.0;
            u_tmp[idx] = r2 * center + r * (east + west + north + south);
          }
        }
      } else {
#pragma omp target teams distribute parallel for collapse(2)                \
    map(present : u [0:total_cells], u_tmp [0:total_cells])                 \
        firstprivate(r, r2, n) thread_limit(kThreadsPerTeam)
        for (int j = 0; j < n; ++j) {
          for (int i = 0; i < n; ++i) {
            const std::size_t row_offset =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
            const std::size_t idx =
                static_cast<std::size_t>(i) + row_offset;
            const double center = u_tmp[idx];
            const double east =
                (i < n - 1) ? u_tmp[idx + 1] : 0.0;
            const double west =
                (i > 0) ? u_tmp[idx - 1] : 0.0;
            const double north =
                (j < n - 1) ? u_tmp[idx + n] : 0.0;
            const double south =
                (j > 0) ? u_tmp[idx - n] : 0.0;
            u[idx] = r2 * center + r * (east + west + north + south);
          }
        }
      }
    }
    toc = omp_get_wtime();

    const double time = dt * static_cast<double>(nsteps);
    double accum = 0.0;

    if ((nsteps & 1) == 0) {
#pragma omp target teams distribute parallel for collapse(2)                \
    reduction(+:accum) map(present : u [0:total_cells])                     \
        firstprivate(time, alpha, dx, length, n) thread_limit(kThreadsPerTeam)
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          const std::size_t row_offset =
              static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
          const std::size_t idx = static_cast<std::size_t>(i) + row_offset;
          const double x = (static_cast<double>(i) + 1.0) * dx;
          const double y = (static_cast<double>(j) + 1.0) * dx;
          const double diff = u[idx] - solution(time, x, y, alpha, length);
          accum += diff * diff;
        }
      }
    } else {
#pragma omp target teams distribute parallel for collapse(2)                \
    reduction(+:accum) map(present : u_tmp [0:total_cells])                 \
        firstprivate(time, alpha, dx, length, n) thread_limit(kThreadsPerTeam)
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          const std::size_t row_offset =
              static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
          const std::size_t idx = static_cast<std::size_t>(i) + row_offset;
          const double x = (static_cast<double>(i) + 1.0) * dx;
          const double y = (static_cast<double>(j) + 1.0) * dx;
          const double diff = u_tmp[idx] - solution(time, x, y, alpha, length);
          accum += diff * diff;
        }
      }
    }
    norm = std::sqrt(accum);

    if ((nsteps & 1) == 0) {
#pragma omp target update from(u[0:total_cells])
    } else {
#pragma omp target update from(u_tmp[0:total_cells])
    }
  }

  if (nsteps & 1) {
    std::memcpy(u, u_tmp, total_cells * sizeof(double));
  }

  double stop = omp_get_wtime();

  GATE_STATS_F64("u", u, total_cells);

  std::printf("Results\n\n");
  std::printf("Error (L2norm): %E\n", norm);
  std::printf("Solve time (s): %lf\n", toc - tic);
  std::printf("Total time (s): %lf\n", stop - start);
  if (toc > tic) {
    const double bytes_moved =
        2.0 * static_cast<double>(total_cells) * static_cast<double>(nsteps) *
        sizeof(double);
    std::printf("Bandwidth (GB/s): %lf\n", 1.0e-9 * bytes_moved / (toc - tic));
  } else {
    std::printf("Bandwidth (GB/s): %lf\n", 0.0);
  }
  std::printf("%s", kLine);

  std::free(u);
  std::free(u_tmp);
  return 0;
}
