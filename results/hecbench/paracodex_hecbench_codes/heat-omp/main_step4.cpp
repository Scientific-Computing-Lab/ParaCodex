#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "gate.h"

#define PI acos(-1.0)
#define LINE "--------------------\n"

#pragma omp declare target
double solution(const double t, const double x, const double y, const double alpha, const double length);
#pragma omp end declare target
double l2norm(const int n, const double *__restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

int main(int argc, char *argv[]) {
  double start = omp_get_wtime();

  int n = 1000;
  int nsteps = 10;

  if (argc == 3) {
    n = atoi(argv[1]);
    if (n < 0) {
      fprintf(stderr, "Error: n must be positive\n");
      exit(EXIT_FAILURE);
    }

    nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      fprintf(stderr, "Error: nsteps must be positive\n");
      exit(EXIT_FAILURE);
    }
  }

  double alpha = 0.1;
  double length = 1000.0;
  double dx = length / (n + 1);
  double dt = 0.5 / nsteps;

  double r = alpha * dt / (dx * dx);

  printf("\n");
  printf(" MMS heat equation\n\n");
  printf(LINE);
  printf("Problem input\n\n");
  printf(" Grid size: %d x %d\n", n, n);
  printf(" Cell width: %E\n", dx);
  printf(" Grid length: %lf x %lf\n", length, length);
  printf("\n");
  printf(" Alpha: %E\n", alpha);
  printf("\n");
  printf(" Steps: %d\n", nsteps);
  printf(" Total time: %E\n", dt * (double)nsteps);
  printf(" Time step: %E\n", dt);
  printf(LINE);

  printf("Stability\n\n");
  printf(" r value: %lf\n", r);
  if (r > 0.5)
    printf(" Warning: unstable\n");
  printf(LINE);

  double *u = (double *)malloc(sizeof(double) * n * n);
  double *u_tmp = (double *)malloc(sizeof(double) * n * n);

  double tic, toc;
  double norm = 0.0;
  const int block_size = 256;
  const int num_devices = omp_get_num_devices();
  const bool use_gpu = num_devices > 0;

  {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        double y = (j + 1) * dx;
        double x = (i + 1) * dx;
        u[i + j * n] = sin(PI * x / length) * sin(PI * y / length);
      }
    }

    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        u_tmp[i + j * n] = 0.0;
      }
    }

    const double r2 = 1.0 - 4.0 * r;
    const int total_points = n * n;
    const int interior_width = (n > 2) ? (n - 2) : 0;
    const int interior_points = interior_width * interior_width;
    const int num_teams_interior = (interior_points > 0) ? ((interior_points + block_size - 1) / block_size) : 0;
    const int row_elements = (n == 1) ? 1 : 2 * n;
    const int num_teams_rows = (row_elements > 0) ? ((row_elements + block_size - 1) / block_size) : 0;
    const int column_elements = (n > 2) ? 2 * (n - 2) : 0;
    const int num_teams_columns = (column_elements > 0) ? ((column_elements + block_size - 1) / block_size) : 0;

    // Persist the field arrays on the GPU when a device is available.
    #pragma omp target data map(tofrom: u[0:n * n], u_tmp[0:n * n]) if (use_gpu)
    {
      tic = omp_get_wtime();

      for (int t = 0; t < nsteps; ++t) {
        double *__restrict u_curr = u;
        double *__restrict u_next = u_tmp;

        if (use_gpu) {
          // Split interior and boundary updates to eliminate per-thread branching on the device.
          if (interior_points > 0) {
            #pragma omp target teams distribute parallel for collapse(2) \
                num_teams(num_teams_interior) thread_limit(block_size) \
                map(present: u_curr[0:total_points], u_next[0:total_points]) \
                firstprivate(r, r2, n)
            for (int j = 1; j < n - 1; ++j) {
              for (int i = 1; i < n - 1; ++i) {
                const int idx = i + j * n;
                const double center = u_curr[idx];
                const double east = u_curr[idx + 1];
                const double west = u_curr[idx - 1];
                const double north = u_curr[idx + n];
                const double south = u_curr[idx - n];
                u_next[idx] = r2 * center + r * (east + west + north + south);
              }
            }
          }

          if (row_elements > 0) {
            const int row_count = (n == 1) ? 1 : 2;
            #pragma omp target teams distribute parallel for collapse(2) \
                num_teams(num_teams_rows) thread_limit(block_size) \
                map(present: u_curr[0:total_points], u_next[0:total_points]) \
                firstprivate(r, r2, n, row_count)
            for (int edge = 0; edge < row_count; ++edge) {
              for (int i = 0; i < n; ++i) {
                const int j = (edge == 0) ? 0 : (n - 1);
                const int idx = i + j * n;
                const double center = u_curr[idx];
                const double east = (i < n - 1) ? u_curr[idx + 1] : 0.0;
                const double west = (i > 0) ? u_curr[idx - 1] : 0.0;
                double north = 0.0;
                double south = 0.0;
                if (edge == 0) {
                  north = (n > 1) ? u_curr[idx + n] : 0.0;
                } else if (n > 1) {
                  south = u_curr[idx - n];
                }
                u_next[idx] = r2 * center + r * (east + west + north + south);
              }
            }
          }

          if (column_elements > 0) {
            #pragma omp target teams distribute parallel for collapse(2) \
                num_teams(num_teams_columns) thread_limit(block_size) \
                map(present: u_curr[0:total_points], u_next[0:total_points]) \
                firstprivate(r, r2, n)
            for (int edge = 0; edge < 2; ++edge) {
              for (int j = 1; j < n - 1; ++j) {
                const int i = (edge == 0) ? 0 : (n - 1);
                const int idx = i + j * n;
                const double center = u_curr[idx];
                const double east = (edge == 0) ? u_curr[idx + 1] : 0.0;
                const double west = (edge == 0) ? 0.0 : u_curr[idx - 1];
                const double north = u_curr[idx + n];
                const double south = u_curr[idx - n];
                u_next[idx] = r2 * center + r * (east + west + north + south);
              }
            }
          }
        } else {
          for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
              const int idx = i + j * n;
              const double center = u_curr[idx];
              const double east = (i < n - 1) ? u_curr[idx + 1] : 0.0;
              const double west = (i > 0) ? u_curr[idx - 1] : 0.0;
              const double north = (j < n - 1) ? u_curr[idx + n] : 0.0;
              const double south = (j > 0) ? u_curr[idx - n] : 0.0;
              u_next[idx] = r2 * center + r * (east + west + north + south);
            }
          }
        }

        double *tmp = u;
        u = u_tmp;
        u_tmp = tmp;
      }

      toc = omp_get_wtime();
    }
  }

  norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

  double stop = omp_get_wtime();

  printf("Results\n\n");
  printf("Error (L2norm): %E\n", norm);
  printf("Solve time (s): %lf\n", toc - tic);
  printf("Total time (s): %lf\n", stop - start);
  printf("Bandwidth (GB/s): %lf\n", 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) / (toc - tic));
  printf(LINE);

  GATE_STATS_F64("u", u, (size_t)(n * n));
  GATE_STATS_F64("l2norm", &norm, 1);

  free(u);
  free(u_tmp);
}

#pragma omp declare target
double solution(const double t, const double x, const double y, const double alpha, const double length) {
  return exp(-2.0 * alpha * PI * PI * t / (length * length)) * sin(PI * x / length) * sin(PI * y / length);
}
#pragma omp end declare target

double l2norm(const int n, const double *__restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {
  double time = dt * (double)nsteps;
  double l2norm = 0.0;
  const bool use_gpu = omp_get_num_devices() > 0;
  const int team_size = 256;
  const int total_points = n * n;
  const int num_teams_launch = (total_points + team_size - 1) / team_size;

  // Offload the L2 norm accumulation to the GPU and collapse the loops for more parallel work.
  #pragma omp target teams distribute parallel for collapse(2) num_teams(num_teams_launch) thread_limit(team_size) map(to: u[0:n * n], time, dx, alpha, length, n) reduction(+: l2norm) if (use_gpu)
  for (int j = 0; j < n; ++j) {
    const double y = (j + 1) * dx;
    for (int i = 0; i < n; ++i) {
      const double x = (i + 1) * dx;
      double answer = solution(time, x, y, alpha, length);
      double diff = u[i + j * n] - answer;
      l2norm += diff * diff;
    }
  }

  return sqrt(l2norm);
}
