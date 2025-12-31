#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "gate.h"

#define PI acos(-1.0)
#define LINE "--------------------\n"

double solution(const double t, const double x, const double y, const double alpha, const double length);
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
  const int block_size = 256;

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

    tic = omp_get_wtime();

    for (int t = 0; t < nsteps; ++t) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          u_tmp[i + j * n] = r2 * u[i + j * n] +
                             r * ((i < n - 1) ? u[i + 1 + j * n] : 0.0) +
                             r * ((i > 0) ? u[i - 1 + j * n] : 0.0) +
                             r * ((j < n - 1) ? u[i + (j + 1) * n] : 0.0) +
                             r * ((j > 0) ? u[i + (j - 1) * n] : 0.0);
        }
      }

      double *tmp = u;
      u = u_tmp;
      u_tmp = tmp;
    }

    toc = omp_get_wtime();
  }

  double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);
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

double solution(const double t, const double x, const double y, const double alpha, const double length) {
  return exp(-2.0 * alpha * PI * PI * t / (length * length)) * sin(PI * x / length) * sin(PI * y / length);
}

double l2norm(const int n, const double *__restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {
  double time = dt * (double)nsteps;
  double l2norm = 0.0;

  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      double answer = solution(time, x, y, alpha, length);
      l2norm += (u[i + j * n] - answer) * (u[i + j * n] - answer);
      x += dx;
    }
    y += dx;
  }

  return sqrt(l2norm);
}
