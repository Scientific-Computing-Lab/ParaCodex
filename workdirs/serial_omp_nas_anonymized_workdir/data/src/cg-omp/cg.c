#ifdef __PGIC__
#undef __GNUC__
#else
#define num_gangs(a)
#define num_workers(a)
#define vector_length(a)
#define gang
#define worker
#define vector
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "globals.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

unsigned int  nz =  (NA*(NONZER+1)*(NONZER+1));
unsigned int  naz = (NA*(NONZER+1));
unsigned int  na = NA;
static int colidx[NZ];
static int rowstr[NA+1];
static int iv[NA];
static int arow[NA];
static int acol[NAZ];

static double aelt[NAZ];
static double a[NZ];
static double x[NA+2];
static double z[NA+2];
static double p[NA+2];
static double q[NA+2];
static double r[NA+2];

static int *d_colidx;
static int *d_rowstr;
static double *d_a;
static double *d_x;
static double *d_z;
static double *d_p;
static double *d_q;
static double *d_r;

static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;

static double amult;
static double tran;

static logical timeron;

static void conj_grad(int colidx[],
                      int rowstr[],
                      double x[],
                      double z[],
                      double a[],
                      double p[],
                      double q[],
                      double r[],
                      double *rnorm);
static void makea(int n,
                  int nz,
                  double a[],
                  int colidx[],
                  int rowstr[],
                  int firstrow,
                  int lastrow,
                  int firstcol,
                  int lastcol,
                  int arow[],
                  int acol[][NONZER+1],
                  double aelt[][NONZER+1],
                  int iv[]);
static void sparse(double a[],
                   int colidx[],
                   int rowstr[],
                   int n,
                   int nz,
                   int nozer,
                   int arow[],
                   int acol[][NONZER+1],
                   double aelt[][NONZER+1],
                   int firstrow,
                   int lastrow,
                   int nzloc[],
                   double rcond,
                   double shift);
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
static int icnvrt(double x, int ipwr2);
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
static int conj_calls = 0;
static int loop_iter = 0;

int main(int argc, char *argv[])
{
  int i, j, k, it;
  int end;

  double zeta;
  double rnorm;
  double norm_temp1, norm_temp2;

  double t, mflops, tmax;
  char Class;
  int verified;
  double zeta_verify_value, epsilon, err;

  char *t_names[T_last];

  for (i = 0; i < T_last; i++) {
    timer_clear(i);
  }
  
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[T_init] = "init";
    t_names[T_bench] = "benchmk";
    t_names[T_conj_grad] = "conjgd";
    fclose(fp);
  } else {
    timeron = false;
  }

  timer_start(T_init);

  firstrow = 0;
  lastrow  = NA-1;
  firstcol = 0;
  lastcol  = NA-1;

  if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10) {
    Class = 'S';
    zeta_verify_value = 8.5971775078648;
  } else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12) {
    Class = 'W';
    zeta_verify_value = 10.362595087124;
  } else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20) {
    Class = 'A';
    zeta_verify_value = 17.130235054029;
  } else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60) {
    Class = 'B';
    zeta_verify_value = 22.712745482631;
  } else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110) {
    Class = 'C';
    zeta_verify_value = 28.973605592845;
  } else if (NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500) {
    Class = 'D';
    zeta_verify_value = 52.514532105794;
  } else if (NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500) {
    Class = 'E';
    zeta_verify_value = 77.522164599383;
  } else {
    Class = 'U';
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - CG Benchmark\n\n");
  printf(" Size: %11d\n", NA);
  printf(" Iterations: %5d\n", NITER);
  printf("\n");

  naa = NA;
  nzz = NZ;

  tran    = 314159265.0;
  amult   = 1220703125.0;
  zeta    = randlc(&tran, amult);

  makea(naa, nzz, a, colidx, rowstr, 
        firstrow, lastrow, firstcol, lastcol, 
        arow, 
        (int (*)[NONZER+1])(void*)acol, 
        (double (*)[NONZER+1])(void*)aelt,
        iv);

  for (j = 0; j < lastrow - firstrow + 1; j++) {
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      colidx[k] = colidx[k] - firstcol;
    }
  }

  for (i = 0; i < NA+1; i++) {
    x[i] = 1.0;
  }

  end = lastcol - firstcol + 1;
  for (j = 0; j < end; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = 0.0;
    p[j] = 0.0;
  }

  zeta = 0.0;

  int device = omp_get_default_device();
  int host_device = omp_get_initial_device();
  size_t mat_bytes = (size_t)NZ * sizeof(double);
  size_t vec_bytes = (size_t)(NA+2) * sizeof(double);
  size_t colidx_bytes = (size_t)NZ * sizeof(int);
  size_t rowstr_bytes = (size_t)(NA+1) * sizeof(int);

  d_colidx = (int*)omp_target_alloc(colidx_bytes, device);
  d_rowstr = (int*)omp_target_alloc(rowstr_bytes, device);
  d_a      = (double*)omp_target_alloc(mat_bytes, device);
  d_x      = (double*)omp_target_alloc(vec_bytes, device);
  d_z      = (double*)omp_target_alloc(vec_bytes, device);
  d_p      = (double*)omp_target_alloc(vec_bytes, device);
  d_q      = (double*)omp_target_alloc(vec_bytes, device);
  d_r      = (double*)omp_target_alloc(vec_bytes, device);

  if (!d_colidx || !d_rowstr || !d_a || !d_x ||
      !d_z || !d_p || !d_q || !d_r) {
    fprintf(stderr, "Failed to allocate device buffers\n");
    exit(EXIT_FAILURE);
  }

  omp_target_memcpy(d_colidx, colidx, colidx_bytes, 0, 0, device, host_device);
  omp_target_memcpy(d_rowstr, rowstr, rowstr_bytes, 0, 0, device, host_device);
  omp_target_memcpy(d_a, a, mat_bytes, 0, 0, device, host_device);
  omp_target_memcpy(d_x, x, vec_bytes, 0, 0, device, host_device);
  omp_target_memcpy(d_z, z, vec_bytes, 0, 0, device, host_device);
  omp_target_memcpy(d_p, p, vec_bytes, 0, 0, device, host_device);
  omp_target_memcpy(d_q, q, vec_bytes, 0, 0, device, host_device);
  omp_target_memcpy(d_r, r, vec_bytes, 0, 0, device, host_device);

  for (it = 1; it <= 1; it++) {
    conj_grad(d_colidx, d_rowstr, d_x, d_z, d_a, d_p, d_q, d_r, &rnorm);

    // Normalize on-device before the benchmark so z/x stay resident on the GPU.
    norm_temp2 = 0.0;
    #pragma omp target teams loop reduction(+:norm_temp2) is_device_ptr(d_z) firstprivate(end)
    for (j = 0; j < end; j++) {
      norm_temp2 = norm_temp2 + d_z[j] * d_z[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    #pragma omp target teams loop is_device_ptr(d_x, d_z) firstprivate(end, norm_temp2)
    for (j = 0; j < end; j++) {
      d_x[j] = norm_temp2 * d_z[j];
    }
  }

  for (i = 0; i < NA+1; i++) {
    x[i] = 1.0;
  }

  omp_target_memcpy(d_x, x, vec_bytes, 0, 0, device, host_device);

  zeta = 0.0;

  timer_stop(T_init);

  printf(" Initialization time = %15.3f seconds\n", timer_read(T_init));

  timer_start(T_bench);

  for (it = 1; it <= NITER; it++) {
    conj_grad(d_colidx, d_rowstr, d_x, d_z, d_a, d_p, d_q, d_r, &rnorm);

    // Reduce dot-products on the GPU to avoid transferring whole z/x arrays.
    norm_temp1 = 0.0;
    #pragma omp target teams loop reduction(+:norm_temp1) is_device_ptr(d_x, d_z) firstprivate(end)
    for (j = 0; j < end; j++) {
      norm_temp1 = norm_temp1 + d_x[j] * d_z[j];
    }

    norm_temp2 = 0.0;
    #pragma omp target teams loop reduction(+:norm_temp2) is_device_ptr(d_z) firstprivate(end)
    for (j = 0; j < end; j++) {
      norm_temp2 = norm_temp2 + d_z[j] * d_z[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    zeta = SHIFT + 1.0 / norm_temp1;
    if (it == 1) 
      printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);

    #pragma omp target teams loop is_device_ptr(d_x, d_z) firstprivate(end, norm_temp2)
    for (j = 0; j < end; j++) {
      d_x[j] = norm_temp2 * d_z[j];
    }
  }

  timer_stop(T_bench);

  t = timer_read(T_bench);

  printf(" Benchmark completed\n");

  epsilon = 1.0e-10;
  if (Class != 'U') {
    err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
    if (err <= epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" Zeta is    %20.13E\n", zeta);
      printf(" Error is   %20.13E\n", err);
    } else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %20.13E\n", zeta);
      printf(" The correct zeta is %20.13E\n", zeta_verify_value);
    }
  } else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
  }

  if (t != 0.0) {
    mflops = (double)(2*NITER*NA)
                   * (3.0+(double)(NONZER*(NONZER+1))
                     + 25.0*(5.0+(double)(NONZER*(NONZER+1)))
                     + 3.0) / t / 1000000.0;
  } else {
    mflops = 0.0;
  }

  print_results("CG", Class, NA, 0, 0,
                NITER, t,
                mflops, "          floating point", 
                verified, NPBVERSION, COMPILETIME,
                CS1, CS2, CS3, CS4, CS5, CS6, CS7);

  if (timeron) {
    tmax = timer_read(T_bench);
    if (tmax == 0.0) tmax = 1.0;
    printf("  SECTION   Time (secs)\n");
    for (i = 0; i < T_last; i++) {
      t = timer_read(i);
      if (i == T_init) {
        printf("  %8s:%9.3f\n", t_names[i], t);
      } else {
        printf("  %8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100.0/tmax);
        if (i == T_conj_grad) {
          t = tmax - t;
          printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest", t, t*100.0/tmax);
        }
      }
    }
  }
  omp_target_free(d_colidx, device);
  omp_target_free(d_rowstr, device);
  omp_target_free(d_a, device);
  omp_target_free(d_x, device);
  omp_target_free(d_z, device);
  omp_target_free(d_p, device);
  omp_target_free(d_q, device);
  omp_target_free(d_r, device);
  printf("conj calls=%d, loop iter = %d. \n", conj_calls, loop_iter);
  return 0;
}

static void conj_grad(int colidx[],
                      int rowstr[],
                      double x[],
                      double z[],
                      double a[],
                      double p[],
                      double q[],
                      double r[],
                      double *rnorm)
{
  int j, k, tmp1, tmp2, tmp3;
  int cgit, cgitmax = 25;
  double rho, rho0, alpha, beta;
  const int rows = lastrow - firstrow + 1;
  const int cols = lastcol - firstcol + 1;
  conj_calls++;

  rho = 0.0;
  // Initialize q/z/r/p and compute rho in one kernel to reduce launches.
  #pragma omp target teams loop reduction(+:rho) is_device_ptr(z, r, p, q, x) firstprivate(rows)
  for (j = 0; j < rows; j++) {
    double xj = x[j];
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = xj;
    p[j] = xj;
    rho += xj * xj;
  }
  
  for (cgit = 1; cgit <= cgitmax; cgit++) {
    loop_iter++;

    double d = 0.0;
    // Assemble q and accumulate p·q inside the same kernel.
    #pragma omp target teams loop reduction(+:d) \
      is_device_ptr(rowstr, colidx, p, q, a) firstprivate(rows)
    for (j = 0; j < rows; j++) {
      tmp1 = rowstr[j];
      tmp2 = rowstr[j+1];
      double sum_loc = 0.0;
      for (k = tmp1; k < tmp2; k++) {
        tmp3 = colidx[k];
        sum_loc += a[k] * p[tmp3];
      }
      q[j] = sum_loc;
      d += p[j] * sum_loc;
    }

    alpha = rho / d;
    rho0 = rho;

    rho = 0.0;
    // Update z/r and recompute rho at once to shave off one kernel launch.
    #pragma omp target teams loop reduction(+:rho) \
      is_device_ptr(z, r, p, q) firstprivate(rows, alpha)
    for (j = 0; j < rows; j++) {
      double qj = q[j];
      double pj = p[j];
      double rj = r[j] - alpha * qj;
      double zj = z[j] + alpha * pj;
      z[j] = zj;
      r[j] = rj;
      rho += rj * rj;
    }

    beta = rho / rho0;

    #pragma omp target teams loop is_device_ptr(p, r) firstprivate(rows, beta)
    for (j = 0; j < rows; j++) {
      p[j] = r[j] + beta * p[j];
    } 
  }

  #pragma omp target teams loop is_device_ptr(rowstr, colidx, z, r, a) firstprivate(rows)
  for (j = 0; j < rows; j++) {
    tmp1 = rowstr[j];
    tmp2 = rowstr[j+1];
    double sum_loc = 0.0;
    for (k = tmp1; k < tmp2; k++) {
      tmp3 = colidx[k];
      sum_loc += a[k] * z[tmp3];
    }
    r[j] = sum_loc;
  }
   
  double diff_sum = 0.0;
  #pragma omp target teams loop reduction(+:diff_sum) is_device_ptr(x, r) firstprivate(cols)
  for (j = 0; j < cols; j++) {
    double diff = x[j] - r[j];
    diff_sum += diff * diff;
  }

  *rnorm = sqrt(diff_sum);
}

static void makea(int n,
                  int nz,
                  double a[],
                  int colidx[],
                  int rowstr[],
                  int firstrow,
                  int lastrow,
                  int firstcol,
                  int lastcol,
                  int arow[],
                  int acol[][NONZER+1],
                  double aelt[][NONZER+1],
                  int iv[])
{
  int iouter, ivelt, nzv, nn1;
  int ivc[NONZER+1];
  double vc[NONZER+1];

  nn1 = 1;
  do {
    nn1 = 2 * nn1;
  } while (nn1 < n);

  for (iouter = 0; iouter < n; iouter++) {
    nzv = NONZER;
    sprnvc(n, nzv, nn1, vc, ivc);
    vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
    arow[iouter] = nzv;
    
    for (ivelt = 0; ivelt < nzv; ivelt++) {
      acol[iouter][ivelt] = ivc[ivelt] - 1;
      aelt[iouter][ivelt] = vc[ivelt];
    }
  }

  sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, 
         aelt, firstrow, lastrow,
         iv, RCOND, SHIFT);
}

static void sparse(double a[],
                   int colidx[],
                   int rowstr[],
                   int n,
                   int nz,
                   int nozer,
                   int arow[],
                   int acol[][NONZER+1],
                   double aelt[][NONZER+1],
                   int firstrow,
                   int lastrow,
                   int nzloc[],
                   double rcond,
                   double shift)
{
  int nrows;

  int i, j, j1, j2, nza, k, kk, nzrow, jcol;
  double size, scale, ratio, va;
  logical cont40;

  nrows = lastrow - firstrow + 1;

  for (j = 0; j < nrows+1; j++) {
    rowstr[j] = 0;
  }

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza] + 1;
      rowstr[j] = rowstr[j] + arow[i];
    }
  }

  rowstr[0] = 0;
  for (j = 1; j < nrows+1; j++) {
    rowstr[j] = rowstr[j] + rowstr[j-1];
  }
  nza = rowstr[nrows] - 1;

  if (nza > nz) {
    printf("Space for matrix elements exceeded in sparse\n");
    printf("nza, nzmax = %d, %d\n", nza, nz);
    exit(EXIT_FAILURE);
  }

  for (j = 0; j < nrows; j++) {
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      a[k] = 0.0;
      colidx[k] = -1;
    }
    nzloc[j] = 0;
  }

  size = 1.0;
  ratio = pow(rcond, (1.0 / (double)(n)));

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza];

      scale = size * aelt[i][nza];
      for (nzrow = 0; nzrow < arow[i]; nzrow++) {
        jcol = acol[i][nzrow];
        va = aelt[i][nzrow] * scale;

        if (jcol == j && j == i) {
          va = va + rcond - shift;
        }

        cont40 = false;
        for (k = rowstr[j]; k < rowstr[j+1]; k++) {
          if (colidx[k] > jcol) {
            for (kk = rowstr[j+1]-2; kk >= k; kk--) {
              if (colidx[kk] > -1) {
                a[kk+1]  = a[kk];
                colidx[kk+1] = colidx[kk];
              }
            }
            colidx[k] = jcol;
            a[k]  = 0.0;
            cont40 = true;
            break;
          } else if (colidx[k] == -1) {
            colidx[k] = jcol;
            cont40 = true;
            break;
          } else if (colidx[k] == jcol) {
            nzloc[j] = nzloc[j] + 1;
            cont40 = true;
            break;
          }
        }
        if (cont40 == false) {
          printf("internal error in sparse: i=%d\n", i);
          exit(EXIT_FAILURE);
        }
        a[k] = a[k] + va;
      }
    }
    size = size * ratio;
  }

  for (j = 1; j < nrows; j++) {
    nzloc[j] = nzloc[j] + nzloc[j-1];
  }

  for (j = 0; j < nrows; j++) {
    if (j > 0) {
      j1 = rowstr[j] - nzloc[j-1];
    } else {
      j1 = 0;
    }
    j2 = rowstr[j+1] - nzloc[j];
    nza = rowstr[j];
    for (k = j1; k < j2; k++) {
      a[k] = a[nza];
      colidx[k] = colidx[nza];
      nza = nza + 1;
    }
  }
  for (j = 1; j < nrows+1; j++) {
    rowstr[j] = rowstr[j] - nzloc[j-1];
  }
  nza = rowstr[nrows] - 1;
}

static void sprnvc(int n, int nz, int nn1, double v[], int iv[])
{
  int nzv, ii, i;
  double vecelt, vecloc;

  nzv = 0;

  while (nzv < nz) {
    vecelt = randlc(&tran, amult);

    vecloc = randlc(&tran, amult);
    i = icnvrt(vecloc, nn1) + 1;
    if (i > n) continue;

    logical was_gen = false;
    for (ii = 0; ii < nzv; ii++) {
      if (iv[ii] == i) {
        was_gen = true;
        break;
      }
    }
    if (was_gen) continue;
    v[nzv] = vecelt;
    iv[nzv] = i;
    nzv = nzv + 1;
  }
}

static int icnvrt(double x, int ipwr2)
{
  return (int)(ipwr2 * x);
}

static void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
  int k;
  logical set;

  set = false;
  for (k = 0; k < *nzv; k++) {
    if (iv[k] == i) {
      v[k] = val;
      set  = true;
    }
  }
  if (set == false) {
    v[*nzv]  = val;
    iv[*nzv] = i;
    *nzv     = *nzv + 1;
  }
}
