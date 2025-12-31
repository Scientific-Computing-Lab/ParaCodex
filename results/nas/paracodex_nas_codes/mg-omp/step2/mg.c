
#ifdef __PGIC__
#undef __GNUC__
#else
#define num_workers(a)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "globals.h"

#include "timers.h"
#include "print_results.h"

#define I3D(array,n1,n2,i3,i2,i1) (array[(i3)*n2*n1 + (i2)*n1 + (i1)])

static void setup(int *n1, int *n2, int *n3);
static void mg3P(double u[], double v[], double r[],
                 double a[4], double c[4], int n1, int n2, int n3);
static void psinv(double * __restrict__ orr, double * __restrict__ ou, int n1, int n2, int n3,
                  double c[4], int k);
static void resid(double * __restrict__ ou, double * __restrict__ ov, double * __restrict__ orr, int n1, int n2, int n3,
                  double a[4], int k);
static void rprj3(double *orr, int m1k, int m2k, int m3k,
                  double *os, int m1j, int m2j, int m3j, int k);
static void interp(double *oz, int mm1, int mm2, int mm3,
                   double *ou, int n1, int n2, int n3, int k);
static void norm2u3(double *orr, int n1, int n2, int n3,
                    double *rnm2, double *rnmu,
                    int nx, int ny, int nz);
static void rep_nrm(double *u, int n1, int n2, int n3, char *title, int kk);
static void comm3(double *ou, int n1, int n2, int n3, int kk);
static void zran3(double *oz, int n1, int n2, int n3, int nx, int ny, int k);
static void showall(double *oz, int n1, int n2, int n3);
static double power(double a, int n);
static void bubble(double ten[][2], int j1[][2], int j2[][2], int j3[][2],
                   int m, int ind);
static void zero3(double *oz, int n1, int n2, int n3);

static double u[NR];
static double v[NR];
static double r[NR];
int gnr = NR;

static int is1, is2, is3, ie1, ie2, ie3;

int main()
{

  int k, it;
  double t, tinit, mflops;

  double a[4], c[4];

  double rnm2, rnmu, old2, oldu, epsilon;
  int n1, n2, n3, nit;
  double nn, verify_value, err;
  logical verified;

  int i;
  char *t_names[T_last];
  double tmax;

  for (i = T_init; i < T_last; i++) {
    timer_clear(i);
  }
  timer_start(T_init);

  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[T_init] = "init";
    t_names[T_bench] = "benchmk";
    t_names[T_mg3P] = "mg3P";
    t_names[T_psinv] = "psinv";
    t_names[T_resid] = "resid";
    t_names[T_rprj3] = "rprj3";
    t_names[T_interp] = "interp";
    t_names[T_norm2] = "norm2";
    t_names[T_comm3] = "comm3";
    fclose(fp);
  } else {
    timeron = false;
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - MG Benchmark\n\n");

  if ((fp = fopen("mg.input", "r")) != NULL) {
    int result;
    printf(" Reading from input file mg.input\n");
    result = fscanf(fp, "%d\n", &lt);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d%d", &nx[lt], &ny[lt], &nz[lt]);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d", &nit);
    while (fgetc(fp) != '\n');
    for (i = 0; i <= 7; i++) {
      result = fscanf(fp, "%d", &debug_vec[i]);
    }
    fclose(fp);
  } else {
    printf(" No input file. Using compiled defaults \n");
    lt = LT_DEFAULT;
    nit = NIT_DEFAULT;
    nx[lt] = NX_DEFAULT;
    ny[lt] = NY_DEFAULT;
    nz[lt] = NZ_DEFAULT;
	
    for (i = 0; i <= 7; i++) {
      debug_vec[i] = DEBUG_DEFAULT;
    }
  }

  if ( (nx[lt] != ny[lt]) || (nx[lt] != nz[lt]) ) {
    Class = 'U';
  } else if ( nx[lt] == 32 && nit == 4 ) {
    Class = 'S';
  } else if ( nx[lt] == 128 && nit == 4 ) {
    Class = 'W';
  } else if ( nx[lt] == 256 && nit == 4 ) {  
    Class = 'A';
  } else if ( nx[lt] == 256 && nit == 20 ) {
    Class = 'B';
  } else if ( nx[lt] == 512 && nit == 20 ) {  
    Class = 'C';
  } else if ( nx[lt] == 1024 && nit == 50 ) {  
    Class = 'D';
  } else if ( nx[lt] == 2048 && nit == 50 ) {  
    Class = 'E';
  } else {
    Class = 'U';
  }

  a[0] = -8.0/3.0;
  a[1] =  0.0;
  a[2] =  1.0/6.0;
  a[3] =  1.0/12.0;
      
  if (Class == 'A' || Class == 'S' || Class =='W') {

    c[0] =  -3.0/8.0;
    c[1] =  +1.0/32.0;
    c[2] =  -1.0/64.0;
    c[3] =   0.0;
  } else {

    c[0] =  -3.0/17.0;
    c[1] =  +1.0/33.0;
    c[2] =  -1.0/61.0;
    c[3] =   0.0;
  }
  lb = 1;
  k  = lt;

  #pragma omp target enter data map(alloc: u[0:NR], v[0:NR], r[0:NR])

{
  setup(&n1, &n2, &n3);
  zero3(u, n1, n2, n3);
  zran3(v, n1, n2, n3, nx[lt], ny[lt], k);
  #pragma omp target update to(v[0:NR])

  norm2u3(v, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);

  printf(" Size: %4dx%4dx%4d  (class %c)\n", nx[lt], ny[lt], nz[lt], Class);
  printf(" Iterations: %3d\n", nit);
  printf("\n");

  resid(u, v, r, n1, n2, n3, a, k);
  norm2u3(r, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);
  old2 = rnm2;
  oldu = rnmu;

  mg3P(u, v, r, a, c, n1, n2, n3);
  resid(u, v, r, n1, n2, n3, a, k);
  setup(&n1, &n2, &n3);
  zero3(u, n1, n2, n3);
  zran3(v, n1, n2, n3, nx[lt], ny[lt], k);
  #pragma omp target update to(v[0:NR])

  timer_stop(T_init);
  tinit = timer_read(T_init);

  printf(" Initialization time: %15.3f seconds\n\n", tinit);

  for (i = T_bench; i < T_last; i++) {
    timer_clear(i);
  }

  timer_start(T_bench);

  resid(u, v, r, n1, n2, n3, a, k);
  norm2u3(r, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);
  old2 = rnm2;
  oldu = rnmu;

  for (it = 1; it <= nit; it++) {
    if ((it == 1) || (it == nit) || ((it % 100) == 0)) {
      printf("  iter %3d\n", it);
    }
    mg3P(u, v, r, a, c, n1, n2, n3);
    resid(u, v, r, n1, n2, n3, a, k);
  }

  norm2u3(r, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);
}
  timer_stop(T_bench);
  #pragma omp target exit data map(from:u[0:NR], r[0:NR]) map(delete:v[0:NR])

  t = timer_read(T_bench);

  verified = false;
  verify_value = 0.0;

  printf("\n Benchmark completed\n");

  epsilon = 1.0e-8;
  if (Class != 'U') {
    if (Class == 'S') {
      verify_value = 0.5307707005734e-04;
    } else if (Class == 'W') {
      verify_value = 0.6467329375339e-05;
    } else if (Class == 'A') {
      verify_value = 0.2433365309069e-05;
    } else if (Class == 'B') {
      verify_value = 0.1800564401355e-05;
    } else if (Class == 'C') {
      verify_value = 0.5706732285740e-06;
    } else if (Class == 'D') {
      verify_value = 0.1583275060440e-09;
    } else if (Class == 'E') {
      verify_value = 0.8157592357404e-10;
    }

    err = fabs( rnm2 - verify_value ) / verify_value;
    
    if (err <= epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" L2 Norm is %20.13E\n", rnm2);
      printf(" Error is   %20.13E\n", err);
    } else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" L2 Norm is             %20.13E\n", rnm2);
      printf(" The correct L2 Norm is %20.13E\n", verify_value);
    }
  } else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
    printf(" L2 Norm is %20.13E\n", rnm2);
  }

  nn = 1.0 * nx[lt] * ny[lt] * nz[lt];

  if (t != 0.0) {
    mflops = 58.0 * nit * nn * 1.0e-6 / t;
  } else {
    mflops = 0.0;
  }

  print_results("MG", Class, nx[lt], ny[lt], nz[lt], 
                nit, t,
                mflops, "          floating point", 
                verified, NPBVERSION, COMPILETIME,
                CS1, CS2, CS3, CS4, CS5, CS6, CS7);

  if (timeron) {
    tmax = timer_read(T_bench);
    if (tmax == 0.0) tmax = 1.0;

    printf("  SECTION   Time (secs)\n");
    for (i = T_bench; i < T_last; i++) {
      t = timer_read(i);
      if (i == T_resid2) {
        t = timer_read(T_resid) - t;
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "mg-resid", t, t*100./tmax);
      } else {
        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100./tmax);
      }
    }
  }
  return 0;
}

static void setup(int *n1, int *n2, int *n3)
{
  int k, j;

  int ax, mi[MAXLEVEL+1][3];
  int ng[MAXLEVEL+1][3];

  ng[lt][0] = nx[lt];
  ng[lt][1] = ny[lt];
  ng[lt][2] = nz[lt];
  for (k = lt-1; k >= 1; k--) {
    for (ax = 0; ax < 3; ax++) {
      ng[k][ax] = ng[k+1][ax]/2;
    }
  }
  for (k = lt; k >= 1; k--) {
    nx[k] = ng[k][0];
    ny[k] = ng[k][1];
    nz[k] = ng[k][2];
  }

  for (k = lt; k >= 1; k--) {
    for (ax = 0; ax < 3; ax++) {
      mi[k][ax] = 2 + ng[k][ax];
    }

    m1[k] = mi[k][0];
    m2[k] = mi[k][1];
    m3[k] = mi[k][2];
  }

  k = lt;
  is1 = 2 + ng[k][0] - ng[lt][0];
  ie1 = 1 + ng[k][0];
  *n1 = 3 + ie1 - is1;
  is2 = 2 + ng[k][1] - ng[lt][1];
  ie2 = 1 + ng[k][1];
  *n2 = 3 + ie2 - is2;
  is3 = 2 + ng[k][2] - ng[lt][2];
  ie3 = 1 + ng[k][2];
  *n3 = 3 + ie3 - is3;

  ir[lt] = 0;
  for (j = lt-1; j >= 1; j--) {
    ir[j] = ir[j+1]+ONE*m1[j+1]*m2[j+1]*m3[j+1];
  }

  if (debug_vec[1] >= 1) {
    printf(" in setup, \n");
    printf(" k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3\n");
    printf("%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d\n", 
        k,lt,ng[k][0],ng[k][1],ng[k][2],*n1,*n2,*n3,is1,is2,is3,ie1,ie2,ie3);
  }
}

static void mg3P(double u[], double v[], double r[],
                 double a[4], double c[4], int n1, int n2, int n3)
{
  int j, k;

  for (k = lt; k >= lb+1; k--) {
    j = k - 1;
    rprj3(&r[ir[k]], m1[k], m2[k], m3[k], 
          &r[ir[j]], m1[j], m2[j], m3[j], k);
  }
  
  k = lb;

  zero3(&u[ir[k]], m1[k], m2[k], m3[k]);
  psinv(&r[ir[k]], &u[ir[k]], m1[k], m2[k], m3[k], c, k);

  for (k = lb+1; k <= lt-1; k++) {
    j = k - 1;

    zero3(&u[ir[k]], m1[k], m2[k], m3[k]);
    interp(&u[ir[j]], m1[j], m2[j], m3[j], &u[ir[k]], m1[k], m2[k], m3[k], k);

    resid(&u[ir[k]], &r[ir[k]], &r[ir[k]], m1[k], m2[k], m3[k], a, k);

    psinv(&r[ir[k]], &u[ir[k]], m1[k], m2[k], m3[k], c, k);
  }

  j = lt - 1;
  k = lt;
  interp(&u[ir[j]], m1[j], m2[j], m3[j], u, n1, n2, n3, k);
  resid(u, v, r, n1, n2, n3, a, k);
  psinv(r, u, n1, n2, n3, c, k);
}

static void psinv(double * __restrict__ orr, double * __restrict__ ou, int n1, int n2, int n3, double c[4], int k)
{

  int i3, i2, i1;
  double c0, c1, c2;

  size_t total = (size_t)n3*n2*n1;

  c0 = c[0];
  c1 = c[1];
  c2 = c[2];

  if (timeron) timer_start(T_psinv);
  /* Fused psinv to remove temporary buffers and extra kernel launch. */
  #pragma omp target teams loop collapse(3) map(present:orr[0:total],ou[0:total])
  for (i3 = 1; i3 < n3-1; i3++) {
    for (i2 = 1; i2 < n2-1; i2++) {
      for (i1 = 1; i1 < n1-1; i1++) {
        double center = I3D(orr, n1, n2, i3, i2, i1);
        double center_left = I3D(orr, n1, n2, i3, i2, i1-1);
        double center_right = I3D(orr, n1, n2, i3, i2, i1+1);

        double r1_center = I3D(orr, n1, n2, i3, i2-1, i1) + I3D(orr, n1, n2, i3, i2+1, i1)
                         + I3D(orr, n1, n2, i3-1, i2, i1) + I3D(orr, n1, n2, i3+1, i2, i1);
        double r1_left = I3D(orr, n1, n2, i3, i2-1, i1-1) + I3D(orr, n1, n2, i3, i2+1, i1-1)
                       + I3D(orr, n1, n2, i3-1, i2, i1-1) + I3D(orr, n1, n2, i3+1, i2, i1-1);
        double r1_right = I3D(orr, n1, n2, i3, i2-1, i1+1) + I3D(orr, n1, n2, i3, i2+1, i1+1)
                        + I3D(orr, n1, n2, i3-1, i2, i1+1) + I3D(orr, n1, n2, i3+1, i2, i1+1);
        double r2_center = I3D(orr, n1, n2, i3-1, i2-1, i1) + I3D(orr, n1, n2, i3-1, i2+1, i1)
                         + I3D(orr, n1, n2, i3+1, i2-1, i1) + I3D(orr, n1, n2, i3+1, i2+1, i1);

        I3D(ou, n1, n2, i3, i2, i1) = I3D(ou, n1, n2, i3, i2, i1)
                                    + c0 * center
                                    + c1 * (center_left + center_right + r1_center)
                                    + c2 * (r2_center + r1_left + r1_right);
      }
    }
  }
  if (timeron) timer_stop(T_psinv);

  comm3(ou, n1, n2, n3, k);

  if (debug_vec[0] >= 1) {
    
    rep_nrm(ou, n1, n2, n3, "   psinv", k);
  }

  if (debug_vec[3] >= k) {
    
    showall(ou, n1, n2, n3);
  }
}

static void resid(double * __restrict__ ou, double * __restrict__ ov, double * __restrict__ orr, int n1, int n2, int n3,
                  double a[4], int k)
{

  int i3, i2, i1;
  double a0, a2, a3;
  
  size_t total = (size_t)n3*n2*n1;

  a0 = a[0];
  a2 = a[2];
  a3 = a[3];

  if (timeron) timer_start(T_resid);
  /* Fused resid to avoid temporary buffers and extra kernel launch. */
  #pragma omp target teams loop collapse(3) map(present:ou[0:total],ov[0:total],orr[0:total])
  for (i3 = 1; i3 < n3-1; i3++) {
    for (i2 = 1; i2 < n2-1; i2++) {
      for (i1 = 1; i1 < n1-1; i1++) {
        double u_center = I3D(ou, n1, n2, i3, i2, i1);
        double u_left = I3D(ou, n1, n2, i3, i2, i1-1);
        double u_right = I3D(ou, n1, n2, i3, i2, i1+1);

        double u1_center = I3D(ou, n1, n2, i3, i2-1, i1) + I3D(ou, n1, n2, i3, i2+1, i1)
                        + I3D(ou, n1, n2, i3-1, i2, i1) + I3D(ou, n1, n2, i3+1, i2, i1);
        double u1_left = I3D(ou, n1, n2, i3, i2-1, i1-1) + I3D(ou, n1, n2, i3, i2+1, i1-1)
                       + I3D(ou, n1, n2, i3-1, i2, i1-1) + I3D(ou, n1, n2, i3+1, i2, i1-1);
        double u1_right = I3D(ou, n1, n2, i3, i2-1, i1+1) + I3D(ou, n1, n2, i3, i2+1, i1+1)
                        + I3D(ou, n1, n2, i3-1, i2, i1+1) + I3D(ou, n1, n2, i3+1, i2, i1+1);

        double u2_center = I3D(ou, n1, n2, i3-1, i2-1, i1) + I3D(ou, n1, n2, i3-1, i2+1, i1)
                         + I3D(ou, n1, n2, i3+1, i2-1, i1) + I3D(ou, n1, n2, i3+1, i2+1, i1);
        double u2_left = I3D(ou, n1, n2, i3-1, i2-1, i1-1) + I3D(ou, n1, n2, i3-1, i2+1, i1-1)
                       + I3D(ou, n1, n2, i3+1, i2-1, i1-1) + I3D(ou, n1, n2, i3+1, i2+1, i1-1);
        double u2_right = I3D(ou, n1, n2, i3-1, i2-1, i1+1) + I3D(ou, n1, n2, i3-1, i2+1, i1+1)
                        + I3D(ou, n1, n2, i3+1, i2-1, i1+1) + I3D(ou, n1, n2, i3+1, i2+1, i1+1);

        I3D(orr, n1, n2, i3, i2, i1) = I3D(ov, n1, n2, i3, i2, i1) 
               - a0 * u_center
               - a2 * (u2_center + u1_left + u1_right)
               - a3 * (u2_left + u2_right);
      }
    }
  }
  if (timeron) timer_stop(T_resid);

  comm3(orr, n1, n2, n3, k);

  if (debug_vec[0] >= 1) {
  
  rep_nrm(orr, n1, n2, n3, "   resid", k);
  }

  if (debug_vec[2] >= k) {
  
  showall(orr, n1, n2, n3);
  }
}

static void rprj3(double *orr, int m1k, int m2k, int m3k,
                  double *os, int m1j, int m2j, int m3j, int k)
{

  int j3, j2, j1, i3, i2, i1, d1, d2, d3, j;

  double *x1, *y1, x2, y2;
  size_t fine_total = (size_t)m3k*m2k*m1k;
  size_t coarse_total = (size_t)m3j*m2j*m1j;
  x1 = (double*)malloc(fine_total*sizeof(double));
  y1 = (double*)malloc(fine_total*sizeof(double));

  if (timeron) timer_start(T_rprj3);
  if (m1k == 3) {
    d1 = 2;
  } else {
    d1 = 1;
  }

  if (m2k == 3) {
    d2 = 2;
  } else {
    d2 = 1;
  }

  if (m3k == 3) {
    d3 = 2;
  } else {
    d3 = 1;
  }

  #pragma omp target data map(present: orr[0:fine_total], os[0:coarse_total]) map(alloc: x1[0:fine_total], y1[0:fine_total])
  {
    #pragma omp target teams loop collapse(3) map(present:orr[0:fine_total],x1[0:fine_total],y1[0:fine_total])
    for (j3 = 1; j3 < m3j-1; j3++) {
      i3 = 2*j3-d3;
      for (j2 = 1; j2 < m2j-1; j2++) {
        i2 = 2*j2-d2;
        for (j1 = 1; j1 < m1j; j1++) {
          i1 = 2*j1-d1;
          I3D(x1, m1k, m2k, i3, i2, i1) = I3D(orr, m1k, m2k, i3+1, i2, i1) + I3D(orr, m1k, m2k, i3+1, i2+2, i1)
                 + I3D(orr, m1k, m2k, i3, i2+1, i1) + I3D(orr, m1k, m2k, i3+2, i2+1, i1);
          I3D(y1, m1k, m2k, i3, i2, i1) = I3D(orr, m1k, m2k, i3, i2, i1)   + I3D(orr, m1k, m2k, i3+2, i2, i1)
                 + I3D(orr, m1k, m2k, i3, i2+2, i1) + I3D(orr, m1k, m2k, i3+2, i2+2, i1);
        }
      }
    }
    #pragma omp target teams loop collapse(3) map(present:orr[0:fine_total],os[0:coarse_total],x1[0:fine_total],y1[0:fine_total])
    for (j3 = 1; j3 < m3j-1; j3++) {
      i3 = 2*j3-d3;
      for (j2 = 1; j2 < m2j-1; j2++) {
        i2 = 2*j2-d2;
        for (j1 = 1; j1 < m1j-1; j1++) {
          i1 = 2*j1-d1;
          y2 = I3D(orr, m1k, m2k, i3, i2, i1+1)   + I3D(orr, m1k, m2k, i3+2, i2, i1+1)
             + I3D(orr, m1k, m2k, i3, i2+2, i1+1) + I3D(orr, m1k, m2k, i3+2, i2+2, i1+1);
          x2 = I3D(orr, m1k, m2k, i3+1, i2, i1+1) + I3D(orr, m1k, m2k, i3+1, i2+2, i1+1)
             + I3D(orr, m1k, m2k, i3, i2+1, i1+1) + I3D(orr, m1k, m2k, i3+2, i2+1, i1+1);
          I3D(os, m1j, m2j, j3, j2, j1) = 
                  0.5 * I3D(orr, m1k, m2k, i3+1, i2+1, i1+1)
                + 0.25 * (I3D(orr, m1k, m2k, i3+1, i2+1, i1) + I3D(orr, m1k, m2k, i3+1, i2+1, i1+2) + x2)
                + 0.125 * ( I3D(x1, m1k, m2k, i3, i2, i1)+ I3D(x1, m1k, m2k, i3, i2, i1+2) + y2)
                + 0.0625 * (I3D(y1, m1k, m2k, i3, i2, i1) + I3D(y1, m1k, m2k, i3, i2, i1+2));
        }
      }
    }
  }
  if (timeron) timer_stop(T_rprj3);

  j = k-1;
  
  comm3(os, m1j, m2j, m3j, j);

  if (debug_vec[0] >= 1) {
    
    rep_nrm(os, m1j, m2j, m3j, "   rprj3", k-1);
  }

  if (debug_vec[4] >= k) {
    
    showall(os, m1j, m2j, m3j);
  }
  free(x1);
  free(y1);
}

static void interp(double *oz, int mm1, int mm2, int mm3,
                   double *ou, int n1, int n2, int n3, int k)
{

  int i3, i2, i1, d1, d2, d3, t1, t2, t3;

  static double *z1, *z2, *z3;
  size_t fine_total = (size_t)mm3*mm2*mm1;
  size_t coarse_total = (size_t)n3*n2*n1;
  z1 = (double*)malloc(fine_total*sizeof(double));
  z2 = (double*)malloc(fine_total*sizeof(double));
  z3 = (double*)malloc(fine_total*sizeof(double));

  if (timeron) timer_start(T_interp);

  #pragma omp target data map(present: oz[0:fine_total], ou[0:coarse_total]) map(alloc: z1[0:fine_total], z2[0:fine_total], z3[0:fine_total])
  {

  if (n1 != 3 && n2 != 3 && n3 != 3) {

    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 0; i3 < mm3-1; i3++) {
      for (i2 = 0; i2 < mm2-1; i2++) {
        for (i1 = 0; i1 < mm1; i1++) {
          I3D(z1, mm1, mm2, i3, i2, i1) = I3D(oz, mm1, mm2, i3, i2+1, i1)   
		  								+ I3D(oz, mm1, mm2, i3, i2, i1);
		  I3D(z2, mm1, mm2, i3, i2, i1) = I3D(oz, mm1, mm2, i3+1, i2, i1)   
		  								+ I3D(oz, mm1, mm2, i3, i2, i1);
		  I3D(z3, mm1, mm2, i3, i2, i1) = I3D(oz, mm1, mm2, i3+1, i2+1, i1) 
		  								+ I3D(oz, mm1, mm2, i3+1, i2, i1)
										+ I3D(z1, mm1, mm2, i3, i2, i1);

        }
      }
    }
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 0; i3 < mm3-1; i3++) {
      for (i2 = 0; i2 < mm2-1; i2++) {
        for (i1 = 0; i1 < mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3, 2*i2, 2*i1)   = I3D(ou, n1, n2, 2*i3, 2*i2, 2*i1)
		  									  + I3D(oz, mm1, mm2, i3, i2, i1);
		  I3D(ou, n1, n2, 2*i3, 2*i2, 2*i1+1) = I3D(ou, n1, n2, 2*i3, 2*i2, 2*i1+1)
		  									  + 0.5*(I3D(oz, mm1, mm2, i3, i2, i1+1)
											       + I3D(oz, mm1, mm2, i3, i2, i1));
        }
      }
    }

    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 0; i3 < mm3-1; i3++) {
      for (i2 = 0; i2 < mm2-1; i2++) {
        for (i1 = 0; i1 < mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3, 2*i2+1, 2*i1) = I3D(ou, n1, n2, 2*i3, 2*i2+1, 2*i1)
											+ 0.5 * I3D(z1, mm1, mm2, i3, i2, i1);
		  I3D(ou, n1, n2, 2*i3, 2*i2+1, 2*i1+1) = I3D(ou, n1, n2, 2*i3, 2*i2+1, 2*i1+1)
		  										+ 0.25 * (I3D(z1, mm1, mm2, i3, i2, i1)
														+ I3D(z1, mm1, mm2, i3, i2, i1+1));
        }
      }
    }
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 0; i3 < mm3-1; i3++) {
      for (i2 = 0; i2 < mm2-1; i2++) {
        for (i1 = 0; i1 < mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3+1, 2*i2, 2*i1) = I3D(ou, n1, n2, 2*i3+1, 2*i2, 2*i1)
		  									  + 0.5 * I3D(z2, mm1, mm2, i3, i2, i1);
		  I3D(ou, n1, n2, 2*i3+1, 2*i2, 2*i1+1) = I3D(ou, n1, n2, 2*i3+1, 2*i2, 2*i1+1)
		  										+ 0.25 * (I3D(z2, mm1, mm2, i3, i2, i1)
														+ I3D(z2, mm1, mm2, i3, i2, i1+1));
         }
       }
     }
 
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 0; i3 < mm3-1; i3++) {
      for (i2 = 0; i2 < mm2-1; i2++) {
        for (i1 = 0; i1 < mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3+1, 2*i2+1, 2*i1) = I3D(ou, n1, n2, 2*i3+1, 2*i2+1, 2*i1)
		  										+ 0.25 * I3D(z3, mm1, mm2, i3, i2, i1);
		  I3D(ou, n1, n2, 2*i3+1, 2*i2+1, 2*i1+1) = I3D(ou, n1, n2, 2*i3+1, 2*i2+1, 2*i1+1)
		  										  + 0.125 * (I3D(z3, mm1, mm2, i3, i2, i1)
												  		   + I3D(z3, mm1, mm2, i3, i2, i1+1));
        }
      }
    }
  } else {
    if (n1 == 3) {
      d1 = 2;
      t1 = 1;
    } else {
      d1 = 1;
      t1 = 0;
    }

    if (n2 == 3) {
      d2 = 2;
      t2 = 1;
    } else {
      d2 = 1;
      t2 = 0;
    }

    if (n3 == 3) {
      d3 = 2;
      t3 = 1;
    } else {
      d3 = 1;
      t3 = 0;
    }

    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = d3; i3 <= mm3-1; i3++) {
      for (i2 = d2; i2 <= mm2-1; i2++) {
        for (i1 = d1; i1 <= mm1-1; i1++) {
		
		 I3D(ou, n1, n2, 2*i3-d3-1, 2*i2-d2-1, 2*i1-d1-1) = 
		 	I3D(ou, n1, n2, 2*i3-d3-1, 2*i2-d2-1, 2*i1-d1-1)
			+ I3D(ou, mm1, mm2, i3-1, i2-1, i1-1);
        }
       }
    }
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = d3; i3 <= mm3-1; i3++) {
      for (i2 = d2; i2 <= mm2-1; i2++) {
        for (i1 = 1; i1 <= mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3-d3-1, 2*i2-d2-1, 2*i1-t1-1) = 
		  	I3D(ou, n1, n2, 2*i3-d3-1, 2*i2-d2-1, 2*i1-t1-1)
			+ 0.5 * (I3D(oz, mm1, mm2, i3-1, i2-1, i1) 
				   + I3D(oz, mm1, mm2, i3-1, i2-1, i1-1));
        }
      }
    }
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = d3; i3 <= mm3-1; i3++) {
      for (i2 = 1; i2 <= mm2-1; i2++) {
        for (i1 = d1; i1 <= mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3-d3-1, 2*i2-t2-1, 2*i1-d1-1) = 
		    I3D(ou, n1, n2, 2*i3-d3-1, 2*i2-t2-1, 2*i1-d1-1)
			+ 0.5 * (I3D(oz, mm1, mm2, i3-1, i2,   i1-1)
				   + I3D(oz, mm1, mm2, i3-1, i2-1, i1-1));
        }
	  }
	 }
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = d3; i3 <= mm3-1; i3++) {
      for (i2 = 1; i2 <= mm2-1; i2++) {
        for (i1 = 1; i1 <= mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3-d3-1, 2*i2-t2-1, 2*i1-t1-1) = 
		    I3D(ou, n1, n2, 2*i3-d3-1, 2*i2-t2-1, 2*i1-t1-1)
			+ 0.25 * (I3D(oz, mm1, mm2, i3-1, i2,   i1)
			        + I3D(oz, mm1, mm2, i3-1, i2-1, i1)
					+ I3D(oz, mm1, mm2, i3-1, i2,   i1-1)
					+ I3D(oz, mm1, mm2, i3-1, i2-1, i1-1));
        }
      }
    }

    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 1; i3 <= mm3-1; i3++) {
      for (i2 = d2; i2 <= mm2-1; i2++) {
        for (i1 = d1; i1 <= mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3-t3-1, 2*i2-d2-1, 2*i1-d1-1) =
		    I3D(ou, n1, n2, 2*i3-t3-1, 2*i2-d2-1, 2*i1-d1-1)
			+ 0.5 * (I3D(oz, mm1, mm2, i3,   i2-1, i1-1)
			       + I3D(oz, mm1, mm2, i3-1, i2-1, i1-1));
        }
      }
    }
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 1; i3 <= mm3-1; i3++) {
      for (i2 = d2; i2 <= mm2-1; i2++) {
        for (i1 = 1; i1 <= mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3-t3-1, 2*i2-d2-1, 2*i1-t1-1) =
		    I3D(ou, n1, n2, 2*i3-t3-1, 2*i2-d2-1, 2*i1-t1-1)
			+ 0.25 * (I3D(oz, mm1, mm2, i3,   i2-1, i1)
			        + I3D(oz, mm1, mm2, i3,   i2-1, i1-1)
					+ I3D(oz, mm1, mm2, i3-1, i2-1, i1)
					+ I3D(oz, mm1, mm2, i3-1, i2-1, i1-1));
        }
      }
    }
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 1; i3 <= mm3-1; i3++) {
      for (i2 = 1; i2 <= mm2-1; i2++) {
        for (i1 = d1; i1 <= mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3-t3-1, 2*i2-t2-1, 2*i1-d1-1) = 
		    I3D(ou, n1, n2, 2*i3-t3-1, 2*i2-t2-1, 2*i1-d1-1)
			+ 0.25 * (I3D(oz, mm1, mm2, i3,   i2,   i1-1)
			        + I3D(oz, mm1, mm2, i3,   i2-1, i1-1)
					+ I3D(oz, mm1, mm2, i3-1, i2,   i1-1)
					+ I3D(oz, mm1, mm2, i3-1, i2-1, i1-1));
        }
       }
     }
    #pragma omp target teams loop collapse(3) map(present:oz[0:fine_total],ou[0:coarse_total],z1[0:fine_total],z2[0:fine_total],z3[0:fine_total])
    for (i3 = 1; i3 <= mm3-1; i3++) {
      for (i2 = 1; i2 <= mm2-1; i2++) {
        for (i1 = 1; i1 <= mm1-1; i1++) {
		
		  I3D(ou, n1, n2, 2*i3-t3-1, 2*i2-t2-1, 2*i1-t1-1) = 
		    I3D(ou, n1, n2, 2*i3-t3-1, 2*i2-t2-1, 2*i1-t1-1)
			+ 0.125 * (I3D(oz, mm1, mm2, i3,   i2,   i1)
			         + I3D(oz, mm1, mm2, i3,   i2-1, i1)
					 + I3D(oz, mm1, mm2, i3,   i2,   i1-1)
					 + I3D(oz, mm1, mm2, i3,   i2-1, i1-1)
					 + I3D(oz, mm1, mm2, i3-1, i2,   i1)
					 + I3D(oz, mm1, mm2, i3-1, i2-1, i1)
					 + I3D(oz, mm1, mm2, i3-1, i2,   i1-1)
					 + I3D(oz, mm1, mm2, i3-1, i2-1, i1-1));
        }
      }
    }

  }
  }

  free(z1);
  free(z2);
  free(z3);
  if (timeron) timer_stop(T_interp);

  if (debug_vec[0] >= 1) {

    rep_nrm(oz, mm1, mm2, mm3, "z: inter", k-1);
    rep_nrm(ou, n1, n2, n3, "u: inter", k);
  }

  if (debug_vec[5] >= k) {

    showall(oz, mm1, mm2, mm3);
    showall(ou, n1, n2, n3);
  }
}

static void norm2u3(double *orr, int n1, int n2, int n3,
                    double *rnm2, double *rnmu,
                    int nx, int ny, int nz)
{

  double s, a;
  double temp;
  int i3, i2, i1;

  double dn;

  if (timeron) timer_start(T_norm2);
  dn = 1.0*nx*ny*nz;

  s = 0.0;
  *rnmu = 0.0;
  temp = 0.0;
  {
    size_t total = (size_t)n3*n2*n1;
    #pragma omp target teams loop collapse(3) reduction(+:s) reduction(max:temp) map(present: orr[0:total])
    for (i3 = 1; i3 < n3-1; i3++) {
      for (i2 = 1; i2 < n2-1; i2++) {
        for (i1 = 1; i1 < n1-1; i1++) {
          double val = I3D(orr, n1, n2, i3, i2, i1);
          s = s + val * val;
          a = fabs(val);
          
          temp = max(temp, a);
        }
      }
    }
  }
  *rnmu = temp;
  *rnm2 = sqrt(s / dn);
  if (timeron) timer_stop(T_norm2);
}

static void rep_nrm(double *u, int n1, int n2, int n3, char *title, int kk)
{
  double rnm2, rnmu;

  norm2u3(u, n1, n2, n3, &rnm2, &rnmu, nx[kk], ny[kk], nz[kk]);
  printf(" Level%2d in %8s: norms =%21.14E%21.14E\n", kk, title, rnm2, rnmu);
}

static void comm3(double *ou, int n1, int n2, int n3, int kk)
{

  int i1, i2, i3;
  size_t total = (size_t)n3*n2*n1;
  if (timeron) timer_start(T_comm3);
  #pragma omp target teams loop collapse(2) map(present: ou[0:total])
  for (i3 = 1; i3 < n3-1; i3++) {
    for (i2 = 1; i2 < n2-1; i2++) {
	
	  I3D(ou, n1, n2, i3, i2, 0)    = I3D(ou, n1, n2, i3, i2, n1-2);
	  I3D(ou, n1, n2, i3, i2, n1-1) = I3D(ou, n1, n2, i3, i2, 1);
    }
  }

  #pragma omp target teams loop collapse(2) map(present: ou[0:total])
  for (i3 = 1; i3 < n3-1; i3++) {
    for (i1 = 0; i1 < n1; i1++) {
	
	 I3D(ou, n1, n2, i3, 0, i1)    = I3D(ou, n1, n2, i3, n2-2, i1);
	 I3D(ou, n1, n2, i3, n2-1, i1) = I3D(ou, n1, n2, i3, 1, i1);
    }
  }

  #pragma omp target teams loop collapse(2) map(present: ou[0:total])
  for (i2 = 0; i2 < n2; i2++) {
    for (i1 = 0; i1 < n1; i1++) {
	
	  I3D(ou, n1, n2, 0, i2, i1) = I3D(ou, n1, n2, n3-2, i2, i1);
	  I3D(ou, n1, n2, n3-1, i2, i1) = I3D(ou, n1, n2, 1, i2, i1);
    }
  }
  if (timeron) timer_stop(T_comm3);
}

inline double randlc( double *x, double a )
{
  const double r23 = 1.1920928955078125e-07;
  const double r46 = r23 * r23;
  const double t23 = 8.388608e+06;
  const double t46 = t23 * t23;

  double t1, t2, t3, t4, a1, a2, x1, x2, z;
  double r;
  t1 = r23 * a;
  a1 = (int) t1;
  a2 = a - t23 * a1;
  
  t1 = r23 * (*x);
  x1 = (int) t1;
  x2 = *x - t23 * x1;
  t1 = a1 * x2 + a2 * x1;
  t2 = (int) (r23 * t1);
  z = t1 - t23 * t2;
  t3 = t23 * z + a2 * x2;
  t4 = (int) (r46 * t3);
  *x = t3 - t46 * t4;
  r = r46 * (*x);
  
  return r;
}

inline void vranlc( int n, double *x, double a, double y[] )
{
	
  const double r23 = 1.1920928955078125e-07;
  const double r46 = r23 * r23;
  const double t23 = 8.388608e+06;
  const double t46 = t23 * t23;

  double t1, t2, t3, t4, a1, a2, x1, x2, z;

  int i;
  t1 = r23 * a;
  a1 = (int) t1;
  a2 = a - t23 * a1;
  for ( i = 0; i < n; i++ ) {
    t1 = r23 * (*x);
    x1 = (int) t1;
    x2 = *x - t23 * x1;
    t1 = a1 * x2 + a2 * x1;
    t2 = (int) (r23 * t1);
    z = t1 - t23 * t2;
    t3 = t23 * z + a2 * x2;
    t4 = (int) (r46 * t3) ;
    *x = t3 - t46 * t4;
    y[i] = r46 * (*x);
  }
}

static void zran3(double *oz, int n1, int n2, int n3, int nx, int ny, int k)
{

  int i0, m0, m1;

  int i1, i2, i3, d1, e1, e2, e3;
  double xx, x0, x1, a1, a2, ai;

  const int mm = 10;
  const double a = pow(5.0, 13.0);
  const double x = 314159265.0;
  double ten[mm][2], best;
  int i, j1[mm][2], j2[mm][2], j3[mm][2];
  int jg[4][mm][2];

  double rdummy;

  a1 = power(a, nx);
  a2 = power(a, nx*ny);

  zero3(oz, n1, n2, n3);
  i = is1-2+nx*(is2-2+ny*(is3-2));

  ai = power(a, i);
  d1 = ie1 - is1 + 1;
  e1 = ie1 - is1 + 2;
  e2 = ie2 - is2 + 2;
  e3 = ie3 - is3 + 2;
  x0 = x;
  rdummy = randlc(&x0, ai);

  for (i3 = 1; i3 < e3; i3++) {
    x1 = x0;
    for (i2 = 1; i2 < e2; i2++) {
      xx = x1;
      
      vranlc(d1, &xx, a, &(I3D(oz, n1, n2, i3, i2, 1)));
      rdummy = randlc(&x1,a1);
    }
    rdummy = randlc(&x0, a2);
  }

  for (i = 0; i < mm; i++) {
    ten[i][1] = 0.0;
    j1[i][1] = 0;
    j2[i][1] = 0;
    j3[i][1] = 0;
    ten[i][0] = 1.0;
    j1[i][0] = 0;
    j2[i][0] = 0;
    j3[i][0] = 0;
  }
  
  for (i3 = 1; i3 < n3-1; i3++) {
    for (i2 = 1; i2 < n2-1; i2++) {
      for (i1 = 1; i1 < n1-1; i1++) {
        
		if(I3D(oz, n1, n2, i3, i2, i1) > ten[0][1]){
          
          ten[0][1] = I3D(oz, n1, n2, i3, i2, i1);
          j1[0][1] = i1;
          j2[0][1] = i2;
          j3[0][1] = i3;
          bubble(ten, j1, j2, j3, mm, 1);
        }
        
        if (I3D(oz, n1, n2, i3, i2, i1) < ten[0][0]) {
          
          ten[0][0] = I3D(oz, n1, n2, i3, i2, i1);
          j1[0][0] = i1;
          j2[0][0] = i2;
          j3[0][0] = i3;
          bubble(ten, j1, j2, j3, mm, 0);
        }
      }
    }
  }

  i1 = mm - 1;
  i0 = mm - 1; 
  for (i = mm - 1; i >= 0; i--) {
    best = 0.0;
    if (best < ten[i1][1]) {
      jg[0][i][1] = 0;
      jg[1][i][1] = is1 - 2 + j1[i1][1];
      jg[2][i][1] = is2 - 2 + j2[i1][1];
      jg[3][i][1] = is3 - 2 + j3[i1][1];
      i1 = i1-1;
    } else {
      jg[0][i][1] = 0;
      jg[1][i][1] = 0;
      jg[2][i][1] = 0;
      jg[3][i][1] = 0;
    }

    best = 1.0;
    if (best > ten[i0][0]) {
      jg[0][i][0] = 0;
      jg[1][i][0] = is1 - 2 + j1[i0][0];
      jg[2][i][0] = is2 - 2 + j2[i0][0];
      jg[3][i][0] = is3 - 2 + j3[i0][0];
      i0 = i0-1;
    } else {
      jg[0][i][0] = 0;
      jg[1][i][0] = 0;
      jg[2][i][0] = 0;
      jg[3][i][0] = 0;
    }

  }

  m1 = 0;
  m0 = 0;

  for (i3 = 0; i3 < n3; i3++) {
    for (i2 = 0; i2 < n2; i2++) {
      for (i1 = 0; i1 < n1; i1++) {
        
        I3D(oz, n1, n2, i3, i2, i1) = 0.0;
      }
    }
  }
  
  for (i = mm-1; i >= m0; i--) {
    
	i3 = jg[3][i][0];
	i2 = jg[2][i][0];
	i1 = jg[1][i][0];
    I3D(oz, n1, n2, i3, i2, i1) = -1.0;
  }

  for (i = mm-1; i >= m1; i--) {
    
	i3 = jg[3][i][1];
	i2 = jg[2][i][1];
	i1 = jg[1][i][1];
    I3D(oz, n1, n2, i3, i2, i1) = +1.0;
  }

  comm3(oz, n1, n2, n3, k);

}

static void showall(double *oz, int n1, int n2, int n3)
{

  int i1, i2, i3;
  int m1, m2, m3;

  m1 = min(n1, 18);
  m2 = min(n2, 14);
  m3 = min(n3, 18);

  printf("   \n");
  for (i3 = 0; i3 < m3; i3++) {
    for (i1 = 0; i1 < m1; i1++) {
      for (i2 = 0; i2 < m2; i2++) {
    
		  printf("%6.3f", I3D(oz, n1, n2, i3, i2, i2));
      }
      printf("\n");
    }
    printf("  - - - - - - - \n");
  }
  printf("   \n");
}

static double power(double a, int n)
{
  double aj;
  int nj;
  double rdummy;
  double power;

  power = 1.0;
  nj = n;
  aj = a;

  while (nj != 0) {
    if ((nj % 2) == 1) rdummy = randlc(&power, aj);
    rdummy = randlc(&aj, aj);
    nj = nj/2;
  }

  return power;
}

static void bubble(double ten[][2], int j1[][2], int j2[][2], int j3[][2],
                   int m, int ind)
{
  double temp;
  int i, j_temp;

  if (ind == 1) {
    for (i = 0; i < m-1; i++) {
      if (ten[i][ind] > ten[i+1][ind]) {
        temp = ten[i+1][ind];
        ten[i+1][ind] = ten[i][ind];
        ten[i][ind] = temp;

        j_temp = j1[i+1][ind];
        j1[i+1][ind] = j1[i][ind];
        j1[i][ind] = j_temp;

        j_temp = j2[i+1][ind];
        j2[i+1][ind] = j2[i][ind];
        j2[i][ind] = j_temp;

        j_temp = j3[i+1][ind];
        j3[i+1][ind] = j3[i][ind];
        j3[i][ind] = j_temp;
      } else {
        return;
      }
    }
  } else {
    for (i = 0; i < m-1; i++) {
      if (ten[i][ind] < ten[i+1][ind]) {

        temp = ten[i+1][ind];
        ten[i+1][ind] = ten[i][ind];
        ten[i][ind] = temp;

        j_temp = j1[i+1][ind];
        j1[i+1][ind] = j1[i][ind];
        j1[i][ind] = j_temp;

        j_temp = j2[i+1][ind];
        j2[i+1][ind] = j2[i][ind];
        j2[i][ind] = j_temp;

        j_temp = j3[i+1][ind];
        j3[i+1][ind] = j3[i][ind];
        j3[i][ind] = j_temp;
      } else {
        return;
      }
    }
  }
}

static void zero3(double *oz, int n1, int n2, int n3)
{

  int i1, i2, i3;
  size_t total = (size_t)n3*n2*n1;

  #pragma omp target teams loop collapse(3) map(present: oz[0:total])
  for (i3 = 0; i3 < n3; i3++) {
    for (i2 = 0; i2 < n2; i2++) {
      for (i1 = 0; i1 < n1; i1++) {
        I3D(oz, n1, n2, i3, i2, i1) = 0.0;
      }
    }
  }
}
