

#ifdef __PGIC__
#undef __GNUC__
#else
#define num_workers(a)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "randdp.h"
#include "type.h"
#include "npbparams.h"
#include "timers.h"
#include "print_results.h"

#define MAX(X,Y)  (((X) > (Y)) ? (X) : (Y))

#pragma omp declare target
double r23;
double r46;
double t23;
double t46;

double randlc_ep( double *x, double a )
{

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
#pragma omp end declare target

int MK;
int MM;
int NN;
double EPSILON;
double A;
double S;
int NK;
int NQ;

int BLKSIZE;

int main() 
{
  double Mops, t1, t2, t3, t4, x1, x2;
  double sx, sy, tm, an, tt, gc;
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  int    np;
  int    i, ik, kk, l, k, nit;
  int    k_offset, j;
  int verified, timers_enabled;
  double q0, q1, q2, q3, q4, q5, q6, q7, q8, q9;
    
    MK =  16;
    MM =  (M - MK);
    NN =       (1 << MM);
    EPSILON =  1.0e-8;
    A =        1220703125.0;
    S =        271828183.0;
    NK = 1 << MK;
    NQ = 10;

    BLKSIZE = 2048;

    r23 = 1.1920928955078125e-07;
    r46 = r23 * r23;
    t23 = 8.388608e+06;
    t46 = t23 * t23;

    #pragma omp target update to(r23, r46, t23, t46)

  double q[10]; 
  double *qq;
  double dum[3] = {1.0, 1.0, 1.0};
  char   size[16];

  int blksize = BLKSIZE;
  int blk, koff, numblks;

  FILE *fp;

  if ((fp = fopen("timer.flag", "r")) == NULL) {
    timers_enabled = 0;
  } else {
    timers_enabled = 1;
    fclose(fp);
  }

  if (NN < blksize) {
     blksize = NN;
  }
  numblks = ceil( (double)NN / (double) blksize);

  qq = (double*)malloc(blksize*NQ*sizeof(double));

  sprintf(size, "%15.0lf", pow(2.0, M+1));
  j = 14;
  if (size[j] == '.') j--;
  size[j+1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n", size);

  verified = 0;

  np = NN; 
printf("NK=%d NN=%d NQ=%d BLKS=%d NBLKS=%d\n",NK,NN,NQ,blksize,numblks);

{
  vranlc(0, &dum[0], dum[1], &dum[2]);
  dum[0] = randlc_ep(&dum[1], dum[2]);

  for (i = 0; i < NQ; i++) {
    q[i] = 0.0;
  }
  Mops = log(sqrt(fabs(MAX(1.0, 1.0))));   

  timer_clear(0);
  timer_clear(1);
  timer_clear(2);
  timer_start(0);

  t1 = A;

  for (i = 0; i < MK + 1; i++) {
    t2 = randlc_ep(&t1, t1);
  }

  an = t1;
  tt = S;
  gc = 0.0;
  sx = 0.0;
  sy = 0.0;
  k_offset = -1;

  for (blk = 0; blk < numblks; ++blk) {

    koff = blk * blksize;
    int cur_blksize = blksize;
    if (koff + cur_blksize > np) {
      cur_blksize = np - koff;
    }
    if (cur_blksize <= 0) break;

    int kk_base = k_offset + koff;
    double block_sx = 0.0;
    double block_sy = 0.0;
    double target_A = A;
    double target_S = S;
    double target_an = an;
    int local_NK = NK;
    int local_NQ = NQ;
    int block_stride = cur_blksize;

#pragma omp target teams map(tofrom: qq[0:block_stride*local_NQ]) map(tofrom: block_sx, block_sy) \
                         map(to: block_stride, kk_base, local_NK, local_NQ, target_A, target_S, target_an)
    {
#pragma omp loop
      for (int idx = 0; idx < block_stride * local_NQ; ++idx) {
        qq[idx] = 0.0;
      }

#pragma omp loop reduction(+:block_sx, block_sy)
      for (int k_iter = 1; k_iter <= block_stride; ++k_iter) {
        int kk_local = kk_base + k_iter;
        double t1_local = target_S;
        double t2_local = target_an;
        int stride = k_iter - 1;

        for (int iter = 1; iter <= 100; ++iter) {
          int ik_local = kk_local / 2;
          if ((2 * ik_local) != kk_local) {
            double tmp_t1 = r23 * t2_local;
            double tmp_a1 = (int)tmp_t1;
            double tmp_a2 = t2_local - t23 * tmp_a1;

            double tmp_t2 = r23 * t1_local;
            double tmp_x1 = (int)tmp_t2;
            double tmp_x2 = t1_local - t23 * tmp_x1;
            double tmp_t3 = tmp_a1 * tmp_x2 + tmp_a2 * tmp_x1;
            double tmp_t4 = (int)(r23 * tmp_t3);
            double tmp_z = tmp_t3 - t23 * tmp_t4;
            double tmp_t5 = t23 * tmp_z + tmp_a2 * tmp_x2;
            double tmp_t6 = (int)(r46 * tmp_t5);
            t1_local = tmp_t5 - t46 * tmp_t6;
          }
          if (ik_local == 0) break;

          double tmp_t7 = r23 * t2_local;
          double tmp_a3 = (int)tmp_t7;
          double tmp_a4 = t2_local - t23 * tmp_a3;

          double tmp_t8 = r23 * t2_local;
          double tmp_x3 = (int)tmp_t8;
          double tmp_x4 = t2_local - t23 * tmp_x3;
          double tmp_t9 = tmp_a3 * tmp_x4 + tmp_a4 * tmp_x3;
          double tmp_t10 = (int)(r23 * tmp_t9);
          double tmp_z2 = tmp_t9 - t23 * tmp_t10;
          double tmp_t11 = t23 * tmp_z2 + tmp_a4 * tmp_x4;
          double tmp_t12 = (int)(r46 * tmp_t11);
          t2_local = tmp_t11 - t46 * tmp_t12;
          kk_local = ik_local;
        }

        double tmp_sx_local = 0.0;
        double tmp_sy_local = 0.0;
        for (int idx = 0; idx < local_NK; ++idx) {
          double x1_val = 2.0 * randlc_ep(&t1_local, target_A) - 1.0;
          double x2_val = 2.0 * randlc_ep(&t1_local, target_A) - 1.0;
          double t_val = x1_val * x1_val + x2_val * x2_val;
          if (t_val <= 1.0) {
            double scale = sqrt(-2.0 * log(t_val) / t_val);
            double t3_val = x1_val * scale;
            double t4_val = x2_val * scale;
            int l_val = MAX(fabs(t3_val), fabs(t4_val));
            qq[l_val * block_stride + stride] += 1.0;
            tmp_sx_local += t3_val;
            tmp_sy_local += t4_val;
          }
        }

        block_sx += tmp_sx_local;
        block_sy += tmp_sy_local;
      }
    }

    sx += block_sx;
    sy += block_sy;

    for (i = 0; i < NQ; i++)
    {
      double sum_qi = 0.0;
      for (k = 0; k < cur_blksize; k++)
        sum_qi += qq[i * cur_blksize + k];

      q[i] += sum_qi;

      gc += sum_qi;
    }
  }
}

  {
    double gate_ep_sums[2] = { sx, sy };
  }

  timer_stop(0);
  tm = timer_read(0);

  nit = 0;
  verified = 1;
  if (M == 24) {
    sx_verify_value = -3.247834652034740e+3;
    sy_verify_value = -6.958407078382297e+3;
  } else if (M == 25) {
    sx_verify_value = -2.863319731645753e+3;
    sy_verify_value = -6.320053679109499e+3;
  } else if (M == 28) {
    sx_verify_value = -4.295875165629892e+3;
    sy_verify_value = -1.580732573678431e+4;
  } else if (M == 30) {
    sx_verify_value =  4.033815542441498e+4;
    sy_verify_value = -2.660669192809235e+4;
  } else if (M == 32) {
    sx_verify_value =  4.764367927995374e+4;
    sy_verify_value = -8.084072988043731e+4;
  } else if (M == 36) {
    sx_verify_value =  1.982481200946593e+5;
    sy_verify_value = -1.020596636361769e+5;
  } else if (M == 40) {
    sx_verify_value = -5.319717441530e+05;
    sy_verify_value = -3.688834557731e+05;
  } else {
    verified = 0;
  }

  if (verified) {
    sx_err = fabs((sx - sx_verify_value) / sx_verify_value);
    sy_err = fabs((sy - sy_verify_value) / sy_verify_value);
    verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
  }

  Mops = pow(2.0, M+1) / tm / 1000000.0;

  printf("\nEP Benchmark Results:\n\n");
  printf("CPU Time =%10.4lf\n", tm);
  printf("N = 2^%5d\n", M);
  printf("No. Gaussian Pairs = %15.0lf\n", gc);
  printf("Sums = %25.15lE %25.15lE\n", sx, sy);
  printf("Counts: \n");
  for (i = 0; i < NQ; i++) {
    printf("%3d%15.0lf\n", i, q[i]);
  }

  print_results("EP", CLASS, M+1, 0, 0, nit,
      tm, Mops, 
      "Random numbers generated",
      verified, NPBVERSION, COMPILETIME, CS1,
      CS2, CS3, CS4, CS5, CS6, CS7);

  if (timers_enabled) {
    if (tm <= 0.0) tm = 1.0;
    tt = timer_read(0);
    printf("\nTotal time:     %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
    tt = timer_read(1);
    printf("Gaussian pairs: %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
    tt = timer_read(2);
    printf("Random numbers: %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
  }

	free(qq);

  return 0;
}
