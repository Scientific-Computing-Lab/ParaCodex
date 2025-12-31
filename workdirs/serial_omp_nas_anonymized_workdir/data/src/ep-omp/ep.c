#ifdef __PGIC__
#undef __GNUC__
#else
#define num_workers(a)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "randdp.h"
#include "type.h"
#include "npbparams.h"
#include "timers.h"
#include "print_results.h"

#define MAX(X,Y)  (((X) > (Y)) ? (X) : (Y))

int MK;
int MM;
int NN;
double EPSILON;
double A;
double S;
int NK;
int NQ;

int BLKSIZE;

double r23;
double r46;
double t23;
double t46;

double randlc_ep( double *__restrict x, double a )
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

int main() 
{
  double Mops, t1, t2, t3, t4, x1, x2;
  double sx, sy, tm, an, tt, gc;
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
int    i, nit;
int    j;
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

  double q[10]; 
  
  double dum[3] = {1.0, 1.0, 1.0};
  char   size[16];
    

  int blksize = BLKSIZE;
  int numblks;

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

  sprintf(size, "%15.0lf", pow(2.0, M+1));
  j = 14;
  if (*((size) + (j)) == '.') j--;
  size[j+1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n", size);

  verified = 0;

  printf("NK=%d NN=%d NQ=%d BLKS=%d NBLKS=%d\n",NK,NN,NQ,blksize,numblks);

  vranlc(0, &dum[0], dum[1], &dum[2]);
  dum[0] = randlc_ep(&dum[1], dum[2]);

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
  const int bins = NQ;
  const int hist_bins = 10;
  const double r23_loc = r23;
  const double r46_loc = r46;
  const double t23_loc = t23;
  const double t46_loc = t46;
  const double A_loc = A;
  const double S_loc = S;
  const double an_loc = an;
  const int NK_loc = NK;
  const int NN_loc = NN;

  int kk;
  // Flattened NN-wide sample loop: each device thread replays the RNG, builds a local histogram, and atomically merges into q
  #pragma omp target data map(to: r23_loc, r46_loc, t23_loc, t46_loc, A_loc, S_loc, an_loc, NK_loc, NN_loc, bins) map(tofrom: q[:NQ], sx, sy, gc)
  {
    #pragma omp target teams loop reduction(+:sx, sy, gc) firstprivate(r23_loc, r46_loc, t23_loc, t46_loc, A_loc, S_loc, an_loc, NK_loc, NN_loc, bins)
    for (kk = 0; kk < NN_loc; ++kk) {
      double local_hist[hist_bins];
      #pragma omp simd
      for (int bin = 0; bin < bins; ++bin) {
        local_hist[bin] = 0.0;
      }

      double tmp_sx_sample = 0.0;
      double tmp_sy_sample = 0.0;
      double sample_count = 0.0;
      double t1_sample = S_loc;
      double t2_sample = an_loc;
      double in_t1_loc = 0.0;
      double in_t2_loc = 0.0;
      double in_t3_loc = 0.0;
      double in_t4_loc = 0.0;
      double in_x1_loc = 0.0;
      double in_x2_loc = 0.0;
      double in_z_loc = 0.0;
      double in_a1_loc = 0.0;
      double in_a2_loc = 0.0;
      int kk_local = kk;
      int ik;

      for (int iter = 1; iter <= 100; ++iter) {
        ik = kk_local / 2;
        if ((2 * ik) != kk_local) {
          in_t1_loc = r23_loc * t2_sample;
          in_a1_loc = (int) in_t1_loc;
          in_a2_loc = t2_sample - t23_loc * in_a1_loc;

          in_t1_loc = r23_loc * t1_sample;
          in_x1_loc = (int) in_t1_loc;
          in_x2_loc = t1_sample - t23_loc * in_x1_loc;
          in_t1_loc = in_a1_loc * in_x2_loc + in_a2_loc * in_x1_loc;
          in_t2_loc = (int)(r23_loc * in_t1_loc);
          in_z_loc = in_t1_loc - t23_loc * in_t2_loc;
          in_t3_loc = t23_loc * in_z_loc + in_a2_loc * in_x2_loc;
          in_t4_loc = (int)(r46_loc * in_t3_loc);
          t1_sample = in_t3_loc - t46_loc * in_t4_loc;
        }
        if (ik == 0) break;

        in_t1_loc = r23_loc * t2_sample;
        in_a1_loc = (int) in_t1_loc;
        in_a2_loc = t2_sample - t23_loc * in_a1_loc;

        in_t1_loc = r23_loc * t2_sample;
        in_x1_loc = (int) in_t1_loc;
        in_x2_loc = t2_sample - t23_loc * in_x1_loc;
        in_t1_loc = in_a1_loc * in_x2_loc + in_a2_loc * in_x1_loc;
        in_t2_loc = (int)(r23_loc * in_t1_loc);
        in_z_loc = in_t1_loc - t23_loc * in_t2_loc;
        in_t3_loc = t23_loc * in_z_loc + in_a2_loc * in_x2_loc;
        in_t4_loc = (int)(r46_loc * in_t3_loc);
        t2_sample = in_t3_loc - t46_loc * in_t4_loc;
        kk_local = ik;
      }

      in_t1_loc = r23_loc * A_loc;
      in_a1_loc = (int) in_t1_loc;
      in_a2_loc = A_loc - t23_loc * in_a1_loc;
      double state = t1_sample;

      for (int i = 0; i < NK_loc; ++i) {
        in_t1_loc = r23_loc * state;
        in_x1_loc = (int) in_t1_loc;
        in_x2_loc = state - t23_loc * in_x1_loc;
        in_t1_loc = in_a1_loc * in_x2_loc + in_a2_loc * in_x1_loc;
        in_t2_loc = (int)(r23_loc * in_t1_loc);
        in_z_loc = in_t1_loc - t23_loc * in_t2_loc;
        in_t3_loc = t23_loc * in_z_loc + in_a2_loc * in_x2_loc;
        in_t4_loc = (int)(r46_loc * in_t3_loc);
        state = in_t3_loc - t46_loc * in_t4_loc;
        double rand1 = r46_loc * state;

        in_t1_loc = r23_loc * state;
        in_x1_loc = (int) in_t1_loc;
        in_x2_loc = state - t23_loc * in_x1_loc;
        in_t1_loc = in_a1_loc * in_x2_loc + in_a2_loc * in_x1_loc;
        in_t2_loc = (int)(r23_loc * in_t1_loc);
        in_z_loc = in_t1_loc - t23_loc * in_t2_loc;
        in_t3_loc = t23_loc * in_z_loc + in_a2_loc * in_x2_loc;
        in_t4_loc = (int)(r46_loc * in_t3_loc);
        state = in_t3_loc - t46_loc * in_t4_loc;
        double rand2 = r46_loc * state;

        double x1_val = 2.0 * rand1 - 1.0;
        double x2_val = 2.0 * rand2 - 1.0;
        double sum_sq = x1_val * x1_val + x2_val * x2_val;
        if (sum_sq <= 1.0) {
          double norm = sqrt(-2.0 * log(sum_sq) / sum_sq);
          double t3_val = x1_val * norm;
          double t4_val = x2_val * norm;
          int l = MAX(fabs(t3_val), fabs(t4_val));
          local_hist[l] += 1.0;
          tmp_sx_sample += t3_val;
          tmp_sy_sample += t4_val;
          sample_count += 1.0;
        }
      }

      for (int bin = 0; bin < bins; ++bin) {
        double bin_val = local_hist[bin];
        if (bin_val != 0.0) {
          #pragma omp atomic update
          q[bin] += bin_val;
        }
      }

      sx += tmp_sx_sample;
      sy += tmp_sy_sample;
      gc += sample_count;
    }
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

  return 0;
}
