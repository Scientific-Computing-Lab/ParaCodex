

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
#define HIST_BINS 10

#pragma omp declare target
int MK;
int MM;
int NN;
double EPSILON;
double A;
double S;
int NK;
int NQ;

double r23;
double r46;
double t23;
double t46;
#pragma omp end declare target

/*
 * Place randlc_ep next to the RNG constants on the device so the kernel can reuse
 * the RNG state without remapping.
 */
#pragma omp declare target
/* Keep this helper inline so the device sees the RNG math without indirect calls. */
static inline double randlc_ep( double *x, double a )

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


int main() 
{
  double Mops, t1, t2, t3, t4, x1, x2;
  double sx, sy, tm, an, tt, gc;
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  int    np;
  int    i, j, nit;
  int    verified, timers_enabled;
  double q[HIST_BINS];
  double dum[3] = {1.0, 1.0, 1.0};
  char   size[16];
  FILE  *fp;

  MK =  16;
  MM =  (M - MK);
  NN =       (1 << MM);
  EPSILON =  1.0e-8;
  A =        1220703125.0;
  S =        271828183.0;
  NK = 1 << MK;
  NQ = HIST_BINS;

  r23 = 1.1920928955078125e-07;
  r46 = r23 * r23;
  t23 = 8.388608e+06;
  t46 = t23 * t23;

  if ((fp = fopen("timer.flag", "r")) == NULL) {
    timers_enabled = 0;
  } else {
    timers_enabled = 1;
    fclose(fp);
  }

  sprintf(size, "%15.0lf", pow(2.0, M+1));
  j = 14;
  if (size[j] == '.') j--;
  size[j+1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n", size);

  verified = 0;

  np = NN;
  printf("NK=%d NN=%d NQ=%d\n", NK, NN, NQ);

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

    const int samples = np;
    const int pairs = NK;
    const int bins = NQ;
    const double a = A;
    const double s_seed = S;
    const double an_seed = an;

    /* Each thread keeps a tiny histogram (< 1KB) and merges via atomic updates. */
    #pragma omp target data map(tofrom: q[0:bins], sx, sy)
    {
      #pragma omp target teams loop reduction(+:sx, sy) firstprivate(samples, pairs, bins, a, s_seed, an_seed)
      for (int kk = 0; kk < samples; ++kk) {
        double t1_state = s_seed;
        double t2_state = an_seed;
        int ktemp = kk;

        while (1) {
          int ik = ktemp / 2;
          if ((ik << 1) != ktemp) {
            (void)randlc_ep(&t1_state, t2_state);
          }
          if (ik == 0) break;
          (void)randlc_ep(&t2_state, t2_state);
          ktemp = ik;
        }

        double local_hist[HIST_BINS];
        for (int bin = 0; bin < bins; ++bin) {
          local_hist[bin] = 0.0;
        }

        double sample_sx = 0.0;
        double sample_sy = 0.0;

        for (int pair = 0; pair < pairs; ++pair) {
          double x1_loc = 2.0 * randlc_ep(&t1_state, a) - 1.0;
          double x2_loc = 2.0 * randlc_ep(&t1_state, a) - 1.0;
          double t1_loc = x1_loc * x1_loc + x2_loc * x2_loc;
          if (t1_loc <= 1.0) {
            double t2_loc = sqrt(-2.0 * log(t1_loc) / t1_loc);
            double t3_loc = x1_loc * t2_loc;
            double t4_loc = x2_loc * t2_loc;
            double t3_abs = t3_loc < 0.0 ? -t3_loc : t3_loc;
            double t4_abs = t4_loc < 0.0 ? -t4_loc : t4_loc;
            int abs_t3 = (int)t3_abs;
            int abs_t4 = (int)t4_abs;
            int bin = abs_t3 > abs_t4 ? abs_t3 : abs_t4;
            if (bin < bins) {
              local_hist[bin] += 1.0;
            }
            sample_sx += t3_loc;
            sample_sy += t4_loc;
          }
        }

        for (int bin = 0; bin < bins; ++bin) {
          double incr = local_hist[bin];
          if (incr != 0.0) {
            #pragma omp atomic update
            q[bin] += incr;
          }
        }

        sx += sample_sx;
        sy += sample_sy;
      }
    }
  }

  {
    double gate_ep_sums[2] = { sx, sy };
  }

  timer_stop(0);
  tm = timer_read(0);

  gc = 0.0;
  for (i = 0; i < NQ; i++) {
    gc += q[i];
  }
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
