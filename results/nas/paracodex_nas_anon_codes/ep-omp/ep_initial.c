

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

int v1;
int v2;
int v3;
double v4;
double v5;
double v6;
int v7;
int v8;

int v9;

double v10;
double v11;
double v12;
double v13;
double f1( double *v14, double v15 );

int main() 
{
  double v26, v16, v17;
  double v27, v28, v29, v30, v31, v32;
  double v33, v34, v35, v36;
  int    v38, v43;
  int    v45;
  int v75, v78;
  int v46, v47;
  double v48, v49, v50, v51, v52, v53, v54, v55, v56, v57;
    
    v1 =  16;
    v2 =  (M - v1);
    v3 =       (1 << v2);
    v4 =  1.0e-8;
    v5 =        1220703125.0;
    v6 =        271828183.0;
    v7 = 1 << v1;
    v8 = 10;

    v9 = 2048;
    v75 = v9;

    v10 = 1.1920928955078125e-07;
    v11 = v10 * v10;
    v12 = 8.388608e+06;
    v13 = v12 * v12;

  double v58[2*(1<<16)];
  double v59[10]; 

  double v73[3] = {1.0, 1.0, 1.0};
  char   v74[16];

  FILE *v79;

  if ((v79 = fopen("timer.flag", "r")) == NULL) {
    v47 = 0;
  } else {
    v47 = 1;
    fclose(v79);
  }

  if (v3 < v75) {
     v75 = v3;
  }
  v78 = ceil( (double)v3 / (double) v75);

  sprintf(v74, "%15.0lf", pow(2.0, M+1));
  v45 = 14;
  if (v74[v45] == '.') v45--;
  v74[v45+1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n", v74);

  v46 = 0;

printf("NK=%d NN=%d NQ=%d BLKS=%d NBLKS=%d\n",v7,v3,v8,v75,v78);

{
  vranlc(0, &v73[0], v73[1], &v73[2]);
  v73[0] = f1(&v73[1], v73[2]);

  for (v38 = 0; v38 < v8; v38++) {
    v59[v38] = 0.0;
  }
  v26 = log(sqrt(fabs(MAX(1.0, 1.0))));   

  timer_clear(0);
  timer_clear(1);
  timer_clear(2);
  timer_start(0);

  v16 = v5;

  for (v38 = 0; v38 < v1 + 1; v38++) {
    v17 = f1(&v16, v16);
  }

  v30 = v16;
  v31 = v6;
  v32 = 0.0;
  v27 = 0.0;
  v28 = 0.0;

  /* Offload per-sample RNG/hist work to the GPU */
#pragma omp target data map(tofrom:v59[0:v8], v27, v28, v32) \
                        map(to:v3, v5, v6, v7, v8, v10, v11, v12, v13, v30)
  {
    #pragma omp target teams loop reduction(+:v27, v28, v32) firstprivate(v5, v6, v7, v8, v10, v11, v12, v13, v30)
    for (int sample = 0; sample < v3; ++sample) {
      const int bin_count = v8; // cache the bin count so the hot loop does not reload v8 repeatedly
      double local_hist[10];
      for (int bin = 0; bin < bin_count; ++bin) {
        local_hist[bin] = 0.0;
      }

      double sample_v16 = v6;
      double sample_v17 = v30;
      int sample_seed = sample;
      double local_sum_x = 0.0;
      double local_sum_y = 0.0;
      double local_count = 0.0;
      double local_v62, local_v63, local_v64, local_v65;
      double local_v66, local_v67, local_v68, local_v69, local_v70;

      for (int iter = 1; iter <= 100; ++iter) {
        int half_seed = sample_seed / 2;
        if ((2 * half_seed) != sample_seed) {
          local_v62 = v10 * sample_v17;
          local_v66 = (int)local_v62;
          local_v67 = sample_v17 - v12 * local_v66;

          local_v62 = v10 * sample_v16;
          local_v68 = (int)local_v62;
          local_v69 = sample_v16 - v12 * local_v68;
          local_v62 = local_v66 * local_v69 + local_v67 * local_v68;
          local_v63 = (int)(v10 * local_v62);
          local_v70 = local_v62 - v12 * local_v63;
          local_v64 = v12 * local_v70 + local_v67 * local_v69;
          local_v65 = (int)(v11 * local_v64);
          sample_v16 = local_v64 - v13 * local_v65;
        }
        if (half_seed == 0) break;

        local_v62 = v10 * sample_v17;
        local_v66 = (int)local_v62;
        local_v67 = sample_v17 - v12 * local_v66;

        local_v62 = v10 * sample_v17;
        local_v68 = (int)local_v62;
        local_v69 = sample_v17 - v12 * local_v68;
        local_v62 = local_v66 * local_v69 + local_v67 * local_v68;
        local_v63 = (int)(v10 * local_v62);
        local_v70 = local_v62 - v12 * local_v63;
        local_v64 = v12 * local_v70 + local_v67 * local_v69;
        local_v65 = (int)(v11 * local_v64);
          sample_v17 = local_v64 - v13 * local_v65;
        sample_seed = half_seed;
      }

      local_v62 = v10 * v5;
      local_v66 = (int)local_v62;
      local_v67 = v5 - v12 * local_v66;

      for (int pair = 0; pair < v7; ++pair) {
        local_v62 = v10 * sample_v16;
        local_v68 = (int)local_v62;
        local_v69 = sample_v16 - v12 * local_v68;
        local_v62 = local_v66 * local_v69 + local_v67 * local_v68;
        local_v63 = (int)(v10 * local_v62);
        local_v70 = local_v62 - v12 * local_v63;
        local_v64 = v12 * local_v70 + local_v67 * local_v69;
        local_v65 = (int)(v11 * local_v64);
        sample_v16 = local_v64 - v13 * local_v65;
        double rand1 = v11 * sample_v16;

        local_v62 = v10 * sample_v16;
        local_v68 = (int)local_v62;
        local_v69 = sample_v16 - v12 * local_v68;
        local_v62 = local_v66 * local_v69 + local_v67 * local_v68;
        local_v63 = (int)(v10 * local_v62);
        local_v70 = local_v62 - v12 * local_v63;
        local_v64 = v12 * local_v70 + local_v67 * local_v69;
        local_v65 = (int)(v11 * local_v64);
        sample_v16 = local_v64 - v13 * local_v65;
        double rand2 = v11 * sample_v16;

        double v22 = 2.0 * rand1 - 1.0;
        double v23 = 2.0 * rand2 - 1.0;
        double radius = v22 * v22 + v23 * v23;
        if (radius <= 1.0) {
          double factor = sqrt(-2.0 * log(radius) / radius);
          double gaussian1 = v22 * factor;
          double gaussian2 = v23 * factor;
          int bin_index = (int)MAX(fabs(gaussian1), fabs(gaussian2));
          if (bin_index < bin_count) {
            local_hist[bin_index] += 1.0;
          }
          local_sum_x += gaussian1;
          local_sum_y += gaussian2;
          local_count += 1.0;
        }
      }

      for (int bin = 0; bin < bin_count; ++bin) {
        double bin_value = local_hist[bin]; // cache the histogram entry to avoid reloading during the atomic
        if (bin_value != 0.0) {
          #pragma omp atomic update
          v59[bin] += bin_value;
        }
      }
      v27 += local_sum_x;
      v28 += local_sum_y;
      v32 += local_count;
    }
  }

  }

  timer_stop(0);
  v29 = timer_read(0);

  v43 = 0;
  v46 = 1;
  if (M == 24) {
    v33 = -3.247834652034740e+3;
    v34 = -6.958407078382297e+3;
  } else if (M == 25) {
    v33 = -2.863319731645753e+3;
    v34 = -6.320053679109499e+3;
  } else if (M == 28) {
    v33 = -4.295875165629892e+3;
    v34 = -1.580732573678431e+4;
  } else if (M == 30) {
    v33 =  4.033815542441498e+4;
    v34 = -2.660669192809235e+4;
  } else if (M == 32) {
    v33 =  4.764367927995374e+4;
    v34 = -8.084072988043731e+4;
  } else if (M == 36) {
    v33 =  1.982481200946593e+5;
    v34 = -1.020596636361769e+5;
  } else if (M == 40) {
    v33 = -5.319717441530e+05;
    v34 = -3.688834557731e+05;
  } else {
    v46 = 0;
  }

  if (v46) {
    v35 = fabs((v27 - v33) / v33);
    v36 = fabs((v28 - v34) / v34);
    v46 = ((v35 <= v4) && (v36 <= v4));
  }

  v26 = pow(2.0, M+1) / v29 / 1000000.0;

  printf("\nEP Benchmark Results:\n\n");
  printf("CPU Time =%10.4lf\n", v29);
  printf("N = 2^%5d\n", M);
  printf("No. Gaussian Pairs = %15.0lf\n", v32);
  printf("Sums = %25.15lE %25.15lE\n", v27, v28);
  printf("Counts: \n");
  for (v38 = 0; v38 < v8; v38++) {
    printf("%3d%15.0lf\n", v38, v59[v38]);
  }

  print_results("EP", CLASS, M+1, 0, 0, v43,
      v29, v26, 
      "Random numbers generated",
      v46, NPBVERSION, COMPILETIME, CS1,
      CS2, CS3, CS4, CS5, CS6, CS7);

  if (v47) {
    if (v29 <= 0.0) v29 = 1.0;
    v31 = timer_read(0);
    printf("\nTotal time:     %9.3lf (%6.2lf)\n", v31, v31*100.0/v29);
    v31 = timer_read(1);
    printf("Gaussian pairs: %9.3lf (%6.2lf)\n", v31, v31*100.0/v29);
    v31 = timer_read(2);
    printf("Random numbers: %9.3lf (%6.2lf)\n", v31, v31*100.0/v29);
  }

  return 0;
}

double f1( double *v14, double v15 )
{

  double v16, v17, v18, v19, v20, v21, v22, v23, v24;
  double v25;

  v16 = v10 * v15;
  v20 = (int) v16;
  v21 = v15 - v12 * v20;

  v16 = v10 * (*v14);
  v22 = (int) v16;
  v23 = *v14 - v12 * v22;
  v16 = v20 * v23 + v21 * v22;
  v17 = (int) (v10 * v16);
  v24 = v16 - v12 * v17;
  v18 = v12 * v24 + v21 * v23;
  v19 = (int) (v11 * v18);
  *v14 = v18 - v13 * v19;
  v25 = v11 * (*v14);

  return v25;
}
