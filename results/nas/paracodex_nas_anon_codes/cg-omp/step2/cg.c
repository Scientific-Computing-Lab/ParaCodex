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

static int v1[NZ];
static int v2[NA+1];
static int v3[NA];
static int v4[NA];
static int v5[NAZ];

static double v6[NAZ];
static double v7[NZ];
static double v8[NA+2];
static double v9[NA+2];
static double v10[NA+2];
static double v11[NA+2];
static double v12[NA+2];

static int *d_v1 = NULL;
static int *d_v2 = NULL;
static double *d_v7 = NULL;
static double *d_v8 = NULL;
static double *d_v9 = NULL;
static double *d_v10 = NULL;
static double *d_v11 = NULL;
static double *d_v12 = NULL;
static int omp_target_device = -1;
static int omp_initial_device = -1;

static int v13;
static int v14;
static int v15;
static int v16;
static int v17;
static int v18;

static double v19;
static double v20;

static logical v21;

static int f5(double v22, int v23);
static void f6(int v24, double v25[], int v26[], int *v27, int v28, double v29);
static void f4(int v30, int v31, int v32, double v33[], int v34[]);
static void f3(double v35[],
                   int v36[],
                   int v37[],
                   int v38,
                   int v39,
                   int v40,
                   int v41[],
                   int v42[][NONZER+1],
                   double v43[][NONZER+1],
                   int v44,
                   int v45,
                   int v46[],
                   double v47,
                   double v48);
static void f2(int v49,
                  int v50,
                  double v51[],
                  int v52[],
                  int v53[],
                  int v54,
                  int v55,
                  int v56,
                  int v57,
                  int v58[],
                  int v59[][NONZER+1],
                  double v60[][NONZER+1],
                  int v61[]);
static void f1(int v62[],
                      int v63[],
                      double v64[],
                      double v65[],
                      double v66[],
                      double v67[],
                      double v68[],
                      double v69[],
                      double *v70);
static void allocate_device_buffers(void);
static void copy_csr_to_device(void);
static void copy_vectors_to_device(void);
static void free_device_buffers(void);
static int v71 = 0;
static int v72 = 0;

int main(int argc, char *argv[])
{
  int v73, v74, v75, v76;
  int v77;

  double v78;
  double v79;
  double v80, v81;

  double v82, v83, v84;
  char v85;
  int v86;
  double v87, v88, v89;

  char *v90[T_last];

  for (v73 = 0; v73 < T_last; v73++) {
    timer_clear(v73);
  }
  
  FILE *v91;
  if ((v91 = fopen("timer.flag", "r")) != NULL) {
    v21 = true;
    v90[T_init] = "init";
    v90[T_bench] = "benchmk";
    v90[T_conj_grad] = "conjgd";
    fclose(v91);
  } else {
    v21 = false;
  }

  timer_start(T_init);

  v15 = 0;
  v16  = NA-1;
  v17 = 0;
  v18  = NA-1;

  if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10) {
    v85 = 'S';
    v87 = 8.5971775078648;
  } else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12) {
    v85 = 'W';
    v87 = 10.362595087124;
  } else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20) {
    v85 = 'A';
    v87 = 17.130235054029;
  } else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60) {
    v85 = 'B';
    v87 = 22.712745482631;
  } else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110) {
    v85 = 'C';
    v87 = 28.973605592845;
  } else if (NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500) {
    v85 = 'D';
    v87 = 52.514532105794;
  } else if (NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500) {
    v85 = 'E';
    v87 = 77.522164599383;
  } else {
    v85 = 'U';
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - CG Benchmark\n\n");
  printf(" Size: %11d\n", NA);
  printf(" Iterations: %5d\n", NITER);
  printf("\n");

  v13 = NA;
  v14 = NZ;

  v20    = 314159265.0;
  v19   = 1220703125.0;
  v78    = randlc(&v20, v19);

  f2(v13, v14, v7, v1, v2, 
        v15, v16, v17, v18, 
        v4, 
        (int (*)[NONZER+1])(void*)v5, 
        (double (*)[NONZER+1])(void*)v6,
        v3);

  for (v74 = 0; v74 < v16 - v15 + 1; v74++) {
    for (v75 = v2[v74]; v75 < v2[v74+1]; v75++) {
      v1[v75] = v1[v75] - v17;
    }
  }

  allocate_device_buffers();
  copy_csr_to_device();

{
  int v92 = NA+1;
  for (v73 = 0; v73 < NA+1; v73++) {
    v8[v73] = 1.0;
  }

  v77 = v18 - v17 + 1;
  for (v74 = 0; v74 < v77; v74++) {
    v11[v74] = 0.0;
    v9[v74] = 0.0;
    v12[v74] = 0.0;
    v10[v74] = 0.0;
  }

  copy_vectors_to_device();

  v78 = 0.0;

  for (v76 = 1; v76 <= 1; v76++) {
    f1(d_v1, d_v2, d_v8, d_v9, d_v7, d_v10, d_v11, d_v12, &v79);

    double warm_norm = 0.0;
    #pragma omp target teams loop reduction(+:warm_norm) is_device_ptr(d_v9)
    for (v74 = 0; v74 < v77; v74++) {
      double tmp = d_v9[v74];
      warm_norm += tmp * tmp;
    }

    double warm_inv = 1.0 / sqrt(warm_norm);
    #pragma omp target teams loop firstprivate(warm_inv) is_device_ptr(d_v8, d_v9)
    for (v74 = 0; v74 < v77; v74++) {
      d_v8[v74] = warm_inv * d_v9[v74];
    }
  }

  v92 = NA+1;
  for (v73 = 0; v73 < NA+1; v73++) {
    v8[v73] = 1.0;
  }

  omp_target_memcpy(d_v8, v8, sizeof(v8), 0, 0, omp_target_device, omp_initial_device);

  v78 = 0.0;

  timer_stop(T_init);

  printf(" Initialization time = %15.3f seconds\n", timer_read(T_init));

  timer_start(T_bench);

  v77 = v18 - v17 + 1;
  for (v76 = 1; v76 <= NITER; v76++) {
    f1(d_v1, d_v2, d_v8, d_v9, d_v7, d_v10, d_v11, d_v12, &v79);

    v80 = 0.0;
    v81 = 0.0;
    #pragma omp target teams loop reduction(+:v80,v81) is_device_ptr(d_v8, d_v9)
    for (v74 = 0; v74 < v77; v74++) {
      double tmp = d_v9[v74];
      v80 += d_v8[v74] * tmp;
      v81 += tmp * tmp;
    }

    v81 = 1.0 / sqrt(v81);

    v78 = SHIFT + 1.0 / v80;
    if (v76 == 1) 
      printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", v76, v79, v78);

    #pragma omp target teams loop firstprivate(v81) is_device_ptr(d_v8, d_v9)
    for (v74 = 0; v74 < v77; v74++) {
      d_v8[v74] = v81 * d_v9[v74];
    }
  }

  timer_stop(T_bench);
}

  v82 = timer_read(T_bench);

  printf(" Benchmark completed\n");

  v88 = 1.0e-10;
  if (v85 != 'U') {
    v89 = fabs(v78 - v87) / v87;
    if (v89 <= v88) {
      v86 = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" Zeta is    %20.13E\n", v78);
      printf(" Error is   %20.13E\n", v89);
    } else {
      v86 = false;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %20.13E\n", v78);
      printf(" The correct zeta is %20.13E\n", v87);
    }
  } else {
    v86 = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
  }

  if (v82 != 0.0) {
    v83 = (double)(2*NITER*NA)
                   * (3.0+(double)(NONZER*(NONZER+1))
                     + 25.0*(5.0+(double)(NONZER*(NONZER+1)))
                     + 3.0) / v82 / 1000000.0;
  } else {
    v83 = 0.0;
  }

  print_results("CG", v85, NA, 0, 0,
                NITER, v82,
                v83, "          floating point", 
                v86, NPBVERSION, COMPILETIME,
                CS1, CS2, CS3, CS4, CS5, CS6, CS7);

  if (v21) {
    v84 = timer_read(T_bench);
    if (v84 == 0.0) v84 = 1.0;
    printf("  SECTION   Time (secs)\n");
    for (v73 = 0; v73 < T_last; v73++) {
      v82 = timer_read(v73);
      if (v73 == T_init) {
        printf("  %8s:%9.3f\n", v90[v73], v82);
      } else {
        printf("  %8s:%9.3f  (%6.2f%%)\n", v90[v73], v82, v82*100.0/v84);
        if (v73 == T_conj_grad) {
          v82 = v84 - v82;
          printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest", v82, v82*100.0/v84);
        }
      }
    }
  }
  printf("conj calls=%d, loop iter = %d. \n", v71, v72);
  free_device_buffers();
  return 0;
}

static int f5(double v22, int v23)
{
  return (int)(v23 * v22);
}

static void f6(int v24, double v25[], int v26[], int *v27, int v28, double v29)
{
  int v75;
  logical v93;

  v93 = false;
  for (v75 = 0; v75 < *v27; v75++) {
    if (v26[v75] == v28) {
      v25[v75] = v29;
      v93  = true;
    }
  }
  if (v93 == false) {
    v25[*v27]  = v29;
    v26[*v27] = v28;
    *v27     = *v27 + 1;
  }
}

static void f4(int v30, int v31, int v32, double v33[], int v34[])
{
  int v94, v95, v73;
  double v96, v97;

  v94 = 0;

  while (v94 < v31) {
    v96 = randlc(&v20, v19);

    v97 = randlc(&v20, v19);
    v73 = f5(v97, v32) + 1;
    if (v73 > v30) continue;

    logical v98 = false;
    for (v95 = 0; v95 < v94; v95++) {
      if (v34[v95] == v73) {
        v98 = true;
        break;
      }
    }
    if (v98) continue;
    v33[v94] = v96;
    v34[v94] = v73;
    v94 = v94 + 1;
  }
}

static void f3(double v35[],
                   int v36[],
                   int v37[],
                   int v38,
                   int v39,
                   int v40,
                   int v41[],
                   int v42[][NONZER+1],
                   double v43[][NONZER+1],
                   int v44,
                   int v45,
                   int v46[],
                   double v47,
                   double v48)
{
  int v99;

  int v73, v74, v100, v101, v102, v75, v103, v104, v105;
  double v106, v107, v108, v109;
  logical v110;

  v99 = v45 - v44 + 1;

  for (v74 = 0; v74 < v99+1; v74++) {
    v37[v74] = 0;
  }

  for (v73 = 0; v73 < v38; v73++) {
    for (v102 = 0; v102 < v41[v73]; v102++) {
      v74 = v42[v73][v102] + 1;
      v37[v74] = v37[v74] + v41[v73];
    }
  }

  v37[0] = 0;
  for (v74 = 1; v74 < v99+1; v74++) {
    v37[v74] = v37[v74] + v37[v74-1];
  }
  v102 = v37[v99] - 1;

  if (v102 > v39) {
    printf("Space for matrix elements exceeded in sparse\n");
    printf("nza, nzmax = %d, %d\n", v102, v39);
    exit(EXIT_FAILURE);
  }

  for (v74 = 0; v74 < v99; v74++) {
    for (v75 = v37[v74]; v75 < v37[v74+1]; v75++) {
      v35[v75] = 0.0;
      v36[v75] = -1;
    }
    v46[v74] = 0;
  }

  v106 = 1.0;
  v108 = pow(v47, (1.0 / (double)(v38)));

  for (v73 = 0; v73 < v38; v73++) {
    for (v102 = 0; v102 < v41[v73]; v102++) {
      v74 = v42[v73][v102];

      v107 = v106 * v43[v73][v102];
      for (v104 = 0; v104 < v41[v73]; v104++) {
        v105 = v42[v73][v104];
        v109 = v43[v73][v104] * v107;

        if (v105 == v74 && v74 == v73) {
          v109 = v109 + v47 - v48;
        }

        v110 = false;
        for (v75 = v37[v74]; v75 < v37[v74+1]; v75++) {
          if (v36[v75] > v105) {
            for (v103 = v37[v74+1]-2; v103 >= v75; v103--) {
              if (v36[v103] > -1) {
                v35[v103+1]  = v35[v103];
                v36[v103+1] = v36[v103];
              }
            }
            v36[v75] = v105;
            v35[v75]  = 0.0;
            v110 = true;
            break;
          } else if (v36[v75] == -1) {
            v36[v75] = v105;
            v110 = true;
            break;
          } else if (v36[v75] == v105) {
            v46[v74] = v46[v74] + 1;
            v110 = true;
            break;
          }
        }
        if (v110 == false) {
          printf("internal error in sparse: i=%d\n", v73);
          exit(EXIT_FAILURE);
        }
        v35[v75] = v35[v75] + v109;
      }
    }
    v106 = v106 * v108;
  }

  for (v74 = 1; v74 < v99; v74++) {
    v46[v74] = v46[v74] + v46[v74-1];
  }

  for (v74 = 0; v74 < v99; v74++) {
    if (v74 > 0) {
      v100 = v37[v74] - v46[v74-1];
    } else {
      v100 = 0;
    }
    v101 = v37[v74+1] - v46[v74];
    v102 = v37[v74];
    for (v75 = v100; v75 < v101; v75++) {
      v35[v75] = v35[v102];
      v36[v75] = v36[v102];
      v102 = v102 + 1;
    }
  }
  for (v74 = 1; v74 < v99+1; v74++) {
    v37[v74] = v37[v74] - v46[v74-1];
  }
  v102 = v37[v99] - 1;
}

static void f2(int v49,
                  int v50,
                  double v51[],
                  int v52[],
                  int v53[],
                  int v54,
                  int v55,
                  int v56,
                  int v57,
                  int v58[],
                  int v59[][NONZER+1],
                  double v60[][NONZER+1],
                  int v61[])
{
  int v111, v112, v113, v114;
  int v115[NONZER+1];
  double v116[NONZER+1];

  v114 = 1;
  do {
    v114 = 2 * v114;
  } while (v114 < v49);

  for (v111 = 0; v111 < v49; v111++) {
    v113 = NONZER;
    f4(v49, v113, v114, v116, v115);
    f6(v49, v116, v115, &v113, v111+1, 0.5);
    v58[v111] = v113;
    
    for (v112 = 0; v112 < v113; v112++) {
      v59[v111][v112] = v115[v112] - 1;
      v60[v111][v112] = v116[v112];
    }
  }

  f3(v51, v52, v53, v49, v50, NONZER, v58, v59, 
         v60, v54, v55,
         v61, RCOND, SHIFT);
}

static void f1(int v62[],
                      int v63[],
                      double v64[],
                      double v65[],
                      double v66[],
                      double v67[],
                      double v68[],
                      double v69[],
                      double *v70)
{
  int v74, v75, v117, v118, v119;
  int v77;
  int v120, v121 = 25;
  double v122, v123, v124, v125, v126, v127;

  v71++;
  v124 = 0.0;

  v77 = v18 - v17 + 1;
  #pragma omp target teams loop is_device_ptr(v65, v68, v64, v67, v69)
  for (v74 = 0; v74 < v13; v74++) {
    v68[v74] = 0.0;
    v65[v74] = 0.0;
    v69[v74] = v64[v74];
    v67[v74] = v69[v74];
  }

  #pragma omp target teams loop reduction(+:v124) is_device_ptr(v69)
  for (v74 = 0; v74 < v77; v74++) {
    v124 += v69[v74] * v69[v74];
  }

  for (v120 = 1; v120 <= v121; v120++) {
    v72++;

    // Combine SpMV with v67·v68 reduction to reuse loaded elements.
    v122 = 0.0;
    v77 = v16 - v17 + 1;
    #pragma omp target teams loop reduction(+:v122) is_device_ptr(v63, v62, v66, v67, v68)
    for (v74 = 0; v74 < v77; v74++) {
      v117 = v63[v74];
      v118 = v63[v74+1];
      v123 = 0.0;
      double row_val = v67[v74];
      for (v75 = v117; v75 < v118; v75++) {
        v119 = v62[v75];
        v123 += v66[v75] * v67[v119];
      }
      v68[v74] = v123;
      v122 += row_val * v123;
    }

    v126 = v124 / v122;
    v125 = v124;
    v124 = 0.0;

    // Merge update of v65/v69 with the dot-product accumulation for v124.
    #pragma omp target teams loop reduction(+:v124) firstprivate(v126) is_device_ptr(v65, v69, v68, v67)
    for (v74 = 0; v74 < v77; v74++) {
      double y_val = v69[v74] - v126 * v68[v74];
      v65[v74] = v65[v74] + v126 * v67[v74];
      v69[v74] = y_val;
      v124 += y_val * y_val;
    }

    v127 = v124 / v125;

    #pragma omp target teams loop firstprivate(v127) is_device_ptr(v67, v69)
    for (v74 = 0; v74 < v77; v74++) {
      v67[v74] = v69[v74] + v127 * v67[v74];
    }
  }

  // Fuse the final SpMV and norm calculation to reuse registered values.
  v123 = 0.0;
  v77 = v16 - v15 + 1;
  #pragma omp target teams loop reduction(+:v123) is_device_ptr(v63, v62, v66, v65, v69, v64)
  for (v74 = 0; v74 < v77; v74++) {
    v117 = v63[v74];
    v118 = v63[v74+1];
    v122 = 0.0;
    for (v75 = v117; v75 < v118; v75++) {
      v119 = v62[v75];
      v122 += v66[v75] * v65[v119];
    }
    v69[v74] = v122;
    double diff = v64[v74] - v122;
    v123 += diff * diff;
  }

  *v70 = sqrt(v123);
}

static void allocate_device_buffers(void)
{
  omp_initial_device = omp_get_initial_device();
  omp_target_device = omp_get_default_device();

  d_v1 = (int *)omp_target_alloc(sizeof(v1), omp_target_device);
  d_v2 = (int *)omp_target_alloc(sizeof(v2), omp_target_device);
  d_v7 = (double *)omp_target_alloc(sizeof(v7), omp_target_device);
  d_v8 = (double *)omp_target_alloc(sizeof(v8), omp_target_device);
  d_v9 = (double *)omp_target_alloc(sizeof(v9), omp_target_device);
  d_v10 = (double *)omp_target_alloc(sizeof(v10), omp_target_device);
  d_v11 = (double *)omp_target_alloc(sizeof(v11), omp_target_device);
  d_v12 = (double *)omp_target_alloc(sizeof(v12), omp_target_device);

  if (!d_v1 || !d_v2 || !d_v7 || !d_v8 || !d_v9 || !d_v10 || !d_v11 || !d_v12) {
    fprintf(stderr, "Failed to allocate device buffers\n");
    exit(EXIT_FAILURE);
  }
}

static void copy_csr_to_device(void)
{
  omp_target_memcpy(d_v1, v1, sizeof(v1), 0, 0, omp_target_device, omp_initial_device);
  omp_target_memcpy(d_v2, v2, sizeof(v2), 0, 0, omp_target_device, omp_initial_device);
  omp_target_memcpy(d_v7, v7, sizeof(v7), 0, 0, omp_target_device, omp_initial_device);
}

static void copy_vectors_to_device(void)
{
  omp_target_memcpy(d_v8, v8, sizeof(v8), 0, 0, omp_target_device, omp_initial_device);
  omp_target_memcpy(d_v9, v9, sizeof(v9), 0, 0, omp_target_device, omp_initial_device);
  omp_target_memcpy(d_v10, v10, sizeof(v10), 0, 0, omp_target_device, omp_initial_device);
  omp_target_memcpy(d_v11, v11, sizeof(v11), 0, 0, omp_target_device, omp_initial_device);
  omp_target_memcpy(d_v12, v12, sizeof(v12), 0, 0, omp_target_device, omp_initial_device);
}

static void free_device_buffers(void)
{
  if (d_v1) {
    omp_target_free(d_v1, omp_target_device);
    d_v1 = NULL;
  }
  if (d_v2) {
    omp_target_free(d_v2, omp_target_device);
    d_v2 = NULL;
  }
  if (d_v7) {
    omp_target_free(d_v7, omp_target_device);
    d_v7 = NULL;
  }
  if (d_v8) {
    omp_target_free(d_v8, omp_target_device);
    d_v8 = NULL;
  }
  if (d_v9) {
    omp_target_free(d_v9, omp_target_device);
    d_v9 = NULL;
  }
  if (d_v10) {
    omp_target_free(d_v10, omp_target_device);
    d_v10 = NULL;
  }
  if (d_v11) {
    omp_target_free(d_v11, omp_target_device);
    d_v11 = NULL;
  }
  if (d_v12) {
    omp_target_free(d_v12, omp_target_device);
    d_v12 = NULL;
  }
}
