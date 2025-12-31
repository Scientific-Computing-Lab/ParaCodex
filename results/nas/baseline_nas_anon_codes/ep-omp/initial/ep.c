

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
  double v26, v16, v17, v18, v19, v22, v23;
  double v27, v28, v29, v30, v31, v32;
  double v33, v34, v35, v36;
  int    v37;
  int    v38, v39, v40, v41, v42, v43;
  int    v44, v45;
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

    v10 = 1.1920928955078125e-07;
    v11 = v10 * v10;
    v12 = 8.388608e+06;
    v13 = v12 * v12;

  double v58[2*(1<<16)];
  double v59[10]; 
  double *v60, *v61;

  double v62, v63, v64, v65;
  double v66, v67, v68, v69, v70;

  double v71, v72;
  double v73[3] = {1.0, 1.0, 1.0};
  char   v74[16];

  int v75 = v9;
  int v76, v77, v78;

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

  v60 = (double*)malloc(2*v7*sizeof(double));
  v61 = (double*)malloc(v75*v8*sizeof(double));

  sprintf(v74, "%15.0lf", pow(2.0, M+1));
  v45 = 14;
  if (v74[v45] == '.') v45--;
  v74[v45+1] = '\0';
  printf("\n\n Computation Test\n");
  printf("\n Number of random numbers generated: %15s\n", v74);

  v46 = 0;

  v37 = v3; 
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
  v44 = -1;

for (v76=0; v76 < v78; ++v76) {

 v77 = v76*v75;

 if (v77 + v75 > v37) {
     v75 = v37 - (v76*v75);
 }
 
 for(v42=0; v42<v75; v42++)
  {
  	for(v38=0; v38<v8; v38++)
		v61[v42*v8 + v38] = 0.0;
  }

  for (v42 = 1; v42 <= v75; v42++) {
    v40 = v44 + v42 + v77; 
    v16 = v6;
    v17 = v30;

    for (v38 = 1; v38 <= 100; v38++) {
      v39 = v40 / 2;
      if ((2 * v39) != v40)
      {
        v62 = v10 * v17;
        v66 = (int)v62;
        v67 = v17 - v12 * v66;
        
        v62 = v10 * v16;
        v68 = (int)v62;
        v69 = v16 - v12 * v68;
        v62 = v66 * v69 + v67 * v68;
        v63 = (int)(v10 * v62);
        v70 = v62 - v12 * v63;
        v64 = v12 * v70 + v67 * v69;
        v65 = (int)(v11 * v64);
        v16 = v64 - v13 * v65;
        v18 = v11 * v16;
      }
      if (v39 == 0) break;
        v62 = v10 * v17;
        v66 = (int)v62;
        v67 = v17 - v12 * v66;
        
        v62 = v10 * v17;
        v68 = (int)v62;
        v69 = v17 - v12 * v68;
        v62 = v66 * v69 + v67 * v68;
        v63 = (int)(v10 * v62);
        v70 = v62 - v12 * v63;
        v64 = v12 * v70 + v67 * v69;
        v65 = (int)(v11 * v64);
        v17 = v64 - v13 * v65;
        v18 = v11 * v17;
      v40 = v39;
    }

    v62 = v10 * v5;
    v66 = (int)v62;
    v67 = v5 - v12 * v66;

    for(v38=0; v38<2*v7; v38++)
    {
		v62 = v10 * v16;
		v68 = (int)v62;
		v69 = v16 - v12 * v68;
		v62 = v66 * v69 + v67 * v68;
		v63 = (int)(v10 * v62);
		v70 = v62 - v12 * v63;
		v64 = v12*v70 + v67 *v69;
		v65 = (int)(v11 * v64);
		v16 = v64 - v13 * v65;
        v60[v38] = v11 * v16;
    }

	v71 = 0.0;
	v72 = 0.0;

    int chunk_size = v75;
    int pair_offset = v42 - 1;
    size_t bin_entries = (size_t)v8 * chunk_size;

    /* Offload Gaussian pair processing and bin updates */
#pragma omp target teams distribute parallel for reduction(+:v71,v72) \
    map(to: v60[:2*v7]) map(tofrom: v61[:bin_entries]) \
    firstprivate(chunk_size, pair_offset)
    for (v38 = 0; v38 < v7; v38++) {
      v22 = 2.0 * v60[2*v38] - 1.0;
      v23 = 2.0 * v60[2*v38 + 1] - 1.0;
      v16 = v22 * v22 + v23 * v23;
      if (v16 <= 1.0) {
        v17   = sqrt(-2.0 * log(v16) / v16);
        v18   = (v22 * v17); 
        v19   = (v23 * v17); 
        v41    = MAX(fabs(v18), fabs(v19));
        int bin_index = v41 * chunk_size + pair_offset;
#pragma omp atomic update
        v61[bin_index] += 1.0;
        v71   = v71 + v18;  
        v72   = v72 + v19;  
      }
    }

    v27 += v71;
    v28 += v72;

  }
}

	for(v38=0; v38<v8; v38++)
	{
		double v80 = 0.0;
		for(v42=0; v42<v75; v42++)
			v80 = v80 + v61[v38*v75 + v42];
		
		v59[v38] += v80;
		
		v32 += v80;
	}
 
}

  {
    double v81[2] = { v27, v28 };
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

  printf("\nComputation Results:\n\n");
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

	free(v60);
	free(v61);

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
