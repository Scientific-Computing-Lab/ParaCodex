#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#define OPENMP
#include "../../common/rodiniaUtilFunctions.h"
#include "../../../gate_sdk/gate.h"

#define GPU_DEVICE 1
#define ERROR_THRESHOLD 0.05

void runTest( int *arr1, int *arr2, int dim1, int dim2, int p1, int dev1);

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

int maximum( int a, int b, int c){
	int val1;
	if( a <= b )
		val1 = b;
	else 
		val1 = a;
	if( val1 <=c )
		return(c);
	else
		return(val1);
}

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

void compareResults(int *cpu, int *gpu, int dim1, int dim2)
{
  int i, cnt1;
  cnt1 = 0;
  for (i=0; i < dim1 * dim2; i++) 
    {
	if (percentDiff(gpu[i], cpu[i]) > ERROR_THRESHOLD) 
	    {
	      cnt1++;
	    }
    }
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, cnt1);
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads>\n", argv[0]);
	fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
	fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
	fprintf(stderr, "\t<num_threads>    - no. of threads\n");
	exit(1);
}

void init(int *arr1_cpu, int *arr1_gpu, int *arr2_cpu, int *arr2_gpu, int dim1, int dim2, int p1)
{
        srand ( 7 );
        for (int i = 0 ; i < dim2; i++){
		for (int j = 0 ; j < dim1; j++){
			arr1_cpu[i*dim2+j] = 0;
			arr1_gpu[i*dim2+j] = 0;
		}
	}
	for( int i=1; i< dim1 ; i++){
	  int val2 = rand() % 10 + 1;
          arr1_cpu[i*dim2] = val2;
	  arr1_gpu[i*dim2] = val2;
	}
        for( int j=1; j< dim2 ; j++){
	  int val2 = rand() % 10 + 1;
          arr1_cpu[j] = val2;
	  arr1_gpu[j] = val2;
	}
	for (int i = 1 ; i < dim2; i++){
		for (int j = 1 ; j < dim1; j++){
		arr2_cpu[i*dim2+j] = blosum62[arr1_cpu[i*dim2]][arr1_cpu[j]];
		arr2_gpu[i*dim2+j] = blosum62[arr1_gpu[i*dim2]][arr1_gpu[j]];
		}
	}
    for( int i = 1; i< dim1 ; i++){
        arr1_cpu[i*dim2] = -i * p1;
	arr1_gpu[i*dim2] = -i * p1;
	for( int j = 1; j< dim2 ; j++){
       	    arr1_cpu[j] = -j * p1;
	    arr1_gpu[j] = -j * p1;
	}
    }
}

void runTest_CPU(int dim2, int dim1, int *arr1, int *arr2, int p1){
	int idx1, i, idx2;
        for( i = 0 ; i < dim2-2 ; i++){
		for( idx2 = 0 ; idx2 <= i ; idx2++){
		 idx1 = (idx2 + 1) * dim2 + (i + 1 - idx2);
		 int val1;
		 if((arr1[idx1-1-dim2]+ arr2[idx1]) <= (arr1[idx1-1]-p1))
	    	    val1 = (arr1[idx1-1]-p1);
		 else 
		    val1 = (arr1[idx1-1-dim2]+ arr2[idx1]);
		 if(val1<=(arr1[idx1-dim2]-p1))
		    arr1[idx1] = (arr1[idx1-dim2]-p1);
		 else 
		    arr1[idx1] = val1;
		}
	}
	for( i = dim2 - 4 ; i >= 0 ; i--){
	       for( idx2 = 0 ; idx2 <= i ; idx2++){
		      idx1 =  ( dim2 - idx2 - 2 ) * dim2 + idx2 + dim2 - i - 2 ;
			 int val1;
			 if((arr1[idx1-1-dim2]+ arr2[idx1]) <= (arr1[idx1-1]-p1))
		    	    val1 = (arr1[idx1-1]-p1);
			 else 
			    val1 = (arr1[idx1-1-dim2]+ arr2[idx1]);
			 if(val1<=(arr1[idx1-dim2]-p1))
			    arr1[idx1] = (arr1[idx1-dim2]-p1);
			 else 
			    arr1[idx1] = val1;
		}
	}
}

void runTest_GPU(int dim2, int dim1, int *arr1, int *arr2, int p1){
	// process forward wavefront diagonals sequentially while parallelizing cells in each diagonal
	for( int i = 0 ; i < dim2-2 ; i++){
		#pragma omp target teams loop
		for( int idx2 = 0 ; idx2 <= i ; idx2++){
			int idx1 = (idx2 + 1) * dim2 + (i + 1 - idx2);
			int val1;
			if((arr1[idx1-1-dim2]+ arr2[idx1]) <= (arr1[idx1-1]-p1))
				val1 = (arr1[idx1-1]-p1);
			else 
				val1 = (arr1[idx1-1-dim2]+ arr2[idx1]);
			if(val1<=(arr1[idx1-dim2]-p1))
				arr1[idx1] = (arr1[idx1-dim2]-p1);
			else 
				arr1[idx1] = val1;
		}
	}
	// mirror sweep runs in reverse order but keeps the same per-diagonal parallelism
	for( int i = dim2 - 4 ; i >= 0 ; i--){
		#pragma omp target teams loop
		for( int idx2 = 0 ; idx2 <= i ; idx2++){
			int idx1 =  ( dim2 - idx2 - 2 ) * dim2 + idx2 + dim2 - i - 2 ;
			int val1;
			if((arr1[idx1-1-dim2]+ arr2[idx1]) <= (arr1[idx1-1]-p1))
				val1 = (arr1[idx1-1]-p1);
			else 
				val1 = (arr1[idx1-1-dim2]+ arr2[idx1]);
			if(val1<=(arr1[idx1-dim2]-p1))
				arr1[idx1] = (arr1[idx1-dim2]-p1);
			else 
				arr1[idx1] = val1;
		}
	}
}

void runTest( int *arr1, int *arr2, int dim1, int dim2, int p1, int dev1) 
{
	if(dev1 == 0)
		runTest_CPU(dim2, dim1, arr1, arr2, p1);
	else
		runTest_GPU(dim2, dim1, arr1, arr2, p1);
#ifdef TRACEBACK
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n");
    for (int i = dim1 - 2,  j = dim1 - 2; i>=0, j>=0;){
		int v1, v3, v2, tb1;
		if ( i == dim1 - 2 && j == dim1 - 2 )
			fprintf(fpo, "%d ", arr1[ i * dim2 + j]);
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			v1 = arr1[(i - 1) * dim2 + j - 1];
		    v2  = arr1[ i * dim2 + j - 1 ];
            v3  = arr1[(i - 1) * dim2 + j];
		}
		else if ( i == 0 ){
		    v1 = v3 = LIMIT;
		    v2  = arr1[ i * dim2 + j - 1 ];
		}
		else if ( j == 0 ){
		    v1 = v2 = LIMIT;
            v3  = arr1[(i - 1) * dim2 + j];
		}
		else{
		}
		int nv1, nv2, nv3;
		nv1 = v1 + arr2[i * dim2 + j];
		nv2 = v2 - p1;
		nv3 = v3 - p1;
		tb1 = maximum(nv1, nv2, nv3);
		if(tb1 == nv1)
			tb1 = v1;
		if(tb1 == nv2)
			tb1 = v2;
		if(tb1 == nv3)
            tb1 = v3;
		fprintf(fpo, "%d ", tb1);
		if(tb1 == v1 )
		{i--; j--; continue;}
        else if(tb1 == v2 )
		{j--; continue;}
        else if(tb1 == v3 )
		{i--; continue;}
		else
		;
	}
	fclose(fpo);
#endif
}

int main( int argc, char** argv) 
{
    double t1, t2;
    int dim1, dim2, p1;
    int *arr1_cpu, *arr1_gpu;
    int *arr2_cpu, *arr2_gpu;

    if (argc == 4)
	{
		dim1 = atoi(argv[1]);
		dim2 = atoi(argv[1]);
		p1 = atoi(argv[2]);
	}
    else{
		usage(argc, argv);
    }

    dim1 = dim1 + 1;
    dim2 = dim2 + 1;

    arr1_cpu = (int *)malloc( dim1 * dim2 * sizeof(int));
    arr1_gpu = (int *)malloc( dim1 * dim2 * sizeof(int));   
    arr2_cpu = (int *)malloc( dim1 * dim2 * sizeof(int) ); 
    arr2_gpu = (int *)malloc( dim1 * dim2 * sizeof(int) ); 

    if (!arr1_cpu)
		fprintf(stderr, "error: can not allocate memory");

    init(arr1_cpu, arr1_gpu, arr2_cpu, arr2_gpu, dim1, dim1, p1);

    printf("Start Needleman-Wunsch\n");

    t1 = rtclock();
    runTest( arr1_cpu, arr2_cpu, dim1, dim2, p1, 0);
    t2 = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t2 - t1); 

    int matrix_size = dim1 * dim2;
    // keep the GPU workspace resident while we time the offloaded sweep
    #pragma omp target data map(tofrom: arr1_gpu[0:matrix_size]) map(to: arr2_gpu[0:matrix_size])
    {
        t1 = rtclock();
        runTest( arr1_gpu, arr2_gpu, dim1, dim2, p1, 1);
        t2 = rtclock();
        fprintf(stdout, "GPU Runtime: %0.6lfs\n", t2 - t1);      
    }

    compareResults(arr1_cpu, arr1_gpu, dim1, dim2);

    int *needle_result = arr1_gpu;
    GATE_CHECKSUM_U32("needle_arr1", (const uint32_t*)needle_result, matrix_size);

    free(arr1_cpu); 
    free(arr1_gpu);
    free(arr2_cpu);
    free(arr2_gpu);    

    return EXIT_SUCCESS;
}
