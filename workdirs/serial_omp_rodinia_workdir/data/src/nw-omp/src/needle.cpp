#define LIMIT -999
//#define TRACE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#define OPENMP
#include "../../common/rodiniaUtilFunctions.h"


#define GPU_DEVICE 1
#define ERROR_THRESHOLD 0.05

//#define NUM_THREAD 4

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int *input_itemsets, int *referrence, int max_rows, int max_cols, int penalty, int dev);

int maximum( int a,
		 int b,
		 int c){

	int k;
	if( a <= b )
		k = b;
	else 
	k = a;

	if( k <=c )
	return(c);
	else
	return(k);
}


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

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads>\n", argv[0]);
	fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
	fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
	fprintf(stderr, "\t<num_threads>    - no. of threads\n");
	exit(1);
}


void compareResults(int *cpu, int *gpu, int max_rows, int max_cols)
{
  int i, fail;
  fail = 0;
	
  // Compare B and B_GPU
  for (i=0; i < max_rows * max_cols; i++) 
    {
	if (percentDiff(gpu[i], cpu[i]) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
    }
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
	
}

void init(int *input_itemsets_cpu, int *input_itemsets_gpu, int *referrence_cpu, int *referrence_gpu, int max_rows, int max_cols, int penalty)
{
        srand ( 7 );

        for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets_cpu[i*max_cols+j] = 0;
			input_itemsets_gpu[i*max_cols+j] = 0;
		}
	}

	for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
	  int al = rand() % 10 + 1;
          input_itemsets_cpu[i*max_cols] = al;
	  input_itemsets_gpu[i*max_cols] = al;
	}
       
        for( int j=1; j< max_cols ; j++){    //please define your own sequence.
	  int al = rand() % 10 + 1;
          input_itemsets_cpu[j] = al;
	  input_itemsets_gpu[j] = al;
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		referrence_cpu[i*max_cols+j] = blosum62[input_itemsets_cpu[i*max_cols]][input_itemsets_cpu[j]];
		referrence_gpu[i*max_cols+j] = blosum62[input_itemsets_gpu[i*max_cols]][input_itemsets_gpu[j]];
		}
	}

    for( int i = 1; i< max_rows ; i++){
        input_itemsets_cpu[i*max_cols] = -i * penalty;
	input_itemsets_gpu[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++){
       	    input_itemsets_cpu[j] = -j * penalty;
	    input_itemsets_gpu[j] = -j * penalty;
	}
    }
}

void runTest_GPU(int max_cols, int max_rows, int *input_itemsets, int *referrence, int penalty){
	int index, i, idx;
	#pragma omp target device(GPU_DEVICE) map(to: referrence[0:max_rows*max_cols]) map(tofrom: input_itemsets[0:max_rows * max_cols])
	{	
		for( i = 0 ; i < max_cols-2 ; i++){

			#pragma omp parallel for
			for( idx = 0 ; idx <= i ; idx++){
				 index = (idx + 1) * max_cols + (i + 1 - idx);

				 int k;
				 if((input_itemsets[index-1-max_cols]+ referrence[index]) <= (input_itemsets[index-1]-penalty))
			    	    k = (input_itemsets[index-1]-penalty);
				 else 
				    k = (input_itemsets[index-1-max_cols]+ referrence[index]);

				 if(k<=(input_itemsets[index-max_cols]-penalty))
				    input_itemsets[index] = (input_itemsets[index-max_cols]-penalty);
				 else 
				    input_itemsets[index] = k;
			}
		}
	}
        
        //Compute bottom-right matrix 
       	#pragma omp target device(GPU_DEVICE) map(to: referrence[0:max_rows * max_cols]) map(tofrom: input_itemsets[0:max_rows * max_cols])
	{
		for( i = max_cols - 4 ; i >= 0 ; i--){
			       #pragma omp parallel for
			       for( idx = 0 ; idx <= i ; idx++){
				      index =  ( max_cols - idx - 2 ) * max_cols + idx + max_cols - i - 2 ;

					 int k;
					 if((input_itemsets[index-1-max_cols]+ referrence[index]) <= (input_itemsets[index-1]-penalty))
				    	    k = (input_itemsets[index-1]-penalty);
					 else 
					    k = (input_itemsets[index-1-max_cols]+ referrence[index]);

					 if(k<=(input_itemsets[index-max_cols]-penalty))
					    input_itemsets[index] = (input_itemsets[index-max_cols]-penalty);
					 else 
					    input_itemsets[index] = k;
				}

		}
	}
}

void runTest_CPU(int max_cols, int max_rows, int *input_itemsets, int *referrence, int penalty){
	int index, i, idx;
	//printf("Processing top-left matrix\n");	
        for( i = 0 ; i < max_cols-2 ; i++){
		for( idx = 0 ; idx <= i ; idx++){
		 index = (idx + 1) * max_cols + (i + 1 - idx);

		 int k;
		 if((input_itemsets[index-1-max_cols]+ referrence[index]) <= (input_itemsets[index-1]-penalty))
	    	    k = (input_itemsets[index-1]-penalty);
		 else 
		    k = (input_itemsets[index-1-max_cols]+ referrence[index]);

		 if(k<=(input_itemsets[index-max_cols]-penalty))
		    input_itemsets[index] = (input_itemsets[index-max_cols]-penalty);
		 else 
		    input_itemsets[index] = k;
		}
	}
        
        //Compute bottom-right matrix 
	for( i = max_cols - 4 ; i >= 0 ; i--){
	       for( idx = 0 ; idx <= i ; idx++){
		      index =  ( max_cols - idx - 2 ) * max_cols + idx + max_cols - i - 2 ;

			 int k;
			 if((input_itemsets[index-1-max_cols]+ referrence[index]) <= (input_itemsets[index-1]-penalty))
		    	    k = (input_itemsets[index-1]-penalty);
			 else 
			    k = (input_itemsets[index-1-max_cols]+ referrence[index]);

			 if(k<=(input_itemsets[index-max_cols]-penalty))
			    input_itemsets[index] = (input_itemsets[index-max_cols]-penalty);
			 else 
			    input_itemsets[index] = k;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int *input_itemsets, int *referrence, int max_rows, int max_cols, int penalty, int dev) 
{

	//Compute top-left matrix 
	if(dev == 0)
		runTest_CPU(max_cols, max_rows, input_itemsets, referrence, penalty);
	else
		runTest_GPU(max_cols, max_rows, input_itemsets, referrence, penalty);

//#define TRACEBACK
#ifdef TRACEBACK
	
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n");
    
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]); //print the first element
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = input_itemsets[(i - 1) * max_cols + j - 1];
		    w  = input_itemsets[ i * max_cols + j - 1 ];
            n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = input_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}

		else
		;
	}
	
	fclose(fpo);

#endif
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    double t_start, t_end;
    int max_rows, max_cols, penalty;
    int *input_itemsets_cpu, *input_itemsets_gpu;
    int *referrence_cpu, *referrence_gpu;

    if (argc == 4)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    else{
		usage(argc, argv);
    }

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;

    input_itemsets_cpu = (int *)malloc( max_rows * max_cols * sizeof(int));
    input_itemsets_gpu = (int *)malloc( max_rows * max_cols * sizeof(int));   

    referrence_cpu = (int *)malloc( max_rows * max_cols * sizeof(int) ); 
    referrence_gpu = (int *)malloc( max_rows * max_cols * sizeof(int) ); 


    if (!input_itemsets_cpu)
		fprintf(stderr, "error: can not allocate memory");

    init(input_itemsets_cpu, input_itemsets_gpu, referrence_cpu, referrence_gpu, max_rows, max_rows, penalty);

    printf("Start Needleman-Wunsch\n");

    t_start = rtclock();
    runTest( input_itemsets_cpu, referrence_cpu, max_rows, max_cols, penalty, 0);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); 

    t_start = rtclock();
    runTest( input_itemsets_gpu, referrence_gpu, max_rows, max_cols, penalty, 1);
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);     


    compareResults(input_itemsets_cpu, input_itemsets_gpu, max_rows, max_cols);

    free(input_itemsets_cpu); 
    free(input_itemsets_gpu);
    free(referrence_cpu);
    free(referrence_gpu);    

    return EXIT_SUCCESS;
}


