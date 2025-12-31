#include <stdio.h>
#include <omp.h>
#include "gate.h"

#define GPU_DEVICE 1

void lud_omp_cpu(float *a, int size)
{
     int i,j,k;
     float sum;

     for (i=0; i <size; i++){
	 for (j=i; j <size; j++){
	     sum=a[i*size+j];
	     for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
	     a[i*size+j]=sum;
	 }

	 for (j=i+1;j<size; j++){
	     sum=a[j*size+i];
	     for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
	     a[j*size+i]=sum/a[i*size+i];
	 }
     }

}


void lud_omp_gpu(float *a, int size)
{
     int i;

     /* Keep matrix resident on device while sweeping along i */
     #pragma omp target data map(tofrom: a[0:size*size])
     {
	 for (i=0; i <size; i++){
	     // Sweep stage i: parallelize U-row and L-column updates on the device.
	     #pragma omp target teams loop thread_limit(128)
	     for (int j=i; j <size; j++){
		 float current = a[i*size+j];
		 float accum = 0.0f;
		 #pragma omp loop reduction(+:accum)
		 for (int k=0; k<i; k++){
		     accum += a[i*size+k]*a[k*size+j];
		 }
		 a[i*size+j]=current - accum;
	     }

	     float pivot = a[i*size+i];
	     #pragma omp target teams loop thread_limit(128)
	     for (int j=i+1;j<size; j++){
		 float current=a[j*size+i];
		 float accum = 0.0f;
		 #pragma omp loop reduction(+:accum)
		 for (int k=0; k<i; k++){
		     accum +=a[j*size+k]*a[k*size+i];
		 }
		 a[j*size+i]=(current - accum)/pivot;
	     }
	 }
     }
}
