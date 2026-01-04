#include <stdio.h>
#include <omp.h>
#define GPU_DEVICE 1

void func1(float *arr, int n)
{
     int i,j,k;
     float acc;
 
     for (i=0; i <n; i++){
	 for (j=i; j <n; j++){
	     acc=arr[i*n+j];
	     for (k=0; k<i; k++) acc -= arr[i*n+k]*arr[k*n+j];
	     arr[i*n+j]=acc;
	 }

	 for (j=i+1;j<n; j++){
	     acc=arr[j*n+i];
	     for (k=0; k<i; k++) acc -=arr[j*n+k]*arr[k*n+i];
	     arr[j*n+i]=acc/arr[i*n+i];
	 }
     }

}


void func2(float *arr, int n)
{
     int i,j,k;
     float acc;
     {
	     for (i=0; i <n; i++){
		 for (j=i; j <n; j++){
		     acc=arr[i*n+j];
		     for (k=0; k<i; k++) acc -= arr[i*n+k]*arr[k*n+j];
		     arr[i*n+j]=acc;
		 }
		 for (j=i+1;j<n; j++){
		     acc=arr[j*n+i];
		     for (k=0; k<i; k++) acc -=arr[j*n+k]*arr[k*n+i];
		     arr[j*n+i]=acc/arr[i*n+i];
		 }
	     }
     }
}

