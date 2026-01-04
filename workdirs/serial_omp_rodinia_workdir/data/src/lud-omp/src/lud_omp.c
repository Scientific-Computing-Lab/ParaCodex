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
     int i, j, k;
     size_t arr_size = (size_t)n * n;

     /* Keep the matrix resident on the device while stepping through i */
     #pragma omp target data map(tofrom: arr[0:arr_size])
     {
         for (i = 0; i < n; ++i) {
             int diag = i;
             int diag_offset = diag * n;

             /* Compute the U row i with teams/threads operating on each column */
             #pragma omp target teams distribute parallel for thread_limit(256)
             for (j = diag; j < n; ++j) {
                 float acc = arr[diag_offset + j];
                 for (k = 0; k < diag; ++k) {
                     acc -= arr[diag_offset + k] * arr[k * n + j];
                 }
                 arr[diag_offset + j] = acc;
             }

             /* Update the L column values below the diagonal */
             #pragma omp target teams distribute parallel for thread_limit(256)
             for (j = diag + 1; j < n; ++j) {
                 float acc = arr[j * n + diag];
                 for (k = 0; k < diag; ++k) {
                     acc -= arr[j * n + k] * arr[k * n + diag];
                 }
                 arr[j * n + diag] = acc / arr[diag_offset + diag];
             }
         }
     }
}
