#include <stddef.h>
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
     const int stride = n;
     const size_t matrix_size = (size_t)stride * stride;
     float *__restrict data = arr;
     // Restrict alias helps the GPU compiler reason about triangular updates.

     #pragma omp target data map(tofrom: data[0:matrix_size])
     {
         for (int i = 0; i < stride; ++i) {
             const size_t row_offset = (size_t)i * stride;

             #pragma omp target teams loop thread_limit(256) map(present: data[0:matrix_size]) firstprivate(i, row_offset)
             for (int j = i; j < stride; ++j) {
                 float acc = data[row_offset + j];
                 #pragma omp loop reduction(+:acc)
                 for (int k = 0; k < i; ++k) {
                     acc -= data[row_offset + k] * data[(size_t)k * stride + j];
                 }
                 data[row_offset + j] = acc;
             }

             #pragma omp target teams loop thread_limit(256) map(present: data[0:matrix_size]) firstprivate(i, row_offset)
             for (int j = i + 1; j < stride; ++j) {
                 const float pivot = data[row_offset + i];
                 float acc = data[(size_t)j * stride + i];
                 #pragma omp loop reduction(+:acc)
                 for (int k = 0; k < i; ++k) {
                     acc -= data[(size_t)j * stride + k] * data[(size_t)k * stride + i];
                 }
                 data[(size_t)j * stride + i] = acc / pivot;
             }
         }
     }
}
