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
        for (i = 0; i < size; i++) {
            // Fuse the U-row and L-column sweeps into a single kernel per pivot.
            #pragma omp target teams thread_limit(128)
            {
                int row_idx = i * size;
                float *row_i = &a[row_idx];
                float *diag_col = &a[i];

                #pragma omp loop
                for (int j = i; j < size; j++) {
                    float sum = 0.0f;
                    float *col_ptr = &a[j];
                    for (int k = 0; k < i; k++, col_ptr += size) {
                        sum += row_i[k] * (*col_ptr);
                    }
                    row_i[j] -= sum;
                }

                float pivot = row_i[i];
                #pragma omp loop
                for (int j = i + 1; j < size; j++) {
                    float sum = 0.0f;
                    float *row_j = &a[j * size];
                    float *col_ptr = diag_col;
                    for (int k = 0; k < i; k++, col_ptr += size) {
                        sum += row_j[k] * (*col_ptr);
                    }
                    row_j[i] = (row_j[i] - sum) / pivot;
                }
            }
        }
     }
}
