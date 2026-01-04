#include <stdio.h>
#include "../../../gate_sdk/gate.h"
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
    /*
     * Keep the working matrix resident on the device for the entire timed region
     * and launch a target kernel for each triangular sweep so that the j loops
     * can execute in parallel while the outer i loop remains sequential.
     */
    #pragma omp target data map(tofrom:a[0:size*size])
    {
        for (int i = 0; i < size; ++i) {
            #pragma omp target teams loop map(present:a[0:size*size])
            for (int j = i; j < size; ++j) {
                float sum = a[i*size + j];
                for (int k = 0; k < i; ++k) {
                    sum -= a[i*size + k] * a[k*size + j];
                }
                a[i*size + j] = sum;
            }

            #pragma omp target teams loop map(present:a[0:size*size])
            for (int j = i + 1; j < size; ++j) {
                float sum = a[j*size + i];
                for (int k = 0; k < i; ++k) {
                    sum -= a[j*size + k] * a[k*size + i];
                }
                a[j*size + i] = sum / a[i*size + i];
            }
        }
    }
}
