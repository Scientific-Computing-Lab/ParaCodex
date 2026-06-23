#include <stdio.h>
#include <omp.h>

void lud_omp(float *a, int size)
{
    int i, j, k;

    printf("running omp target\n");

    for (i = 0; i < size; i++) {
        #pragma omp target teams distribute parallel for map(tofrom: a[0:size*size]) private(k)
        for (j = i; j < size; j++) {
            float sum = a[i*size+j];
            for (k = 0; k < i; k++) {
                sum -= a[i*size+k] * a[k*size+j];
            }
            a[i*size+j] = sum;
        }

        #pragma omp target teams distribute parallel for map(tofrom: a[0:size*size]) private(k)
        for (j = i + 1; j < size; j++) {
            float sum = a[j*size+i];
            for (k = 0; k < i; k++) {
                sum -= a[j*size+k] * a[k*size+i];
            }
            a[j*size+i] = sum / a[i*size+i];
        }
    }
}
