#include <stdio.h>
#include <stddef.h>
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
     size_t matrix_size = (size_t)n * n;
     size_t bytes = matrix_size * sizeof(float);
     int target_device = omp_get_default_device();
     int host_device = omp_get_initial_device();

     float *d_arr = (float *)omp_target_alloc(bytes, target_device);
     if (!d_arr) {
         fprintf(stderr, "func2: failed to allocate GPU buffer\n");
         return;
     }

     omp_target_memcpy(d_arr, arr, bytes, 0, 0, target_device, host_device);

     for (int i = 0; i < n; ++i) {
         size_t row = (size_t)i * n;

         #pragma omp target teams loop device(GPU_DEVICE) is_device_ptr(d_arr)
         for (int j = i; j < n; ++j) {
             float acc = d_arr[row + j];
             for (int k = 0; k < i; ++k) {
                 acc -= d_arr[row + k] * d_arr[(size_t)k * n + j];
             }
             d_arr[row + j] = acc;
         }

         #pragma omp target teams loop device(GPU_DEVICE) is_device_ptr(d_arr)
         for (int j = i + 1; j < n; ++j) {
             float acc = d_arr[(size_t)j * n + i];
             for (int k = 0; k < i; ++k) {
                 acc -= d_arr[(size_t)j * n + k] * d_arr[(size_t)k * n + i];
             }
             d_arr[(size_t)j * n + i] = acc / d_arr[row + i];
         }
     }

     omp_target_memcpy(arr, d_arr, bytes, 0, 0, host_device, target_device);
     omp_target_free(d_arr, target_device);
}
