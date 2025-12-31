


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#define INSTANCES 224   

#define ATTRIBUTES 4096 

#define THREADS 128    


struct char4 { char x; char y; char z; char w; };




void CPU(int * data, int * distance) {
  

    for (int i = 0; i < INSTANCES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      for (int k = 0; k < ATTRIBUTES; k++) {
        distance[i + INSTANCES * j] += 
          (data[i * ATTRIBUTES + k] != data[j * ATTRIBUTES + k]);
      }
    }
  }
}

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }
  
  const int iterations = atoi(argv[1]);

  

  int *data; 
  char *data_char;
  int *cpu_distance, *gpu_distance;

  

  double start_cpu, stop_cpu;
  double start_gpu, stop_gpu;
  double elapsedTime; 
  struct timeval tp;
  struct timezone tzp;
  
 
  int status;

  

  srand(2);

  

  data = (int *)malloc(INSTANCES * ATTRIBUTES * sizeof(int));
  data_char = (char *)malloc(INSTANCES * ATTRIBUTES * sizeof(char));
  cpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));
  gpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));

  

    for (int i = 0; i < ATTRIBUTES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      data[i + ATTRIBUTES * j] = data_char[i + ATTRIBUTES * j] = random() % 3;
    }
  }

  

  bzero(cpu_distance,INSTANCES*INSTANCES*sizeof(int));
  gettimeofday(&tp, &tzp);
  start_cpu = tp.tv_sec*1000000+tp.tv_usec;
  CPU(data, cpu_distance);
  gettimeofday(&tp, &tzp);
  stop_cpu = tp.tv_sec*1000000+tp.tv_usec;
  elapsedTime = stop_cpu - start_cpu;
  printf("CPU time: %f (us)\n",elapsedTime);

    {
    for (int n = 0; n < iterations; n++) {
      

      bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
        
      gettimeofday(&tp, &tzp);
      start_gpu = tp.tv_sec*1000000+tp.tv_usec;
  
            {
                {
          int idx = omp_get_thread_num();
          int gx = omp_get_team_num() % INSTANCES;
          int gy = omp_get_team_num() / INSTANCES;
      
          for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
            char4 j = *(char4 *)(data_char + i + ATTRIBUTES*gx);
            char4 k = *(char4 *)(data_char + i + ATTRIBUTES*gy);
      
            

            char count = 0;
      
            if(j.x ^ k.x) 
              count++; 
            if(j.y ^ k.y)
              count++;
            if(j.z ^ k.z)
              count++;
            if(j.w ^ k.w)
              count++;
      
            

                        gpu_distance[ INSTANCES*gx + gy ] += count;
          }
        }
      }
  
      gettimeofday(&tp, &tzp);
      stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
      elapsedTime += stop_gpu - start_gpu;
  
          }
  
    printf("Average kernel execution time (w/o shared memory): %f (us)\n", elapsedTime / iterations);
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    if (status != 0) printf("FAIL\n");
    else printf("PASS\n");
  
    elapsedTime = 0; 
    for (int n = 0; n < iterations; n++) {
      

      bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
        
      gettimeofday(&tp, &tzp);
      start_gpu = tp.tv_sec*1000000+tp.tv_usec;
  
            {
        int dist[THREADS];
                {
          int idx = omp_get_thread_num();
          int gx = omp_get_team_num() % INSTANCES;
          int gy = omp_get_team_num() / INSTANCES;
      
          dist[idx] = 0;
                
          for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
            char4 j = *(char4 *)(data_char + i + ATTRIBUTES*gx);
            char4 k = *(char4 *)(data_char + i + ATTRIBUTES*gy);
      
            

            char count = 0;
      
            if(j.x ^ k.x) 
              count++; 
            if(j.y ^ k.y)
              count++;
            if(j.z ^ k.z)
              count++;
            if(j.w ^ k.w)
              count++;
      
            dist[idx] += count;
          }
      
        

                
        
 
          if(idx == 0) {
            for(int i = 1; i < THREADS; i++) {
              dist[0] += dist[i];
            }
      
            

            gpu_distance[INSTANCES*gy + gx] = dist[0];
          }
        }
      }
  
      gettimeofday(&tp, &tzp);
      stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
      elapsedTime += stop_gpu - start_gpu;
  
          }
  
    printf("Average kernel execution time (w/ shared memory): %f (us)\n", elapsedTime / iterations);
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    if (status != 0) printf("FAIL\n");
    else printf("PASS\n");
  }

  free(cpu_distance);
  free(gpu_distance);
  free(data_char);
  free(data);
  return status;
}