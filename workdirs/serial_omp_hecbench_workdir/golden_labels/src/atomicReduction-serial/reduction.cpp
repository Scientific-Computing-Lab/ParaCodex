


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cmath>

int main(int argc, char** argv)
{
  int arrayLength = 52428800;
  int block_sizes[] = {128, 256, 512, 1024};
  int N = 32;

  if (argc == 3) {
    arrayLength=atoi(argv[1]);
    N=atoi(argv[2]);
  }

  std::cout << "Array size: " << arrayLength*sizeof(int)/1024.0/1024.0 << " MB"<<std::endl;
  std::cout << "Repeat the kernel execution: " << N << " times" << std::endl;

  int* array=(int*)malloc(arrayLength*sizeof(int));
  int checksum =0;
  for(int i=0;i<arrayLength;i++) {
    array[i]=rand()%2;
    checksum+=array[i];
  }

  

  std::chrono::high_resolution_clock::time_point t1, t2;

  float GB=(float)arrayLength*sizeof(int)*N;
  int sum;

    {
    

    for(int n=0;n<N;n++) {
      sum = 0;
                  for (int i = 0; i < arrayLength; i++) {
        sum += array[i];
      }
    }

    for (size_t k = 0; k < sizeof(block_sizes) / sizeof(int); k++) {
      int threads = block_sizes[k];
      int blocks=std::min((arrayLength+threads-1)/threads,2048);

      

      t1 = std::chrono::high_resolution_clock::now();
      for(int n=0;n<N;n++) {
        sum = 0;
                        for (int i = 0; i < arrayLength; i++) {
          sum += array[i];
        }
      }
            t2 = std::chrono::high_resolution_clock::now();
      double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;


      printf("%d %d\n", sum, checksum);
      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;

      t1 = std::chrono::high_resolution_clock::now();
      for(int n=0;n<N;n++) {
        sum = 0;
                        for (int i = 0; i < arrayLength; i=i+2) { 
          sum += array[i] + array[i+1];
        }
      }
            t2 = std::chrono::high_resolution_clock::now();
      times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;

      t1 = std::chrono::high_resolution_clock::now();
      for(int n=0;n<N;n++) {
        sum = 0;
                        for (int i = 0; i < arrayLength; i=i+4) { 
          sum += array[i] + array[i+1] + array[i+2] + array[i+3];
        }
      }
            t2 = std::chrono::high_resolution_clock::now();
      times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;

      t1 = std::chrono::high_resolution_clock::now();
      for(int n=0;n<N;n++) {
        sum = 0;
                        for (int i = 0; i < arrayLength; i=i+8) { 
          sum += array[i] + array[i+1] + array[i+2] + array[i+3] + 
                 array[i+4] + array[i+5] + array[i+6] + array[i+7];
        }
      }
            t2 = std::chrono::high_resolution_clock::now();
      times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;

      t1 = std::chrono::high_resolution_clock::now();
      for(int n=0;n<N;n++) {
        sum = 0;
                        for (int i = 0; i < arrayLength; i=i+16) { 
          sum += array[i] + array[i+1] + array[i+2] + array[i+3] + 
                 array[i+4] + array[i+5] + array[i+6] + array[i+7] +
                 array[i+8] + array[i+9] + array[i+10] + array[i+11] +
                 array[i+12] + array[i+13] + array[i+14] + array[i+15];
        }
      }
            t2 = std::chrono::high_resolution_clock::now();
      times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;
    }
  }
  free(array);
  return 0;
}