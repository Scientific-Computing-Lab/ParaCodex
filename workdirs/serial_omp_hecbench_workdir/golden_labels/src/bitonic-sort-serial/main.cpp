







































































#include <math.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <limits>

void ParallelBitonicSort(int input[], int n) {

  

  int size = pow(2, n);

  

    {
    auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < n; step++) {
      

      for (int stage = step; stage >= 0; stage--) {
        

        

        

        

        int seq_len = pow(2, stage + 1);
        

        int two_power = 1 << (step - stage);

        

                for (int i = 0; i < size; i++) {
          

          int seq_num = i / seq_len;

          

          int swapped_ele = -1;

          

          

          

          

          int h_len = seq_len / 2;

          if (i < (seq_len * seq_num) + h_len) swapped_ele = i + h_len;

          

          int odd = seq_num / two_power;

          

          

          bool increasing = ((odd % 2) == 0);

          

          if (swapped_ele != -1) {
            if (((input[i] > input[swapped_ele]) && increasing) ||
                ((input[i] < input[swapped_ele]) && !increasing)) {
              int temp = input[i];
              input[i] = input[swapped_ele];
              input[swapped_ele] = temp;
            }
          }
        }
      }  

    } 


    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %f (ms)\n", time * 1e-6f);
  }
}



void SwapElements(int step, int stage, int num_sequence, int seq_len,
                  int *array) {
  for (int seq_num = 0; seq_num < num_sequence; seq_num++) {
    int odd = seq_num / (pow(2, (step - stage)));
    bool increasing = ((odd % 2) == 0);

    int h_len = seq_len / 2;

    

    for (int i = seq_num * seq_len; i < seq_num * seq_len + h_len; i++) {
      int swapped_ele = i + h_len;

      if (((array[i] > array[swapped_ele]) && increasing) ||
          ((array[i] < array[swapped_ele]) && !increasing)) {
        int temp = array[i];
        array[i] = array[swapped_ele];
        array[swapped_ele] = temp;
      }
    }  

  }    

}





inline void BitonicSort(int a[], int n) {
  


  

  for (int step = 0; step < n; step++) {
    

    for (int stage = step; stage >= 0; stage--) {
      

      int num_sequence = pow(2, (n - stage - 1));
      

      int sequence_len = pow(2, stage + 1);

      SwapElements(step, stage, num_sequence, sequence_len, a);
    }
  }
}



void DisplayArray(int a[], int array_size) {
  for (int i = 0; i < array_size; ++i) std::cout << a[i] << " ";
  std::cout << "\n";
}

void Usage(std::string prog_name, int exponent) {
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: " << prog_name << " n k \n\n";
  std::cout << " n: Integer exponent presenting the size of the input array. "
               "The number of element in\n";
  std::cout << "    the array must be power of 2 (e.g., 1, 2, 4, ...). Please "
               "enter the corresponding\n";
  std::cout << "    exponent between 0 and " << exponent - 1 << ".\n";
  std::cout << " k: Seed used to generate a random sequence.\n";
}

int main(int argc, char *argv[]) {
  int n, seed, size;
  int exp_max = log2(std::numeric_limits<int>::max());

  

  try {
    n = std::stoi(argv[1]);

    

    if (n < 0 || n >= exp_max) {
      Usage(argv[0], exp_max);
      return -1;
    }

    seed = std::stoi(argv[2]);
    size = pow(2, n);
  } catch (...) {
    Usage(argv[0], exp_max);
    return -1;
  }

  std::cout << "\nArray size: " << size << ", seed: " << seed << "\n";

  size_t size_bytes = size * sizeof(int);

  

  int *data_cpu = (int *)malloc(size_bytes);

  

  int *data_gpu = (int *)malloc(size_bytes);

  

  srand(seed);

  for (int i = 0; i < size; i++) {
    data_gpu[i] = data_cpu[i] = rand() % 1000;
  }

  std::cout << "Bitonic sort (parallel)..\n";
  ParallelBitonicSort(data_gpu, n);

  std::cout << "Bitonic sort (serial)..\n";
  BitonicSort(data_cpu, n);

  

  int unequal = memcmp(data_gpu, data_cpu, size_bytes);
  std::cout << (unequal ? "FAIL" : "PASS") << std::endl;

  

  free(data_cpu);
  free(data_gpu);

  return 0;
}