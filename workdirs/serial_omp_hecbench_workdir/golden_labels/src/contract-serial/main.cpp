#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>

const int nContractions = 18;  


template <typename T>
void contraction (
  const T *__restrict tensor,
  const T *__restrict adj,
        T *__restrict value,
  const int output_size, 
  const int N, 
  const int nChanels)
{
    for (int tid = 0; tid < output_size; tid++) {
    int C = nChanels;
    int B = N * C;
    int A = N * B;
    int Y = nChanels * nContractions;

    int f = (tid % Y) % nChanels;
    int Case = (tid % Y) / nChanels + 1;
    int y = (tid / Y) % N;
    int x = (tid / Y) / N;

    int a, b, c, d, e;
    T adj_value;

    T sum = (T)0;

    

    

    


    

    if (Case == 1) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    

    if (Case == 2) {    
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (b = 0; b < N; ++b) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }  
    }

    

    if (Case == 3) {    
      b = x;
      c = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            for (a = 0; a < N; ++a) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }  
    }

    

    if (Case == 4) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (a = 0; a < N; ++a) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    

    if (Case == 5) {    
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (a = 0; a < N; ++a) {
          for (b = 0; b < N; ++b) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    

    

    


    

    if (Case == 6) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          c = d;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }

    

    if (Case == 7) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        e = d;
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    if (Case == 8) {
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (b = 0; b < N; ++b) {
            c = b;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    if (Case == 9) {
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          b = e;
          for (c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    if (Case == 10) {
      b = x;
      c = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            a = d;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    if (Case == 11) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (a = 0; a < N; ++a) {
            c = a;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    if (Case == 12) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          a = e;
          for (int c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    if (Case == 13) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          c = e;
          for (int a = 0; a < N; ++a) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    if (Case == 14) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int a = 0; a < N; ++a) {
          b = a;
          for (int c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    if (Case == 15) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int b = 0; b < N; ++b) {
          c = b;
          for (int a = 0; a < N; ++a) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    

    

    


    

    if (Case == 16) {
      a = x;
      d = y;

      for (int e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          b = e;
          c = e;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }  

    

    if (Case == 17) {
      b = x;
      d = y;

      for (int e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          a = e;
          c = e;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }

    

    if (Case == 18) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int a = 0; a < N; ++a) {
          b = a;
          c = a;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }
    value[tid] = sum;
  }
}

int rounded_division(int number1, int number2) {
  if (number1 % number2 == 0)
    return number1 / number2;
  return number1 / number2 + 1;
}

template <typename T>
void contract (const int max_N, const int max_C, const int repeat) {
  

  const size_t tensor_size = (size_t)max_N * max_N * max_N * max_C;
  const size_t tensor_size_byte = tensor_size * sizeof(T);

  T* tensor_value = (T*) malloc (tensor_size_byte);
  for (size_t i = 0; i < tensor_size; i++)
    tensor_value[i] = 1;

  

  const size_t adj_size = max_N * max_N;
  const size_t adj_size_byte = adj_size * sizeof(T);
  
  

  T* adj_value = (T*) malloc (adj_size_byte);
  for (size_t i = 0; i < adj_size; i++) adj_value[i] = 1;

  

  const size_t output_size = max_N * max_N * max_C * nContractions;
  const size_t output_size_byte = max_N * max_N * max_C * nContractions * sizeof(T);

  T* value = (T*) malloc (output_size_byte);

  

    {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      contraction(tensor_value, adj_value, value, output_size, max_N, max_C);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
  }

  double checksum = 0;
  for (size_t i = 0; i < output_size; i++) checksum += value[i];
  printf("Checksum: %lf min:%lf max:%lf\n", checksum, 
         *std::min_element(value, value+output_size),
         *std::max_element(value, value+output_size));

  free(value);
  free(tensor_value);
  free(adj_value);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <dimension> <repeat>\n", argv[0]);
    return 1;
  }
 
  int max_N = atoi(argv[1]);
  int max_C = nContractions;
  int repeat = atoi(argv[2]);

  contract<float>(max_N, max_C, repeat);
  contract<double>(max_N, max_C, repeat);

  return 0;
}