#include <iostream>
#include <limits>
#include <cmath>
#include <chrono>

using namespace std;






constexpr int m_size = 512 * 8;  

constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int K = m_size / 2;

#include "verify.cpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int i, j;

  

  float(*a_host)[N] = new float[M][N];
  float(*b_host)[K] = new float[N][K];
  

  float(*c_host)[K] = new float[M][K];
  

  float(*c_back)[K] = new float[M][K];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_host[i][j] = 1.f / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < K; j++)
      b_host[i][j] = rand() % 256;

  for (j = 0; j < K; j++) { 
    float sum = 0;
    for (i = 0; i < N; i++)
      sum += b_host[i][j];
    for (i = 0; i < N; i++)
      b_host[i][j] /= sum;
  }

  

  


  cout << "Problem size: c(" << M << "," << K << ") = a(" << M << "," << N
       << ") * b(" << N << "," << K << ")\n";

    {
    for (int m = 1; m <= 4; m++) {
      printf("Minkowski distance with p = %d\n", m);
      const float p = (float)m;
      const float one_over_p = 1.f / p;

      auto start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++) {
                for (int i = 0; i < M; i++) {
          for (int j = 0; j < K; j++) {
            float sum = 0.f;
            for (int k = 0; k < N; k++) {
              sum += powf(fabsf(a_host[i][k] - b_host[k][j]), p);
            }
            c_back[i][j] = powf(sum, one_over_p);
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

            #ifdef VERIFY
      VerifyResult(a_host, b_host, c_host, c_back, p, one_over_p);
      #endif
    }
  }

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  return 0;
}
