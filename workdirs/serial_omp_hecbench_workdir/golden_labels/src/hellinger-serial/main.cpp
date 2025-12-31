#include <iostream>
#include <new>
#include <cmath>
#include <chrono>

#ifdef DOUBLE_PRECISION
  #define SQRT sqrt
  #define FABS fabs
  #define FP double
#else
  #define SQRT sqrtf
  #define FABS fabsf
  #define FP float
#endif






constexpr int m_size = 768 * 8;  

constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

#ifdef VERIFY
#include "verify.h"
#endif

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int i, j;

  

  FP(*a_host)[N] = new FP[M][N];
  FP(*b_host)[P] = new FP[N][P];
  

  FP(*c_host)[P] = new FP[M][P];
  

  FP(*c_back)[P] = new FP[M][P];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_host[i][j] = (FP)1.0 / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++)
      b_host[i][j] = rand() % 256;

  for (j = 0; j < P; j++) { 
    FP sum = 0;
    for (i = 0; i < N; i++)
      sum += b_host[i][j];
    for (i = 0; i < N; i++)
      b_host[i][j] /= sum;
  }

  

  


  std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
            << ") * b(" << N << "," << P << ")\n";

    {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
            for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
          FP sum = (FP)0.0;
          for (int k = 0; k < N; k++) {
            sum += SQRT(a_host[i][k] * b_host[k][j]);
          }
          const FP value = (FP)1.0 - sum;
          const FP gate = (!std::signbit(value));
          c_back[i][j] = SQRT(gate * value);
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / repeat << " (s)\n";
  }

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  return 0;
}