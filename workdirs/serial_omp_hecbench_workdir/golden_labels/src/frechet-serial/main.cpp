


#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

#include <random>
#include <chrono>

#define n_d 10000 

#include "norm1.h"
#include "norm2.h"
#include "norm3.h"

void discrete_frechet_distance(const int s, const int n_1, const int n_2, const int repeat)
{
  double *ca, *c1, *c2;
  int k; 


  int ca_size = n_1*n_2*sizeof(double);
  int c1_size = n_1*n_d*sizeof(double);
  int c2_size = n_2*n_d*sizeof(double);

  

  ca = (double *) malloc (ca_size);

  

  c1 = (double *) malloc (c1_size);
  c2 = (double *) malloc (c2_size);

  

  for (k = 0; k < n_1*n_2; k++)
  {
    ca[k] = -1.0;
  }

  std::mt19937 gen(19937);
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  for (k = 0; k < n_1 * n_d; k++)
  {
    c1[k] = dis(gen);
  }

  for (k = 0; k < n_2 * n_d; k++)
  {
    c2[k] = dis(gen);
  }

  auto start = std::chrono::steady_clock::now();

  if (s == 0)
    for (k = 0; k < repeat; k++)
      distance_norm1(n_1, n_2, ca, c1, c2);

  else if (s == 1)
    for (k = 0; k < repeat; k++)
      distance_norm2(n_1, n_2, ca, c1, c2);

  else if (s == 2)
    for (k = 0; k < repeat; k++)
      distance_norm3(n_1, n_2, ca, c1, c2);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  double checkSum = 0;
  for (k = 0; k < n_1 * n_2; k++)
    checkSum += ca[k];
  printf("checkSum: %lf\n", checkSum);

  

  free(ca);
  free(c1);
  free(c2);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <n_1> <n_2> <repeat>\n", argv[0]); 
    printf("  n_1: number of points of the 1st curve");
    printf("  n_2: number of points of the 2nd curve");
    return 1;
  }

  

  const int n_1 = atoi(argv[1]);
  const int n_2 = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  for (int i = 0; i < 3; i++)
    discrete_frechet_distance(i, n_1, n_2, repeat);

  return 0;
}
