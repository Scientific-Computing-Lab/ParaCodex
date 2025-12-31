#include <omp.h>
#include <cstdio>
int main(){
  int x = 0;
#pragma omp target map(tofrom:x)
  x = 42;
  printf("%d\n", x);
  return 0;
}
