#include <cstdio>
#include <omp.h>
int main(){
  int n = 10;
  int a[10];
  #pragma omp target teams distribute parallel for map(tofrom:a[0:n])
  for(int i=0;i<n;++i){
    a[i]=i*i;
  }
  for(int i=0;i<n;++i) printf("%d ", a[i]);
  printf("\n");
  return 0;
}
