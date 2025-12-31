#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <chrono>

































inline float
createRoundingFactor(float max, int n) {
  float delta = (max * (float)n) / (1.f - 2.f * (float)n * FLT_EPSILON);

  

  

  

  

  int exp;
  frexpf(delta, &exp);

  

  return ldexpf(1.f, exp);
}
  






inline float
truncateWithRoundingFactor(float roundingFactor, float x) {
  return (roundingFactor + x) -  

         roundingFactor;         

}

void sumArray (
  const float factor, 
  const   int length,
  const float *__restrict x,
        float *__restrict r)
{
    for (int i = 0; i < length; i++) {
    float q = truncateWithRoundingFactor(factor, x[i]);
        *r += q; 

  }
}
  
void sumArrays (
  const int nArrays,
  const int length,
  const float *__restrict x,
        float *__restrict r,
  const float *__restrict maxVal)
{
    for (int i = 0; i < nArrays; i++) {
    x += i * length;
    float factor = createRoundingFactor(maxVal[i], length);
    float s = 0;
    for (int n = length-1; n >= 0; n--)  

      s += truncateWithRoundingFactor(factor, x[n]);
    r[i] = s;
  }
}
  
int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of arrays> <length of each array>\n", argv[0]); 
    return 1;
  }

  const int nArrays = atoi(argv[1]);
  const int nElems = atoi(argv[2]);
  const size_t narray_size = sizeof(float) * nArrays;
  const size_t array_size = narray_size * nElems;

  

  float *arrays = (float*) malloc (array_size);
  

  float *maxVal = (float*) malloc (narray_size);
  

  float *result = (float*) malloc (narray_size);
  

  float *factor = (float*) malloc (narray_size);
  

  float *result_ref = (float*) malloc (narray_size);

  srand(123);

  

  float *arr = arrays;
  for (int n = 0; n < nArrays; n++) {
    float max = 0;
    for (int i = 0; i < nElems; i++) {
      arr[i] = (float)rand() / (float)RAND_MAX;
      if (rand() % 2) arr[i] = -1.f * arr[i];
      max = fmaxf(fabs(arr[i]), max);
    }
    factor[n] = createRoundingFactor(max, nElems);
    maxVal[n] = max;
    arr += nElems;
  }

  

  arr = arrays;
  for (int n = 0; n < nArrays; n++) {
    result_ref[n] = 0;
    for (int i = 0; i < nElems; i++)
      result_ref[n] += truncateWithRoundingFactor(factor[n], arr[i]);
    arr += nElems;
  }

  bool ok;
    {
    

        for (int i = 0; i < nArrays; i++)
      result[i] = 0.f;

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < nArrays; n++) {
      

      sumArray (factor[n], nElems, arrays + n * nElems, result + n);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (sumArray): %f (s)\n", (time * 1e-9f) / nArrays);

    

        ok = !memcmp(result_ref, result, narray_size);
    printf("%s\n", ok ? "PASS" : "FAIL");
    
    start = std::chrono::steady_clock::now();

    

    sumArrays (nArrays, nElems, arrays, result, maxVal);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Kernel execution time (sumArrays): %f (s)\n", time * 1e-9f);

    

        ok = !memcmp(result_ref, result, narray_size);
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  free(arrays);
  free(maxVal);
  free(result);
  free(factor);
  free(result_ref);

  return 0;
}