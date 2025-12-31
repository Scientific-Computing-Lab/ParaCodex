















































































































#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "reference.h"















void Extrema(const double* history, const int history_length, double *result, int& result_length)
{
  result[0] = history[0];

  int eidx = 0;
  for (int i = 1; i < history_length - 1; i++)
    if ((history[i] > result[eidx] && history[i] > history[i + 1]) ||
        (history[i] < result[eidx] && history[i] < history[i + 1]))
      result[++eidx] = history[i];

  result[++eidx] = history[history_length - 1];
  result_length = eidx + 1;
}













void Execute(const double* history, const int history_length,
             double *extrema, int* points, double3 *results,
             int *results_length )
{
  int extrema_length = 0;
  Extrema(history, history_length, extrema, extrema_length);

  int pidx = -1, eidx = -1, ridx = -1;

  for (int i = 0; i < extrema_length; i++)
  {
    points[++pidx] = ++eidx;
    double xRange, yRange;
    while (pidx >= 2 && (xRange = fabs(extrema[points[pidx - 1]] - extrema[points[pidx]]))
           >= (yRange = fabs(extrema[points[pidx - 2]] - extrema[points[pidx - 1]])))
    {
      double yMean = 0.5 * (extrema[points[pidx - 2]] + extrema[points[pidx - 1]]);

      if (pidx == 2)
      {
        results[++ridx] = { 0.5, yRange, yMean };
        points[0] = points[1];
        points[1] = points[2];
        pidx = 1;
      }
      else
      {
        results[++ridx] = { 1.0, yRange, yMean };
        points[pidx - 2] = points[pidx];
        pidx -= 2;
      }
    }
  }

  for (int i = 0; i <= pidx - 1; i++)
  {
    double range = fabs(extrema[points[i]] - extrema[points[i + 1]]);
    double mean = 0.5 * (extrema[points[i]] + extrema[points[i + 1]]);
    results[++ridx] = { 0.5, range, mean };
  }

  *results_length = ridx + 1;
}

int main(int argc, char* argv[]) {
  const int num_history = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int *history_lengths = (int*) malloc ((num_history + 1) * sizeof(int));
  int *result_lengths = (int*) malloc (num_history * sizeof(int));
  int *ref_result_lengths = (int*) malloc (num_history * sizeof(int));
  
  srand(123);

  

  const int scale = 100;
  size_t total_length  = 0;

  int n;
  for (n = 0; n < num_history; n++) {
     history_lengths[n] = total_length;
     total_length += (rand() % 10 + 1) * scale;
  }
  history_lengths[n] = total_length;
  
  printf("Total history length = %zu\n", total_length);

  double *history = (double*) malloc (total_length * sizeof(double));
  for (size_t i = 0; i < total_length; i++) {
    history[i] = rand() / (double)RAND_MAX;
  }

  double *extrema = (double*) malloc (total_length * sizeof(double));
  double3 *results = (double3*) malloc (total_length * sizeof(double3));
  int *points = (int*) malloc (total_length * sizeof(int));
  
    {
    auto start = std::chrono::steady_clock::now();

    for (n = 0; n < repeat; n++) {
            for (int i = 0; i < num_history; i++) {
        const int offset = history_lengths[i];
        const int history_length = history_lengths[i+1] - offset;
        Execute(history + offset, 
                history_length,
                extrema + offset,
                points + offset,
                results + offset,
                result_lengths + i);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
  }

  reference (
    history, 
    history_lengths,
    extrema,
    points,
    results,
    ref_result_lengths,
    num_history
  );

  int error = memcmp(ref_result_lengths, result_lengths, num_history * sizeof(int));
  printf("%s\n", error ? "FAIL" : "PASS");

  free(history);
  free(history_lengths);
  free(extrema);
  free(points);
  free(results);
  free(result_lengths);
  free(ref_result_lengths);

  return 0;
}