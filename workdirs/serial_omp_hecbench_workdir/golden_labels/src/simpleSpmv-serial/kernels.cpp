#include <stdlib.h>
#include <chrono>
#include "mv.h"



long mv_dense_parallel(const int repeat,
                       const int bs,
                       const size_t num_rows,
                       const REAL* x,
                             REAL* matrix,
                             REAL* y)
{
  long time;

    {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
            for (size_t i = 0; i < num_rows; i++) {
        REAL temp = 0;
        for (size_t j = 0; j < num_rows; j++) {
          if (matrix[i * num_rows + j] != (REAL)0) 
            temp += matrix[i * num_rows + j] * x[j];
        }
        y[i] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  return time;
}



long mv_csr_parallel(const int repeat,
                     const int bs,
                     const size_t num_rows,
                     const size_t *row_indices,
                     const size_t *col_indices,
                     const REAL* values,
                     const REAL* x,
                     const size_t nnz,
                     REAL* matrix,
                     REAL* y)
{
  long time;

    {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
            for (size_t i = 0; i < num_rows; i++) {
        size_t row_start = row_indices[i];
        size_t row_end = row_indices[i+1];

        REAL temp = 0;
        for(size_t j = row_start; j < row_end; j++){
          temp += values[j] * x[col_indices[j]];
        }
        y[i] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  return time;
}





size_t prevPowerOf2(size_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v >> 1;
}



long vector_mv_csr_parallel(const int repeat,
                            const int bs,
                            const size_t num_rows,
                            const size_t *row_indices,
                            const size_t *col_indices,
                            const REAL* values,
                            const REAL* x,
                            const size_t nnz,
                            REAL* matrix,
                            REAL* y)
{
  long time;

  int nnz_per_row = nnz / num_rows;
  int threads_per_row = prevPowerOf2(nnz_per_row);
  int warpSize = 32;
  

  threads_per_row = threads_per_row > warpSize ? warpSize : threads_per_row;
  int rows_per_block = bs / threads_per_row;
  if (rows_per_block == 0) rows_per_block = 1;
  int num_blocks = (num_rows + rows_per_block - 1) / rows_per_block;

    {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
            for (size_t i = 0; i < num_rows; i++) {
        size_t row_start = row_indices[i];
        size_t row_end = row_indices[i+1];

        REAL temp = 0;
                for(size_t j = row_start; j < row_end; j++){
          temp += values[j] * x[col_indices[j]];
        }
        y[i] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  return time;
}