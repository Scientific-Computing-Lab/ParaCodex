#include <stdio.h>

#define BS 16

#pragma omp declare target
static void lud_diagonal_block(float *a, int size, int offset)
{
    int i, j, k;

    for (i = 0; i < BS; i++) {
        for (j = i; j < BS; j++) {
            float sum = a[(offset + i) * size + offset + j];
            for (k = 0; k < i; k++) {
                sum -= a[(offset + i) * size + offset + k] *
                       a[(offset + k) * size + offset + j];
            }
            a[(offset + i) * size + offset + j] = sum;
        }

        {
            float inv = 1.f / a[(offset + i) * size + offset + i];
            for (j = i + 1; j < BS; j++) {
                float sum = a[(offset + j) * size + offset + i];
                for (k = 0; k < i; k++) {
                    sum -= a[(offset + j) * size + offset + k] *
                           a[(offset + k) * size + offset + i];
                }
                a[(offset + j) * size + offset + i] = sum * inv;
            }
        }
    }
}

static void lud_perimeter_block(float *a, int size, int offset, int chunk_idx)
{
    int i, j, k;
    int i_global = offset;
    int j_global = offset + BS * (chunk_idx + 1);
    float temp[BS][BS];

    for (i = 0; i < BS; i++) {
        for (j = 0; j < BS; j++) {
            temp[i][j] = a[(offset + i) * size + offset + j];
        }
    }

    for (j = 0; j < BS; j++) {
        for (i = 0; i < BS; i++) {
            float sum = 0.f;
            for (k = 0; k < i; k++) {
                sum += temp[i][k] * a[(offset + k) * size + j_global + j];
            }
            a[(i_global + i) * size + j_global + j] -= sum;
        }
    }

    j_global = offset;
    i_global += BS * (chunk_idx + 1);
    for (i = 0; i < BS; i++) {
        for (j = 0; j < BS; j++) {
            float sum = 0.f;
            for (k = 0; k < j; k++) {
                sum += a[(i_global + i) * size + offset + k] * temp[k][j];
            }
            a[(i_global + i) * size + j_global + j] =
                (a[(i_global + i) * size + j_global + j] - sum) /
                a[(offset + j) * size + offset + j];
        }
    }
}

static void lud_interior_block(float *a, int size, int offset, int chunk_idx,
                               int chunks_in_inter_row)
{
    int i, j, k;
    int i_global = offset + BS * (1 + chunk_idx / chunks_in_inter_row);
    int j_global = offset + BS * (1 + chunk_idx % chunks_in_inter_row);
    float temp_top[BS][BS];
    float temp_left[BS][BS];
    float sum[BS];

    for (j = 0; j < BS; j++) {
        sum[j] = 0.f;
    }

    for (i = 0; i < BS; i++) {
        for (j = 0; j < BS; j++) {
            temp_top[i][j] = a[(offset + i) * size + j_global + j];
            temp_left[i][j] = a[(i_global + i) * size + offset + j];
        }
    }

    for (i = 0; i < BS; i++) {
        for (k = 0; k < BS; k++) {
            for (j = 0; j < BS; j++) {
                sum[j] += temp_left[i][k] * temp_top[k][j];
            }
        }
        for (j = 0; j < BS; j++) {
            a[(i_global + i) * size + j_global + j] -= sum[j];
            sum[j] = 0.f;
        }
    }
}

#pragma omp end declare target

void lud_omp(float *a, int size)
{
    int offset;
    int total = size * size;

    printf("running omp offload\n");

#pragma omp target data map(tofrom : a[0:total])
    {
        for (offset = 0; offset < size - BS; offset += BS) {
            int size_inter = size - offset - BS;
            int chunks_in_inter_row = size_inter / BS;
            int chunks_per_inter = chunks_in_inter_row * chunks_in_inter_row;
            int chunk_idx;

#pragma omp target map(present, alloc : a[0:total]) firstprivate(size, offset)
            {
                lud_diagonal_block(a, size, offset);
            }

#pragma omp target teams distribute parallel for map(present, alloc : a[0:total]) firstprivate(size, offset, chunks_in_inter_row)
            for (chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++) {
                lud_perimeter_block(a, size, offset, chunk_idx);
            }

#pragma omp target teams distribute parallel for map(present, alloc : a[0:total]) firstprivate(size, offset, chunks_in_inter_row, chunks_per_inter)
            for (chunk_idx = 0; chunk_idx < chunks_per_inter; chunk_idx++) {
                lud_interior_block(a, size, offset, chunk_idx,
                                   chunks_in_inter_row);
            }
        }

#pragma omp target map(present, alloc : a[0:total]) firstprivate(size, offset)
        {
            lud_diagonal_block(a, size, offset);
        }
    }
}
