#include <chrono>
#include <cmath>
#include <cstdlib>

#include "gate.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  double start = rtclock();

  DATA_TYPE* A =
      static_cast<DATA_TYPE*>(malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE)));
  DATA_TYPE* B_host = static_cast<DATA_TYPE*>(
      malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE)));
  DATA_TYPE* B = static_cast<DATA_TYPE*>(
      malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE)));
  DATA_TYPE* C = static_cast<DATA_TYPE*>(malloc(4 * 4 * sizeof(DATA_TYPE)));

  for (int i = 0; i < MAP_SIZE; ++i) {
    for (int j = 0; j < MAP_SIZE; ++j) {
      A[i * MAP_SIZE + j] = rand() / static_cast<float>(RAND_MAX);
    }
  }

  WinogradConv2D_2x2_filter_transformation(C);

  const int tile_n = (MAP_SIZE - 2 + 1) / 2;

  size_t globalWorkSize[2] = {
      static_cast<size_t>(
          ceil(static_cast<float>(tile_n) /
               static_cast<float>(DIM_LOCAL_WORK_GROUP_X))) *
          DIM_LOCAL_WORK_GROUP_X,
      static_cast<size_t>(
          ceil(static_cast<float>(tile_n) /
               static_cast<float>(DIM_LOCAL_WORK_GROUP_Y))) *
          DIM_LOCAL_WORK_GROUP_Y};

  size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};

  size_t cpu_global_size[2];
  size_t gpu_global_size[2];
  size_t global_offset[2];

  bool pass = true;

  double co_time = 0.0;

  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {
    cpu_global_size[0] =
        cpu_offset *
        static_cast<size_t>(
            ceil(static_cast<float>(tile_n) /
                 static_cast<float>(DIM_LOCAL_WORK_GROUP_X))) /
        100 * DIM_LOCAL_WORK_GROUP_X;
    cpu_global_size[1] = globalWorkSize[1];

    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    const int tile_i_size = gpu_global_size[0];
    const int tile_j_size = gpu_global_size[1];
    const int offset_i = global_offset[0];
    const int offset_j = global_offset[1];

    bool cpu_run = false;
    bool gpu_run = false;
    if (cpu_global_size[0] > 0) {
      cpu_run = true;
    }
    if (gpu_global_size[0] > 0) {
      gpu_run = true;
    }

    double co_start = rtclock();

    if (gpu_run) {
      for (int tile_j = 0; tile_j < tile_j_size; tile_j++) {
        for (int tile_i = 0; tile_i < tile_i_size; tile_i++) {
          DATA_TYPE input_tile[4][4];
          DATA_TYPE tmp_tile[4][4];
          DATA_TYPE transformed_tile[4][4];
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              int x = 2 * (tile_i + offset_i) + i;
              int y = 2 * (tile_j + offset_j) + j;
              if (x >= MAP_SIZE || y >= MAP_SIZE) {
                input_tile[i][j] = 0;
                continue;
              }
              input_tile[i][j] = A[x * MAP_SIZE + y];
            }
          }

          for (int j = 0; j < 4; j++) {
            tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
            tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
            tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
            tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
          }

          for (int i = 0; i < 4; i++) {
            transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
            transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
            transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
            transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
          }

          DATA_TYPE multiplied_tile[4][4];
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              multiplied_tile[i][j] =
                  transformed_tile[i][j] * C[i * 4 + j];
            }
          }

          DATA_TYPE tmp_tile_1[2][4];
          DATA_TYPE final_tile[2][2];

          for (int j = 0; j < 4; j++) {
            tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] +
                               multiplied_tile[2][j];
            tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] -
                               multiplied_tile[3][j];
          }

          for (int i = 0; i < 2; i++) {
            final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] +
                               tmp_tile_1[i][2];
            final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] -
                               tmp_tile_1[i][3];
          }

          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              int x = 2 * (tile_i + offset_i) + i;
              int y = 2 * (tile_j + offset_j) + j;
              if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
                continue;
              }
              B[x * (MAP_SIZE - 2) + y] = final_tile[i][j];
            }
          }
        }
      }
    }

    if (cpu_run) {
      WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);
    }

    co_time += rtclock() - co_start;

#ifdef VERBOSE
    if (cpu_run)
      printf("run on host\n");
    if (gpu_run)
      printf("run on device\n");
    printf("CPU workload size : %d\n", cpu_offset);
#endif

    WinogradConv2D_2x2(A, B_host, C);
    pass &= compareResults(B_host, B);
  }

  printf("%s\n", pass ? "PASS" : "FAIL");

  const size_t output_elems = (MAP_SIZE - 2) * (MAP_SIZE - 2);
  GATE_STATS_F32("B_host", B_host, output_elems);
  GATE_STATS_F32("B", B, output_elems);

  free(A);
  free(B);
  free(B_host);
  free(C);

  double end = rtclock();
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (end - start));

  return 0;
}
