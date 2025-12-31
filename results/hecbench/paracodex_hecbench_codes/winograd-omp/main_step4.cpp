#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "gate.h"
#include "utils.h"

static inline bool compare_host_segment(const DATA_TYPE* reference,
                                        const DATA_TYPE* candidate,
                                        size_t start_index,
                                        size_t count) {
  for (size_t idx = 0; idx < count; ++idx) {
    const size_t pos = start_index + idx;
    if (percentDiff(reference[pos], candidate[pos]) >
        PERCENT_DIFF_ERROR_THRESHOLD) {
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
  double start = rtclock();

  DATA_TYPE* A =
      static_cast<DATA_TYPE*>(malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE)));
  DATA_TYPE* B_host = static_cast<DATA_TYPE*>(
      malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE)));
  DATA_TYPE* B = static_cast<DATA_TYPE*>(
      malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE)));
  DATA_TYPE* C = static_cast<DATA_TYPE*>(malloc(4 * 4 * sizeof(DATA_TYPE)));
  DATA_TYPE* random_values =
      static_cast<DATA_TYPE*>(malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE)));

  const size_t input_elems =
      static_cast<size_t>(MAP_SIZE) * static_cast<size_t>(MAP_SIZE);
  const size_t output_elems = static_cast<size_t>(MAP_SIZE - 2) *
                              static_cast<size_t>(MAP_SIZE - 2);
  const size_t filter_elems = 4 * 4;

  for (int i = 0; i < MAP_SIZE; ++i) {
    for (int j = 0; j < MAP_SIZE; ++j) {
      const DATA_TYPE value = rand() / static_cast<float>(RAND_MAX);
      random_values[i * MAP_SIZE + j] = value;
      A[i * MAP_SIZE + j] = value;
    }
  }

  WinogradConv2D_2x2_filter_transformation(C);

  const int out_map_size = MAP_SIZE - 2;
  const int tile_n = (out_map_size + 1) / 2;

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

  // Precompute the golden output once to reuse during validation.
  WinogradConv2D_2x2(A, B_host, C);

  const int device_count = omp_get_num_devices();
  const bool has_device = device_count > 0;

  if (has_device) {
    const int sm_count_estimate = 36;  // RTX 4060 Laptop GPU (Ada SM89)
    // Shape the team/thread mix so the Ada GPU sees enough concurrent work
    // without oversubscribing small tiles.

    // Keep the repeatedly used inputs resident on the GPU and reuse the output
    // buffer without remapping each launch.
#pragma omp target data map(to: A[0:input_elems], C[0:filter_elems], \
                                B_host[0:output_elems]) \
    map(alloc: B[0:output_elems])
    {
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

        const bool cpu_run = cpu_global_size[0] > 0;
        const bool gpu_run = gpu_global_size[0] > 0;

        const int cpu_rows_candidate =
            2 * static_cast<int>(cpu_global_size[0]);
        const int cpu_rows =
            (cpu_rows_candidate < out_map_size) ? cpu_rows_candidate
                                                : out_map_size;
        const int gpu_rows = out_map_size - cpu_rows;
        const size_t gpu_row_offset =
            static_cast<size_t>(cpu_rows) * static_cast<size_t>(out_map_size);
        const size_t device_gpu_elems_sz =
            static_cast<size_t>(gpu_rows) * static_cast<size_t>(out_map_size);

        double co_start = rtclock();

        if (gpu_run) {
          const int total_gpu_tiles = tile_i_size * tile_j_size;
          int gpu_thread_limit = 128;
          if (total_gpu_tiles < sm_count_estimate * 4) {
            gpu_thread_limit = 64;
          }
          int gpu_num_teams =
              (total_gpu_tiles + gpu_thread_limit - 1) / gpu_thread_limit;
          if (gpu_num_teams < 1) {
            gpu_num_teams = 1;
          }
          const int gpu_occupancy_target = sm_count_estimate * 6;
          if (total_gpu_tiles >= gpu_occupancy_target &&
              gpu_num_teams < gpu_occupancy_target) {
            gpu_num_teams = gpu_occupancy_target;
          }
          if (total_gpu_tiles > 0 && gpu_num_teams > total_gpu_tiles) {
            gpu_num_teams = total_gpu_tiles;
          }
          // Bias toward many resident teams with up to four warps per team to
          // improve latency hiding on the Ada Lovelace (SM89) GPU while
          // shrinking the team size when the GPU slice is small.
#pragma omp target teams distribute parallel for collapse(2) \
        num_teams(gpu_num_teams) \
        thread_limit(gpu_thread_limit) \
        map(present: A[0:input_elems], C[0:filter_elems], \
                     B[0:output_elems]) \
        firstprivate(tile_i_size, tile_j_size, offset_i, offset_j)
          for (int tile_i = 0; tile_i < tile_i_size; tile_i++) {
            for (int tile_j = 0; tile_j < tile_j_size; tile_j++) {
              // Favor contiguous memory access by marching along output columns
              // inside the inner loop and by using a fast path for interior tiles.
              const int base_x = 2 * (tile_i + offset_i);
              const int base_y = 2 * (tile_j + offset_j);
              const bool full_tile =
                  (base_x + 3) < MAP_SIZE && (base_y + 3) < MAP_SIZE;
              const bool full_output =
                  base_x <= (MAP_SIZE - 4) && base_y <= (MAP_SIZE - 4);

              DATA_TYPE input_tile[4][4];
              DATA_TYPE work_tile[4][4];

              if (full_tile) {
                const size_t input_stride = MAP_SIZE;
                const size_t input_base =
                    static_cast<size_t>(base_x) * input_stride +
                    static_cast<size_t>(base_y);
                const DATA_TYPE* tile_ptr = A + input_base;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  const DATA_TYPE* row_ptr =
                      tile_ptr + static_cast<size_t>(i) * input_stride;
                  input_tile[i][0] = row_ptr[0];
                  input_tile[i][1] = row_ptr[1];
                  input_tile[i][2] = row_ptr[2];
                  input_tile[i][3] = row_ptr[3];
                }
              } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  const int x = base_x + i;
                  const bool x_valid = x < MAP_SIZE;
#pragma unroll
                  for (int j = 0; j < 4; ++j) {
                    const int y = base_y + j;
                    input_tile[i][j] =
                        (x_valid && y < MAP_SIZE)
                            ? A[static_cast<size_t>(x) * MAP_SIZE +
                                static_cast<size_t>(y)]
                            : static_cast<DATA_TYPE>(0);
                  }
                }
              }

#pragma unroll
              for (int j = 0; j < 4; ++j) {
                const DATA_TYPE r0 = input_tile[0][j];
                const DATA_TYPE r1 = input_tile[1][j];
                const DATA_TYPE r2 = input_tile[2][j];
                const DATA_TYPE r3 = input_tile[3][j];
                work_tile[0][j] = r0 - r2;
                work_tile[1][j] = r1 + r2;
                work_tile[2][j] = r2 - r1;
                work_tile[3][j] = r1 - r3;
              }

#pragma unroll
              for (int i = 0; i < 4; ++i) {
                const DATA_TYPE t0 = work_tile[i][0];
                const DATA_TYPE t1 = work_tile[i][1];
                const DATA_TYPE t2 = work_tile[i][2];
                const DATA_TYPE t3 = work_tile[i][3];

                const DATA_TYPE u0 = t0 - t2;
                const DATA_TYPE u1 = t1 + t2;
                const DATA_TYPE u2 = t2 - t1;
                const DATA_TYPE u3 = t1 - t3;

                work_tile[i][0] = u0 * C[i * 4 + 0];
                work_tile[i][1] = u1 * C[i * 4 + 1];
                work_tile[i][2] = u2 * C[i * 4 + 2];
                work_tile[i][3] = u3 * C[i * 4 + 3];
              }

              const DATA_TYPE s0 = work_tile[0][0] + work_tile[1][0] +
                                   work_tile[2][0];
              const DATA_TYPE s1 = work_tile[0][1] + work_tile[1][1] +
                                   work_tile[2][1];
              const DATA_TYPE s2 = work_tile[0][2] + work_tile[1][2] +
                                   work_tile[2][2];
              const DATA_TYPE s3 = work_tile[0][3] + work_tile[1][3] +
                                   work_tile[2][3];

              const DATA_TYPE t0 = work_tile[1][0] - work_tile[2][0] -
                                   work_tile[3][0];
              const DATA_TYPE t1 = work_tile[1][1] - work_tile[2][1] -
                                   work_tile[3][1];
              const DATA_TYPE t2 = work_tile[1][2] - work_tile[2][2] -
                                   work_tile[3][2];
              const DATA_TYPE t3 = work_tile[1][3] - work_tile[2][3] -
                                   work_tile[3][3];

              const DATA_TYPE out00 = s0 + s1 + s2;
              const DATA_TYPE out01 = s1 - s2 - s3;
              const DATA_TYPE out10 = t0 + t1 + t2;
              const DATA_TYPE out11 = t1 - t2 - t3;

              if (full_output) {
                const size_t out_stride = MAP_SIZE - 2;
                const size_t out_base =
                    static_cast<size_t>(base_x) * out_stride +
                    static_cast<size_t>(base_y);
                DATA_TYPE* out_ptr = B + out_base;
                out_ptr[0] = out00;
                out_ptr[1] = out01;
                out_ptr += out_stride;
                out_ptr[0] = out10;
                out_ptr[1] = out11;
              } else {
                const int out_limit = MAP_SIZE - 2;
                if (base_x < out_limit && base_y < out_limit) {
                  B[static_cast<size_t>(base_x) * out_limit +
                    static_cast<size_t>(base_y)] = out00;
                }
                if (base_x < out_limit && (base_y + 1) < out_limit) {
                  B[static_cast<size_t>(base_x) * out_limit +
                    static_cast<size_t>(base_y + 1)] = out01;
                }
                if ((base_x + 1) < out_limit && base_y < out_limit) {
                  B[static_cast<size_t>(base_x + 1) * out_limit +
                    static_cast<size_t>(base_y)] = out10;
                }
                if ((base_x + 1) < out_limit && (base_y + 1) < out_limit) {
                  B[static_cast<size_t>(base_x + 1) * out_limit +
                    static_cast<size_t>(base_y + 1)] = out11;
                }
              }
            }
          }
        }

        if (gpu_run && gpu_rows > 0) {
#pragma omp target update from(B[gpu_row_offset:device_gpu_elems_sz])
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

        bool iteration_pass = true;

        if (gpu_run && gpu_rows > 0) {
          int mismatch_flag = 0;
          const int device_cols = out_map_size;
          const int device_gpu_elems = gpu_rows * device_cols;
          const size_t device_row_offset = gpu_row_offset;
          int check_thread_limit = 64;
          if (device_gpu_elems >= sm_count_estimate * 12) {
            check_thread_limit = 128;
          }
          int check_num_teams =
              (device_gpu_elems + check_thread_limit - 1) / check_thread_limit;
          if (check_num_teams < 1) {
            check_num_teams = 1;
          }
          const int check_occupancy_target = sm_count_estimate * 6;
          if (device_gpu_elems >= check_occupancy_target &&
              check_num_teams < check_occupancy_target) {
            check_num_teams = check_occupancy_target;
          }
          if (device_gpu_elems > 0 && check_num_teams > device_gpu_elems) {
            check_num_teams = device_gpu_elems;
          }
#pragma omp target teams distribute parallel for \
        num_teams(check_num_teams) \
        thread_limit(check_thread_limit) \
        map(present: B[0:output_elems], B_host[0:output_elems]) \
        map(tofrom: mismatch_flag) \
        firstprivate(device_cols, device_gpu_elems, device_row_offset)
          for (int elem = 0; elem < device_gpu_elems; ++elem) {
            const int row = elem / device_cols;
            const int col = elem - row * device_cols;
            const size_t idx =
                device_row_offset +
                static_cast<size_t>(row) * static_cast<size_t>(device_cols) +
                static_cast<size_t>(col);
            const DATA_TYPE gpu_val = B[idx];
            const DATA_TYPE ref_val = B_host[idx];
            DATA_TYPE percent = 0.0f;
            const DATA_TYPE abs_gpu = (gpu_val < static_cast<DATA_TYPE>(0))
                                          ? -gpu_val
                                          : gpu_val;
            const DATA_TYPE abs_ref = (ref_val < static_cast<DATA_TYPE>(0))
                                          ? -ref_val
                                          : ref_val;
            if (!((abs_gpu < static_cast<DATA_TYPE>(0.01f)) &&
                  (abs_ref < static_cast<DATA_TYPE>(0.01f)))) {
              DATA_TYPE numerator = gpu_val - ref_val;
              if (numerator < static_cast<DATA_TYPE>(0)) {
                numerator = -numerator;
              }
              DATA_TYPE denominator =
                  gpu_val + static_cast<DATA_TYPE>(SMALL_FLOAT_VAL);
              if (denominator < static_cast<DATA_TYPE>(0)) {
                denominator = -denominator;
              }
              percent = static_cast<DATA_TYPE>(100.0f) *
                        (numerator / denominator);
              if (percent < static_cast<DATA_TYPE>(0)) {
                percent = -percent;
              }
            }
            if (percent > PERCENT_DIFF_ERROR_THRESHOLD) {
#pragma omp atomic write
              mismatch_flag = 1;
            }
          }
          if (mismatch_flag != 0) {
            iteration_pass = false;
          }
        }

        if (cpu_run && cpu_rows > 0) {
          const size_t cpu_elem_count =
              static_cast<size_t>(cpu_rows) * static_cast<size_t>(out_map_size);
          if (!compare_host_segment(B_host, B, 0, cpu_elem_count)) {
            iteration_pass = false;
          }
        }

        pass &= iteration_pass;
      }
    }
  } else {
    for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {
      cpu_global_size[0] = globalWorkSize[0];
      cpu_global_size[1] = globalWorkSize[1];
      gpu_global_size[0] = 0;
      gpu_global_size[1] = 0;

      double co_start = rtclock();

      WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);

      co_time += rtclock() - co_start;

#ifdef VERBOSE
      printf("run on host\n");
      printf("CPU workload size : %d\n", cpu_offset);
#endif

      pass &= compareResults(B_host, B);
    }
  }

  printf("%s\n", pass ? "PASS" : "FAIL");

  GATE_STATS_F32("B_host", B_host, output_elems);
  GATE_STATS_F32("B", B, output_elems);

  free(A);
  free(B);
  free(B_host);
  free(C);
  free(random_values);

  double end = rtclock();
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (end - start));

  return 0;
}
