#include <omp.h>
#include <cstdlib>
#include <vector>

#include "gate.h"
#include "utils.h"

namespace {

double run_gpu_winograd(const DATA_TYPE* input,
                        DATA_TYPE* output,
                        const DATA_TYPE* transformed_filter) {
  const int map_size = MAP_SIZE;
  const int out_map_size = map_size - 2;
  const int tile_n = (out_map_size + 1) / 2;
  const size_t input_elems = static_cast<size_t>(map_size) * map_size;
  const size_t output_elems = static_cast<size_t>(out_map_size) * out_map_size;
  const size_t filter_elems = 16;
  const int team_size = DIM_LOCAL_WORK_GROUP_X * DIM_LOCAL_WORK_GROUP_Y;
  const bool has_device = omp_get_num_devices() > 0;

  double kernel_time = 0.0;

  #pragma omp target data map(to: input[0:input_elems], transformed_filter[0:filter_elems]) \
                          map(from: output[0:output_elems]) if (has_device)
  {
    double kernel_start = rtclock();

    #pragma omp target teams distribute parallel for collapse(2) thread_limit(team_size) if (has_device)
    for (int tile_i = 0; tile_i < tile_n; ++tile_i) {
      for (int tile_j = 0; tile_j < tile_n; ++tile_j) {
        DATA_TYPE input_tile[4][4];
        DATA_TYPE tmp_tile[4][4];
        DATA_TYPE transformed_tile[4][4];
        DATA_TYPE multiplied_tile[4][4];
        DATA_TYPE tmp_tile_1[2][4];
        DATA_TYPE final_tile[2][2];

        const int base_x = tile_i << 1;
        const int base_y = tile_j << 1;

        for (int i = 0; i < 4; ++i) {
          const int x = base_x + i;
          const bool valid_x = x < map_size;
          const int row_offset = valid_x ? x * map_size : 0;
          for (int j = 0; j < 4; ++j) {
            const int y = base_y + j;
            if (valid_x && y < map_size) {
              input_tile[i][j] = input[row_offset + y];
            } else {
              input_tile[i][j] = 0.0f;
            }
          }
        }

        for (int j = 0; j < 4; ++j) {
          const DATA_TYPE d0 = input_tile[0][j];
          const DATA_TYPE d1 = input_tile[1][j];
          const DATA_TYPE d2 = input_tile[2][j];
          const DATA_TYPE d3 = input_tile[3][j];
          tmp_tile[0][j] = d0 - d2;
          tmp_tile[1][j] = d1 + d2;
          tmp_tile[2][j] = -d1 + d2;
          tmp_tile[3][j] = d1 - d3;
        }

        for (int i = 0; i < 4; ++i) {
          const DATA_TYPE t0 = tmp_tile[i][0];
          const DATA_TYPE t1 = tmp_tile[i][1];
          const DATA_TYPE t2 = tmp_tile[i][2];
          const DATA_TYPE t3 = tmp_tile[i][3];
          transformed_tile[i][0] = t0 - t2;
          transformed_tile[i][1] = t1 + t2;
          transformed_tile[i][2] = -t1 + t2;
          transformed_tile[i][3] = t1 - t3;
        }

        for (int i = 0; i < 4; ++i) {
          const DATA_TYPE* filter_row = transformed_filter + (i << 2);
          for (int j = 0; j < 4; ++j) {
            multiplied_tile[i][j] = transformed_tile[i][j] * filter_row[j];
          }
        }

        for (int j = 0; j < 4; ++j) {
          const DATA_TYPE m0 = multiplied_tile[0][j];
          const DATA_TYPE m1 = multiplied_tile[1][j];
          const DATA_TYPE m2 = multiplied_tile[2][j];
          const DATA_TYPE m3 = multiplied_tile[3][j];
          tmp_tile_1[0][j] = m0 + m1 + m2;
          tmp_tile_1[1][j] = m1 - m2 - m3;
        }

        for (int i = 0; i < 2; ++i) {
          const DATA_TYPE v0 = tmp_tile_1[i][0];
          const DATA_TYPE v1 = tmp_tile_1[i][1];
          const DATA_TYPE v2 = tmp_tile_1[i][2];
          const DATA_TYPE v3 = tmp_tile_1[i][3];
          final_tile[i][0] = v0 + v1 + v2;
          final_tile[i][1] = v1 - v2 - v3;
        }

        for (int i = 0; i < 2; ++i) {
          const int ox = base_x + i;
          if (ox >= out_map_size) continue;
          const int out_row = ox * out_map_size;
          for (int j = 0; j < 2; ++j) {
            const int oy = base_y + j;
            if (oy >= out_map_size) continue;
            output[out_row + oy] = final_tile[i][j];
          }
        }
      }
    }

    kernel_time = rtclock() - kernel_start;
  }

  return kernel_time;
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  const int map_size = MAP_SIZE;
  const int out_map_size = map_size - 2;
  const size_t input_elems = static_cast<size_t>(map_size) * map_size;
  const size_t output_elems = static_cast<size_t>(out_map_size) * out_map_size;

  std::vector<DATA_TYPE> A(input_elems);
  std::vector<DATA_TYPE> B(output_elems, 0.0f);
  std::vector<DATA_TYPE> B_host(output_elems, 0.0f);
  std::vector<DATA_TYPE> C(16);

  srand(0);
  for (size_t idx = 0; idx < input_elems; ++idx) {
    A[idx] = static_cast<DATA_TYPE>(rand()) / static_cast<DATA_TYPE>(RAND_MAX);
  }

  WinogradConv2D_2x2_filter_transformation(C.data());

  double start = rtclock();

  double co_time = run_gpu_winograd(A.data(), B.data(), C.data());

  double end = rtclock();

  WinogradConv2D_2x2(A.data(), B_host.data(), C.data());

  bool pass = compareResults(B_host.data(), B.data());

  GATE_STATS_F32("winograd/B", B.data(), output_elems);
  GATE_STATS_F32("winograd/B_host", B_host.data(), output_elems);

  printf("%s\n", pass ? "PASS" : "FAIL");
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (end - start));

  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
