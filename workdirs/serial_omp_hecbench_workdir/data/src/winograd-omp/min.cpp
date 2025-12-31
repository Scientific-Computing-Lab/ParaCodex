#include <cmath>
#include <cstdlib>
#define MAP_SIZE 512
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8
float A[MAP_SIZE*MAP_SIZE];
float B[(MAP_SIZE-2)*(MAP_SIZE-2)];
float C[16];
int main(){
  const size_t input_elems = MAP_SIZE*MAP_SIZE;
  const size_t filter_elems = 16;
  const size_t output_elems = (MAP_SIZE-2)*(MAP_SIZE-2);
  const int tile_n = (MAP_SIZE-2 + 1)/2;
  size_t globalWorkSize0 = ((size_t)ceil((float)tile_n / (float)DIM_LOCAL_WORK_GROUP_X))*DIM_LOCAL_WORK_GROUP_X;
  size_t globalWorkSize1 = ((size_t)ceil((float)tile_n / (float)DIM_LOCAL_WORK_GROUP_Y))*DIM_LOCAL_WORK_GROUP_Y;
  size_t cpu_global_size0 = 0; 
  size_t gpu_global_size0 = globalWorkSize0 - cpu_global_size0;
  size_t gpu_global_size1 = globalWorkSize1;
  size_t global_offset0 = cpu_global_size0;
  size_t global_offset1 = 0;
  const int tile_i_size = gpu_global_size0;
  const int tile_j_size = gpu_global_size1;
  const int offset_i = global_offset0;
  const int offset_j = global_offset1;
#pragma omp target data map(to:A[0:input_elems],C[0:filter_elems]) map(alloc:B[0:output_elems])
  {
#pragma omp target teams distribute parallel for collapse(2) \
    map(present:A[0:input_elems],C[0:filter_elems],B[0:output_elems]) \
    firstprivate(tile_i_size,tile_j_size,offset_i,offset_j)
    for(int tile_j=0; tile_j<tile_j_size; ++tile_j){
      for(int tile_i=0; tile_i<tile_i_size; ++tile_i){
        float input_tile[4][4];
        float tmp_tile[4][4];
        float transformed_tile[4][4];
        for (int i=0;i<4;i++){
          for(int j=0;j<4;j++){
            int x = 2*(tile_i+offset_i)+i;
            int y = 2*(tile_j+offset_j)+j;
            if (x >= MAP_SIZE || y >= MAP_SIZE){
              input_tile[i][j] = 0;
              continue;
            }
            input_tile[i][j] = A[x*MAP_SIZE+y];
          }
        }
        for(int j=0;j<4;j++){
          tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
          tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
          tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
          tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
        }
        for(int i=0;i<4;i++){
          transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
          transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
          transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
          transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
        }
        float multiplied_tile[4][4];
        for(int i=0;i<4;i++){
          for(int j=0;j<4;j++){
            multiplied_tile[i][j] = transformed_tile[i][j] * C[i*4+j];
          }
        }
        float tmp_tile_1[2][4];
        float final_tile[2][2];
        for(int j=0;j<4;j++){
          tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
          tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
        }
        for(int i=0;i<2;i++){
          final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
          final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
        }
        for (int i=0;i<2;i++){
          for(int j=0;j<2;j++){
            int x = 2*(tile_i+offset_i)+i;
            int y = 2*(tile_j+offset_j)+j;
            if (x >= MAP_SIZE-2 || y >= MAP_SIZE-2) continue;
            B[x*(MAP_SIZE-2)+y] = final_tile[i][j];
          }
        }
      }
    }
  }
  return 0;
}
