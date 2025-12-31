


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

#include <iostream>

#define WARMUPS         1000
#define NSPEEDS         9
#define LOCALSIZEX      128
#define LOCALSIZEY      1



#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"



typedef struct
{
  int   nx;            

  int   ny;            

  int   maxIters;      

  int   reynolds_dim;  

  float density;       

  float accel;         

  float omega;         

} t_param;



typedef struct
{
  float speeds[NSPEEDS];
} t_speed;






int initialise(const char* paramfile, const char* obstaclefile,
    t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
    int** obstacles_ptr, float** av_vels_ptr);



int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);



int finalise(t_speed* cells_ptr, t_speed* tmp_cells_ptr,
    int* obstacles_ptr, float* av_vels_ptr);



float total_density(const t_param params, t_speed* cells);



float av_velocity(const t_param params, t_speed* cells, int* obstacles);



float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);



void die(const char* message, const int line, const char* file);
void usage(const char* exe);


bool 
isGreater(const float x, const float y) 
{
  return x > y ? 1 : 0;
}

int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    

  char*    obstaclefile = NULL; 

  t_param  params;              

  t_speed* cells     = NULL;    

  t_speed* tmp_cells = NULL;    

  int*     obstacles = NULL;

  float*   av_vels   = NULL;    

  struct timeval timstr;        

  double tic, toc;              


  

  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  

  initialise(paramfile, obstaclefile, &params, &cells, 
      &tmp_cells, &obstacles, &av_vels);

  

  unsigned int Ny = params.ny;
  unsigned int Nx = params.nx;
  unsigned int MaxIters = params.maxIters;

  float *speeds0 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *speeds1 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *speeds2 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *speeds3 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *speeds4 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *speeds5 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *speeds6 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *speeds7 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *speeds8 = (float*) malloc (sizeof(float) * Ny*Nx);

  float *tmp_speeds0 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *tmp_speeds1 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *tmp_speeds2 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *tmp_speeds3 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *tmp_speeds4 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *tmp_speeds5 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *tmp_speeds6 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *tmp_speeds7 = (float*) malloc (sizeof(float) * Ny*Nx);
  float *tmp_speeds8 = (float*) malloc (sizeof(float) * Ny*Nx);

  float *tot_up = (float*) malloc (sizeof(float) * (Ny/LOCALSIZEY) * (Nx/LOCALSIZEX) * MaxIters);
  int *tot_cellsp = (int*) malloc (sizeof(int) * (Ny/LOCALSIZEY) * (Nx/LOCALSIZEX) * MaxIters);

  

  

  for (int jj = 0; jj < Ny; jj++)
  {
    for (int ii = 0; ii < Nx; ii++)
    {
      speeds0[ii + jj*Nx] = cells[ii + jj*Nx].speeds[0];
      speeds1[ii + jj*Nx] = cells[ii + jj*Nx].speeds[1];
      speeds2[ii + jj*Nx] = cells[ii + jj*Nx].speeds[2];
      speeds3[ii + jj*Nx] = cells[ii + jj*Nx].speeds[3];
      speeds4[ii + jj*Nx] = cells[ii + jj*Nx].speeds[4];
      speeds5[ii + jj*Nx] = cells[ii + jj*Nx].speeds[5];
      speeds6[ii + jj*Nx] = cells[ii + jj*Nx].speeds[6];
      speeds7[ii + jj*Nx] = cells[ii + jj*Nx].speeds[7];
      speeds8[ii + jj*Nx] = cells[ii + jj*Nx].speeds[8];
    }
  }

  

  float omega = params.omega;
  float densityaccel = params.density*params.accel;

  int teams = Nx / LOCALSIZEX * Ny / LOCALSIZEY;
  int threads = LOCALSIZEX * LOCALSIZEY;

  {

  

  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec * 1e6 + timstr.tv_usec;

  for (int tt = 0; tt < MaxIters; tt++) {
    if (tt == WARMUPS - 1) {
      

      gettimeofday(&timstr, NULL);
      tic = timstr.tv_sec * 1e6 + timstr.tv_usec;
    }
        {
      float local_sum[LOCALSIZEX*LOCALSIZEY];
      float local_sum2[LOCALSIZEX*LOCALSIZEY];
            {
        const int lid = omp_get_thread_num();  
        const int tid = omp_get_team_num();  
        const int dim = omp_get_num_threads();  
        const int gid = dim * tid + lid;

        

        const int ii = gid % Nx;
        const int jj = gid / Nx;

        const float c_sq_inv = 3.f;
        const float c_sq = 1.f/c_sq_inv; 

        const float temp1 = 4.5f;
        const float w1 = 1.f/9.f;
        const float w0 = 4.f * w1;  

        const float w2 = 1.f/36.f; 

        const float w11 = densityaccel * w1;
        const float w21 = densityaccel * w2;

        

        const int y_n = (jj + 1) % Ny;
        const int x_e = (ii + 1) % Nx;
        const int y_s = (jj == 0) ? (jj + Ny - 1) : (jj - 1);
        const int x_w = (ii == 0) ? (ii + Nx - 1) : (ii - 1);

        


        float tmp_s0 = speeds0[ii + jj*Nx];
        float tmp_s1 = (jj == Ny-2 && (!obstacles[x_w + jj*Nx] && isGreater((speeds3[x_w + jj*Nx] - w11) , 0.f) && isGreater((speeds6[x_w + jj*Nx] - w21) , 0.f) && isGreater((speeds7[x_w + jj*Nx] - w21) , 0.f))) ? speeds1[x_w + jj*Nx]+w11 : speeds1[x_w + jj*Nx];
        float tmp_s2 = speeds2[ii + y_s*Nx];
        float tmp_s3 = (jj == Ny-2 && (!obstacles[x_e + jj*Nx] && isGreater((speeds3[x_e + jj*Nx] - w11) , 0.f) && isGreater((speeds6[x_e + jj*Nx] - w21) , 0.f) && isGreater((speeds7[x_e + jj*Nx] - w21) , 0.f))) ? speeds3[x_e + jj*Nx]-w11 : speeds3[x_e + jj*Nx];
        float tmp_s4 = speeds4[ii + y_n*Nx];
        float tmp_s5 = (y_s == Ny-2 && (!obstacles[x_w + y_s*Nx] && isGreater((speeds3[x_w + y_s*Nx] - w11) , 0.f) && isGreater((speeds6[x_w + y_s*Nx] - w21) , 0.f) && isGreater((speeds7[x_w + y_s*Nx] - w21) , 0.f))) ? speeds5[x_w + y_s*Nx]+w21 : speeds5[x_w + y_s*Nx];
        float tmp_s6 = (y_s == Ny-2 && (!obstacles[x_e + y_s*Nx] && isGreater((speeds3[x_e + y_s*Nx] - w11) , 0.f) && isGreater((speeds6[x_e + y_s*Nx] - w21) , 0.f) && isGreater((speeds7[x_e + y_s*Nx] - w21) , 0.f))) ? speeds6[x_e + y_s*Nx]-w21 : speeds6[x_e + y_s*Nx];
        float tmp_s7 = (y_n == Ny-2 && (!obstacles[x_e + y_n*Nx] && isGreater((speeds3[x_e + y_n*Nx] - w11) , 0.f) && isGreater((speeds6[x_e + y_n*Nx] - w21) , 0.f) && isGreater((speeds7[x_e + y_n*Nx] - w21) , 0.f))) ? speeds7[x_e + y_n*Nx]-w21 : speeds7[x_e + y_n*Nx];
        float tmp_s8 = (y_n == Ny-2 && (!obstacles[x_w + y_n*Nx] && isGreater((speeds3[x_w + y_n*Nx] - w11) , 0.f) && isGreater((speeds6[x_w + y_n*Nx] - w21) , 0.f) && isGreater((speeds7[x_w + y_n*Nx] - w21) , 0.f))) ? speeds8[x_w + y_n*Nx]+w21 : speeds8[x_w + y_n*Nx];

        

        float local_density = tmp_s0 + tmp_s1 + tmp_s2 + tmp_s3 + tmp_s4  + tmp_s5  + tmp_s6  + tmp_s7  + tmp_s8;
        const float local_density_recip = 1.f/(local_density);
        

        float u_x = (tmp_s1
            + tmp_s5
            + tmp_s8
            - tmp_s3
            - tmp_s6
            - tmp_s7)
          * local_density_recip;
        

        float u_y = (tmp_s2
            + tmp_s5
            + tmp_s6
            - tmp_s4
            - tmp_s8
            - tmp_s7)
          * local_density_recip;

        

        const float temp2 = - (u_x * u_x + u_y * u_y)/(2.f * c_sq);

        

        float d_equ[NSPEEDS];
        

        d_equ[0] = w0 * local_density
          * (1.f + temp2);
        

        d_equ[1] = w1 * local_density * (1.f + u_x * c_sq_inv
            + (u_x * u_x) * temp1
            + temp2);
        d_equ[2] = w1 * local_density * (1.f + u_y * c_sq_inv
            + (u_y * u_y) * temp1
            + temp2);
        d_equ[3] = w1 * local_density * (1.f - u_x * c_sq_inv
            + (u_x * u_x) * temp1
            + temp2);
        d_equ[4] = w1 * local_density * (1.f - u_y * c_sq_inv
            + (u_y * u_y) * temp1
            + temp2);
        

        d_equ[5] = w2 * local_density * (1.f + (u_x + u_y) * c_sq_inv
            + ((u_x + u_y) * (u_x + u_y)) * temp1
            + temp2);
        d_equ[6] = w2 * local_density * (1.f + (-u_x + u_y) * c_sq_inv
            + ((-u_x + u_y) * (-u_x + u_y)) * temp1
            + temp2);
        d_equ[7] = w2 * local_density * (1.f + (-u_x - u_y) * c_sq_inv
            + ((-u_x - u_y) * (-u_x - u_y)) * temp1
            + temp2);
        d_equ[8] = w2 * local_density * (1.f + (u_x - u_y) * c_sq_inv
            + ((u_x - u_y) * (u_x - u_y)) * temp1
            + temp2);

        float tmp;
        int expression = obstacles[ii + jj*Nx];
        tmp_s0 = expression ? tmp_s0 : (tmp_s0 + omega * (d_equ[0] - tmp_s0));
        tmp = tmp_s1;
        tmp_s1 = expression ? tmp_s3 : (tmp_s1 + omega * (d_equ[1] - tmp_s1));
        tmp_s3 = expression ? tmp : (tmp_s3 + omega * (d_equ[3] - tmp_s3));
        tmp = tmp_s2;
        tmp_s2 = expression ? tmp_s4 : (tmp_s2 + omega * (d_equ[2] - tmp_s2));
        tmp_s4 = expression ? tmp : (tmp_s4 + omega * (d_equ[4] - tmp_s4));
        tmp = tmp_s5;
        tmp_s5 = expression ? tmp_s7 : (tmp_s5 + omega * (d_equ[5] - tmp_s5));
        tmp_s7 = expression ? tmp : (tmp_s7 + omega * (d_equ[7] - tmp_s7));
        tmp = tmp_s6;
        tmp_s6 = expression ? tmp_s8 : (tmp_s6 + omega * (d_equ[6] - tmp_s6));
        tmp_s8 = expression ? tmp : (tmp_s8 + omega * (d_equ[8] - tmp_s8));

        

        local_density = 1.f/((tmp_s0 + tmp_s1 + tmp_s2 + tmp_s3 + tmp_s4 + tmp_s5 + tmp_s6 + tmp_s7 + tmp_s8));

        

        u_x = (tmp_s1
            + tmp_s5
            + tmp_s8
            - tmp_s3
            - tmp_s6
            - tmp_s7)
          * local_density;
        

        u_y = (tmp_s2
            + tmp_s5
            + tmp_s6
            - tmp_s4
            - tmp_s7
            - tmp_s8)
          * local_density;

        tmp_speeds0[ii + jj*Nx] = tmp_s0;
        tmp_speeds1[ii + jj*Nx] = tmp_s1;
        tmp_speeds2[ii + jj*Nx] = tmp_s2;
        tmp_speeds3[ii + jj*Nx] = tmp_s3;
        tmp_speeds4[ii + jj*Nx] = tmp_s4;
        tmp_speeds5[ii + jj*Nx] = tmp_s5;
        tmp_speeds6[ii + jj*Nx] = tmp_s6;
        tmp_speeds7[ii + jj*Nx] = tmp_s7;
        tmp_speeds8[ii + jj*Nx] = tmp_s8;


        int local_idi = lid % LOCALSIZEX; 
        int local_idj = lid / LOCALSIZEX;
        int local_sizei = LOCALSIZEX;
        int local_sizej = LOCALSIZEY;
        

        local_sum[local_idi + local_idj*local_sizei] = (obstacles[ii + jj*Nx]) ? 0 : hypotf(u_x,u_y);
        

        local_sum2[local_idi + local_idj*local_sizei] = (obstacles[ii + jj*Nx]) ? 0 : 1 ;


        int group_id = tid % (Nx/LOCALSIZEX);
        int group_id2 = tid / (Nx/LOCALSIZEX);
        int group_size = (Nx/LOCALSIZEX);
        int group_size2 = (Ny/LOCALSIZEY);
        if(local_idi == 0 && local_idj == 0){
          float sum = 0.0f;
          int sum2 = 0;
          for(int i = 0; i<local_sizei*local_sizej; i++){
            sum += local_sum[i];
            sum2 += local_sum2[i];
          }
          tot_up[group_id+group_id2*group_size+tt*group_size*group_size2] = sum;
          tot_cellsp[group_id+group_id2*group_size+tt*group_size*group_size2] = sum2;
        }
      }
    }

    float* speed_tmp = speeds0;
    speeds0 = tmp_speeds0;
    tmp_speeds0 = speed_tmp;

    speed_tmp = speeds1;
    speeds1 = tmp_speeds1;
    tmp_speeds1 = speed_tmp;

    speed_tmp = speeds2;
    speeds2 = tmp_speeds2;
    tmp_speeds2 = speed_tmp;

    speed_tmp = speeds3;
    speeds3 = tmp_speeds3;
    tmp_speeds3 = speed_tmp;

    speed_tmp = speeds4;
    speeds4 = tmp_speeds4;
    tmp_speeds4 = speed_tmp;

    speed_tmp = speeds5;
    speeds5 = tmp_speeds5;
    tmp_speeds5 = speed_tmp;

    speed_tmp = speeds6;
    speeds6 = tmp_speeds6;
    tmp_speeds6 = speed_tmp;

    speed_tmp = speeds7;
    speeds7 = tmp_speeds7;
    tmp_speeds7 = speed_tmp;

    speed_tmp = speeds8;
    speeds8 = tmp_speeds8;
    tmp_speeds8 = speed_tmp;
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec * 1e6 + timstr.tv_usec;
  printf("After warmup for %d iterations, ", WARMUPS);
  printf("average kernel execution time over %d iterations:\t\t\t%.6lf (us)\n",
         MaxIters - WARMUPS, (toc - tic) / (MaxIters - WARMUPS));

  } 


  float tot_u = 0;
  int tot_cells = 0;
  for (int tt = 0; tt < MaxIters; tt++){
    tot_u = 0;
    tot_cells = 0;
    for(int i = 0; i < Nx/LOCALSIZEX*Ny/LOCALSIZEY; i++){
      tot_u += tot_up[i+tt*Nx/LOCALSIZEX*Ny/LOCALSIZEY];
      tot_cells += tot_cellsp[i+tt*Nx/LOCALSIZEX*Ny/LOCALSIZEY];
      

    }
    av_vels[tt] = tot_u/tot_cells;
  }

  

  for (int jj = 0; jj < Ny; jj++)
  {
    for (int ii = 0; ii < Nx; ii++)
    {
      cells[ii + jj*Nx].speeds[0] = speeds0[ii + jj*Nx];
      cells[ii + jj*Nx].speeds[1] = speeds1[ii + jj*Nx];
      cells[ii + jj*Nx].speeds[2] = speeds2[ii + jj*Nx];
      cells[ii + jj*Nx].speeds[3] = speeds3[ii + jj*Nx];
      cells[ii + jj*Nx].speeds[4] = speeds4[ii + jj*Nx];
      cells[ii + jj*Nx].speeds[5] = speeds5[ii + jj*Nx];
      cells[ii + jj*Nx].speeds[6] = speeds6[ii + jj*Nx];
      cells[ii + jj*Nx].speeds[7] = speeds7[ii + jj*Nx];
      cells[ii + jj*Nx].speeds[8] = speeds8[ii + jj*Nx];
    }
  }

  

  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  write_values(params, cells, obstacles, av_vels);
  finalise(cells, tmp_cells, obstacles, av_vels);

  free(speeds0);
  free(speeds1);
  free(speeds2);
  free(speeds3);
  free(speeds4);
  free(speeds5);
  free(speeds6);
  free(speeds7);
  free(speeds8);
  free(tot_up);
  free(tot_cellsp);

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  

  float tot_u;          


  

  tot_u = 0.f;

  

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      

      if (!obstacles[ii + jj*params.nx])
      {
        

        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        

        float u_x = (cells[ii + jj*params.nx].speeds[1]
            + cells[ii + jj*params.nx].speeds[5]
            + cells[ii + jj*params.nx].speeds[8]
            - (cells[ii + jj*params.nx].speeds[3]
              + cells[ii + jj*params.nx].speeds[6]
              + cells[ii + jj*params.nx].speeds[7]))
          / local_density;
        

        float u_y = (cells[ii + jj*params.nx].speeds[2]
            + cells[ii + jj*params.nx].speeds[5]
            + cells[ii + jj*params.nx].speeds[6]
            - (cells[ii + jj*params.nx].speeds[4]
              + cells[ii + jj*params.nx].speeds[7]
              + cells[ii + jj*params.nx].speeds[8]))
          / local_density;
        

        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        

        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
    t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
    int** obstacles_ptr, float** av_vels_ptr){
  char   message[1024];  

  FILE*  fp;             

  int    xx, yy;         

  int    blocked;        

  int    retval;         


  

  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }
  

  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  

  fclose(fp);

  


  

  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  

  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  

  *obstacles_ptr = (int*) malloc (sizeof(int) * params->ny * params->nx);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  

  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      

      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      

      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      

      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  

  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  

  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    

    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    

    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  

  fclose(fp);

  

  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(t_speed* cells_ptr, t_speed* tmp_cells_ptr,
    int* obstacles_ptr, float* av_vels_ptr)
{
  

  free(cells_ptr);
  free(tmp_cells_ptr);
  free(obstacles_ptr);
  free(av_vels_ptr);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  


  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     

  const float c_sq = 1.f / 3.f; 

  float local_density;         

  float pressure;              

  float u_x;                   

  float u_y;                   

  float u;                     


  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      

      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      

      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        

        u_x = (cells[ii + jj*params.nx].speeds[1]
            + cells[ii + jj*params.nx].speeds[5]
            + cells[ii + jj*params.nx].speeds[8]
            - (cells[ii + jj*params.nx].speeds[3]
              + cells[ii + jj*params.nx].speeds[6]
              + cells[ii + jj*params.nx].speeds[7]))
          / local_density;
        

        u_y = (cells[ii + jj*params.nx].speeds[2]
            + cells[ii + jj*params.nx].speeds[5]
            + cells[ii + jj*params.nx].speeds[6]
            - (cells[ii + jj*params.nx].speeds[4]
              + cells[ii + jj*params.nx].speeds[7]
              + cells[ii + jj*params.nx].speeds[8]))
          / local_density;
        

        u = sqrtf((u_x * u_x) + (u_y * u_y));
        

        pressure = local_density * c_sq;
      }

      

      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}