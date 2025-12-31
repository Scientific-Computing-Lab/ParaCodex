


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>



#define NUM 512



#define BLOCK_SIZE 128



#define DOUBLE

#ifdef DOUBLE
#define Real double

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0

#define SMALL 1.0e-10;



const Real Re_num = 1000.0;



const Real omega = 1.7;



const Real mix_param = 0.9;



const Real tau = 0.5;



const Real gx = 0.0;
const Real gy = 0.0;



#define xLength 1.0
#define yLength 1.0

#else

#define Real float



#define fmin fminf
#define fmax fmaxf
#define fabs fabsf
#define sqrt sqrtf

#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#define FOUR 4.0f
#define SMALL 1.0e-10f;



const Real Re_num = 1000.0f;



const Real omega = 1.7f;



const Real mix_param = 0.9f;



const Real tau = 0.5f;



const Real gx = 0.0f;
const Real gy = 0.0f;



#define xLength 1.0f
#define yLength 1.0f
#endif



const Real dx = xLength / NUM;
const Real dy = yLength / NUM;













#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]



void set_BCs_host (Real* u, Real* v) 
{
  int ind;

  

  for (ind = 0; ind < NUM + 2; ++ind) {

    

    u(0, ind) = ZERO;
    v(0, ind) = -v(1, ind);

    

    u(NUM, ind) = ZERO;
    v(NUM + 1, ind) = -v(NUM, ind);

    

    u(ind, 0) = -u(ind, 1);
    v(ind, 0) = ZERO;

    

    u(ind, NUM + 1) = TWO - u(ind, NUM);
    v(ind, NUM) = ZERO;

    if (ind == NUM) {
      

      u(0, 0) = ZERO;
      v(0, 0) = -v(1, 0);
      u(0, NUM + 1) = ZERO;
      v(0, NUM + 1) = -v(1, NUM + 1);

      

      u(NUM, 0) = ZERO;
      v(NUM + 1, 0) = -v(NUM, 0);
      u(NUM, NUM + 1) = ZERO;
      v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);

      

      u(0, 0) = -u(0, 1);
      v(0, 0) = ZERO;
      u(NUM + 1, 0) = -u(NUM + 1, 1);
      v(NUM + 1, 0) = ZERO;

      

      u(0, NUM + 1) = TWO - u(0, NUM);
      v(0, NUM) = ZERO;
      u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
      v(ind, NUM + 1) = ZERO;
    } 


  } 


} 





int main (int argc, char *argv[])
{
  

  int iter = 0;

  const int it_max = 1000000;

  

  const Real tol = 0.001;

  

  const Real time_start = 0.0;
  const Real time_end = 0.001; 


  

  Real dt = 0.02;

  int size = (NUM + 2) * (NUM + 2);
  int size_pres = ((NUM / 2) + 2) * (NUM + 2);

  

  Real* F;
  Real* u;
  Real* G;
  Real* v;

  F = (Real *) calloc (size, sizeof(Real));
  u = (Real *) calloc (size, sizeof(Real));
  G = (Real *) calloc (size, sizeof(Real));
  v = (Real *) calloc (size, sizeof(Real));

  for (int i = 0; i < size; ++i) {
    F[i] = ZERO;
    u[i] = ZERO;
    G[i] = ZERO;
    v[i] = ZERO;
  }

  

  Real* pres_red;
  Real* pres_black;

  pres_red = (Real *) calloc (size_pres, sizeof(Real));
  pres_black = (Real *) calloc (size_pres, sizeof(Real));

  for (int i = 0; i < size_pres; ++i) {
    pres_red[i] = ZERO;
    pres_black[i] = ZERO;
  }

  

  printf("Problem size: %d x %d \n", NUM, NUM);

  

  Real* res_arr;

  int size_res = NUM / (2 * BLOCK_SIZE) * NUM;
  res_arr = (Real *) calloc (size_res, sizeof(Real));

  

  Real* max_u_arr;
  Real* max_v_arr;
  int size_max = size_res;

  max_u_arr = (Real *) calloc (size_max, sizeof(Real));
  max_v_arr = (Real *) calloc (size_max, sizeof(Real));

  

  Real* pres_sum;
  pres_sum = (Real *) calloc (size_res, sizeof(Real));

  

  set_BCs_host (u, v);

  Real max_u = SMALL;
  Real max_v = SMALL;
  

    for (int col = 0; col < NUM + 2; ++col) {
        for (int row = 1; row < NUM + 2; ++row) {
      max_u = fmax(max_u, fabs( u(col, row) ));
    }
  }

    for (int col = 1; col < NUM + 2; ++col) {
        for (int row = 0; row < NUM + 2; ++row) {
      max_v = fmax(max_v, fabs( v(col, row) ));
    }
  }

  {
    Real time = time_start;

    

    Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));

    auto start = std::chrono::steady_clock::now();

    

    while (time < time_end) {

      

      dt = fmin((dx / max_u), (dy / max_v));
      dt = tau * fmin(dt_Re, dt);

      if ((time + dt) >= time_end) {
        dt = time_end - time;
      }

      

      

      #include "calculate_F.h"

      

      #include "calculate_G.h"

      

      

      #include "sum_pressure.h"

      

      
      Real p0_norm = ZERO;
            for (int i = 0; i < size_res; ++i) {
        p0_norm += pres_sum[i];
      }
      


      p0_norm = sqrt(p0_norm / ((Real)(NUM * NUM)));
      if (p0_norm < 0.0001) {
        p0_norm = 1.0;
      }

      Real norm_L2;

      

      

      for (iter = 1; iter <= it_max; ++iter) {

        

        

        #include "set_horz_pres_BCs.h"

        

        #include "set_vert_pres_BCs.h"

        

        

        #include "red_kernel.h"

        

        

        #include "black_kernel.h"

        

        

        #include "calc_residual.h"

                

        


        norm_L2 = ZERO;

                for (int i = 0; i < size_res; ++i) {
          norm_L2 += res_arr[i];
        }

      


        

        norm_L2 = sqrt(norm_L2 / ((Real)(NUM * NUM))) / p0_norm;

        

        if (norm_L2 < tol) {
          break;
        }  
      } 


      printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);

      


      

      #include "calculate_u.h"

      

      
      

      #include "calculate_v.h"

      

      
      

      max_v = SMALL;
      max_u = SMALL;

            for (int i = 0; i < size_max; ++i) {
        Real test_u = max_u_arr[i];
        max_u = fmax(max_u, test_u);

        Real test_v = max_v_arr[i];
        max_v = fmax(max_v, test_v);
      }

      

      

      #include "set_BCs.h"

      

      time += dt;

      

      


    } 


    auto end = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("\nTotal execution time of the iteration loop: %f (s)\n", elapsed_time * 1e-9f);
  }
  


  

  FILE * pfile;
  pfile = fopen("velocity_gpu.dat", "w");
  fprintf(pfile, "#x\ty\tu\tv\n");
  if (pfile != NULL) {
    for (int row = 0; row < NUM; ++row) {
      for (int col = 0; col < NUM; ++col) {

        Real u_ij = u[(col * NUM) + row];
        Real u_im1j;
        if (col == 0) {
          u_im1j = 0.0;
        } else {
          u_im1j = u[(col - 1) * NUM + row];
        }

        u_ij = (u_ij + u_im1j) / 2.0;

        Real v_ij = v[(col * NUM) + row];
        Real v_ijm1;
        if (row == 0) {
          v_ijm1 = 0.0;
        } else {
          v_ijm1 = v[(col * NUM) + row - 1];
        }

        v_ij = (v_ij + v_ijm1) / 2.0;

        fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
      }
    }
  }

  fclose(pfile);

  free(pres_red);
  free(pres_black);
  free(u);
  free(v);
  free(F);
  free(G);
  free(max_u_arr);
  free(max_v_arr);
  free(res_arr);
  free(pres_sum);
  return 0;
}