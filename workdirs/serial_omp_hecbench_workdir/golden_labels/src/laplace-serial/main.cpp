


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "timer.h"



#define NUM 1024



#define BLOCK_SIZE 256

#define Real float
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f



const Real omega = 1.85f;



void fill_coeffs (int rowmax, int colmax, Real th_cond, Real dx, Real dy,
    Real width, Real TN, Real * aP, Real * aW, Real * aE, 
    Real * aS, Real * aN, Real * b)
{
  int col, row;
  for (col = 0; col < colmax; ++col) {
    for (row = 0; row < rowmax; ++row) {
      int ind = col * rowmax + row;

      b[ind] = ZERO;
      Real SP = ZERO;

      if (col == 0) {
        

        aW[ind] = ZERO;
        SP = -TWO * th_cond * width * dy / dx;
      } else {
        aW[ind] = th_cond * width * dy / dx;
      }

      if (col == colmax - 1) {
        

        aE[ind] = ZERO;
        SP = -TWO * th_cond * width * dy / dx;
      } else {
        aE[ind] = th_cond * width * dy / dx;
      }

      if (row == 0) {
        

        aS[ind] = ZERO;
        SP = -TWO * th_cond * width * dx / dy;
      } else {
        aS[ind] = th_cond * width * dx / dy;
      }

      if (row == rowmax - 1) {
        

        aN[ind] = ZERO;
        b[ind] = TWO * th_cond * width * dx * TN / dy;
        SP = -TWO * th_cond * width * dx / dy;
      } else {
        aN[ind] = th_cond * width * dx / dy;
      }

      aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
    } 

  } 

} 




int main (void) {

  

  Real L = 1.0;
  Real H = 1.0;
  Real width = 0.01;

  

  Real th_cond = 1.0;

  

  Real TN = 1.0;

  

  Real tol = 1.e-6;

  

  

  int num_rows = (NUM / 2) + 2;
  int num_cols = NUM + 2;
  int size_temp = num_rows * num_cols;
  int size = NUM * NUM;

  

  Real dx = L / NUM;
  Real dy = H / NUM;

  

  int iter;
  int it_max = 1e6;

  

  Real *aP, *aW, *aE, *aS, *aN, *b;
  Real *temp_red, *temp_black;

  

  aP = (Real *) calloc (size, sizeof(Real));
  aW = (Real *) calloc (size, sizeof(Real));
  aE = (Real *) calloc (size, sizeof(Real));
  aS = (Real *) calloc (size, sizeof(Real));
  aN = (Real *) calloc (size, sizeof(Real));

  

  b = (Real *) calloc (size, sizeof(Real));

  

  temp_red = (Real *) calloc (size_temp, sizeof(Real));
  temp_black = (Real *) calloc (size_temp, sizeof(Real));

  

  fill_coeffs (NUM, NUM, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);

  int i;
  for (i = 0; i < size_temp; ++i) {
    temp_red[i] = ZERO;
    temp_black[i] = ZERO;
  }

  

  Real *bl_norm_L2;

  

  int size_norm = size_temp;
  bl_norm_L2 = (Real *) calloc (size_norm, sizeof(Real));
  for (i = 0; i < size_norm; ++i) {
    bl_norm_L2[i] = ZERO;
  }

  

  printf("Problem size: %d x %d \n", NUM, NUM);

  

    {
    StartTimer();
  
    for (iter = 1; iter <= it_max; ++iter) {
  
      Real norm_L2 = ZERO;
  
            for (int row = 1; row <= NUM/2; row++) {
        for (int col = 1; col <= NUM; col++) {
          int ind_red = col * ((NUM >> 1) + 2) + row;  					

          int ind = 2 * row - (col & 1) - 1 + NUM * (col - 1);	

  
          Real temp_old = temp_red[ind_red];
  
          Real res = b[ind] + (aW[ind] * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
                + aE[ind] * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
                + aS[ind] * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
                + aN[ind] * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)]);
  
          Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);
  
          temp_red[ind_red] = temp_new;
          res = temp_new - temp_old;
  
          bl_norm_L2[ind_red] = res * res;
        }
      }
      

            for (int i = 0; i < size_norm; ++i) {
        norm_L2 += bl_norm_L2[i];
      }
  
            for (int row = 1; row <= NUM/2; row++) {
        for (int col = 1; col <= NUM; col++) {
          int ind_black = col * ((NUM >> 1) + 2) + row; 

          int ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1); 

  
          Real temp_old = temp_black[ind_black];
  
          Real res = b[ind] + (aW[ind] * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
                + aE[ind] * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
                + aS[ind] * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
                + aN[ind] * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)]);
  
          Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);
  
          temp_black[ind_black] = temp_new;
          res = temp_new - temp_old;
  
          bl_norm_L2[ind_black] = res * res;
        }
      }
      

            for (int i = 0; i < size_norm; ++i)
        norm_L2 += bl_norm_L2[i];
  
      

      norm_L2 = sqrt(norm_L2 / ((Real)size));
  
      if (iter % 1000 == 0) printf("%5d, %0.6f\n", iter, norm_L2);
  
      

      if (norm_L2 < tol) break;
    }
  
    double runtime = GetTimer();
    printf("Total time for %i iterations: %f s\n", iter, runtime / 1000.0);
  }

  

  FILE * pfile;
  pfile = fopen("temperature.dat", "w");

  if (pfile != NULL) {
    fprintf(pfile, "#x\ty\ttemp(K)\n");

    int row, col;
    for (row = 1; row < NUM + 1; ++row) {
      for (col = 1; col < NUM + 1; ++col) {
        Real x_pos = (col - 1) * dx + (dx / 2);
        Real y_pos = (row - 1) * dy + (dy / 2);

        if ((row + col) % 2 == 0) {
          

          int ind = col * num_rows + (row + (col % 2)) / 2;
          fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_red[ind]);
        } else {
          

          int ind = col * num_rows + (row + ((col + 1) % 2)) / 2;
          fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_black[ind]);
        }	
      }
      fprintf(pfile, "\n");
    }
  }
  fclose(pfile);

  free(aP);
  free(aW);
  free(aE);
  free(aS);
  free(aN);
  free(b);
  free(temp_red);
  free(temp_black);
  free(bl_norm_L2);

  return 0;
}