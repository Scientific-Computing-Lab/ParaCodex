
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "header.h"
#include "timers.h"
#include "print_results.h"
#undef min
#undef max

double elapsed_time;
int grid_points[3];
logical timeron;

double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, 
       dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, 
       dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, 
       ce[5][13], dxmax, dymax, dzmax, xxcon1, xxcon2, 
       xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
       dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
       yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
       zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, 
       dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, 
       dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2, 
       c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1,
       dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
       c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
       c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

double us     [KMAX][JMAXP+1][IMAXP+1];
double vs     [KMAX][JMAXP+1][IMAXP+1];
double ws     [KMAX][JMAXP+1][IMAXP+1];
double qs     [KMAX][JMAXP+1][IMAXP+1];
double rho_i  [KMAX][JMAXP+1][IMAXP+1];
double square [KMAX][JMAXP+1][IMAXP+1];
double forcing[5][KMAX][JMAXP+1][IMAXP+1];
double u[5][KMAX][JMAXP+1][IMAXP+1];
double rhs[5][KMAX][JMAXP+1][IMAXP+1];

double cuf[PROBLEM_SIZE+1];
double q  [PROBLEM_SIZE+1];
double ue [PROBLEM_SIZE+1][5];
double buf[PROBLEM_SIZE+1][5];

double fjac[PROBLEM_SIZE+1][5][5];
double njac[PROBLEM_SIZE+1][5][5];
double lhs [PROBLEM_SIZE+1][3][5][5];
double tmp1, tmp2, tmp3;

int main(int argc, char *argv[])
{
  int i, niter, step;
  double navg, mflops, n3;

  double tmax, t, trecs[t_last+1];
  logical verified;
  char Class;
  char *t_names[t_last+1];

  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[t_total] = "total";
    t_names[t_rhsx] = "rhsx";
    t_names[t_rhsy] = "rhsy";
    t_names[t_rhsz] = "rhsz";
    t_names[t_rhs] = "rhs";
    t_names[t_xsolve] = "xsolve";
    t_names[t_ysolve] = "ysolve";
    t_names[t_zsolve] = "zsolve";
    t_names[t_rdis1] = "redist1";
    t_names[t_rdis2] = "redist2";
    t_names[t_add] = "add";
    fclose(fp);
  } else {
    timeron = false;
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - BT Benchmark\n\n");

  if ((fp = fopen("inputbt.data", "r")) != NULL) {
    int result;
    printf(" Reading from input file inputbt.data\n");
    result = fscanf(fp, "%d", &niter);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf", &dt);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d%d\n", 
        &grid_points[0], &grid_points[1], &grid_points[2]);
    fclose(fp);
  } else {
    printf(" No input file inputbt.data. Using compiled defaults\n");
    niter = NITER_DEFAULT;
    dt    = DT_DEFAULT;
    grid_points[0] = PROBLEM_SIZE;
    grid_points[1] = PROBLEM_SIZE;
    grid_points[2] = PROBLEM_SIZE;
  }

  printf(" Size: %4dx%4dx%4d\n",
      grid_points[0], grid_points[1], grid_points[2]);
  printf(" Iterations: %4d    dt: %10.6f\n", niter, dt);
  printf("\n");
  
  if ( (grid_points[0] > IMAX) ||
       (grid_points[1] > JMAX) ||
       (grid_points[2] > KMAX) ) {
    printf(" %d, %d, %d\n", grid_points[0], grid_points[1], grid_points[2]);
    printf(" Problem size too big for compiled array sizes\n");
    return 0;
  }
  
  set_constants();

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }

{
  initialize();

  exact_rhs();

  adi();
  initialize();
  
  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  timer_start(1);

  for (step = 1; step <= niter; step++) {
    if ((step % 20) == 0 || step == 1) {
      printf(" Time step %4d\n", step);
    }

    adi();
  }

  timer_stop(1);
  tmax = timer_read(1);

  verify(niter, &Class, &verified);

  n3 = 1.0*grid_points[0]*grid_points[1]*grid_points[2];
  navg = (grid_points[0]+grid_points[1]+grid_points[2])/3.0;
  if(tmax != 0.0) {
    mflops = 1.0e-6 * (double)niter *
      (3478.8 * n3 - 17655.7 * (navg*navg) + 28023.7 * navg)
      / tmax;
  } else {
    mflops = 0.0;
  }

}
  print_results("BT", Class, grid_points[0], 
                grid_points[1], grid_points[2], niter,
                tmax, mflops, "          floating point", 
                verified, NPBVERSION,COMPILETIME, CS1, CS2, CS3, CS4, CS5, 
                CS6, "(none)");

  return 0;
}


void add()
{
  int i, j, k, m;
  int gp22, gp12, gp02;

  gp22 = grid_points[2]-2;
  gp12 = grid_points[1]-2;
  gp02 = grid_points[0]-2;

  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
	  
          u[0][k][j][i] = u[0][k][j][i] + rhs[0][k][j][i];
          u[1][k][j][i] = u[1][k][j][i] + rhs[1][k][j][i];
          u[2][k][j][i] = u[2][k][j][i] + rhs[2][k][j][i];
          u[3][k][j][i] = u[3][k][j][i] + rhs[3][k][j][i];
          u[4][k][j][i] = u[4][k][j][i] + rhs[4][k][j][i];
      }
    }
  }
}

void error_norm(double rms[5])
{
  int i, j, k, m, d;
  double xi, eta, zeta, u_exact[5], add;

  for (m = 0; m < 5; m++) {
    rms[m] = 0.0;
  }

  for (k = 0; k <= grid_points[2]-1; k++) {
    zeta = (double)(k) * dnzm1;
    for (j = 0; j <= grid_points[1]-1; j++) {
      eta = (double)(j) * dnym1;
      for (i = 0; i <= grid_points[0]-1; i++) {
        xi = (double)(i) * dnxm1;
        exact_solution(xi, eta, zeta, u_exact);

        for (m = 0; m < 5; m++) {
          add = u[m][k][j][i]-u_exact[m];
          rms[m] = rms[m] + add*add;
        }
      }
    }
  }

  for (m = 0; m < 5; m++) {
    for (d = 0; d < 3; d++) {
      rms[m] = rms[m] / (double)(grid_points[d]-2);
    }
    rms[m] = sqrt(rms[m]);
  }
}

void rhs_norm(double rms[5])
{
  int i, j, k, d, m;
  double add;

  for (m = 0; m < 5; m++) {
    rms[m] = 0.0;
  } 

  for (k = 1; k <= grid_points[2]-2; k++) {
    for (j = 1; j <= grid_points[1]-2; j++) {
      for (i = 1; i <= grid_points[0]-2; i++) {
        for (m = 0; m < 5; m++) {
          add = rhs[m][k][j][i];
          rms[m] = rms[m] + add*add;
        } 
      } 
    } 
  } 

  for (m = 0; m < 5; m++) {
    for (d = 0; d < 3; d++) {
      rms[m] = rms[m] / (double)(grid_points[d]-2);
    } 
    rms[m] = sqrt(rms[m]);
  } 
}

void exact_rhs()
{
  double dtemp[5], xi, eta, zeta, dtpp;
  int m, i, j, k, ip1, im1, jp1, jm1, km1, kp1;

  for (k = 0; k <= grid_points[2]-1; k++) {
    for (j = 0; j <= grid_points[1]-1; j++) {
      for (i = 0; i <= grid_points[0]-1; i++) {
        for (m = 0; m < 5; m++) {
          forcing[m][k][j][i] = 0.0;
        }
      }
    }
  }

  for (k = 1; k <= grid_points[2]-2; k++) {
    zeta = (double)(k) * dnzm1;
    for (j = 1; j <= grid_points[1]-2; j++) {
      eta = (double)(j) * dnym1;

      for (i = 0; i <= grid_points[0]-1; i++) {
        xi = (double)(i) * dnxm1;

        exact_solution(xi, eta, zeta, dtemp);
        for (m = 0; m < 5; m++) {
          ue[i][m] = dtemp[m];
        }

        dtpp = 1.0 / dtemp[0];

        for (m = 1; m < 5; m++) {
          buf[i][m] = dtpp * dtemp[m];
        }

        cuf[i]    = buf[i][1] * buf[i][1];
        buf[i][0] = cuf[i] + buf[i][2] * buf[i][2] + buf[i][3] * buf[i][3];
        q[i] = 0.5*(buf[i][1]*ue[i][1] + buf[i][2]*ue[i][2] +
                    buf[i][3]*ue[i][3]);
      }

      for (i = 1; i <= grid_points[0]-2; i++) {
        im1 = i-1;
        ip1 = i+1;

        forcing[0][k][j][i] = forcing[0][k][j][i] -
          tx2*( ue[ip1][1]-ue[im1][1] )+
          dx1tx1*(ue[ip1][0]-2.0*ue[i][0]+ue[im1][0]);

        forcing[1][k][j][i] = forcing[1][k][j][i] - tx2 * (
            (ue[ip1][1]*buf[ip1][1]+c2*(ue[ip1][4]-q[ip1]))-
            (ue[im1][1]*buf[im1][1]+c2*(ue[im1][4]-q[im1])))+
          xxcon1*(buf[ip1][1]-2.0*buf[i][1]+buf[im1][1])+
          dx2tx1*( ue[ip1][1]-2.0* ue[i][1]+ue[im1][1]);

        forcing[2][k][j][i] = forcing[2][k][j][i] - tx2 * (
            ue[ip1][2]*buf[ip1][1]-ue[im1][2]*buf[im1][1])+
          xxcon2*(buf[ip1][2]-2.0*buf[i][2]+buf[im1][2])+
          dx3tx1*( ue[ip1][2]-2.0*ue[i][2] +ue[im1][2]);

        forcing[3][k][j][i] = forcing[3][k][j][i] - tx2*(
            ue[ip1][3]*buf[ip1][1]-ue[im1][3]*buf[im1][1])+
          xxcon2*(buf[ip1][3]-2.0*buf[i][3]+buf[im1][3])+
          dx4tx1*( ue[ip1][3]-2.0* ue[i][3]+ ue[im1][3]);

        forcing[4][k][j][i] = forcing[4][k][j][i] - tx2*(
            buf[ip1][1]*(c1*ue[ip1][4]-c2*q[ip1])-
            buf[im1][1]*(c1*ue[im1][4]-c2*q[im1]))+
          0.5*xxcon3*(buf[ip1][0]-2.0*buf[i][0]+
              buf[im1][0])+
          xxcon4*(cuf[ip1]-2.0*cuf[i]+cuf[im1])+
          xxcon5*(buf[ip1][4]-2.0*buf[i][4]+buf[im1][4])+
          dx5tx1*( ue[ip1][4]-2.0* ue[i][4]+ ue[im1][4]);
      }

      for (m = 0; m < 5; m++) {
        i = 1;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (5.0*ue[i][m] - 4.0*ue[i+1][m] +ue[i+2][m]);
        i = 2;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (-4.0*ue[i-1][m] + 6.0*ue[i][m] -
            4.0*ue[i+1][m] +     ue[i+2][m]);
      }

      for (i = 3; i <= grid_points[0]-4; i++) {
        for (m = 0; m < 5; m++) {
          forcing[m][k][j][i] = forcing[m][k][j][i] - dssp*
            (ue[i-2][m] - 4.0*ue[i-1][m] +
             6.0*ue[i][m] - 4.0*ue[i+1][m] + ue[i+2][m]);
        }
      }

      for (m = 0; m < 5; m++) {
        i = grid_points[0]-3;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (ue[i-2][m] - 4.0*ue[i-1][m] +
           6.0*ue[i][m] - 4.0*ue[i+1][m]);
        i = grid_points[0]-2;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (ue[i-2][m] - 4.0*ue[i-1][m] + 5.0*ue[i][m]);
      }
    }
  }

  for (k = 1; k <= grid_points[2]-2; k++) {
    zeta = (double)(k) * dnzm1;
    for (i = 1; i <= grid_points[0]-2; i++) {
      xi = (double)(i) * dnxm1;

      for (j = 0; j <= grid_points[1]-1; j++) {
        eta = (double)(j) * dnym1;

        exact_solution(xi, eta, zeta, dtemp);
        for (m = 0; m < 5; m++) {
          ue[j][m] = dtemp[m];
        }

        dtpp = 1.0/dtemp[0];

        for (m = 1; m < 5; m++) {
          buf[j][m] = dtpp * dtemp[m];
        }

        cuf[j]    = buf[j][2] * buf[j][2];
        buf[j][0] = cuf[j] + buf[j][1] * buf[j][1] + buf[j][3] * buf[j][3];
        q[j] = 0.5*(buf[j][1]*ue[j][1] + buf[j][2]*ue[j][2] +
                    buf[j][3]*ue[j][3]);
      }

      for (j = 1; j <= grid_points[1]-2; j++) {
        jm1 = j-1;
        jp1 = j+1;

        forcing[0][k][j][i] = forcing[0][k][j][i] -
          ty2*( ue[jp1][2]-ue[jm1][2] )+
          dy1ty1*(ue[jp1][0]-2.0*ue[j][0]+ue[jm1][0]);

        forcing[1][k][j][i] = forcing[1][k][j][i] - ty2*(
            ue[jp1][1]*buf[jp1][2]-ue[jm1][1]*buf[jm1][2])+
          yycon2*(buf[jp1][1]-2.0*buf[j][1]+buf[jm1][1])+
          dy2ty1*( ue[jp1][1]-2.0* ue[j][1]+ ue[jm1][1]);

        forcing[2][k][j][i] = forcing[2][k][j][i] - ty2*(
            (ue[jp1][2]*buf[jp1][2]+c2*(ue[jp1][4]-q[jp1]))-
            (ue[jm1][2]*buf[jm1][2]+c2*(ue[jm1][4]-q[jm1])))+
          yycon1*(buf[jp1][2]-2.0*buf[j][2]+buf[jm1][2])+
          dy3ty1*( ue[jp1][2]-2.0*ue[j][2] +ue[jm1][2]);

        forcing[3][k][j][i] = forcing[3][k][j][i] - ty2*(
            ue[jp1][3]*buf[jp1][2]-ue[jm1][3]*buf[jm1][2])+
          yycon2*(buf[jp1][3]-2.0*buf[j][3]+buf[jm1][3])+
          dy4ty1*( ue[jp1][3]-2.0*ue[j][3]+ ue[jm1][3]);

        forcing[4][k][j][i] = forcing[4][k][j][i] - ty2*(
            buf[jp1][2]*(c1*ue[jp1][4]-c2*q[jp1])-
            buf[jm1][2]*(c1*ue[jm1][4]-c2*q[jm1]))+
          0.5*yycon3*(buf[jp1][0]-2.0*buf[j][0]+
              buf[jm1][0])+
          yycon4*(cuf[jp1]-2.0*cuf[j]+cuf[jm1])+
          yycon5*(buf[jp1][4]-2.0*buf[j][4]+buf[jm1][4])+
          dy5ty1*(ue[jp1][4]-2.0*ue[j][4]+ue[jm1][4]);
      }

      for (m = 0; m < 5; m++) {
        j = 1;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (5.0*ue[j][m] - 4.0*ue[j+1][m] +ue[j+2][m]);
        j = 2;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (-4.0*ue[j-1][m] + 6.0*ue[j][m] -
           4.0*ue[j+1][m] +       ue[j+2][m]);
      }

      for (j = 3; j <= grid_points[1]-4; j++) {
        for (m = 0; m < 5; m++) {
          forcing[m][k][j][i] = forcing[m][k][j][i] - dssp*
            (ue[j-2][m] - 4.0*ue[j-1][m] +
             6.0*ue[j][m] - 4.0*ue[j+1][m] + ue[j+2][m]);
        }
      }

      for (m = 0; m < 5; m++) {
        j = grid_points[1]-3;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (ue[j-2][m] - 4.0*ue[j-1][m] +
           6.0*ue[j][m] - 4.0*ue[j+1][m]);
        j = grid_points[1]-2;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (ue[j-2][m] - 4.0*ue[j-1][m] + 5.0*ue[j][m]);
      }
    }
  }

  for (j = 1; j <= grid_points[1]-2; j++) {
    eta = (double)(j) * dnym1;
    for (i = 1; i <= grid_points[0]-2; i++) {
      xi = (double)(i) * dnxm1;

      for (k = 0; k <= grid_points[2]-1; k++) {
        zeta = (double)(k) * dnzm1;

        exact_solution(xi, eta, zeta, dtemp);
        for (m = 0; m < 5; m++) {
          ue[k][m] = dtemp[m];
        }

        dtpp = 1.0/dtemp[0];

        for (m = 1; m < 5; m++) {
          buf[k][m] = dtpp * dtemp[m];
        }

        cuf[k]    = buf[k][3] * buf[k][3];
        buf[k][0] = cuf[k] + buf[k][1] * buf[k][1] + buf[k][2] * buf[k][2];
        q[k] = 0.5*(buf[k][1]*ue[k][1] + buf[k][2]*ue[k][2] +
                    buf[k][3]*ue[k][3]);
      }

      for (k = 1; k <= grid_points[2]-2; k++) {
        km1 = k-1;
        kp1 = k+1;

        forcing[0][k][j][i] = forcing[0][k][j][i] -
          tz2*( ue[kp1][3]-ue[km1][3] )+
          dz1tz1*(ue[kp1][0]-2.0*ue[k][0]+ue[km1][0]);

        forcing[1][k][j][i] = forcing[1][k][j][i] - tz2 * (
            ue[kp1][1]*buf[kp1][3]-ue[km1][1]*buf[km1][3])+
          zzcon2*(buf[kp1][1]-2.0*buf[k][1]+buf[km1][1])+
          dz2tz1*( ue[kp1][1]-2.0* ue[k][1]+ ue[km1][1]);

        forcing[2][k][j][i] = forcing[2][k][j][i] - tz2 * (
            ue[kp1][2]*buf[kp1][3]-ue[km1][2]*buf[km1][3])+
          zzcon2*(buf[kp1][2]-2.0*buf[k][2]+buf[km1][2])+
          dz3tz1*(ue[kp1][2]-2.0*ue[k][2]+ue[km1][2]);

        forcing[3][k][j][i] = forcing[3][k][j][i] - tz2 * (
            (ue[kp1][3]*buf[kp1][3]+c2*(ue[kp1][4]-q[kp1]))-
            (ue[km1][3]*buf[km1][3]+c2*(ue[km1][4]-q[km1])))+
          zzcon1*(buf[kp1][3]-2.0*buf[k][3]+buf[km1][3])+
          dz4tz1*( ue[kp1][3]-2.0*ue[k][3] +ue[km1][3]);

        forcing[4][k][j][i] = forcing[4][k][j][i] - tz2 * (
            buf[kp1][3]*(c1*ue[kp1][4]-c2*q[kp1])-
            buf[km1][3]*(c1*ue[km1][4]-c2*q[km1]))+
          0.5*zzcon3*(buf[kp1][0]-2.0*buf[k][0]
              +buf[km1][0])+
          zzcon4*(cuf[kp1]-2.0*cuf[k]+cuf[km1])+
          zzcon5*(buf[kp1][4]-2.0*buf[k][4]+buf[km1][4])+
          dz5tz1*( ue[kp1][4]-2.0*ue[k][4]+ ue[km1][4]);
      }

      for (m = 0; m < 5; m++) {
        k = 1;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (5.0*ue[k][m] - 4.0*ue[k+1][m] +ue[k+2][m]);
        k = 2;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (-4.0*ue[k-1][m] + 6.0*ue[k][m] -
           4.0*ue[k+1][m] +       ue[k+2][m]);
      }

      for (k = 3; k <= grid_points[2]-4; k++) {
        for (m = 0; m < 5; m++) {
          forcing[m][k][j][i] = forcing[m][k][j][i] - dssp*
            (ue[k-2][m] - 4.0*ue[k-1][m] +
             6.0*ue[k][m] - 4.0*ue[k+1][m] + ue[k+2][m]);
        }
      }

      for (m = 0; m < 5; m++) {
        k = grid_points[2]-3;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (ue[k-2][m] - 4.0*ue[k-1][m] +
           6.0*ue[k][m] - 4.0*ue[k+1][m]);
        k = grid_points[2]-2;
        forcing[m][k][j][i] = forcing[m][k][j][i] - dssp *
          (ue[k-2][m] - 4.0*ue[k-1][m] + 5.0*ue[k][m]);
      }

    }
  }

  for (k = 1; k <= grid_points[2]-2; k++) {
    for (j = 1; j <= grid_points[1]-2; j++) {
      for (i = 1; i <= grid_points[0]-2; i++) {
        for (m = 0; m < 5; m++) {
          forcing[m][k][j][i] = -1.0 * forcing[m][k][j][i];
        }
      }
    }
  }
}

void initialize()
{
  int i, j, k, m, ix, iy, iz;
  double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];

  for (k = 0; k <= grid_points[2]-1; k++) {
    for (j = 0; j <= grid_points[1]-1; j++) {
      for (i = 0; i <= grid_points[0]-1; i++) {
        for (m = 0; m < 5; m++) {
          u[m][k][j][i] = 1.0;
        }
      }
    }
  }

  for (k = 0; k <= grid_points[2]-1; k++) {
    zeta = (double)(k) * dnzm1;
    for (j = 0; j <= grid_points[1]-1; j++) {
      eta = (double)(j) * dnym1;
      for (i = 0; i <= grid_points[0]-1; i++) {
        xi = (double)(i) * dnxm1;

        for (ix = 0; ix < 2; ix++) {
          exact_solution((double)ix, eta, zeta, &Pface[ix][0][0]);
        }

        for (iy = 0; iy < 2; iy++) {
          exact_solution(xi, (double)iy , zeta, &Pface[iy][1][0]);
        }

        for (iz = 0; iz < 2; iz++) {
          exact_solution(xi, eta, (double)iz, &Pface[iz][2][0]);
        }

        for (m = 0; m < 5; m++) {
          Pxi   = xi   * Pface[1][0][m] + (1.0-xi)   * Pface[0][0][m];
          Peta  = eta  * Pface[1][1][m] + (1.0-eta)  * Pface[0][1][m];
          Pzeta = zeta * Pface[1][2][m] + (1.0-zeta) * Pface[0][2][m];

          u[m][k][j][i] = Pxi + Peta + Pzeta - 
                          Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + 
                          Pxi*Peta*Pzeta;
        }
      }
    }
  }

  i = 0;
  xi = 0.0;
  for (k = 0; k <= grid_points[2]-1; k++) {
    zeta = (double)(k) * dnzm1;
    for (j = 0; j <= grid_points[1]-1; j++) {
      eta = (double)(j) * dnym1;
      exact_solution(xi, eta, zeta, temp);
      for (m = 0; m < 5; m++) {
        u[m][k][j][i] = temp[m];
      }
    }
  }

  i = grid_points[0]-1;
  xi = 1.0;
  for (k = 0; k <= grid_points[2]-1; k++) {
    zeta = (double)(k) * dnzm1;
    for (j = 0; j <= grid_points[1]-1; j++) {
      eta = (double)(j) * dnym1;
      exact_solution(xi, eta, zeta, temp);
      for (m = 0; m < 5; m++) {
        u[m][k][j][i] = temp[m];
      }
    }
  }

  j = 0;
  eta = 0.0;
  for (k = 0; k <= grid_points[2]-1; k++) {
    zeta = (double)(k) * dnzm1;
    for (i = 0; i <= grid_points[0]-1; i++) {
      xi = (double)(i) * dnxm1;
      exact_solution(xi, eta, zeta, temp);
      for (m = 0; m < 5; m++) {
        u[m][k][j][i] = temp[m];
      }
    }
  }

  j = grid_points[1]-1;
  eta = 1.0;
  for (k = 0; k <= grid_points[2]-1; k++) {
    zeta = (double)(k) * dnzm1;
    for (i = 0; i <= grid_points[0]-1; i++) {
      xi = (double)(i) * dnxm1;
      exact_solution(xi, eta, zeta, temp);
      for (m = 0; m < 5; m++) {
        u[m][k][j][i] = temp[m];
      }
    }
  }

  k = 0;
  zeta = 0.0;
  for (j = 0; j <= grid_points[1]-1; j++) {
    eta = (double)(j) * dnym1;
    for (i =0; i <= grid_points[0]-1; i++) {
      xi = (double)(i) *dnxm1;
      exact_solution(xi, eta, zeta, temp);
      for (m = 0; m < 5; m++) {
        u[m][k][j][i] = temp[m];
      }
    }
  }

  k = grid_points[2]-1;
  zeta = 1.0;
  for (j = 0; j <= grid_points[1]-1; j++) {
    eta = (double)(j) * dnym1;
    for (i = 0; i <= grid_points[0]-1; i++) {
      xi = (double)(i) * dnxm1;
      exact_solution(xi, eta, zeta, temp);
      for (m = 0; m < 5; m++) {
        u[m][k][j][i] = temp[m];
      }
    }
  }
}

#include "header.h"
#undef min
#undef max

#include<stdio.h>

void compute_rhs()
{
  int i, j, k, m;
  double rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
  int gp0, gp1, gp2;
  int gp01,gp11,gp21;
  int gp02,gp12,gp22;

  gp0 = grid_points[0];
  gp1 = grid_points[1];
  gp2 = grid_points[2];
  gp01 = grid_points[0]-1;
  gp11 = grid_points[1]-1;
  gp21 = grid_points[2]-1;
  gp02 = grid_points[0]-2;
  gp12 = grid_points[1]-2;
  gp22 = grid_points[2]-2;

  for (k = 0; k <= gp21; k++) {
    for (j = 0; j <= gp11; j++) {
      for (i = 0; i <= gp01; i++) {
        rho_inv = 1.0/u[0][k][j][i];
        rho_i[k][j][i] = rho_inv;
        us[k][j][i] = u[1][k][j][i] * rho_inv;
        vs[k][j][i] = u[2][k][j][i] * rho_inv;
        ws[k][j][i] = u[3][k][j][i] * rho_inv;
        square[k][j][i] = 0.5* (
            u[1][k][j][i]*u[1][k][j][i] + 
            u[2][k][j][i]*u[2][k][j][i] +
            u[3][k][j][i]*u[3][k][j][i] ) * rho_inv;
        qs[k][j][i] = square[k][j][i] * rho_inv;
      }
    }
  }

  for (k = 0; k <= gp21; k++) {
    for (j = 0; j <= gp11; j++) {
      for (i = 0; i <= gp01; i++) {
          rhs[0][k][j][i] = forcing[0][k][j][i];
          rhs[1][k][j][i] = forcing[1][k][j][i];
          rhs[2][k][j][i] = forcing[2][k][j][i];
          rhs[3][k][j][i] = forcing[3][k][j][i];
          rhs[4][k][j][i] = forcing[4][k][j][i];
      }
    }
  }

  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
        uijk = us[k][j][i];
        up1  = us[k][j][i+1];
        um1  = us[k][j][i-1];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dx1tx1 * 
          (u[0][k][j][i+1] - 2.0*u[0][k][j][i] + 
           u[0][k][j][i-1]) -
          tx2 * (u[1][k][j][i+1] - u[1][k][j][i-1]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dx2tx1 * 
          (u[1][k][j][i+1] - 2.0*u[1][k][j][i] + 
           u[1][k][j][i-1]) +
          xxcon2*con43 * (up1 - 2.0*uijk + um1) -
          tx2 * (u[1][k][j][i+1]*up1 - 
              u[1][k][j][i-1]*um1 +
              (u[4][k][j][i+1]- square[k][j][i+1]-
               u[4][k][j][i-1]+ square[k][j][i-1])*
              c2);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dx3tx1 * 
          (u[2][k][j][i+1] - 2.0*u[2][k][j][i] +
           u[2][k][j][i-1]) +
          xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] +
              vs[k][j][i-1]) -
          tx2 * (u[2][k][j][i+1]*up1 - 
              u[2][k][j][i-1]*um1);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dx4tx1 * 
          (u[3][k][j][i+1] - 2.0*u[3][k][j][i] +
           u[3][k][j][i-1]) +
          xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] +
              ws[k][j][i-1]) -
          tx2 * (u[3][k][j][i+1]*up1 - 
              u[3][k][j][i-1]*um1);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dx5tx1 * 
          (u[4][k][j][i+1] - 2.0*u[4][k][j][i] +
           u[4][k][j][i-1]) +
          xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] +
              qs[k][j][i-1]) +
          xxcon4 * (up1*up1 -       2.0*uijk*uijk + 
              um1*um1) +
          xxcon5 * (u[4][k][j][i+1]*rho_i[k][j][i+1] - 
              2.0*u[4][k][j][i]*rho_i[k][j][i] +
              u[4][k][j][i-1]*rho_i[k][j][i-1]) -
          tx2 * ( (c1*u[4][k][j][i+1] - 
                c2*square[k][j][i+1])*up1 -
              (c1*u[4][k][j][i-1] - 
               c2*square[k][j][i-1])*um1 );
      }
    }
  }
    
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      	i = 1;
        rhs[0][k][j][i] = rhs[0][k][j][i]- dssp * 
          ( 5.0*u[0][k][j][i] - 4.0*u[0][k][j][i+1] +
            u[0][k][j][i+2]);
        rhs[1][k][j][i] = rhs[1][k][j][i]- dssp * 
          ( 5.0*u[1][k][j][i] - 4.0*u[1][k][j][i+1] +
            u[1][k][j][i+2]);
        rhs[2][k][j][i] = rhs[2][k][j][i]- dssp * 
          ( 5.0*u[2][k][j][i] - 4.0*u[2][k][j][i+1] +
            u[2][k][j][i+2]);
        rhs[3][k][j][i] = rhs[3][k][j][i]- dssp * 
          ( 5.0*u[3][k][j][i] - 4.0*u[3][k][j][i+1] +
            u[3][k][j][i+2]);
        rhs[4][k][j][i] = rhs[4][k][j][i]- dssp * 
          ( 5.0*u[4][k][j][i] - 4.0*u[4][k][j][i+1] +
            u[4][k][j][i+2]);

      	i = 2;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
          (-4.0*u[0][k][j][i-1] + 6.0*u[0][k][j][i] -
           4.0*u[0][k][j][i+1] + u[0][k][j][i+2]);
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
          (-4.0*u[1][k][j][i-1] + 6.0*u[1][k][j][i] -
           4.0*u[1][k][j][i+1] + u[1][k][j][i+2]);
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
          (-4.0*u[2][k][j][i-1] + 6.0*u[2][k][j][i] -
           4.0*u[2][k][j][i+1] + u[2][k][j][i+2]);
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
          (-4.0*u[3][k][j][i-1] + 6.0*u[3][k][j][i] -
           4.0*u[3][k][j][i+1] + u[3][k][j][i+2]);
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
          (-4.0*u[4][k][j][i-1] + 6.0*u[4][k][j][i] -
           4.0*u[4][k][j][i+1] + u[4][k][j][i+2]);
    }
  }
  
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 3; i <= gp02-2; i++) {
          rhs[0][k][j][i] = rhs[0][k][j][i] - dssp*
		    (  u[0][k][j][i-2] - 4.0*u[0][k][j][i-1] + 
               6.0*u[0][k][j][i] - 4.0*u[0][k][j][i+1] + 
               u[0][k][j][i+2] );
          rhs[1][k][j][i] = rhs[1][k][j][i] - dssp*
		    (  u[1][k][j][i-2] - 4.0*u[1][k][j][i-1] + 
               6.0*u[1][k][j][i] - 4.0*u[1][k][j][i+1] + 
               u[1][k][j][i+2] );
          rhs[2][k][j][i] = rhs[2][k][j][i] - dssp*
		    (  u[2][k][j][i-2] - 4.0*u[2][k][j][i-1] + 
               6.0*u[2][k][j][i] - 4.0*u[2][k][j][i+1] + 
               u[2][k][j][i+2] );
          rhs[3][k][j][i] = rhs[3][k][j][i] - dssp*
		    (  u[3][k][j][i-2] - 4.0*u[3][k][j][i-1] + 
               6.0*u[3][k][j][i] - 4.0*u[3][k][j][i+1] + 
               u[3][k][j][i+2] );
          rhs[4][k][j][i] = rhs[4][k][j][i] - dssp*
		    (  u[4][k][j][i-2] - 4.0*u[4][k][j][i-1] + 
               6.0*u[4][k][j][i] - 4.0*u[4][k][j][i+1] + 
               u[4][k][j][i+2] );
      }
    }
  }
  
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      	i = gp0-3;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
          ( u[0][k][j][i-2] - 4.0*u[0][k][j][i-1] + 
            6.0*u[0][k][j][i] - 4.0*u[0][k][j][i+1] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
          ( u[1][k][j][i-2] - 4.0*u[1][k][j][i-1] + 
            6.0*u[1][k][j][i] - 4.0*u[1][k][j][i+1] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
          ( u[2][k][j][i-2] - 4.0*u[2][k][j][i-1] + 
            6.0*u[2][k][j][i] - 4.0*u[2][k][j][i+1] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
          ( u[3][k][j][i-2] - 4.0*u[3][k][j][i-1] + 
            6.0*u[3][k][j][i] - 4.0*u[3][k][j][i+1] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
          ( u[4][k][j][i-2] - 4.0*u[4][k][j][i-1] + 
            6.0*u[4][k][j][i] - 4.0*u[4][k][j][i+1] );

      	i = gp02;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
          ( u[0][k][j][i-2] - 4.*u[0][k][j][i-1] +
            5.*u[0][k][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
          ( u[1][k][j][i-2] - 4.*u[1][k][j][i-1] +
            5.*u[1][k][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
          ( u[2][k][j][i-2] - 4.*u[2][k][j][i-1] +
            5.*u[2][k][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
          ( u[3][k][j][i-2] - 4.*u[3][k][j][i-1] +
            5.*u[3][k][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
          ( u[4][k][j][i-2] - 4.*u[4][k][j][i-1] +
            5.*u[4][k][j][i] );
    }
  }

  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
        vijk = vs[k][j][i];
        vp1  = vs[k][j+1][i];
        vm1  = vs[k][j-1][i];
        rhs[0][k][j][i] = rhs[0][k][j][i] + dy1ty1 * 
          (u[0][k][j+1][i] - 2.0*u[0][k][j][i] + 
           u[0][k][j-1][i]) -
          ty2 * (u[2][k][j+1][i] - u[2][k][j-1][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i] + dy2ty1 * 
          (u[1][k][j+1][i] - 2.0*u[1][k][j][i] + 
           u[1][k][j-1][i]) +
          yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + 
              us[k][j-1][i]) -
          ty2 * (u[1][k][j+1][i]*vp1 - 
              u[1][k][j-1][i]*vm1);
        rhs[2][k][j][i] = rhs[2][k][j][i] + dy3ty1 * 
          (u[2][k][j+1][i] - 2.0*u[2][k][j][i] + 
           u[2][k][j-1][i]) +
          yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
          ty2 * (u[2][k][j+1][i]*vp1 - 
              u[2][k][j-1][i]*vm1 +
              (u[4][k][j+1][i] - square[k][j+1][i] - 
               u[4][k][j-1][i] + square[k][j-1][i])
              *c2);
        rhs[3][k][j][i] = rhs[3][k][j][i] + dy4ty1 * 
          (u[3][k][j+1][i] - 2.0*u[3][k][j][i] + 
           u[3][k][j-1][i]) +
          yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + 
              ws[k][j-1][i]) -
          ty2 * (u[3][k][j+1][i]*vp1 - 
              u[3][k][j-1][i]*vm1);
        rhs[4][k][j][i] = rhs[4][k][j][i] + dy5ty1 * 
          (u[4][k][j+1][i] - 2.0*u[4][k][j][i] + 
           u[4][k][j-1][i]) +
          yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + 
              qs[k][j-1][i]) +
          yycon4 * (vp1*vp1       - 2.0*vijk*vijk + 
              vm1*vm1) +
          yycon5 * (u[4][k][j+1][i]*rho_i[k][j+1][i] - 
              2.0*u[4][k][j][i]*rho_i[k][j][i] +
              u[4][k][j-1][i]*rho_i[k][j-1][i]) -
          ty2 * ((c1*u[4][k][j+1][i] - 
                c2*square[k][j+1][i]) * vp1 -
              (c1*u[4][k][j-1][i] - 
               c2*square[k][j-1][i]) * vm1);
      }
    }
  }
    
  for (k = 1; k <= gp22; k++) {
    for (i = 1; i <= gp02; i++) {
    	j = 1;
        
		rhs[0][k][j][i] = rhs[0][k][j][i]- dssp * 
          ( 5.0*u[0][k][j][i] - 4.0*u[0][k][j+1][i] +
            u[0][k][j+2][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i]- dssp * 
          ( 5.0*u[1][k][j][i] - 4.0*u[1][k][j+1][i] +
            u[1][k][j+2][i]);
        rhs[2][k][j][i] = rhs[2][k][j][i]- dssp * 
          ( 5.0*u[2][k][j][i] - 4.0*u[2][k][j+1][i] +
            u[2][k][j+2][i]);
        rhs[3][k][j][i] = rhs[3][k][j][i]- dssp * 
          ( 5.0*u[3][k][j][i] - 4.0*u[3][k][j+1][i] +
            u[3][k][j+2][i]);
        rhs[4][k][j][i] = rhs[4][k][j][i]- dssp * 
          ( 5.0*u[4][k][j][i] - 4.0*u[4][k][j+1][i] +
            u[4][k][j+2][i]);

    	j = 2;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
          (-4.0*u[0][k][j-1][i] + 6.0*u[0][k][j][i] -
           4.0*u[0][k][j+1][i] + u[0][k][j+2][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
          (-4.0*u[1][k][j-1][i] + 6.0*u[1][k][j][i] -
           4.0*u[1][k][j+1][i] + u[1][k][j+2][i]);
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
          (-4.0*u[2][k][j-1][i] + 6.0*u[2][k][j][i] -
           4.0*u[2][k][j+1][i] + u[2][k][j+2][i]);
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
          (-4.0*u[3][k][j-1][i] + 6.0*u[3][k][j][i] -
           4.0*u[3][k][j+1][i] + u[3][k][j+2][i]);
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
          (-4.0*u[4][k][j-1][i] + 6.0*u[4][k][j][i] -
           4.0*u[4][k][j+1][i] + u[4][k][j+2][i]);
    }
  }
  
  for (k = 1; k <= gp22; k++) {
    for (j = 3; j <= gp1-4; j++) {
      for (i = 1; i <= gp02; i++) {
          rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
            (  u[0][k][j-2][i] - 4.0*u[0][k][j-1][i] + 
               6.0*u[0][k][j][i] - 4.0*u[0][k][j+1][i] + 
               u[0][k][j+2][i] );
          rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
            (  u[1][k][j-2][i] - 4.0*u[1][k][j-1][i] + 
               6.0*u[1][k][j][i] - 4.0*u[1][k][j+1][i] + 
               u[1][k][j+2][i] );
          rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
            (  u[2][k][j-2][i] - 4.0*u[2][k][j-1][i] + 
               6.0*u[2][k][j][i] - 4.0*u[2][k][j+1][i] + 
               u[2][k][j+2][i] );
          rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
            (  u[3][k][j-2][i] - 4.0*u[3][k][j-1][i] + 
               6.0*u[3][k][j][i] - 4.0*u[3][k][j+1][i] + 
               u[3][k][j+2][i] );
          rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
            (  u[4][k][j-2][i] - 4.0*u[4][k][j-1][i] + 
               6.0*u[4][k][j][i] - 4.0*u[4][k][j+1][i] + 
               u[4][k][j+2][i] );
      }
    }
  }

  for (k = 1; k <= gp22; k++) {
    for (i = 1; i <= gp02; i++) {
		j = gp1-3;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
          ( u[0][k][j-2][i] - 4.0*u[0][k][j-1][i] + 
            6.0*u[0][k][j][i] - 4.0*u[0][k][j+1][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
          ( u[1][k][j-2][i] - 4.0*u[1][k][j-1][i] + 
            6.0*u[1][k][j][i] - 4.0*u[1][k][j+1][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
          ( u[2][k][j-2][i] - 4.0*u[2][k][j-1][i] + 
            6.0*u[2][k][j][i] - 4.0*u[2][k][j+1][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
          ( u[3][k][j-2][i] - 4.0*u[3][k][j-1][i] + 
            6.0*u[3][k][j][i] - 4.0*u[3][k][j+1][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
          ( u[4][k][j-2][i] - 4.0*u[4][k][j-1][i] + 
            6.0*u[4][k][j][i] - 4.0*u[4][k][j+1][i] );
    
		j = gp12;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
          ( u[0][k][j-2][i] - 4.*u[0][k][j-1][i] +
            5.*u[0][k][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
          ( u[1][k][j-2][i] - 4.*u[1][k][j-1][i] +
            5.*u[1][k][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
          ( u[2][k][j-2][i] - 4.*u[2][k][j-1][i] +
            5.*u[2][k][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
          ( u[3][k][j-2][i] - 4.*u[3][k][j-1][i] +
            5.*u[3][k][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
          ( u[4][k][j-2][i] - 4.*u[4][k][j-1][i] +
            5.*u[4][k][j][i] );
    }
  }
  
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
        wijk = ws[k][j][i];
        wp1  = ws[k+1][j][i];
        wm1  = ws[k-1][j][i];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dz1tz1 * 
          (u[0][k+1][j][i] - 2.0*u[0][k][j][i] + 
           u[0][k-1][j][i]) -
          tz2 * (u[3][k+1][j][i] - u[3][k-1][j][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i] + dz2tz1 * 
          (u[1][k+1][j][i] - 2.0*u[1][k][j][i] + 
           u[1][k-1][j][i]) +
          zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + 
              us[k-1][j][i]) -
          tz2 * (u[1][k+1][j][i]*wp1 - 
              u[1][k-1][j][i]*wm1);
        rhs[2][k][j][i] = rhs[2][k][j][i] + dz3tz1 * 
          (u[2][k+1][j][i] - 2.0*u[2][k][j][i] + 
           u[2][k-1][j][i]) +
          zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + 
              vs[k-1][j][i]) -
          tz2 * (u[2][k+1][j][i]*wp1 - 
              u[2][k-1][j][i]*wm1);
        rhs[3][k][j][i] = rhs[3][k][j][i] + dz4tz1 * 
          (u[3][k+1][j][i] - 2.0*u[3][k][j][i] + 
           u[3][k-1][j][i]) +
          zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
          tz2 * (u[3][k+1][j][i]*wp1 - 
              u[3][k-1][j][i]*wm1 +
              (u[4][k+1][j][i] - square[k+1][j][i] - 
               u[4][k-1][j][i] + square[k-1][j][i])
              *c2);
        rhs[4][k][j][i] = rhs[4][k][j][i] + dz5tz1 * 
          (u[4][k+1][j][i] - 2.0*u[4][k][j][i] + 
           u[4][k-1][j][i]) +
          zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + 
              qs[k-1][j][i]) +
          zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + 
              wm1*wm1) +
          zzcon5 * (u[4][k+1][j][i]*rho_i[k+1][j][i] - 
              2.0*u[4][k][j][i]*rho_i[k][j][i] +
              u[4][k-1][j][i]*rho_i[k-1][j][i]) -
          tz2 * ( (c1*u[4][k+1][j][i] - 
                c2*square[k+1][j][i])*wp1 -
              (c1*u[4][k-1][j][i] - 
               c2*square[k-1][j][i])*wm1);
      }
    }
  }
  
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
  		k = 1;
        rhs[0][k][j][i] = rhs[0][k][j][i]- dssp * 
          ( 5.0*u[0][k][j][i] - 4.0*u[0][k+1][j][i] +
            u[0][k+2][j][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i]- dssp * 
          ( 5.0*u[1][k][j][i] - 4.0*u[1][k+1][j][i] +
            u[1][k+2][j][i]);
        rhs[2][k][j][i] = rhs[2][k][j][i]- dssp * 
          ( 5.0*u[2][k][j][i] - 4.0*u[2][k+1][j][i] +
            u[2][k+2][j][i]);
        rhs[3][k][j][i] = rhs[3][k][j][i]- dssp * 
          ( 5.0*u[3][k][j][i] - 4.0*u[3][k+1][j][i] +
            u[3][k+2][j][i]);
        rhs[4][k][j][i] = rhs[4][k][j][i]- dssp * 
          ( 5.0*u[4][k][j][i] - 4.0*u[4][k+1][j][i] +
            u[4][k+2][j][i]);
  		
		k = 2;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
          (-4.0*u[0][k-1][j][i] + 6.0*u[0][k][j][i] -
           4.0*u[0][k+1][j][i] + u[0][k+2][j][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
          (-4.0*u[1][k-1][j][i] + 6.0*u[1][k][j][i] -
           4.0*u[1][k+1][j][i] + u[1][k+2][j][i]);
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
          (-4.0*u[2][k-1][j][i] + 6.0*u[2][k][j][i] -
           4.0*u[2][k+1][j][i] + u[2][k+2][j][i]);
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
          (-4.0*u[3][k-1][j][i] + 6.0*u[3][k][j][i] -
           4.0*u[3][k+1][j][i] + u[3][k+2][j][i]);
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
          (-4.0*u[4][k-1][j][i] + 6.0*u[4][k][j][i] -
           4.0*u[4][k+1][j][i] + u[4][k+2][j][i]);
    }
  }

  for (k = 3; k <= gp2-4; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
          rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
            (  u[0][k-2][j][i] - 4.0*u[0][k-1][j][i] + 
               6.0*u[0][k][j][i] - 4.0*u[0][k+1][j][i] + 
               u[0][k+2][j][i] );
          rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
            (  u[1][k-2][j][i] - 4.0*u[1][k-1][j][i] + 
               6.0*u[1][k][j][i] - 4.0*u[1][k+1][j][i] + 
               u[1][k+2][j][i] );
          rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
            (  u[2][k-2][j][i] - 4.0*u[2][k-1][j][i] + 
               6.0*u[2][k][j][i] - 4.0*u[2][k+1][j][i] + 
               u[2][k+2][j][i] );
          rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
            (  u[3][k-2][j][i] - 4.0*u[3][k-1][j][i] + 
               6.0*u[3][k][j][i] - 4.0*u[3][k+1][j][i] + 
               u[3][k+2][j][i] );
          rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
            (  u[4][k-2][j][i] - 4.0*u[4][k-1][j][i] + 
               6.0*u[4][k][j][i] - 4.0*u[4][k+1][j][i] + 
               u[4][k+2][j][i] );
      }
    }
  }

  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
		k = gp2-3;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
          ( u[0][k-2][j][i] - 4.0*u[0][k-1][j][i] + 
            6.0*u[0][k][j][i] - 4.0*u[0][k+1][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
          ( u[1][k-2][j][i] - 4.0*u[1][k-1][j][i] + 
            6.0*u[1][k][j][i] - 4.0*u[1][k+1][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
          ( u[2][k-2][j][i] - 4.0*u[2][k-1][j][i] + 
            6.0*u[2][k][j][i] - 4.0*u[2][k+1][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
          ( u[3][k-2][j][i] - 4.0*u[3][k-1][j][i] + 
            6.0*u[3][k][j][i] - 4.0*u[3][k+1][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
          ( u[4][k-2][j][i] - 4.0*u[4][k-1][j][i] + 
            6.0*u[4][k][j][i] - 4.0*u[4][k+1][j][i] );
  
  		k = gp22;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
          ( u[0][k-2][j][i] - 4.*u[0][k-1][j][i] +
            5.*u[0][k][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
          ( u[1][k-2][j][i] - 4.*u[1][k-1][j][i] +
            5.*u[1][k][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
          ( u[2][k-2][j][i] - 4.*u[2][k-1][j][i] +
            5.*u[2][k][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
          ( u[3][k-2][j][i] - 4.*u[3][k-1][j][i] +
            5.*u[3][k][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
          ( u[4][k-2][j][i] - 4.*u[4][k-1][j][i] +
            5.*u[4][k][j][i] );
    }
  }

  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
          rhs[0][k][j][i] = rhs[0][k][j][i] * dt;
          rhs[1][k][j][i] = rhs[1][k][j][i] * dt;
          rhs[2][k][j][i] = rhs[2][k][j][i] * dt;
          rhs[3][k][j][i] = rhs[3][k][j][i] * dt;
          rhs[4][k][j][i] = rhs[4][k][j][i] * dt;
      }
    }
  }

  static int rhs_stage_logged = 0;
  if (!rhs_stage_logged) {
    rhs_stage_logged = 1;
  }

  static int rhs_debug_printed = 0;
  if (!rhs_debug_printed) {
    rhs_debug_printed = 1;
    printf("DEBUG rhs sample: %e %e %e %e %e\\n",
           rhs[0][1][1][1], rhs[1][1][1][1], rhs[2][1][1][1],
           rhs[3][1][1][1], rhs[4][1][1][1]);
    printf("DEBUG rhs last: %e %e %e %e %e\\n",
           rhs[0][gp22][gp12][gp02], rhs[1][gp22][gp12][gp02],
           rhs[2][gp22][gp12][gp02], rhs[3][gp22][gp12][gp02],
           rhs[4][gp22][gp12][gp02]);
  }
  static int rhs_dumped = 0;
  if (!rhs_dumped) {
    rhs_dumped = 1;
    FILE *rhs_file = fopen("/tmp/golden_rhs.txt", "w");
    if (rhs_file) {
      for (k = 1; k <= gp22; k++) {
        for (j = 1; j <= gp12; j++) {
          for (i = 1; i <= gp02; i++) {
            for (m = 0; m < 5; m++) {
              fprintf(rhs_file, "%d %d %d %d %24.16e\n",
                      m, k, j, i, rhs[m][k][j][i]);
            }
          }
        }
      }
      fclose(rhs_file);
    }
  }

}

void x_solve()
{
  int i, j, k, m, n, isize, z;

  int gp22, gp12;

  double fjacX[5][5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1];
  double njacX[5][5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1];
  double lhsX[5][5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1];
  
  double temp1,temp2,temp3,pivot,coeff;
  
  gp22 = grid_points[2]-2;
  gp12 = grid_points[1]-2;

  isize = grid_points[0]-1;

      for (i = 0; i <= isize; i++) {
    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
        temp1 = rho_i[k][j][i];
        temp2 = temp1 * temp1;
        temp3 = temp1 * temp2;
        
        fjacX[0][0][i][j][k] = 0.0;
        fjacX[0][1][i][j][k] = 1.0;
        fjacX[0][2][i][j][k] = 0.0;
        fjacX[0][3][i][j][k] = 0.0;
        fjacX[0][4][i][j][k] = 0.0;

        fjacX[1][0][i][j][k] = -(u[1][k][j][i] * temp2 * u[1][k][j][i])
          + c2 * qs[k][j][i];
        fjacX[1][1][i][j][k] = ( 2.0 - c2 ) * ( u[1][k][j][i] / u[0][k][j][i] );
        fjacX[1][2][i][j][k] = - c2 * ( u[2][k][j][i] * temp1 );
        fjacX[1][3][i][j][k] = - c2 * ( u[3][k][j][i] * temp1 );
        fjacX[1][4][i][j][k] = c2;

        fjacX[2][0][i][j][k] = - ( u[1][k][j][i]*u[2][k][j][i] ) * temp2;
        fjacX[2][1][i][j][k] = u[2][k][j][i] * temp1;
        fjacX[2][2][i][j][k] = u[1][k][j][i] * temp1;
        fjacX[2][3][i][j][k] = 0.0;
        fjacX[2][4][i][j][k] = 0.0;

        fjacX[3][0][i][j][k] = - ( u[1][k][j][i]*u[3][k][j][i] ) * temp2;
        fjacX[3][1][i][j][k] = u[3][k][j][i] * temp1;
        fjacX[3][2][i][j][k] = 0.0;
        fjacX[3][3][i][j][k] = u[1][k][j][i] * temp1;
        fjacX[3][4][i][j][k] = 0.0;

        fjacX[4][0][i][j][k] = ( c2 * 2.0 * square[k][j][i] - c1 * u[4][k][j][i] )
          * ( u[1][k][j][i] * temp2 );
        fjacX[4][1][i][j][k] = c1 *  u[4][k][j][i] * temp1 
          - c2 * ( u[1][k][j][i]*u[1][k][j][i] * temp2 + qs[k][j][i] );
        fjacX[4][2][i][j][k] = - c2 * ( u[2][k][j][i]*u[1][k][j][i] ) * temp2;
        fjacX[4][3][i][j][k] = - c2 * ( u[3][k][j][i]*u[1][k][j][i] ) * temp2;
        fjacX[4][4][i][j][k] = c1 * ( u[1][k][j][i] * temp1 );

        njacX[0][0][i][j][k] = 0.0;
        njacX[0][1][i][j][k] = 0.0;
        njacX[0][2][i][j][k] = 0.0;
        njacX[0][3][i][j][k] = 0.0;
        njacX[0][4][i][j][k] = 0.0;

        njacX[1][0][i][j][k] = - con43 * c3c4 * temp2 * u[1][k][j][i];
        njacX[1][1][i][j][k] =   con43 * c3c4 * temp1;
        njacX[1][2][i][j][k] =   0.0;
        njacX[1][3][i][j][k] =   0.0;
        njacX[1][4][i][j][k] =   0.0;

        njacX[2][0][i][j][k] = - c3c4 * temp2 * u[2][k][j][i];
        njacX[2][1][i][j][k] =   0.0;
        njacX[2][2][i][j][k] =   c3c4 * temp1;
        njacX[2][3][i][j][k] =   0.0;
        njacX[2][4][i][j][k] =   0.0;

        njacX[3][0][i][j][k] = - c3c4 * temp2 * u[3][k][j][i];
        njacX[3][1][i][j][k] =   0.0;
        njacX[3][2][i][j][k] =   0.0;
        njacX[3][3][i][j][k] =   c3c4 * temp1;
        njacX[3][4][i][j][k] =   0.0;

        njacX[4][0][i][j][k] = - ( con43 * c3c4
            - c1345 ) * temp3 * (u[1][k][j][i]*u[1][k][j][i])
          - ( c3c4 - c1345 ) * temp3 * (u[2][k][j][i]*u[2][k][j][i])
          - ( c3c4 - c1345 ) * temp3 * (u[3][k][j][i]*u[3][k][j][i])
          - c1345 * temp2 * u[4][k][j][i];

        njacX[4][1][i][j][k] = ( con43 * c3c4
            - c1345 ) * temp2 * u[1][k][j][i];
        njacX[4][2][i][j][k] = ( c3c4 - c1345 ) * temp2 * u[2][k][j][i];
        njacX[4][3][i][j][k] = ( c3c4 - c1345 ) * temp2 * u[3][k][j][i];
        njacX[4][4][i][j][k] = ( c1345 ) * temp1;
      }
	}
  }

    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
  		for (n = 0; n < 5; n++) {
		   for (m = 0; m < 5; m++){
      			lhsX[m][n][0][0][j][k] = 0.0;
      			lhsX[m][n][1][0][j][k] = 0.0;
      			lhsX[m][n][2][0][j][k] = 0.0;
      			lhsX[m][n][0][isize][j][k] = 0.0;
      			lhsX[m][n][1][isize][j][k] = 0.0;
      			lhsX[m][n][2][isize][j][k] = 0.0;
      		}
  		}
	}
  }

    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
    		lhsX[0][0][1][0][j][k] = 1.0;
    		lhsX[0][0][1][isize][j][k] = 1.0;
    		lhsX[1][1][1][0][j][k] = 1.0;
    		lhsX[1][1][1][isize][j][k] = 1.0;
    		lhsX[2][2][1][0][j][k] = 1.0;
    		lhsX[2][2][1][isize][j][k] = 1.0;
    		lhsX[3][3][1][0][j][k] = 1.0;
    		lhsX[3][3][1][isize][j][k] = 1.0;
    		lhsX[4][4][1][0][j][k] = 1.0;
    		lhsX[4][4][1][isize][j][k] = 1.0;
	}
  }
  
 	   for (i = 1; i <= isize-1; i++) {
    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
        temp1 = dt * tx1;
        temp2 = dt * tx2;

        lhsX[0][0][AA][i][j][k] = - temp2 * fjacX[0][0][i-1][j][k]
          - temp1 * njacX[0][0][i-1][j][k]
          - temp1 * dx1; 
        lhsX[0][1][AA][i][j][k] = - temp2 * fjacX[0][1][i-1][j][k]
          - temp1 * njacX[0][1][i-1][j][k];
        lhsX[0][2][AA][i][j][k] = - temp2 * fjacX[0][2][i-1][j][k]
          - temp1 * njacX[0][2][i-1][j][k];
        lhsX[0][3][AA][i][j][k] = - temp2 * fjacX[0][3][i-1][j][k]
          - temp1 * njacX[0][3][i-1][j][k];
        lhsX[0][4][AA][i][j][k] = - temp2 * fjacX[0][4][i-1][j][k]
          - temp1 * njacX[0][4][i-1][j][k];

        lhsX[1][0][AA][i][j][k] = - temp2 * fjacX[1][0][i-1][j][k]
          - temp1 * njacX[1][0][i-1][j][k];
        lhsX[1][1][AA][i][j][k] = - temp2 * fjacX[1][1][i-1][j][k]
          - temp1 * njacX[1][1][i-1][j][k]
          - temp1 * dx2;
        lhsX[1][2][AA][i][j][k] = - temp2 * fjacX[1][2][i-1][j][k]
          - temp1 * njacX[1][2][i-1][j][k];
        lhsX[1][3][AA][i][j][k] = - temp2 * fjacX[1][3][i-1][j][k]
          - temp1 * njacX[1][3][i-1][j][k];
        lhsX[1][4][AA][i][j][k] = - temp2 * fjacX[1][4][i-1][j][k]
          - temp1 * njacX[1][4][i-1][j][k];

        lhsX[2][0][AA][i][j][k] = - temp2 * fjacX[2][0][i-1][j][k]
          - temp1 * njacX[2][0][i-1][j][k];
        lhsX[2][1][AA][i][j][k] = - temp2 * fjacX[2][1][i-1][j][k]
          - temp1 * njacX[2][1][i-1][j][k];
        lhsX[2][2][AA][i][j][k] = - temp2 * fjacX[2][2][i-1][j][k]
          - temp1 * njacX[2][2][i-1][j][k]
          - temp1 * dx3;
        lhsX[2][3][AA][i][j][k] = - temp2 * fjacX[2][3][i-1][j][k]
          - temp1 * njacX[2][3][i-1][j][k];
        lhsX[2][4][AA][i][j][k] = - temp2 * fjacX[2][4][i-1][j][k]
          - temp1 * njacX[2][4][i-1][j][k];

        lhsX[3][0][AA][i][j][k] = - temp2 * fjacX[3][0][i-1][j][k]
          - temp1 * njacX[3][0][i-1][j][k];
        lhsX[3][1][AA][i][j][k] = - temp2 * fjacX[3][1][i-1][j][k]
          - temp1 * njacX[3][1][i-1][j][k];
        lhsX[3][2][AA][i][j][k] = - temp2 * fjacX[3][2][i-1][j][k]
          - temp1 * njacX[3][2][i-1][j][k];
        lhsX[3][3][AA][i][j][k] = - temp2 * fjacX[3][3][i-1][j][k]
          - temp1 * njacX[3][3][i-1][j][k]
          - temp1 * dx4;
        lhsX[3][4][AA][i][j][k] = - temp2 * fjacX[3][4][i-1][j][k]
          - temp1 * njacX[3][4][i-1][j][k];

        lhsX[4][0][AA][i][j][k] = - temp2 * fjacX[4][0][i-1][j][k]
          - temp1 * njacX[4][0][i-1][j][k];
        lhsX[4][1][AA][i][j][k] = - temp2 * fjacX[4][1][i-1][j][k]
          - temp1 * njacX[4][1][i-1][j][k];
        lhsX[4][2][AA][i][j][k] = - temp2 * fjacX[4][2][i-1][j][k]
          - temp1 * njacX[4][2][i-1][j][k];
        lhsX[4][3][AA][i][j][k] = - temp2 * fjacX[4][3][i-1][j][k]
          - temp1 * njacX[4][3][i-1][j][k];
        lhsX[4][4][AA][i][j][k] = - temp2 * fjacX[4][4][i-1][j][k]
          - temp1 * njacX[4][4][i-1][j][k]
          - temp1 * dx5;

        lhsX[0][0][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[0][0][i][j][k]
          + temp1 * 2.0 * dx1;
        lhsX[0][1][BB][i][j][k] = temp1 * 2.0 * njacX[0][1][i][j][k];
        lhsX[0][2][BB][i][j][k] = temp1 * 2.0 * njacX[0][2][i][j][k];
        lhsX[0][3][BB][i][j][k] = temp1 * 2.0 * njacX[0][3][i][j][k];
        lhsX[0][4][BB][i][j][k] = temp1 * 2.0 * njacX[0][4][i][j][k];

        lhsX[1][0][BB][i][j][k] = temp1 * 2.0 * njacX[1][0][i][j][k];
        lhsX[1][1][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[1][1][i][j][k]
          + temp1 * 2.0 * dx2;
        lhsX[1][2][BB][i][j][k] = temp1 * 2.0 * njacX[1][2][i][j][k];
        lhsX[1][3][BB][i][j][k] = temp1 * 2.0 * njacX[1][3][i][j][k];
        lhsX[1][4][BB][i][j][k] = temp1 * 2.0 * njacX[1][4][i][j][k];

        lhsX[2][0][BB][i][j][k] = temp1 * 2.0 * njacX[2][0][i][j][k];
        lhsX[2][1][BB][i][j][k] = temp1 * 2.0 * njacX[2][1][i][j][k];
        lhsX[2][2][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[2][2][i][j][k]
          + temp1 * 2.0 * dx3;
        lhsX[2][3][BB][i][j][k] = temp1 * 2.0 * njacX[2][3][i][j][k];
        lhsX[2][4][BB][i][j][k] = temp1 * 2.0 * njacX[2][4][i][j][k];

        lhsX[3][0][BB][i][j][k] = temp1 * 2.0 * njacX[3][0][i][j][k];
        lhsX[3][1][BB][i][j][k] = temp1 * 2.0 * njacX[3][1][i][j][k];
        lhsX[3][2][BB][i][j][k] = temp1 * 2.0 * njacX[3][2][i][j][k];
        lhsX[3][3][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[3][3][i][j][k]
          + temp1 * 2.0 * dx4;
        lhsX[3][4][BB][i][j][k] = temp1 * 2.0 * njacX[3][4][i][j][k];

        lhsX[4][0][BB][i][j][k] = temp1 * 2.0 * njacX[4][0][i][j][k];
        lhsX[4][1][BB][i][j][k] = temp1 * 2.0 * njacX[4][1][i][j][k];
        lhsX[4][2][BB][i][j][k] = temp1 * 2.0 * njacX[4][2][i][j][k];
        lhsX[4][3][BB][i][j][k] = temp1 * 2.0 * njacX[4][3][i][j][k];
        lhsX[4][4][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[4][4][i][j][k]
          + temp1 * 2.0 * dx5;

        lhsX[0][0][CC][i][j][k] =  temp2 * fjacX[0][0][i+1][j][k]
          - temp1 * njacX[0][0][i+1][j][k]
          - temp1 * dx1;
        lhsX[0][1][CC][i][j][k] =  temp2 * fjacX[0][1][i+1][j][k]
          - temp1 * njacX[0][1][i+1][j][k];
        lhsX[0][2][CC][i][j][k] =  temp2 * fjacX[0][2][i+1][j][k]
          - temp1 * njacX[0][2][i+1][j][k];
        lhsX[0][3][CC][i][j][k] =  temp2 * fjacX[0][3][i+1][j][k]
          - temp1 * njacX[0][3][i+1][j][k];
        lhsX[0][4][CC][i][j][k] =  temp2 * fjacX[0][4][i+1][j][k]
          - temp1 * njacX[0][4][i+1][j][k];

        lhsX[1][0][CC][i][j][k] =  temp2 * fjacX[1][0][i+1][j][k]
          - temp1 * njacX[1][0][i+1][j][k];
        lhsX[1][1][CC][i][j][k] =  temp2 * fjacX[1][1][i+1][j][k]
          - temp1 * njacX[1][1][i+1][j][k]
          - temp1 * dx2;
        lhsX[1][2][CC][i][j][k] =  temp2 * fjacX[1][2][i+1][j][k]
          - temp1 * njacX[1][2][i+1][j][k];
        lhsX[1][3][CC][i][j][k] =  temp2 * fjacX[1][3][i+1][j][k]
          - temp1 * njacX[1][3][i+1][j][k];
        lhsX[1][4][CC][i][j][k] =  temp2 * fjacX[1][4][i+1][j][k]
          - temp1 * njacX[1][4][i+1][j][k];

        lhsX[2][0][CC][i][j][k] =  temp2 * fjacX[2][0][i+1][j][k]
          - temp1 * njacX[2][0][i+1][j][k];
        lhsX[2][1][CC][i][j][k] =  temp2 * fjacX[2][1][i+1][j][k]
          - temp1 * njacX[2][1][i+1][j][k];
        lhsX[2][2][CC][i][j][k] =  temp2 * fjacX[2][2][i+1][j][k]
          - temp1 * njacX[2][2][i+1][j][k]
          - temp1 * dx3;
        lhsX[2][3][CC][i][j][k] =  temp2 * fjacX[2][3][i+1][j][k]
          - temp1 * njacX[2][3][i+1][j][k];
        lhsX[2][4][CC][i][j][k] =  temp2 * fjacX[2][4][i+1][j][k]
          - temp1 * njacX[2][4][i+1][j][k];

        lhsX[3][0][CC][i][j][k] =  temp2 * fjacX[3][0][i+1][j][k]
          - temp1 * njacX[3][0][i+1][j][k];
        lhsX[3][1][CC][i][j][k] =  temp2 * fjacX[3][1][i+1][j][k]
          - temp1 * njacX[3][1][i+1][j][k];
        lhsX[3][2][CC][i][j][k] =  temp2 * fjacX[3][2][i+1][j][k]
          - temp1 * njacX[3][2][i+1][j][k];
        lhsX[3][3][CC][i][j][k] =  temp2 * fjacX[3][3][i+1][j][k]
          - temp1 * njacX[3][3][i+1][j][k]
          - temp1 * dx4;
        lhsX[3][4][CC][i][j][k] =  temp2 * fjacX[3][4][i+1][j][k]
          - temp1 * njacX[3][4][i+1][j][k];

        lhsX[4][0][CC][i][j][k] =  temp2 * fjacX[4][0][i+1][j][k]
          - temp1 * njacX[4][0][i+1][j][k];
        lhsX[4][1][CC][i][j][k] =  temp2 * fjacX[4][1][i+1][j][k]
          - temp1 * njacX[4][1][i+1][j][k];
        lhsX[4][2][CC][i][j][k] =  temp2 * fjacX[4][2][i+1][j][k]
          - temp1 * njacX[4][2][i+1][j][k];
        lhsX[4][3][CC][i][j][k] =  temp2 * fjacX[4][3][i+1][j][k]
          - temp1 * njacX[4][3][i+1][j][k];
        lhsX[4][4][CC][i][j][k] =  temp2 * fjacX[4][4][i+1][j][k]
          - temp1 * njacX[4][4][i+1][j][k]
          - temp1 * dx5;
      }
    }
  }

    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {

  pivot = 1.00/lhsX[0][0][BB][0][j][k];
  lhsX[0][1][BB][0][j][k] = lhsX[0][1][BB][0][j][k]*pivot;
  lhsX[0][2][BB][0][j][k] = lhsX[0][2][BB][0][j][k]*pivot;
  lhsX[0][3][BB][0][j][k] = lhsX[0][3][BB][0][j][k]*pivot;
  lhsX[0][4][BB][0][j][k] = lhsX[0][4][BB][0][j][k]*pivot;
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k]*pivot;
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k]*pivot;
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k]*pivot;
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k]*pivot;
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k]*pivot;
  rhs[0][k][j][0]   = rhs[0][k][j][0]  *pivot;

  coeff = lhsX[1][0][BB][0][j][k];
  lhsX[1][1][BB][0][j][k]= lhsX[1][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
  lhsX[1][2][BB][0][j][k]= lhsX[1][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
  lhsX[1][3][BB][0][j][k]= lhsX[1][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
  lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
  rhs[1][k][j][0]   = rhs[1][k][j][0]   - coeff*rhs[0][k][j][0];

  coeff = lhsX[2][0][BB][0][j][k];
  lhsX[2][1][BB][0][j][k]= lhsX[2][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
  lhsX[2][2][BB][0][j][k]= lhsX[2][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
  lhsX[2][3][BB][0][j][k]= lhsX[2][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
  lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
  rhs[2][k][j][0]   = rhs[2][k][j][0]   - coeff*rhs[0][k][j][0];

  coeff = lhsX[3][0][BB][0][j][k];
  lhsX[3][1][BB][0][j][k]= lhsX[3][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
  lhsX[3][2][BB][0][j][k]= lhsX[3][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
  lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
  lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
  rhs[3][k][j][0]   = rhs[3][k][j][0]   - coeff*rhs[0][k][j][0];

  coeff = lhsX[4][0][BB][0][j][k];
  lhsX[4][1][BB][0][j][k]= lhsX[4][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
  lhsX[4][2][BB][0][j][k]= lhsX[4][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
  lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
  lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
  rhs[4][k][j][0]   = rhs[4][k][j][0]   - coeff*rhs[0][k][j][0];

  pivot = 1.00/lhsX[1][1][BB][0][j][k];
  lhsX[1][2][BB][0][j][k] = lhsX[1][2][BB][0][j][k]*pivot;
  lhsX[1][3][BB][0][j][k] = lhsX[1][3][BB][0][j][k]*pivot;
  lhsX[1][4][BB][0][j][k] = lhsX[1][4][BB][0][j][k]*pivot;
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k]*pivot;
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k]*pivot;
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k]*pivot;
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k]*pivot;
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k]*pivot;
  rhs[1][k][j][0]   = rhs[1][k][j][0]  *pivot;

  coeff = lhsX[0][1][BB][0][j][k];
  lhsX[0][2][BB][0][j][k]= lhsX[0][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
  lhsX[0][3][BB][0][j][k]= lhsX[0][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
  lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
  rhs[0][k][j][0]   = rhs[0][k][j][0]   - coeff*rhs[1][k][j][0];

  coeff = lhsX[2][1][BB][0][j][k];
  lhsX[2][2][BB][0][j][k]= lhsX[2][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
  lhsX[2][3][BB][0][j][k]= lhsX[2][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
  lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
  rhs[2][k][j][0]   = rhs[2][k][j][0]   - coeff*rhs[1][k][j][0];

  coeff = lhsX[3][1][BB][0][j][k];
  lhsX[3][2][BB][0][j][k]= lhsX[3][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
  lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
  lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
  rhs[3][k][j][0]   = rhs[3][k][j][0]   - coeff*rhs[1][k][j][0];

  coeff = lhsX[4][1][BB][0][j][k];
  lhsX[4][2][BB][0][j][k]= lhsX[4][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
  lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
  lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
  rhs[4][k][j][0]   = rhs[4][k][j][0]   - coeff*rhs[1][k][j][0];

  pivot = 1.00/lhsX[2][2][BB][0][j][k];
  lhsX[2][3][BB][0][j][k] = lhsX[2][3][BB][0][j][k]*pivot;
  lhsX[2][4][BB][0][j][k] = lhsX[2][4][BB][0][j][k]*pivot;
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k]*pivot;
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k]*pivot;
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k]*pivot;
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k]*pivot;
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k]*pivot;
  rhs[2][k][j][0]   = rhs[2][k][j][0]  *pivot;

  coeff = lhsX[0][2][BB][0][j][k];
  lhsX[0][3][BB][0][j][k]= lhsX[0][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
  lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
  rhs[0][k][j][0]   = rhs[0][k][j][0]   - coeff*rhs[2][k][j][0];

  coeff = lhsX[1][2][BB][0][j][k];
  lhsX[1][3][BB][0][j][k]= lhsX[1][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
  lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
  rhs[1][k][j][0]   = rhs[1][k][j][0]   - coeff*rhs[2][k][j][0];

  coeff = lhsX[3][2][BB][0][j][k];
  lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
  lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
  rhs[3][k][j][0]   = rhs[3][k][j][0]   - coeff*rhs[2][k][j][0];

  coeff = lhsX[4][2][BB][0][j][k];
  lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
  lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
  rhs[4][k][j][0]   = rhs[4][k][j][0]   - coeff*rhs[2][k][j][0];

  pivot = 1.00/lhsX[3][3][BB][0][j][k];
  lhsX[3][4][BB][0][j][k] = lhsX[3][4][BB][0][j][k]*pivot;
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k]*pivot;
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k]*pivot;
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k]*pivot;
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k]*pivot;
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k]*pivot;
  rhs[3][k][j][0]   = rhs[3][k][j][0]  *pivot;

  coeff = lhsX[0][3][BB][0][j][k];
  lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
  rhs[0][k][j][0]   = rhs[0][k][j][0]   - coeff*rhs[3][k][j][0];

  coeff = lhsX[1][3][BB][0][j][k];
  lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
  rhs[1][k][j][0]   = rhs[1][k][j][0]   - coeff*rhs[3][k][j][0];

  coeff = lhsX[2][3][BB][0][j][k];
  lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
  rhs[2][k][j][0]   = rhs[2][k][j][0]   - coeff*rhs[3][k][j][0];

  coeff = lhsX[4][3][BB][0][j][k];
  lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
  rhs[4][k][j][0]   = rhs[4][k][j][0]   - coeff*rhs[3][k][j][0];

  pivot = 1.00/lhsX[4][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k]*pivot;
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k]*pivot;
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k]*pivot;
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k]*pivot;
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k]*pivot;
  rhs[4][k][j][0]   = rhs[4][k][j][0]  *pivot;

  coeff = lhsX[0][4][BB][0][j][k];
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
  rhs[0][k][j][0]   = rhs[0][k][j][0]   - coeff*rhs[4][k][j][0];

  coeff = lhsX[1][4][BB][0][j][k];
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
  rhs[1][k][j][0]   = rhs[1][k][j][0]   - coeff*rhs[4][k][j][0];

  coeff = lhsX[2][4][BB][0][j][k];
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
  rhs[2][k][j][0]   = rhs[2][k][j][0]   - coeff*rhs[4][k][j][0];

  coeff = lhsX[3][4][BB][0][j][k];
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
  rhs[3][k][j][0]   = rhs[3][k][j][0]   - coeff*rhs[4][k][j][0];
	
	}
  }

    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
      for (i = 1; i <= isize-1; i++) {
        
  rhs[0][k][j][i] = rhs[0][k][j][i] - lhsX[0][0][AA][i][j][k]*rhs[0][k][j][i-1]
                    - lhsX[0][1][AA][i][j][k]*rhs[1][k][j][i-1]
                    - lhsX[0][2][AA][i][j][k]*rhs[2][k][j][i-1]
                    - lhsX[0][3][AA][i][j][k]*rhs[3][k][j][i-1]
                    - lhsX[0][4][AA][i][j][k]*rhs[4][k][j][i-1];
  rhs[1][k][j][i] = rhs[1][k][j][i] - lhsX[1][0][AA][i][j][k]*rhs[0][k][j][i-1]
                    - lhsX[1][1][AA][i][j][k]*rhs[1][k][j][i-1]
                    - lhsX[1][2][AA][i][j][k]*rhs[2][k][j][i-1]
                    - lhsX[1][3][AA][i][j][k]*rhs[3][k][j][i-1]
                    - lhsX[1][4][AA][i][j][k]*rhs[4][k][j][i-1];
  rhs[2][k][j][i] = rhs[2][k][j][i] - lhsX[2][0][AA][i][j][k]*rhs[0][k][j][i-1]
                    - lhsX[2][1][AA][i][j][k]*rhs[1][k][j][i-1]
                    - lhsX[2][2][AA][i][j][k]*rhs[2][k][j][i-1]
                    - lhsX[2][3][AA][i][j][k]*rhs[3][k][j][i-1]
                    - lhsX[2][4][AA][i][j][k]*rhs[4][k][j][i-1];
  rhs[3][k][j][i] = rhs[3][k][j][i] - lhsX[3][0][AA][i][j][k]*rhs[0][k][j][i-1]
                    - lhsX[3][1][AA][i][j][k]*rhs[1][k][j][i-1]
                    - lhsX[3][2][AA][i][j][k]*rhs[2][k][j][i-1]
                    - lhsX[3][3][AA][i][j][k]*rhs[3][k][j][i-1]
                    - lhsX[3][4][AA][i][j][k]*rhs[4][k][j][i-1];
  rhs[4][k][j][i] = rhs[4][k][j][i] - lhsX[4][0][AA][i][j][k]*rhs[0][k][j][i-1]
                    - lhsX[4][1][AA][i][j][k]*rhs[1][k][j][i-1]
                    - lhsX[4][2][AA][i][j][k]*rhs[2][k][j][i-1]
                    - lhsX[4][3][AA][i][j][k]*rhs[3][k][j][i-1]
                    - lhsX[4][4][AA][i][j][k]*rhs[4][k][j][i-1];
		
  lhsX[0][0][BB][i][j][k] = lhsX[0][0][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[1][0][BB][i][j][k] = lhsX[1][0][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[2][0][BB][i][j][k] = lhsX[2][0][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[3][0][BB][i][j][k] = lhsX[3][0][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[4][0][BB][i][j][k] = lhsX[4][0][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[0][1][BB][i][j][k] = lhsX[0][1][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[1][1][BB][i][j][k] = lhsX[1][1][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[2][1][BB][i][j][k] = lhsX[2][1][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[3][1][BB][i][j][k] = lhsX[3][1][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[4][1][BB][i][j][k] = lhsX[4][1][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[0][2][BB][i][j][k] = lhsX[0][2][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[1][2][BB][i][j][k] = lhsX[1][2][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[2][2][BB][i][j][k] = lhsX[2][2][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[3][2][BB][i][j][k] = lhsX[3][2][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[4][2][BB][i][j][k] = lhsX[4][2][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[0][3][BB][i][j][k] = lhsX[0][3][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[1][3][BB][i][j][k] = lhsX[1][3][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[2][3][BB][i][j][k] = lhsX[2][3][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[3][3][BB][i][j][k] = lhsX[3][3][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[4][3][BB][i][j][k] = lhsX[4][3][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[0][4][BB][i][j][k] = lhsX[0][4][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
  lhsX[1][4][BB][i][j][k] = lhsX[1][4][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
  lhsX[2][4][BB][i][j][k] = lhsX[2][4][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
  lhsX[3][4][BB][i][j][k] = lhsX[3][4][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
  lhsX[4][4][BB][i][j][k] = lhsX[4][4][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];

  pivot = 1.00/lhsX[0][0][BB][i][j][k];
  lhsX[0][1][BB][i][j][k] = lhsX[0][1][BB][i][j][k]*pivot;
  lhsX[0][2][BB][i][j][k] = lhsX[0][2][BB][i][j][k]*pivot;
  lhsX[0][3][BB][i][j][k] = lhsX[0][3][BB][i][j][k]*pivot;
  lhsX[0][4][BB][i][j][k] = lhsX[0][4][BB][i][j][k]*pivot;
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k]*pivot;
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k]*pivot;
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k]*pivot;
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k]*pivot;
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k]*pivot;
  rhs[0][k][j][i]   = rhs[0][k][j][i]  *pivot;

  coeff = lhsX[1][0][BB][i][j][k];
  lhsX[1][1][BB][i][j][k]= lhsX[1][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
  lhsX[1][2][BB][i][j][k]= lhsX[1][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
  lhsX[1][3][BB][i][j][k]= lhsX[1][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
  lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsX[2][0][BB][i][j][k];
  lhsX[2][1][BB][i][j][k]= lhsX[2][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
  lhsX[2][2][BB][i][j][k]= lhsX[2][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
  lhsX[2][3][BB][i][j][k]= lhsX[2][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
  lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsX[3][0][BB][i][j][k];
  lhsX[3][1][BB][i][j][k]= lhsX[3][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
  lhsX[3][2][BB][i][j][k]= lhsX[3][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
  lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
  lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsX[4][0][BB][i][j][k];
  lhsX[4][1][BB][i][j][k]= lhsX[4][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
  lhsX[4][2][BB][i][j][k]= lhsX[4][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
  lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
  lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[0][k][j][i];

  pivot = 1.00/lhsX[1][1][BB][i][j][k];
  lhsX[1][2][BB][i][j][k] = lhsX[1][2][BB][i][j][k]*pivot;
  lhsX[1][3][BB][i][j][k] = lhsX[1][3][BB][i][j][k]*pivot;
  lhsX[1][4][BB][i][j][k] = lhsX[1][4][BB][i][j][k]*pivot;
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k]*pivot;
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k]*pivot;
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k]*pivot;
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k]*pivot;
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k]*pivot;
  rhs[1][k][j][i]   = rhs[1][k][j][i]  *pivot;

  coeff = lhsX[0][1][BB][i][j][k];
  lhsX[0][2][BB][i][j][k]= lhsX[0][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
  lhsX[0][3][BB][i][j][k]= lhsX[0][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
  lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsX[2][1][BB][i][j][k];
  lhsX[2][2][BB][i][j][k]= lhsX[2][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
  lhsX[2][3][BB][i][j][k]= lhsX[2][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
  lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsX[3][1][BB][i][j][k];
  lhsX[3][2][BB][i][j][k]= lhsX[3][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
  lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
  lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsX[4][1][BB][i][j][k];
  lhsX[4][2][BB][i][j][k]= lhsX[4][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
  lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
  lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[1][k][j][i];

  pivot = 1.00/lhsX[2][2][BB][i][j][k];
  lhsX[2][3][BB][i][j][k] = lhsX[2][3][BB][i][j][k]*pivot;
  lhsX[2][4][BB][i][j][k] = lhsX[2][4][BB][i][j][k]*pivot;
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k]*pivot;
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k]*pivot;
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k]*pivot;
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k]*pivot;
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k]*pivot;
  rhs[2][k][j][i]   = rhs[2][k][j][i]  *pivot;

  coeff = lhsX[0][2][BB][i][j][k];
  lhsX[0][3][BB][i][j][k]= lhsX[0][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
  lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsX[1][2][BB][i][j][k];
  lhsX[1][3][BB][i][j][k]= lhsX[1][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
  lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsX[3][2][BB][i][j][k];
  lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
  lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsX[4][2][BB][i][j][k];
  lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
  lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[2][k][j][i];

  pivot = 1.00/lhsX[3][3][BB][i][j][k];
  lhsX[3][4][BB][i][j][k] = lhsX[3][4][BB][i][j][k]*pivot;
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k]*pivot;
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k]*pivot;
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k]*pivot;
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k]*pivot;
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k]*pivot;
  rhs[3][k][j][i]   = rhs[3][k][j][i]  *pivot;

  coeff = lhsX[0][3][BB][i][j][k];
  lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsX[1][3][BB][i][j][k];
  lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsX[2][3][BB][i][j][k];
  lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsX[4][3][BB][i][j][k];
  lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[3][k][j][i];

  pivot = 1.00/lhsX[4][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k]*pivot;
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k]*pivot;
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k]*pivot;
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k]*pivot;
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k]*pivot;
  rhs[4][k][j][i]   = rhs[4][k][j][i]  *pivot;

  coeff = lhsX[0][4][BB][i][j][k];
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsX[1][4][BB][i][j][k];
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsX[2][4][BB][i][j][k];
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsX[3][4][BB][i][j][k];
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[4][k][j][i];

      }
    }
  }
      
    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
	
  rhs[0][k][j][isize] = rhs[0][k][j][isize] - lhsX[0][0][AA][isize][j][k]*rhs[0][k][j][isize-1]
                    - lhsX[0][1][AA][isize][j][k]*rhs[1][k][j][isize-1]
                    - lhsX[0][2][AA][isize][j][k]*rhs[2][k][j][isize-1]
                    - lhsX[0][3][AA][isize][j][k]*rhs[3][k][j][isize-1]
                    - lhsX[0][4][AA][isize][j][k]*rhs[4][k][j][isize-1];
  rhs[1][k][j][isize] = rhs[1][k][j][isize] - lhsX[1][0][AA][isize][j][k]*rhs[0][k][j][isize-1]
                    - lhsX[1][1][AA][isize][j][k]*rhs[1][k][j][isize-1]
                    - lhsX[1][2][AA][isize][j][k]*rhs[2][k][j][isize-1]
                    - lhsX[1][3][AA][isize][j][k]*rhs[3][k][j][isize-1]
                    - lhsX[1][4][AA][isize][j][k]*rhs[4][k][j][isize-1];
  rhs[2][k][j][isize] = rhs[2][k][j][isize] - lhsX[2][0][AA][isize][j][k]*rhs[0][k][j][isize-1]
                    - lhsX[2][1][AA][isize][j][k]*rhs[1][k][j][isize-1]
                    - lhsX[2][2][AA][isize][j][k]*rhs[2][k][j][isize-1]
                    - lhsX[2][3][AA][isize][j][k]*rhs[3][k][j][isize-1]
                    - lhsX[2][4][AA][isize][j][k]*rhs[4][k][j][isize-1];
  rhs[3][k][j][isize] = rhs[3][k][j][isize] - lhsX[3][0][AA][isize][j][k]*rhs[0][k][j][isize-1]
                    - lhsX[3][1][AA][isize][j][k]*rhs[1][k][j][isize-1]
                    - lhsX[3][2][AA][isize][j][k]*rhs[2][k][j][isize-1]
                    - lhsX[3][3][AA][isize][j][k]*rhs[3][k][j][isize-1]
                    - lhsX[3][4][AA][isize][j][k]*rhs[4][k][j][isize-1];
  rhs[4][k][j][isize] = rhs[4][k][j][isize] - lhsX[4][0][AA][isize][j][k]*rhs[0][k][j][isize-1]
                    - lhsX[4][1][AA][isize][j][k]*rhs[1][k][j][isize-1]
                    - lhsX[4][2][AA][isize][j][k]*rhs[2][k][j][isize-1]
                    - lhsX[4][3][AA][isize][j][k]*rhs[3][k][j][isize-1]
                    - lhsX[4][4][AA][isize][j][k]*rhs[4][k][j][isize-1];

	 }
  }
      
    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
	
  lhsX[0][0][BB][isize][j][k] = lhsX[0][0][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[1][0][BB][isize][j][k] = lhsX[1][0][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[2][0][BB][isize][j][k] = lhsX[2][0][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[3][0][BB][isize][j][k] = lhsX[3][0][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[4][0][BB][isize][j][k] = lhsX[4][0][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[0][1][BB][isize][j][k] = lhsX[0][1][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[1][1][BB][isize][j][k] = lhsX[1][1][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[2][1][BB][isize][j][k] = lhsX[2][1][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[3][1][BB][isize][j][k] = lhsX[3][1][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[4][1][BB][isize][j][k] = lhsX[4][1][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[0][2][BB][isize][j][k] = lhsX[0][2][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[1][2][BB][isize][j][k] = lhsX[1][2][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[2][2][BB][isize][j][k] = lhsX[2][2][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[3][2][BB][isize][j][k] = lhsX[3][2][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[4][2][BB][isize][j][k] = lhsX[4][2][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[0][3][BB][isize][j][k] = lhsX[0][3][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[1][3][BB][isize][j][k] = lhsX[1][3][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[2][3][BB][isize][j][k] = lhsX[2][3][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[3][3][BB][isize][j][k] = lhsX[3][3][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[4][3][BB][isize][j][k] = lhsX[4][3][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[0][4][BB][isize][j][k] = lhsX[0][4][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
  lhsX[1][4][BB][isize][j][k] = lhsX[1][4][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
  lhsX[2][4][BB][isize][j][k] = lhsX[2][4][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
  lhsX[3][4][BB][isize][j][k] = lhsX[3][4][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
  lhsX[4][4][BB][isize][j][k] = lhsX[4][4][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];

    }
  }
      
    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
	
  pivot = 1.00/lhsX[0][0][BB][isize][j][k];
  lhsX[0][1][BB][isize][j][k] = lhsX[0][1][BB][isize][j][k]*pivot;
  lhsX[0][2][BB][isize][j][k] = lhsX[0][2][BB][isize][j][k]*pivot;
  lhsX[0][3][BB][isize][j][k] = lhsX[0][3][BB][isize][j][k]*pivot;
  lhsX[0][4][BB][isize][j][k] = lhsX[0][4][BB][isize][j][k]*pivot;
  rhs[0][k][j][isize]   = rhs[0][k][j][isize]  *pivot;

  coeff = lhsX[1][0][BB][isize][j][k];
  lhsX[1][1][BB][isize][j][k]= lhsX[1][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
  lhsX[1][2][BB][isize][j][k]= lhsX[1][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
  lhsX[1][3][BB][isize][j][k]= lhsX[1][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
  lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
  rhs[1][k][j][isize]   = rhs[1][k][j][isize]   - coeff*rhs[0][k][j][isize];

  coeff = lhsX[2][0][BB][isize][j][k];
  lhsX[2][1][BB][isize][j][k]= lhsX[2][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
  lhsX[2][2][BB][isize][j][k]= lhsX[2][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
  lhsX[2][3][BB][isize][j][k]= lhsX[2][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
  lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
  rhs[2][k][j][isize]   = rhs[2][k][j][isize]   - coeff*rhs[0][k][j][isize];

  coeff = lhsX[3][0][BB][isize][j][k];
  lhsX[3][1][BB][isize][j][k]= lhsX[3][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
  lhsX[3][2][BB][isize][j][k]= lhsX[3][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
  lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
  lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
  rhs[3][k][j][isize]   = rhs[3][k][j][isize]   - coeff*rhs[0][k][j][isize];

  coeff = lhsX[4][0][BB][isize][j][k];
  lhsX[4][1][BB][isize][j][k]= lhsX[4][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
  lhsX[4][2][BB][isize][j][k]= lhsX[4][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
  lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
  lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
  rhs[4][k][j][isize]   = rhs[4][k][j][isize]   - coeff*rhs[0][k][j][isize];

  pivot = 1.00/lhsX[1][1][BB][isize][j][k];
  lhsX[1][2][BB][isize][j][k] = lhsX[1][2][BB][isize][j][k]*pivot;
  lhsX[1][3][BB][isize][j][k] = lhsX[1][3][BB][isize][j][k]*pivot;
  lhsX[1][4][BB][isize][j][k] = lhsX[1][4][BB][isize][j][k]*pivot;
  rhs[1][k][j][isize]   = rhs[1][k][j][isize]  *pivot;

  coeff = lhsX[0][1][BB][isize][j][k];
  lhsX[0][2][BB][isize][j][k]= lhsX[0][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
  lhsX[0][3][BB][isize][j][k]= lhsX[0][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
  lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
  rhs[0][k][j][isize]   = rhs[0][k][j][isize]   - coeff*rhs[1][k][j][isize];

  coeff = lhsX[2][1][BB][isize][j][k];
  lhsX[2][2][BB][isize][j][k]= lhsX[2][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
  lhsX[2][3][BB][isize][j][k]= lhsX[2][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
  lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
  rhs[2][k][j][isize]   = rhs[2][k][j][isize]   - coeff*rhs[1][k][j][isize];

  coeff = lhsX[3][1][BB][isize][j][k];
  lhsX[3][2][BB][isize][j][k]= lhsX[3][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
  lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
  lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
  rhs[3][k][j][isize]   = rhs[3][k][j][isize]   - coeff*rhs[1][k][j][isize];

  coeff = lhsX[4][1][BB][isize][j][k];
  lhsX[4][2][BB][isize][j][k]= lhsX[4][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
  lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
  lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
  rhs[4][k][j][isize]   = rhs[4][k][j][isize]   - coeff*rhs[1][k][j][isize];

  pivot = 1.00/lhsX[2][2][BB][isize][j][k];
  lhsX[2][3][BB][isize][j][k] = lhsX[2][3][BB][isize][j][k]*pivot;
  lhsX[2][4][BB][isize][j][k] = lhsX[2][4][BB][isize][j][k]*pivot;
  rhs[2][k][j][isize]   = rhs[2][k][j][isize]  *pivot;

  coeff = lhsX[0][2][BB][isize][j][k];
  lhsX[0][3][BB][isize][j][k]= lhsX[0][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
  lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
  rhs[0][k][j][isize]   = rhs[0][k][j][isize]   - coeff*rhs[2][k][j][isize];

  coeff = lhsX[1][2][BB][isize][j][k];
  lhsX[1][3][BB][isize][j][k]= lhsX[1][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
  lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
  rhs[1][k][j][isize]   = rhs[1][k][j][isize]   - coeff*rhs[2][k][j][isize];

  coeff = lhsX[3][2][BB][isize][j][k];
  lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
  lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
  rhs[3][k][j][isize]   = rhs[3][k][j][isize]   - coeff*rhs[2][k][j][isize];

  coeff = lhsX[4][2][BB][isize][j][k];
  lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
  lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
  rhs[4][k][j][isize]   = rhs[4][k][j][isize]   - coeff*rhs[2][k][j][isize];

  pivot = 1.00/lhsX[3][3][BB][isize][j][k];
  lhsX[3][4][BB][isize][j][k] = lhsX[3][4][BB][isize][j][k]*pivot;
  rhs[3][k][j][isize]   = rhs[3][k][j][isize]  *pivot;

  coeff = lhsX[0][3][BB][isize][j][k];
  lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
  rhs[0][k][j][isize]   = rhs[0][k][j][isize]   - coeff*rhs[3][k][j][isize];

  coeff = lhsX[1][3][BB][isize][j][k];
  lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
  rhs[1][k][j][isize]   = rhs[1][k][j][isize]   - coeff*rhs[3][k][j][isize];

  coeff = lhsX[2][3][BB][isize][j][k];
  lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
  rhs[2][k][j][isize]   = rhs[2][k][j][isize]   - coeff*rhs[3][k][j][isize];

  coeff = lhsX[4][3][BB][isize][j][k];
  lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
  rhs[4][k][j][isize]   = rhs[4][k][j][isize]   - coeff*rhs[3][k][j][isize];

  pivot = 1.00/lhsX[4][4][BB][isize][j][k];
  rhs[4][k][j][isize]   = rhs[4][k][j][isize]  *pivot;

  coeff = lhsX[0][4][BB][isize][j][k];
  rhs[0][k][j][isize]   = rhs[0][k][j][isize]   - coeff*rhs[4][k][j][isize];

  coeff = lhsX[1][4][BB][isize][j][k];
  rhs[1][k][j][isize]   = rhs[1][k][j][isize]   - coeff*rhs[4][k][j][isize];

  coeff = lhsX[2][4][BB][isize][j][k];
  rhs[2][k][j][isize]   = rhs[2][k][j][isize]   - coeff*rhs[4][k][j][isize];

  coeff = lhsX[3][4][BB][isize][j][k];
  rhs[3][k][j][isize]   = rhs[3][k][j][isize]   - coeff*rhs[4][k][j][isize];

	}
  }

      for (i = isize-1; i >=0; i--) {
    for (j = 1; j <= gp12; j++) {
  for (k = 1; k <= gp22; k++) {
    
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsX[0][0][CC][i][j][k]*rhs[0][k][j][i+1];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsX[0][1][CC][i][j][k]*rhs[1][k][j][i+1];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsX[0][2][CC][i][j][k]*rhs[2][k][j][i+1];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsX[0][3][CC][i][j][k]*rhs[3][k][j][i+1];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsX[0][4][CC][i][j][k]*rhs[4][k][j][i+1];
            
			rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsX[1][0][CC][i][j][k]*rhs[0][k][j][i+1];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsX[1][1][CC][i][j][k]*rhs[1][k][j][i+1];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsX[1][2][CC][i][j][k]*rhs[2][k][j][i+1];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsX[1][3][CC][i][j][k]*rhs[3][k][j][i+1];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsX[1][4][CC][i][j][k]*rhs[4][k][j][i+1];
			
			rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsX[2][0][CC][i][j][k]*rhs[0][k][j][i+1];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsX[2][1][CC][i][j][k]*rhs[1][k][j][i+1];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsX[2][2][CC][i][j][k]*rhs[2][k][j][i+1];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsX[2][3][CC][i][j][k]*rhs[3][k][j][i+1];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsX[2][4][CC][i][j][k]*rhs[4][k][j][i+1];
			
			rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsX[3][0][CC][i][j][k]*rhs[0][k][j][i+1];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsX[3][1][CC][i][j][k]*rhs[1][k][j][i+1];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsX[3][2][CC][i][j][k]*rhs[2][k][j][i+1];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsX[3][3][CC][i][j][k]*rhs[3][k][j][i+1];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsX[3][4][CC][i][j][k]*rhs[4][k][j][i+1];
			
			rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsX[4][0][CC][i][j][k]*rhs[0][k][j][i+1];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsX[4][1][CC][i][j][k]*rhs[1][k][j][i+1];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsX[4][2][CC][i][j][k]*rhs[2][k][j][i+1];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsX[4][3][CC][i][j][k]*rhs[3][k][j][i+1];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsX[4][4][CC][i][j][k]*rhs[4][k][j][i+1];
	 
      }
    }
  }
}

void y_solve()
{
  int i, j, k, m, n, jsize, z;
  double pivot, coeff;
  int gp22, gp02;
  double fjacY[5][5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1];
  double njacY[5][5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1];
  double lhsY[5][5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1];
  double temp1, temp2, temp3;

  gp22 = grid_points[2]-2;
  gp02 = grid_points[0]-2;

  jsize = grid_points[1]-1;

      for (j = 0; j <= jsize; j++) {
    for (i = 1; i <= gp02; i++) {
  for (k = 1; k <= gp22; k++) {
        temp1 = rho_i[k][j][i];
        temp2 = temp1 * temp1;
        temp3 = temp1 * temp2;

        fjacY[0][0][j][i][k] = 0.0;
        fjacY[0][1][j][i][k] = 0.0;
        fjacY[0][2][j][i][k] = 1.0;
        fjacY[0][3][j][i][k] = 0.0;
        fjacY[0][4][j][i][k] = 0.0;

        fjacY[1][0][j][i][k] = - ( u[1][k][j][i]*u[2][k][j][i] ) * temp2;
        fjacY[1][1][j][i][k] = u[2][k][j][i] * temp1;
        fjacY[1][2][j][i][k] = u[1][k][j][i] * temp1;
        fjacY[1][3][j][i][k] = 0.0;
        fjacY[1][4][j][i][k] = 0.0;

        fjacY[2][0][j][i][k] = - ( u[2][k][j][i]*u[2][k][j][i]*temp2)
          + c2 * qs[k][j][i];
        fjacY[2][1][j][i][k] = - c2 *  u[1][k][j][i] * temp1;
        fjacY[2][2][j][i][k] = ( 2.0 - c2 ) *  u[2][k][j][i] * temp1;
        fjacY[2][3][j][i][k] = - c2 * u[3][k][j][i] * temp1;
        fjacY[2][4][j][i][k] = c2;

        fjacY[3][0][j][i][k] = - ( u[2][k][j][i]*u[3][k][j][i] ) * temp2;
        fjacY[3][1][j][i][k] = 0.0;
        fjacY[3][2][j][i][k] = u[3][k][j][i] * temp1;
        fjacY[3][3][j][i][k] = u[2][k][j][i] * temp1;
        fjacY[3][4][j][i][k] = 0.0;

        fjacY[4][0][j][i][k] = ( c2 * 2.0 * square[k][j][i] - c1 * u[4][k][j][i] )
          * u[2][k][j][i] * temp2;
        fjacY[4][1][j][i][k] = - c2 * u[1][k][j][i]*u[2][k][j][i] * temp2;
        fjacY[4][2][j][i][k] = c1 * u[4][k][j][i] * temp1 
          - c2 * ( qs[k][j][i] + u[2][k][j][i]*u[2][k][j][i] * temp2 );
        fjacY[4][3][j][i][k] = - c2 * ( u[2][k][j][i]*u[3][k][j][i] ) * temp2;
        fjacY[4][4][j][i][k] = c1 * u[2][k][j][i] * temp1;

        njacY[0][0][j][i][k] = 0.0;
        njacY[0][1][j][i][k] = 0.0;
        njacY[0][2][j][i][k] = 0.0;
        njacY[0][3][j][i][k] = 0.0;
        njacY[0][4][j][i][k] = 0.0;

        njacY[1][0][j][i][k] = - c3c4 * temp2 * u[1][k][j][i];
        njacY[1][1][j][i][k] =   c3c4 * temp1;
        njacY[1][2][j][i][k] =   0.0;
        njacY[1][3][j][i][k] =   0.0;
        njacY[1][4][j][i][k] =   0.0;

        njacY[2][0][j][i][k] = - con43 * c3c4 * temp2 * u[2][k][j][i];
        njacY[2][1][j][i][k] =   0.0;
        njacY[2][2][j][i][k] =   con43 * c3c4 * temp1;
        njacY[2][3][j][i][k] =   0.0;
        njacY[2][4][j][i][k] =   0.0;

        njacY[3][0][j][i][k] = - c3c4 * temp2 * u[3][k][j][i];
        njacY[3][1][j][i][k] =   0.0;
        njacY[3][2][j][i][k] =   0.0;
        njacY[3][3][j][i][k] =   c3c4 * temp1;
        njacY[3][4][j][i][k] =   0.0;

        njacY[4][0][j][i][k] = - (  c3c4
            - c1345 ) * temp3 * (u[1][k][j][i]*u[1][k][j][i])
          - ( con43 * c3c4
              - c1345 ) * temp3 * (u[2][k][j][i]*u[2][k][j][i])
          - ( c3c4 - c1345 ) * temp3 * (u[3][k][j][i]*u[3][k][j][i])
          - c1345 * temp2 * u[4][k][j][i];

        njacY[4][1][j][i][k] = (  c3c4 - c1345 ) * temp2 * u[1][k][j][i];
        njacY[4][2][j][i][k] = ( con43 * c3c4 - c1345 ) * temp2 * u[2][k][j][i];
        njacY[4][3][j][i][k] = ( c3c4 - c1345 ) * temp2 * u[3][k][j][i];
        njacY[4][4][j][i][k] = ( c1345 ) * temp1;
      }
	}
  }

    for (i = 1; i <= gp02; i++) {
  for (k = 1; k <= gp22; k++) {
  		for (n = 0; n < 5; n++) {
    		for (m = 0; m < 5; m++) {
      			lhsY[m][n][0][0][i][k] = 0.0;
      			lhsY[m][n][1][0][i][k] = 0.0;
      			lhsY[m][n][2][0][i][k] = 0.0;
      			lhsY[m][n][0][jsize][i][k] = 0.0;
      			lhsY[m][n][1][jsize][i][k] = 0.0;
      			lhsY[m][n][2][jsize][i][k] = 0.0;
    		}
  		}	
	}
  }

    for (i = 1; i <= gp02; i++) {
  for (k = 1; k <= gp22; k++) {
    		lhsY[0][0][1][0][i][k] = 1.0;
    		lhsY[0][0][1][jsize][i][k] = 1.0;
    		lhsY[1][1][1][0][i][k] = 1.0;
    		lhsY[1][1][1][jsize][i][k] = 1.0;
    		lhsY[2][2][1][0][i][k] = 1.0;
    		lhsY[2][2][1][jsize][i][k] = 1.0;
    		lhsY[3][3][1][0][i][k] = 1.0;
    		lhsY[3][3][1][jsize][i][k] = 1.0;
    		lhsY[4][4][1][0][i][k] = 1.0;
    		lhsY[4][4][1][jsize][i][k] = 1.0;
	}
  }
      
	  for (j = 1; j <= jsize-1; j++) {
    for (i = 1; i <= gp02; i++) {
  for (k = 1; k <= gp22; k++) {
        temp1 = dt * ty1;
        temp2 = dt * ty2;

        lhsY[0][0][AA][j][i][k] = - temp2 * fjacY[0][0][j-1][i][k]
          - temp1 * njacY[0][0][j-1][i][k]
          - temp1 * dy1; 
        lhsY[0][1][AA][j][i][k] = - temp2 * fjacY[0][1][j-1][i][k]
          - temp1 * njacY[0][1][j-1][i][k];
        lhsY[0][2][AA][j][i][k] = - temp2 * fjacY[0][2][j-1][i][k]
          - temp1 * njacY[0][2][j-1][i][k];
        lhsY[0][3][AA][j][i][k] = - temp2 * fjacY[0][3][j-1][i][k]
          - temp1 * njacY[0][3][j-1][i][k];
        lhsY[0][4][AA][j][i][k] = - temp2 * fjacY[0][4][j-1][i][k]
          - temp1 * njacY[0][4][j-1][i][k];

        lhsY[1][0][AA][j][i][k] = - temp2 * fjacY[1][0][j-1][i][k]
          - temp1 * njacY[1][0][j-1][i][k];
        lhsY[1][1][AA][j][i][k] = - temp2 * fjacY[1][1][j-1][i][k]
          - temp1 * njacY[1][1][j-1][i][k]
          - temp1 * dy2;
        lhsY[1][2][AA][j][i][k] = - temp2 * fjacY[1][2][j-1][i][k]
          - temp1 * njacY[1][2][j-1][i][k];
        lhsY[1][3][AA][j][i][k] = - temp2 * fjacY[1][3][j-1][i][k]
          - temp1 * njacY[1][3][j-1][i][k];
        lhsY[1][4][AA][j][i][k] = - temp2 * fjacY[1][4][j-1][i][k]
          - temp1 * njacY[1][4][j-1][i][k];

        lhsY[2][0][AA][j][i][k] = - temp2 * fjacY[2][0][j-1][i][k]
          - temp1 * njacY[2][0][j-1][i][k];
        lhsY[2][1][AA][j][i][k] = - temp2 * fjacY[2][1][j-1][i][k]
          - temp1 * njacY[2][1][j-1][i][k];
        lhsY[2][2][AA][j][i][k] = - temp2 * fjacY[2][2][j-1][i][k]
          - temp1 * njacY[2][2][j-1][i][k]
          - temp1 * dy3;
        lhsY[2][3][AA][j][i][k] = - temp2 * fjacY[2][3][j-1][i][k]
          - temp1 * njacY[2][3][j-1][i][k];
        lhsY[2][4][AA][j][i][k] = - temp2 * fjacY[2][4][j-1][i][k]
          - temp1 * njacY[2][4][j-1][i][k];

        lhsY[3][0][AA][j][i][k] = - temp2 * fjacY[3][0][j-1][i][k]
          - temp1 * njacY[3][0][j-1][i][k];
        lhsY[3][1][AA][j][i][k] = - temp2 * fjacY[3][1][j-1][i][k]
          - temp1 * njacY[3][1][j-1][i][k];
        lhsY[3][2][AA][j][i][k] = - temp2 * fjacY[3][2][j-1][i][k]
          - temp1 * njacY[3][2][j-1][i][k];
        lhsY[3][3][AA][j][i][k] = - temp2 * fjacY[3][3][j-1][i][k]
          - temp1 * njacY[3][3][j-1][i][k]
          - temp1 * dy4;
        lhsY[3][4][AA][j][i][k] = - temp2 * fjacY[3][4][j-1][i][k]
          - temp1 * njacY[3][4][j-1][i][k];

        lhsY[4][0][AA][j][i][k] = - temp2 * fjacY[4][0][j-1][i][k]
          - temp1 * njacY[4][0][j-1][i][k];
        lhsY[4][1][AA][j][i][k] = - temp2 * fjacY[4][1][j-1][i][k]
          - temp1 * njacY[4][1][j-1][i][k];
        lhsY[4][2][AA][j][i][k] = - temp2 * fjacY[4][2][j-1][i][k]
          - temp1 * njacY[4][2][j-1][i][k];
        lhsY[4][3][AA][j][i][k] = - temp2 * fjacY[4][3][j-1][i][k]
          - temp1 * njacY[4][3][j-1][i][k];
        lhsY[4][4][AA][j][i][k] = - temp2 * fjacY[4][4][j-1][i][k]
          - temp1 * njacY[4][4][j-1][i][k]
          - temp1 * dy5;

        lhsY[0][0][BB][j][i][k] = 1.0
          + temp1 * 2.0 * njacY[0][0][j][i][k]
          + temp1 * 2.0 * dy1;
        lhsY[0][1][BB][j][i][k] = temp1 * 2.0 * njacY[0][1][j][i][k];
        lhsY[0][2][BB][j][i][k] = temp1 * 2.0 * njacY[0][2][j][i][k];
        lhsY[0][3][BB][j][i][k] = temp1 * 2.0 * njacY[0][3][j][i][k];
        lhsY[0][4][BB][j][i][k] = temp1 * 2.0 * njacY[0][4][j][i][k];

        lhsY[1][0][BB][j][i][k] = temp1 * 2.0 * njacY[1][0][j][i][k];
        lhsY[1][1][BB][j][i][k] = 1.0
          + temp1 * 2.0 * njacY[1][1][j][i][k]
          + temp1 * 2.0 * dy2;
        lhsY[1][2][BB][j][i][k] = temp1 * 2.0 * njacY[1][2][j][i][k];
        lhsY[1][3][BB][j][i][k] = temp1 * 2.0 * njacY[1][3][j][i][k];
        lhsY[1][4][BB][j][i][k] = temp1 * 2.0 * njacY[1][4][j][i][k];

        lhsY[2][0][BB][j][i][k] = temp1 * 2.0 * njacY[2][0][j][i][k];
        lhsY[2][1][BB][j][i][k] = temp1 * 2.0 * njacY[2][1][j][i][k];
        lhsY[2][2][BB][j][i][k] = 1.0
          + temp1 * 2.0 * njacY[2][2][j][i][k]
          + temp1 * 2.0 * dy3;
        lhsY[2][3][BB][j][i][k] = temp1 * 2.0 * njacY[2][3][j][i][k];
        lhsY[2][4][BB][j][i][k] = temp1 * 2.0 * njacY[2][4][j][i][k];

        lhsY[3][0][BB][j][i][k] = temp1 * 2.0 * njacY[3][0][j][i][k];
        lhsY[3][1][BB][j][i][k] = temp1 * 2.0 * njacY[3][1][j][i][k];
        lhsY[3][2][BB][j][i][k] = temp1 * 2.0 * njacY[3][2][j][i][k];
        lhsY[3][3][BB][j][i][k] = 1.0
          + temp1 * 2.0 * njacY[3][3][j][i][k]
          + temp1 * 2.0 * dy4;
        lhsY[3][4][BB][j][i][k] = temp1 * 2.0 * njacY[3][4][j][i][k];

        lhsY[4][0][BB][j][i][k] = temp1 * 2.0 * njacY[4][0][j][i][k];
        lhsY[4][1][BB][j][i][k] = temp1 * 2.0 * njacY[4][1][j][i][k];
        lhsY[4][2][BB][j][i][k] = temp1 * 2.0 * njacY[4][2][j][i][k];
        lhsY[4][3][BB][j][i][k] = temp1 * 2.0 * njacY[4][3][j][i][k];
        lhsY[4][4][BB][j][i][k] = 1.0
          + temp1 * 2.0 * njacY[4][4][j][i][k] 
          + temp1 * 2.0 * dy5;

        lhsY[0][0][CC][j][i][k] =  temp2 * fjacY[0][0][j+1][i][k]
          - temp1 * njacY[0][0][j+1][i][k]
          - temp1 * dy1;
        lhsY[0][1][CC][j][i][k] =  temp2 * fjacY[0][1][j+1][i][k]
          - temp1 * njacY[0][1][j+1][i][k];
        lhsY[0][2][CC][j][i][k] =  temp2 * fjacY[0][2][j+1][i][k]
          - temp1 * njacY[0][2][j+1][i][k];
        lhsY[0][3][CC][j][i][k] =  temp2 * fjacY[0][3][j+1][i][k]
          - temp1 * njacY[0][3][j+1][i][k];
        lhsY[0][4][CC][j][i][k] =  temp2 * fjacY[0][4][j+1][i][k]
          - temp1 * njacY[0][4][j+1][i][k];

        lhsY[1][0][CC][j][i][k] =  temp2 * fjacY[1][0][j+1][i][k]
          - temp1 * njacY[1][0][j+1][i][k];
        lhsY[1][1][CC][j][i][k] =  temp2 * fjacY[1][1][j+1][i][k]
          - temp1 * njacY[1][1][j+1][i][k]
          - temp1 * dy2;
        lhsY[1][2][CC][j][i][k] =  temp2 * fjacY[1][2][j+1][i][k]
          - temp1 * njacY[1][2][j+1][i][k];
        lhsY[1][3][CC][j][i][k] =  temp2 * fjacY[1][3][j+1][i][k]
          - temp1 * njacY[1][3][j+1][i][k];
        lhsY[1][4][CC][j][i][k] =  temp2 * fjacY[1][4][j+1][i][k]
          - temp1 * njacY[1][4][j+1][i][k];

        lhsY[2][0][CC][j][i][k] =  temp2 * fjacY[2][0][j+1][i][k]
          - temp1 * njacY[2][0][j+1][i][k];
        lhsY[2][1][CC][j][i][k] =  temp2 * fjacY[2][1][j+1][i][k]
          - temp1 * njacY[2][1][j+1][i][k];
        lhsY[2][2][CC][j][i][k] =  temp2 * fjacY[2][2][j+1][i][k]
          - temp1 * njacY[2][2][j+1][i][k]
          - temp1 * dy3;
        lhsY[2][3][CC][j][i][k] =  temp2 * fjacY[2][3][j+1][i][k]
          - temp1 * njacY[2][3][j+1][i][k];
        lhsY[2][4][CC][j][i][k] =  temp2 * fjacY[2][4][j+1][i][k]
          - temp1 * njacY[2][4][j+1][i][k];

        lhsY[3][0][CC][j][i][k] =  temp2 * fjacY[3][0][j+1][i][k]
          - temp1 * njacY[3][0][j+1][i][k];
        lhsY[3][1][CC][j][i][k] =  temp2 * fjacY[3][1][j+1][i][k]
          - temp1 * njacY[3][1][j+1][i][k];
        lhsY[3][2][CC][j][i][k] =  temp2 * fjacY[3][2][j+1][i][k]
          - temp1 * njacY[3][2][j+1][i][k];
        lhsY[3][3][CC][j][i][k] =  temp2 * fjacY[3][3][j+1][i][k]
          - temp1 * njacY[3][3][j+1][i][k]
          - temp1 * dy4;
        lhsY[3][4][CC][j][i][k] =  temp2 * fjacY[3][4][j+1][i][k]
          - temp1 * njacY[3][4][j+1][i][k];

        lhsY[4][0][CC][j][i][k] =  temp2 * fjacY[4][0][j+1][i][k]
          - temp1 * njacY[4][0][j+1][i][k];
        lhsY[4][1][CC][j][i][k] =  temp2 * fjacY[4][1][j+1][i][k]
          - temp1 * njacY[4][1][j+1][i][k];
        lhsY[4][2][CC][j][i][k] =  temp2 * fjacY[4][2][j+1][i][k]
          - temp1 * njacY[4][2][j+1][i][k];
        lhsY[4][3][CC][j][i][k] =  temp2 * fjacY[4][3][j+1][i][k]
          - temp1 * njacY[4][3][j+1][i][k];
        lhsY[4][4][CC][j][i][k] =  temp2 * fjacY[4][4][j+1][i][k]
          - temp1 * njacY[4][4][j+1][i][k]
          - temp1 * dy5;
      }
	}
  }
      
    for (i = 1; i <= gp02; i++) {
  for (k = 1; k <= gp22; k++) {

  pivot = 1.00/lhsY[0][0][BB][0][i][k];
  lhsY[0][1][BB][0][i][k] = lhsY[0][1][BB][0][i][k]*pivot;
  lhsY[0][2][BB][0][i][k] = lhsY[0][2][BB][0][i][k]*pivot;
  lhsY[0][3][BB][0][i][k] = lhsY[0][3][BB][0][i][k]*pivot;
  lhsY[0][4][BB][0][i][k] = lhsY[0][4][BB][0][i][k]*pivot;
  lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k]*pivot;
  lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k]*pivot;
  lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k]*pivot;
  lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k]*pivot;
  lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k]*pivot;
  rhs[0][k][0][i]   = rhs[0][k][0][i]  *pivot;

  coeff = lhsY[1][0][BB][0][i][k];
  lhsY[1][1][BB][0][i][k]= lhsY[1][1][BB][0][i][k] - coeff*lhsY[0][1][BB][0][i][k];
  lhsY[1][2][BB][0][i][k]= lhsY[1][2][BB][0][i][k] - coeff*lhsY[0][2][BB][0][i][k];
  lhsY[1][3][BB][0][i][k]= lhsY[1][3][BB][0][i][k] - coeff*lhsY[0][3][BB][0][i][k];
  lhsY[1][4][BB][0][i][k]= lhsY[1][4][BB][0][i][k] - coeff*lhsY[0][4][BB][0][i][k];
  lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k] - coeff*lhsY[0][0][CC][0][i][k];
  lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k] - coeff*lhsY[0][1][CC][0][i][k];
  lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k] - coeff*lhsY[0][2][CC][0][i][k];
  lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k] - coeff*lhsY[0][3][CC][0][i][k];
  lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k] - coeff*lhsY[0][4][CC][0][i][k];
  rhs[1][k][0][i]   = rhs[1][k][0][i]   - coeff*rhs[0][k][0][i];

  coeff = lhsY[2][0][BB][0][i][k];
  lhsY[2][1][BB][0][i][k]= lhsY[2][1][BB][0][i][k] - coeff*lhsY[0][1][BB][0][i][k];
  lhsY[2][2][BB][0][i][k]= lhsY[2][2][BB][0][i][k] - coeff*lhsY[0][2][BB][0][i][k];
  lhsY[2][3][BB][0][i][k]= lhsY[2][3][BB][0][i][k] - coeff*lhsY[0][3][BB][0][i][k];
  lhsY[2][4][BB][0][i][k]= lhsY[2][4][BB][0][i][k] - coeff*lhsY[0][4][BB][0][i][k];
  lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k] - coeff*lhsY[0][0][CC][0][i][k];
  lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k] - coeff*lhsY[0][1][CC][0][i][k];
  lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k] - coeff*lhsY[0][2][CC][0][i][k];
  lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k] - coeff*lhsY[0][3][CC][0][i][k];
  lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k] - coeff*lhsY[0][4][CC][0][i][k];
  rhs[2][k][0][i]   = rhs[2][k][0][i]   - coeff*rhs[0][k][0][i];

  coeff = lhsY[3][0][BB][0][i][k];
  lhsY[3][1][BB][0][i][k]= lhsY[3][1][BB][0][i][k] - coeff*lhsY[0][1][BB][0][i][k];
  lhsY[3][2][BB][0][i][k]= lhsY[3][2][BB][0][i][k] - coeff*lhsY[0][2][BB][0][i][k];
  lhsY[3][3][BB][0][i][k]= lhsY[3][3][BB][0][i][k] - coeff*lhsY[0][3][BB][0][i][k];
  lhsY[3][4][BB][0][i][k]= lhsY[3][4][BB][0][i][k] - coeff*lhsY[0][4][BB][0][i][k];
  lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k] - coeff*lhsY[0][0][CC][0][i][k];
  lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k] - coeff*lhsY[0][1][CC][0][i][k];
  lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k] - coeff*lhsY[0][2][CC][0][i][k];
  lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k] - coeff*lhsY[0][3][CC][0][i][k];
  lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k] - coeff*lhsY[0][4][CC][0][i][k];
  rhs[3][k][0][i]   = rhs[3][k][0][i]   - coeff*rhs[0][k][0][i];

  coeff = lhsY[4][0][BB][0][i][k];
  lhsY[4][1][BB][0][i][k]= lhsY[4][1][BB][0][i][k] - coeff*lhsY[0][1][BB][0][i][k];
  lhsY[4][2][BB][0][i][k]= lhsY[4][2][BB][0][i][k] - coeff*lhsY[0][2][BB][0][i][k];
  lhsY[4][3][BB][0][i][k]= lhsY[4][3][BB][0][i][k] - coeff*lhsY[0][3][BB][0][i][k];
  lhsY[4][4][BB][0][i][k]= lhsY[4][4][BB][0][i][k] - coeff*lhsY[0][4][BB][0][i][k];
  lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k] - coeff*lhsY[0][0][CC][0][i][k];
  lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k] - coeff*lhsY[0][1][CC][0][i][k];
  lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k] - coeff*lhsY[0][2][CC][0][i][k];
  lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k] - coeff*lhsY[0][3][CC][0][i][k];
  lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k] - coeff*lhsY[0][4][CC][0][i][k];
  rhs[4][k][0][i]   = rhs[4][k][0][i]   - coeff*rhs[0][k][0][i];

  pivot = 1.00/lhsY[1][1][BB][0][i][k];
  lhsY[1][2][BB][0][i][k] = lhsY[1][2][BB][0][i][k]*pivot;
  lhsY[1][3][BB][0][i][k] = lhsY[1][3][BB][0][i][k]*pivot;
  lhsY[1][4][BB][0][i][k] = lhsY[1][4][BB][0][i][k]*pivot;
  lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k]*pivot;
  lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k]*pivot;
  lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k]*pivot;
  lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k]*pivot;
  lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k]*pivot;
  rhs[1][k][0][i]   = rhs[1][k][0][i]  *pivot;

  coeff = lhsY[0][1][BB][0][i][k];
  lhsY[0][2][BB][0][i][k]= lhsY[0][2][BB][0][i][k] - coeff*lhsY[1][2][BB][0][i][k];
  lhsY[0][3][BB][0][i][k]= lhsY[0][3][BB][0][i][k] - coeff*lhsY[1][3][BB][0][i][k];
  lhsY[0][4][BB][0][i][k]= lhsY[0][4][BB][0][i][k] - coeff*lhsY[1][4][BB][0][i][k];
  lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k] - coeff*lhsY[1][0][CC][0][i][k];
  lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k] - coeff*lhsY[1][1][CC][0][i][k];
  lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k] - coeff*lhsY[1][2][CC][0][i][k];
  lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k] - coeff*lhsY[1][3][CC][0][i][k];
  lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k] - coeff*lhsY[1][4][CC][0][i][k];
  rhs[0][k][0][i]   = rhs[0][k][0][i]   - coeff*rhs[1][k][0][i];

  coeff = lhsY[2][1][BB][0][i][k];
  lhsY[2][2][BB][0][i][k]= lhsY[2][2][BB][0][i][k] - coeff*lhsY[1][2][BB][0][i][k];
  lhsY[2][3][BB][0][i][k]= lhsY[2][3][BB][0][i][k] - coeff*lhsY[1][3][BB][0][i][k];
  lhsY[2][4][BB][0][i][k]= lhsY[2][4][BB][0][i][k] - coeff*lhsY[1][4][BB][0][i][k];
  lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k] - coeff*lhsY[1][0][CC][0][i][k];
  lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k] - coeff*lhsY[1][1][CC][0][i][k];
  lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k] - coeff*lhsY[1][2][CC][0][i][k];
  lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k] - coeff*lhsY[1][3][CC][0][i][k];
  lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k] - coeff*lhsY[1][4][CC][0][i][k];
  rhs[2][k][0][i]   = rhs[2][k][0][i]   - coeff*rhs[1][k][0][i];

  coeff = lhsY[3][1][BB][0][i][k];
  lhsY[3][2][BB][0][i][k]= lhsY[3][2][BB][0][i][k] - coeff*lhsY[1][2][BB][0][i][k];
  lhsY[3][3][BB][0][i][k]= lhsY[3][3][BB][0][i][k] - coeff*lhsY[1][3][BB][0][i][k];
  lhsY[3][4][BB][0][i][k]= lhsY[3][4][BB][0][i][k] - coeff*lhsY[1][4][BB][0][i][k];
  lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k] - coeff*lhsY[1][0][CC][0][i][k];
  lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k] - coeff*lhsY[1][1][CC][0][i][k];
  lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k] - coeff*lhsY[1][2][CC][0][i][k];
  lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k] - coeff*lhsY[1][3][CC][0][i][k];
  lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k] - coeff*lhsY[1][4][CC][0][i][k];
  rhs[3][k][0][i]   = rhs[3][k][0][i]   - coeff*rhs[1][k][0][i];

  coeff = lhsY[4][1][BB][0][i][k];
  lhsY[4][2][BB][0][i][k]= lhsY[4][2][BB][0][i][k] - coeff*lhsY[1][2][BB][0][i][k];
  lhsY[4][3][BB][0][i][k]= lhsY[4][3][BB][0][i][k] - coeff*lhsY[1][3][BB][0][i][k];
  lhsY[4][4][BB][0][i][k]= lhsY[4][4][BB][0][i][k] - coeff*lhsY[1][4][BB][0][i][k];
  lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k] - coeff*lhsY[1][0][CC][0][i][k];
  lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k] - coeff*lhsY[1][1][CC][0][i][k];
  lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k] - coeff*lhsY[1][2][CC][0][i][k];
  lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k] - coeff*lhsY[1][3][CC][0][i][k];
  lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k] - coeff*lhsY[1][4][CC][0][i][k];
  rhs[4][k][0][i]   = rhs[4][k][0][i]   - coeff*rhs[1][k][0][i];

  pivot = 1.00/lhsY[2][2][BB][0][i][k];
  lhsY[2][3][BB][0][i][k] = lhsY[2][3][BB][0][i][k]*pivot;
  lhsY[2][4][BB][0][i][k] = lhsY[2][4][BB][0][i][k]*pivot;
  lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k]*pivot;
  lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k]*pivot;
  lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k]*pivot;
  lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k]*pivot;
  lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k]*pivot;
  rhs[2][k][0][i]   = rhs[2][k][0][i]  *pivot;

  coeff = lhsY[0][2][BB][0][i][k];
  lhsY[0][3][BB][0][i][k]= lhsY[0][3][BB][0][i][k] - coeff*lhsY[2][3][BB][0][i][k];
  lhsY[0][4][BB][0][i][k]= lhsY[0][4][BB][0][i][k] - coeff*lhsY[2][4][BB][0][i][k];
  lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k] - coeff*lhsY[2][0][CC][0][i][k];
  lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k] - coeff*lhsY[2][1][CC][0][i][k];
  lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k] - coeff*lhsY[2][2][CC][0][i][k];
  lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k] - coeff*lhsY[2][3][CC][0][i][k];
  lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k] - coeff*lhsY[2][4][CC][0][i][k];
  rhs[0][k][0][i]   = rhs[0][k][0][i]   - coeff*rhs[2][k][0][i];

  coeff = lhsY[1][2][BB][0][i][k];
  lhsY[1][3][BB][0][i][k]= lhsY[1][3][BB][0][i][k] - coeff*lhsY[2][3][BB][0][i][k];
  lhsY[1][4][BB][0][i][k]= lhsY[1][4][BB][0][i][k] - coeff*lhsY[2][4][BB][0][i][k];
  lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k] - coeff*lhsY[2][0][CC][0][i][k];
  lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k] - coeff*lhsY[2][1][CC][0][i][k];
  lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k] - coeff*lhsY[2][2][CC][0][i][k];
  lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k] - coeff*lhsY[2][3][CC][0][i][k];
  lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k] - coeff*lhsY[2][4][CC][0][i][k];
  rhs[1][k][0][i]   = rhs[1][k][0][i]   - coeff*rhs[2][k][0][i];

  coeff = lhsY[3][2][BB][0][i][k];
  lhsY[3][3][BB][0][i][k]= lhsY[3][3][BB][0][i][k] - coeff*lhsY[2][3][BB][0][i][k];
  lhsY[3][4][BB][0][i][k]= lhsY[3][4][BB][0][i][k] - coeff*lhsY[2][4][BB][0][i][k];
  lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k] - coeff*lhsY[2][0][CC][0][i][k];
  lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k] - coeff*lhsY[2][1][CC][0][i][k];
  lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k] - coeff*lhsY[2][2][CC][0][i][k];
  lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k] - coeff*lhsY[2][3][CC][0][i][k];
  lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k] - coeff*lhsY[2][4][CC][0][i][k];
  rhs[3][k][0][i]   = rhs[3][k][0][i]   - coeff*rhs[2][k][0][i];

  coeff = lhsY[4][2][BB][0][i][k];
  lhsY[4][3][BB][0][i][k]= lhsY[4][3][BB][0][i][k] - coeff*lhsY[2][3][BB][0][i][k];
  lhsY[4][4][BB][0][i][k]= lhsY[4][4][BB][0][i][k] - coeff*lhsY[2][4][BB][0][i][k];
  lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k] - coeff*lhsY[2][0][CC][0][i][k];
  lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k] - coeff*lhsY[2][1][CC][0][i][k];
  lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k] - coeff*lhsY[2][2][CC][0][i][k];
  lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k] - coeff*lhsY[2][3][CC][0][i][k];
  lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k] - coeff*lhsY[2][4][CC][0][i][k];
  rhs[4][k][0][i]   = rhs[4][k][0][i]   - coeff*rhs[2][k][0][i];

  pivot = 1.00/lhsY[3][3][BB][0][i][k];
  lhsY[3][4][BB][0][i][k] = lhsY[3][4][BB][0][i][k]*pivot;
  lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k]*pivot;
  lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k]*pivot;
  lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k]*pivot;
  lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k]*pivot;
  lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k]*pivot;
  rhs[3][k][0][i]   = rhs[3][k][0][i]  *pivot;

  coeff = lhsY[0][3][BB][0][i][k];
  lhsY[0][4][BB][0][i][k]= lhsY[0][4][BB][0][i][k] - coeff*lhsY[3][4][BB][0][i][k];
  lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k] - coeff*lhsY[3][0][CC][0][i][k];
  lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k] - coeff*lhsY[3][1][CC][0][i][k];
  lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k] - coeff*lhsY[3][2][CC][0][i][k];
  lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k] - coeff*lhsY[3][3][CC][0][i][k];
  lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k] - coeff*lhsY[3][4][CC][0][i][k];
  rhs[0][k][0][i]   = rhs[0][k][0][i]   - coeff*rhs[3][k][0][i];

  coeff = lhsY[1][3][BB][0][i][k];
  lhsY[1][4][BB][0][i][k]= lhsY[1][4][BB][0][i][k] - coeff*lhsY[3][4][BB][0][i][k];
  lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k] - coeff*lhsY[3][0][CC][0][i][k];
  lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k] - coeff*lhsY[3][1][CC][0][i][k];
  lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k] - coeff*lhsY[3][2][CC][0][i][k];
  lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k] - coeff*lhsY[3][3][CC][0][i][k];
  lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k] - coeff*lhsY[3][4][CC][0][i][k];
  rhs[1][k][0][i]   = rhs[1][k][0][i]   - coeff*rhs[3][k][0][i];

  coeff = lhsY[2][3][BB][0][i][k];
  lhsY[2][4][BB][0][i][k]= lhsY[2][4][BB][0][i][k] - coeff*lhsY[3][4][BB][0][i][k];
  lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k] - coeff*lhsY[3][0][CC][0][i][k];
  lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k] - coeff*lhsY[3][1][CC][0][i][k];
  lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k] - coeff*lhsY[3][2][CC][0][i][k];
  lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k] - coeff*lhsY[3][3][CC][0][i][k];
  lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k] - coeff*lhsY[3][4][CC][0][i][k];
  rhs[2][k][0][i]   = rhs[2][k][0][i]   - coeff*rhs[3][k][0][i];

  coeff = lhsY[4][3][BB][0][i][k];
  lhsY[4][4][BB][0][i][k]= lhsY[4][4][BB][0][i][k] - coeff*lhsY[3][4][BB][0][i][k];
  lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k] - coeff*lhsY[3][0][CC][0][i][k];
  lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k] - coeff*lhsY[3][1][CC][0][i][k];
  lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k] - coeff*lhsY[3][2][CC][0][i][k];
  lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k] - coeff*lhsY[3][3][CC][0][i][k];
  lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k] - coeff*lhsY[3][4][CC][0][i][k];
  rhs[4][k][0][i]   = rhs[4][k][0][i]   - coeff*rhs[3][k][0][i];

  pivot = 1.00/lhsY[4][4][BB][0][i][k];
  lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k]*pivot;
  lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k]*pivot;
  lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k]*pivot;
  lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k]*pivot;
  lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k]*pivot;
  rhs[4][k][0][i]   = rhs[4][k][0][i]  *pivot;

  coeff = lhsY[0][4][BB][0][i][k];
  lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k] - coeff*lhsY[4][0][CC][0][i][k];
  lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k] - coeff*lhsY[4][1][CC][0][i][k];
  lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k] - coeff*lhsY[4][2][CC][0][i][k];
  lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k] - coeff*lhsY[4][3][CC][0][i][k];
  lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k] - coeff*lhsY[4][4][CC][0][i][k];
  rhs[0][k][0][i]   = rhs[0][k][0][i]   - coeff*rhs[4][k][0][i];

  coeff = lhsY[1][4][BB][0][i][k];
  lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k] - coeff*lhsY[4][0][CC][0][i][k];
  lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k] - coeff*lhsY[4][1][CC][0][i][k];
  lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k] - coeff*lhsY[4][2][CC][0][i][k];
  lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k] - coeff*lhsY[4][3][CC][0][i][k];
  lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k] - coeff*lhsY[4][4][CC][0][i][k];
  rhs[1][k][0][i]   = rhs[1][k][0][i]   - coeff*rhs[4][k][0][i];

  coeff = lhsY[2][4][BB][0][i][k];
  lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k] - coeff*lhsY[4][0][CC][0][i][k];
  lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k] - coeff*lhsY[4][1][CC][0][i][k];
  lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k] - coeff*lhsY[4][2][CC][0][i][k];
  lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k] - coeff*lhsY[4][3][CC][0][i][k];
  lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k] - coeff*lhsY[4][4][CC][0][i][k];
  rhs[2][k][0][i]   = rhs[2][k][0][i]   - coeff*rhs[4][k][0][i];

  coeff = lhsY[3][4][BB][0][i][k];
  lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k] - coeff*lhsY[4][0][CC][0][i][k];
  lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k] - coeff*lhsY[4][1][CC][0][i][k];
  lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k] - coeff*lhsY[4][2][CC][0][i][k];
  lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k] - coeff*lhsY[4][3][CC][0][i][k];
  lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k] - coeff*lhsY[4][4][CC][0][i][k];
  rhs[3][k][0][i]   = rhs[3][k][0][i]   - coeff*rhs[4][k][0][i];

	}
  }
      
    for (i = 1; i <= gp02; i++) {
  for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= jsize-1; j++) {
        
  rhs[0][k][j][i] = rhs[0][k][j][i] - lhsY[0][0][AA][j][i][k]*rhs[0][k][j-1][i]
                    - lhsY[0][1][AA][j][i][k]*rhs[1][k][j-1][i]
                    - lhsY[0][2][AA][j][i][k]*rhs[2][k][j-1][i]
                    - lhsY[0][3][AA][j][i][k]*rhs[3][k][j-1][i]
                    - lhsY[0][4][AA][j][i][k]*rhs[4][k][j-1][i];
  rhs[1][k][j][i] = rhs[1][k][j][i] - lhsY[1][0][AA][j][i][k]*rhs[0][k][j-1][i]
                    - lhsY[1][1][AA][j][i][k]*rhs[1][k][j-1][i]
                    - lhsY[1][2][AA][j][i][k]*rhs[2][k][j-1][i]
                    - lhsY[1][3][AA][j][i][k]*rhs[3][k][j-1][i]
                    - lhsY[1][4][AA][j][i][k]*rhs[4][k][j-1][i];
  rhs[2][k][j][i] = rhs[2][k][j][i] - lhsY[2][0][AA][j][i][k]*rhs[0][k][j-1][i]
                    - lhsY[2][1][AA][j][i][k]*rhs[1][k][j-1][i]
                    - lhsY[2][2][AA][j][i][k]*rhs[2][k][j-1][i]
                    - lhsY[2][3][AA][j][i][k]*rhs[3][k][j-1][i]
                    - lhsY[2][4][AA][j][i][k]*rhs[4][k][j-1][i];
  rhs[3][k][j][i] = rhs[3][k][j][i] - lhsY[3][0][AA][j][i][k]*rhs[0][k][j-1][i]
                    - lhsY[3][1][AA][j][i][k]*rhs[1][k][j-1][i]
                    - lhsY[3][2][AA][j][i][k]*rhs[2][k][j-1][i]
                    - lhsY[3][3][AA][j][i][k]*rhs[3][k][j-1][i]
                    - lhsY[3][4][AA][j][i][k]*rhs[4][k][j-1][i];
  rhs[4][k][j][i] = rhs[4][k][j][i] - lhsY[4][0][AA][j][i][k]*rhs[0][k][j-1][i]
                    - lhsY[4][1][AA][j][i][k]*rhs[1][k][j-1][i]
                    - lhsY[4][2][AA][j][i][k]*rhs[2][k][j-1][i]
                    - lhsY[4][3][AA][j][i][k]*rhs[3][k][j-1][i]
                    - lhsY[4][4][AA][j][i][k]*rhs[4][k][j-1][i];

  lhsY[0][0][BB][j][i][k] = lhsY[0][0][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
                              - lhsY[0][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
                              - lhsY[0][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
                              - lhsY[0][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
                              - lhsY[0][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
  lhsY[1][0][BB][j][i][k] = lhsY[1][0][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
                              - lhsY[1][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
                              - lhsY[1][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
                              - lhsY[1][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
                              - lhsY[1][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
  lhsY[2][0][BB][j][i][k] = lhsY[2][0][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
                              - lhsY[2][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
                              - lhsY[2][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
                              - lhsY[2][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
                              - lhsY[2][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
  lhsY[3][0][BB][j][i][k] = lhsY[3][0][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
                              - lhsY[3][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
                              - lhsY[3][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
                              - lhsY[3][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
                              - lhsY[3][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
  lhsY[4][0][BB][j][i][k] = lhsY[4][0][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
                              - lhsY[4][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
                              - lhsY[4][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
                              - lhsY[4][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
                              - lhsY[4][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
  lhsY[0][1][BB][j][i][k] = lhsY[0][1][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
                              - lhsY[0][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
                              - lhsY[0][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
                              - lhsY[0][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
                              - lhsY[0][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
  lhsY[1][1][BB][j][i][k] = lhsY[1][1][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
                              - lhsY[1][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
                              - lhsY[1][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
                              - lhsY[1][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
                              - lhsY[1][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
  lhsY[2][1][BB][j][i][k] = lhsY[2][1][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
                              - lhsY[2][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
                              - lhsY[2][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
                              - lhsY[2][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
                              - lhsY[2][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
  lhsY[3][1][BB][j][i][k] = lhsY[3][1][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
                              - lhsY[3][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
                              - lhsY[3][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
                              - lhsY[3][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
                              - lhsY[3][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
  lhsY[4][1][BB][j][i][k] = lhsY[4][1][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
                              - lhsY[4][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
                              - lhsY[4][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
                              - lhsY[4][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
                              - lhsY[4][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
  lhsY[0][2][BB][j][i][k] = lhsY[0][2][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
                              - lhsY[0][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
                              - lhsY[0][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
                              - lhsY[0][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
                              - lhsY[0][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
  lhsY[1][2][BB][j][i][k] = lhsY[1][2][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
                              - lhsY[1][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
                              - lhsY[1][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
                              - lhsY[1][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
                              - lhsY[1][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
  lhsY[2][2][BB][j][i][k] = lhsY[2][2][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
                              - lhsY[2][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
                              - lhsY[2][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
                              - lhsY[2][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
                              - lhsY[2][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
  lhsY[3][2][BB][j][i][k] = lhsY[3][2][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
                              - lhsY[3][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
                              - lhsY[3][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
                              - lhsY[3][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
                              - lhsY[3][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
  lhsY[4][2][BB][j][i][k] = lhsY[4][2][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
                              - lhsY[4][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
                              - lhsY[4][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
                              - lhsY[4][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
                              - lhsY[4][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
  lhsY[0][3][BB][j][i][k] = lhsY[0][3][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
                              - lhsY[0][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
                              - lhsY[0][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
                              - lhsY[0][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
                              - lhsY[0][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
  lhsY[1][3][BB][j][i][k] = lhsY[1][3][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
                              - lhsY[1][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
                              - lhsY[1][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
                              - lhsY[1][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
                              - lhsY[1][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
  lhsY[2][3][BB][j][i][k] = lhsY[2][3][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
                              - lhsY[2][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
                              - lhsY[2][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
                              - lhsY[2][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
                              - lhsY[2][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
  lhsY[3][3][BB][j][i][k] = lhsY[3][3][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
                              - lhsY[3][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
                              - lhsY[3][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
                              - lhsY[3][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
                              - lhsY[3][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
  lhsY[4][3][BB][j][i][k] = lhsY[4][3][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
                              - lhsY[4][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
                              - lhsY[4][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
                              - lhsY[4][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
                              - lhsY[4][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
  lhsY[0][4][BB][j][i][k] = lhsY[0][4][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
                              - lhsY[0][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
                              - lhsY[0][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
                              - lhsY[0][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
                              - lhsY[0][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];
  lhsY[1][4][BB][j][i][k] = lhsY[1][4][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
                              - lhsY[1][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
                              - lhsY[1][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
                              - lhsY[1][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
                              - lhsY[1][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];
  lhsY[2][4][BB][j][i][k] = lhsY[2][4][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
                              - lhsY[2][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
                              - lhsY[2][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
                              - lhsY[2][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
                              - lhsY[2][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];
  lhsY[3][4][BB][j][i][k] = lhsY[3][4][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
                              - lhsY[3][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
                              - lhsY[3][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
                              - lhsY[3][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
                              - lhsY[3][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];
  lhsY[4][4][BB][j][i][k] = lhsY[4][4][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
                              - lhsY[4][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
                              - lhsY[4][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
                              - lhsY[4][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
                              - lhsY[4][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];

  pivot = 1.00/lhsY[0][0][BB][j][i][k];
  lhsY[0][1][BB][j][i][k] = lhsY[0][1][BB][j][i][k]*pivot;
  lhsY[0][2][BB][j][i][k] = lhsY[0][2][BB][j][i][k]*pivot;
  lhsY[0][3][BB][j][i][k] = lhsY[0][3][BB][j][i][k]*pivot;
  lhsY[0][4][BB][j][i][k] = lhsY[0][4][BB][j][i][k]*pivot;
  lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k]*pivot;
  lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k]*pivot;
  lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k]*pivot;
  lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k]*pivot;
  lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k]*pivot;
  rhs[0][k][j][i]   = rhs[0][k][j][i]  *pivot;

  coeff = lhsY[1][0][BB][j][i][k];
  lhsY[1][1][BB][j][i][k]= lhsY[1][1][BB][j][i][k] - coeff*lhsY[0][1][BB][j][i][k];
  lhsY[1][2][BB][j][i][k]= lhsY[1][2][BB][j][i][k] - coeff*lhsY[0][2][BB][j][i][k];
  lhsY[1][3][BB][j][i][k]= lhsY[1][3][BB][j][i][k] - coeff*lhsY[0][3][BB][j][i][k];
  lhsY[1][4][BB][j][i][k]= lhsY[1][4][BB][j][i][k] - coeff*lhsY[0][4][BB][j][i][k];
  lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k] - coeff*lhsY[0][0][CC][j][i][k];
  lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k] - coeff*lhsY[0][1][CC][j][i][k];
  lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k] - coeff*lhsY[0][2][CC][j][i][k];
  lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k] - coeff*lhsY[0][3][CC][j][i][k];
  lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k] - coeff*lhsY[0][4][CC][j][i][k];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsY[2][0][BB][j][i][k];
  lhsY[2][1][BB][j][i][k]= lhsY[2][1][BB][j][i][k] - coeff*lhsY[0][1][BB][j][i][k];
  lhsY[2][2][BB][j][i][k]= lhsY[2][2][BB][j][i][k] - coeff*lhsY[0][2][BB][j][i][k];
  lhsY[2][3][BB][j][i][k]= lhsY[2][3][BB][j][i][k] - coeff*lhsY[0][3][BB][j][i][k];
  lhsY[2][4][BB][j][i][k]= lhsY[2][4][BB][j][i][k] - coeff*lhsY[0][4][BB][j][i][k];
  lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k] - coeff*lhsY[0][0][CC][j][i][k];
  lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k] - coeff*lhsY[0][1][CC][j][i][k];
  lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k] - coeff*lhsY[0][2][CC][j][i][k];
  lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k] - coeff*lhsY[0][3][CC][j][i][k];
  lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k] - coeff*lhsY[0][4][CC][j][i][k];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsY[3][0][BB][j][i][k];
  lhsY[3][1][BB][j][i][k]= lhsY[3][1][BB][j][i][k] - coeff*lhsY[0][1][BB][j][i][k];
  lhsY[3][2][BB][j][i][k]= lhsY[3][2][BB][j][i][k] - coeff*lhsY[0][2][BB][j][i][k];
  lhsY[3][3][BB][j][i][k]= lhsY[3][3][BB][j][i][k] - coeff*lhsY[0][3][BB][j][i][k];
  lhsY[3][4][BB][j][i][k]= lhsY[3][4][BB][j][i][k] - coeff*lhsY[0][4][BB][j][i][k];
  lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k] - coeff*lhsY[0][0][CC][j][i][k];
  lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k] - coeff*lhsY[0][1][CC][j][i][k];
  lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k] - coeff*lhsY[0][2][CC][j][i][k];
  lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k] - coeff*lhsY[0][3][CC][j][i][k];
  lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k] - coeff*lhsY[0][4][CC][j][i][k];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsY[4][0][BB][j][i][k];
  lhsY[4][1][BB][j][i][k]= lhsY[4][1][BB][j][i][k] - coeff*lhsY[0][1][BB][j][i][k];
  lhsY[4][2][BB][j][i][k]= lhsY[4][2][BB][j][i][k] - coeff*lhsY[0][2][BB][j][i][k];
  lhsY[4][3][BB][j][i][k]= lhsY[4][3][BB][j][i][k] - coeff*lhsY[0][3][BB][j][i][k];
  lhsY[4][4][BB][j][i][k]= lhsY[4][4][BB][j][i][k] - coeff*lhsY[0][4][BB][j][i][k];
  lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k] - coeff*lhsY[0][0][CC][j][i][k];
  lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k] - coeff*lhsY[0][1][CC][j][i][k];
  lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k] - coeff*lhsY[0][2][CC][j][i][k];
  lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k] - coeff*lhsY[0][3][CC][j][i][k];
  lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k] - coeff*lhsY[0][4][CC][j][i][k];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[0][k][j][i];

  pivot = 1.00/lhsY[1][1][BB][j][i][k];
  lhsY[1][2][BB][j][i][k] = lhsY[1][2][BB][j][i][k]*pivot;
  lhsY[1][3][BB][j][i][k] = lhsY[1][3][BB][j][i][k]*pivot;
  lhsY[1][4][BB][j][i][k] = lhsY[1][4][BB][j][i][k]*pivot;
  lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k]*pivot;
  lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k]*pivot;
  lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k]*pivot;
  lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k]*pivot;
  lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k]*pivot;
  rhs[1][k][j][i]   = rhs[1][k][j][i]  *pivot;

  coeff = lhsY[0][1][BB][j][i][k];
  lhsY[0][2][BB][j][i][k]= lhsY[0][2][BB][j][i][k] - coeff*lhsY[1][2][BB][j][i][k];
  lhsY[0][3][BB][j][i][k]= lhsY[0][3][BB][j][i][k] - coeff*lhsY[1][3][BB][j][i][k];
  lhsY[0][4][BB][j][i][k]= lhsY[0][4][BB][j][i][k] - coeff*lhsY[1][4][BB][j][i][k];
  lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k] - coeff*lhsY[1][0][CC][j][i][k];
  lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k] - coeff*lhsY[1][1][CC][j][i][k];
  lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k] - coeff*lhsY[1][2][CC][j][i][k];
  lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k] - coeff*lhsY[1][3][CC][j][i][k];
  lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k] - coeff*lhsY[1][4][CC][j][i][k];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsY[2][1][BB][j][i][k];
  lhsY[2][2][BB][j][i][k]= lhsY[2][2][BB][j][i][k] - coeff*lhsY[1][2][BB][j][i][k];
  lhsY[2][3][BB][j][i][k]= lhsY[2][3][BB][j][i][k] - coeff*lhsY[1][3][BB][j][i][k];
  lhsY[2][4][BB][j][i][k]= lhsY[2][4][BB][j][i][k] - coeff*lhsY[1][4][BB][j][i][k];
  lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k] - coeff*lhsY[1][0][CC][j][i][k];
  lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k] - coeff*lhsY[1][1][CC][j][i][k];
  lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k] - coeff*lhsY[1][2][CC][j][i][k];
  lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k] - coeff*lhsY[1][3][CC][j][i][k];
  lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k] - coeff*lhsY[1][4][CC][j][i][k];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsY[3][1][BB][j][i][k];
  lhsY[3][2][BB][j][i][k]= lhsY[3][2][BB][j][i][k] - coeff*lhsY[1][2][BB][j][i][k];
  lhsY[3][3][BB][j][i][k]= lhsY[3][3][BB][j][i][k] - coeff*lhsY[1][3][BB][j][i][k];
  lhsY[3][4][BB][j][i][k]= lhsY[3][4][BB][j][i][k] - coeff*lhsY[1][4][BB][j][i][k];
  lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k] - coeff*lhsY[1][0][CC][j][i][k];
  lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k] - coeff*lhsY[1][1][CC][j][i][k];
  lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k] - coeff*lhsY[1][2][CC][j][i][k];
  lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k] - coeff*lhsY[1][3][CC][j][i][k];
  lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k] - coeff*lhsY[1][4][CC][j][i][k];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsY[4][1][BB][j][i][k];
  lhsY[4][2][BB][j][i][k]= lhsY[4][2][BB][j][i][k] - coeff*lhsY[1][2][BB][j][i][k];
  lhsY[4][3][BB][j][i][k]= lhsY[4][3][BB][j][i][k] - coeff*lhsY[1][3][BB][j][i][k];
  lhsY[4][4][BB][j][i][k]= lhsY[4][4][BB][j][i][k] - coeff*lhsY[1][4][BB][j][i][k];
  lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k] - coeff*lhsY[1][0][CC][j][i][k];
  lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k] - coeff*lhsY[1][1][CC][j][i][k];
  lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k] - coeff*lhsY[1][2][CC][j][i][k];
  lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k] - coeff*lhsY[1][3][CC][j][i][k];
  lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k] - coeff*lhsY[1][4][CC][j][i][k];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[1][k][j][i];

  pivot = 1.00/lhsY[2][2][BB][j][i][k];
  lhsY[2][3][BB][j][i][k] = lhsY[2][3][BB][j][i][k]*pivot;
  lhsY[2][4][BB][j][i][k] = lhsY[2][4][BB][j][i][k]*pivot;
  lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k]*pivot;
  lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k]*pivot;
  lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k]*pivot;
  lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k]*pivot;
  lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k]*pivot;
  rhs[2][k][j][i]   = rhs[2][k][j][i]  *pivot;

  coeff = lhsY[0][2][BB][j][i][k];
  lhsY[0][3][BB][j][i][k]= lhsY[0][3][BB][j][i][k] - coeff*lhsY[2][3][BB][j][i][k];
  lhsY[0][4][BB][j][i][k]= lhsY[0][4][BB][j][i][k] - coeff*lhsY[2][4][BB][j][i][k];
  lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k] - coeff*lhsY[2][0][CC][j][i][k];
  lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k] - coeff*lhsY[2][1][CC][j][i][k];
  lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k] - coeff*lhsY[2][2][CC][j][i][k];
  lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k] - coeff*lhsY[2][3][CC][j][i][k];
  lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k] - coeff*lhsY[2][4][CC][j][i][k];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsY[1][2][BB][j][i][k];
  lhsY[1][3][BB][j][i][k]= lhsY[1][3][BB][j][i][k] - coeff*lhsY[2][3][BB][j][i][k];
  lhsY[1][4][BB][j][i][k]= lhsY[1][4][BB][j][i][k] - coeff*lhsY[2][4][BB][j][i][k];
  lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k] - coeff*lhsY[2][0][CC][j][i][k];
  lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k] - coeff*lhsY[2][1][CC][j][i][k];
  lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k] - coeff*lhsY[2][2][CC][j][i][k];
  lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k] - coeff*lhsY[2][3][CC][j][i][k];
  lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k] - coeff*lhsY[2][4][CC][j][i][k];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsY[3][2][BB][j][i][k];
  lhsY[3][3][BB][j][i][k]= lhsY[3][3][BB][j][i][k] - coeff*lhsY[2][3][BB][j][i][k];
  lhsY[3][4][BB][j][i][k]= lhsY[3][4][BB][j][i][k] - coeff*lhsY[2][4][BB][j][i][k];
  lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k] - coeff*lhsY[2][0][CC][j][i][k];
  lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k] - coeff*lhsY[2][1][CC][j][i][k];
  lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k] - coeff*lhsY[2][2][CC][j][i][k];
  lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k] - coeff*lhsY[2][3][CC][j][i][k];
  lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k] - coeff*lhsY[2][4][CC][j][i][k];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsY[4][2][BB][j][i][k];
  lhsY[4][3][BB][j][i][k]= lhsY[4][3][BB][j][i][k] - coeff*lhsY[2][3][BB][j][i][k];
  lhsY[4][4][BB][j][i][k]= lhsY[4][4][BB][j][i][k] - coeff*lhsY[2][4][BB][j][i][k];
  lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k] - coeff*lhsY[2][0][CC][j][i][k];
  lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k] - coeff*lhsY[2][1][CC][j][i][k];
  lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k] - coeff*lhsY[2][2][CC][j][i][k];
  lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k] - coeff*lhsY[2][3][CC][j][i][k];
  lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k] - coeff*lhsY[2][4][CC][j][i][k];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[2][k][j][i];

  pivot = 1.00/lhsY[3][3][BB][j][i][k];
  lhsY[3][4][BB][j][i][k] = lhsY[3][4][BB][j][i][k]*pivot;
  lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k]*pivot;
  lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k]*pivot;
  lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k]*pivot;
  lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k]*pivot;
  lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k]*pivot;
  rhs[3][k][j][i]   = rhs[3][k][j][i]  *pivot;

  coeff = lhsY[0][3][BB][j][i][k];
  lhsY[0][4][BB][j][i][k]= lhsY[0][4][BB][j][i][k] - coeff*lhsY[3][4][BB][j][i][k];
  lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k] - coeff*lhsY[3][0][CC][j][i][k];
  lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k] - coeff*lhsY[3][1][CC][j][i][k];
  lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k] - coeff*lhsY[3][2][CC][j][i][k];
  lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k] - coeff*lhsY[3][3][CC][j][i][k];
  lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k] - coeff*lhsY[3][4][CC][j][i][k];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsY[1][3][BB][j][i][k];
  lhsY[1][4][BB][j][i][k]= lhsY[1][4][BB][j][i][k] - coeff*lhsY[3][4][BB][j][i][k];
  lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k] - coeff*lhsY[3][0][CC][j][i][k];
  lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k] - coeff*lhsY[3][1][CC][j][i][k];
  lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k] - coeff*lhsY[3][2][CC][j][i][k];
  lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k] - coeff*lhsY[3][3][CC][j][i][k];
  lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k] - coeff*lhsY[3][4][CC][j][i][k];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsY[2][3][BB][j][i][k];
  lhsY[2][4][BB][j][i][k]= lhsY[2][4][BB][j][i][k] - coeff*lhsY[3][4][BB][j][i][k];
  lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k] - coeff*lhsY[3][0][CC][j][i][k];
  lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k] - coeff*lhsY[3][1][CC][j][i][k];
  lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k] - coeff*lhsY[3][2][CC][j][i][k];
  lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k] - coeff*lhsY[3][3][CC][j][i][k];
  lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k] - coeff*lhsY[3][4][CC][j][i][k];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsY[4][3][BB][j][i][k];
  lhsY[4][4][BB][j][i][k]= lhsY[4][4][BB][j][i][k] - coeff*lhsY[3][4][BB][j][i][k];
  lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k] - coeff*lhsY[3][0][CC][j][i][k];
  lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k] - coeff*lhsY[3][1][CC][j][i][k];
  lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k] - coeff*lhsY[3][2][CC][j][i][k];
  lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k] - coeff*lhsY[3][3][CC][j][i][k];
  lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k] - coeff*lhsY[3][4][CC][j][i][k];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[3][k][j][i];

  pivot = 1.00/lhsY[4][4][BB][j][i][k];
  lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k]*pivot;
  lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k]*pivot;
  lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k]*pivot;
  lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k]*pivot;
  lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k]*pivot;
  rhs[4][k][j][i]   = rhs[4][k][j][i]  *pivot;

  coeff = lhsY[0][4][BB][j][i][k];
  lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k] - coeff*lhsY[4][0][CC][j][i][k];
  lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k] - coeff*lhsY[4][1][CC][j][i][k];
  lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k] - coeff*lhsY[4][2][CC][j][i][k];
  lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k] - coeff*lhsY[4][3][CC][j][i][k];
  lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k] - coeff*lhsY[4][4][CC][j][i][k];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsY[1][4][BB][j][i][k];
  lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k] - coeff*lhsY[4][0][CC][j][i][k];
  lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k] - coeff*lhsY[4][1][CC][j][i][k];
  lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k] - coeff*lhsY[4][2][CC][j][i][k];
  lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k] - coeff*lhsY[4][3][CC][j][i][k];
  lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k] - coeff*lhsY[4][4][CC][j][i][k];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsY[2][4][BB][j][i][k];
  lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k] - coeff*lhsY[4][0][CC][j][i][k];
  lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k] - coeff*lhsY[4][1][CC][j][i][k];
  lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k] - coeff*lhsY[4][2][CC][j][i][k];
  lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k] - coeff*lhsY[4][3][CC][j][i][k];
  lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k] - coeff*lhsY[4][4][CC][j][i][k];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsY[3][4][BB][j][i][k];
  lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k] - coeff*lhsY[4][0][CC][j][i][k];
  lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k] - coeff*lhsY[4][1][CC][j][i][k];
  lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k] - coeff*lhsY[4][2][CC][j][i][k];
  lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k] - coeff*lhsY[4][3][CC][j][i][k];
  lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k] - coeff*lhsY[4][4][CC][j][i][k];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[4][k][j][i];
      }
	}
  }
      
  for (k = 1; k <= gp22; k++) {
    for (i = 1; i <= gp02; i++) {
	
  rhs[0][k][jsize][i] = rhs[0][k][jsize][i] - lhsY[0][0][AA][jsize][i][k]*rhs[0][k][jsize-1][i]
                    - lhsY[0][1][AA][jsize][i][k]*rhs[1][k][jsize-1][i]
                    - lhsY[0][2][AA][jsize][i][k]*rhs[2][k][jsize-1][i]
                    - lhsY[0][3][AA][jsize][i][k]*rhs[3][k][jsize-1][i]
                    - lhsY[0][4][AA][jsize][i][k]*rhs[4][k][jsize-1][i];
  rhs[1][k][jsize][i] = rhs[1][k][jsize][i] - lhsY[1][0][AA][jsize][i][k]*rhs[0][k][jsize-1][i]
                    - lhsY[1][1][AA][jsize][i][k]*rhs[1][k][jsize-1][i]
                    - lhsY[1][2][AA][jsize][i][k]*rhs[2][k][jsize-1][i]
                    - lhsY[1][3][AA][jsize][i][k]*rhs[3][k][jsize-1][i]
                    - lhsY[1][4][AA][jsize][i][k]*rhs[4][k][jsize-1][i];
  rhs[2][k][jsize][i] = rhs[2][k][jsize][i] - lhsY[2][0][AA][jsize][i][k]*rhs[0][k][jsize-1][i]
                    - lhsY[2][1][AA][jsize][i][k]*rhs[1][k][jsize-1][i]
                    - lhsY[2][2][AA][jsize][i][k]*rhs[2][k][jsize-1][i]
                    - lhsY[2][3][AA][jsize][i][k]*rhs[3][k][jsize-1][i]
                    - lhsY[2][4][AA][jsize][i][k]*rhs[4][k][jsize-1][i];
  rhs[3][k][jsize][i] = rhs[3][k][jsize][i] - lhsY[3][0][AA][jsize][i][k]*rhs[0][k][jsize-1][i]
                    - lhsY[3][1][AA][jsize][i][k]*rhs[1][k][jsize-1][i]
                    - lhsY[3][2][AA][jsize][i][k]*rhs[2][k][jsize-1][i]
                    - lhsY[3][3][AA][jsize][i][k]*rhs[3][k][jsize-1][i]
                    - lhsY[3][4][AA][jsize][i][k]*rhs[4][k][jsize-1][i];
  rhs[4][k][jsize][i] = rhs[4][k][jsize][i] - lhsY[4][0][AA][jsize][i][k]*rhs[0][k][jsize-1][i]
                    - lhsY[4][1][AA][jsize][i][k]*rhs[1][k][jsize-1][i]
                    - lhsY[4][2][AA][jsize][i][k]*rhs[2][k][jsize-1][i]
                    - lhsY[4][3][AA][jsize][i][k]*rhs[3][k][jsize-1][i]
                    - lhsY[4][4][AA][jsize][i][k]*rhs[4][k][jsize-1][i];
	}
  }
      
    for (i = 1; i <= gp02; i++) {
  for (k = 1; k <= gp22; k++) {
	
  lhsY[0][0][BB][jsize][i][k] = lhsY[0][0][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
                              - lhsY[0][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
                              - lhsY[0][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
                              - lhsY[0][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
                              - lhsY[0][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
  lhsY[1][0][BB][jsize][i][k] = lhsY[1][0][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
                              - lhsY[1][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
                              - lhsY[1][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
                              - lhsY[1][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
                              - lhsY[1][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
  lhsY[2][0][BB][jsize][i][k] = lhsY[2][0][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
                              - lhsY[2][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
                              - lhsY[2][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
                              - lhsY[2][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
                              - lhsY[2][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
  lhsY[3][0][BB][jsize][i][k] = lhsY[3][0][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
                              - lhsY[3][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
                              - lhsY[3][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
                              - lhsY[3][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
                              - lhsY[3][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
  lhsY[4][0][BB][jsize][i][k] = lhsY[4][0][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
                              - lhsY[4][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
                              - lhsY[4][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
                              - lhsY[4][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
                              - lhsY[4][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
  lhsY[0][1][BB][jsize][i][k] = lhsY[0][1][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
                              - lhsY[0][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
                              - lhsY[0][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
                              - lhsY[0][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
                              - lhsY[0][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
  lhsY[1][1][BB][jsize][i][k] = lhsY[1][1][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
                              - lhsY[1][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
                              - lhsY[1][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
                              - lhsY[1][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
                              - lhsY[1][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
  lhsY[2][1][BB][jsize][i][k] = lhsY[2][1][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
                              - lhsY[2][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
                              - lhsY[2][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
                              - lhsY[2][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
                              - lhsY[2][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
  lhsY[3][1][BB][jsize][i][k] = lhsY[3][1][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
                              - lhsY[3][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
                              - lhsY[3][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
                              - lhsY[3][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
                              - lhsY[3][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
  lhsY[4][1][BB][jsize][i][k] = lhsY[4][1][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
                              - lhsY[4][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
                              - lhsY[4][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
                              - lhsY[4][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
                              - lhsY[4][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
  lhsY[0][2][BB][jsize][i][k] = lhsY[0][2][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
                              - lhsY[0][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
                              - lhsY[0][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
                              - lhsY[0][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
                              - lhsY[0][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
  lhsY[1][2][BB][jsize][i][k] = lhsY[1][2][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
                              - lhsY[1][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
                              - lhsY[1][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
                              - lhsY[1][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
                              - lhsY[1][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
  lhsY[2][2][BB][jsize][i][k] = lhsY[2][2][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
                              - lhsY[2][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
                              - lhsY[2][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
                              - lhsY[2][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
                              - lhsY[2][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
  lhsY[3][2][BB][jsize][i][k] = lhsY[3][2][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
                              - lhsY[3][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
                              - lhsY[3][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
                              - lhsY[3][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
                              - lhsY[3][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
  lhsY[4][2][BB][jsize][i][k] = lhsY[4][2][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
                              - lhsY[4][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
                              - lhsY[4][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
                              - lhsY[4][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
                              - lhsY[4][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
  lhsY[0][3][BB][jsize][i][k] = lhsY[0][3][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
                              - lhsY[0][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
                              - lhsY[0][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
                              - lhsY[0][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
                              - lhsY[0][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
  lhsY[1][3][BB][jsize][i][k] = lhsY[1][3][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
                              - lhsY[1][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
                              - lhsY[1][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
                              - lhsY[1][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
                              - lhsY[1][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
  lhsY[2][3][BB][jsize][i][k] = lhsY[2][3][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
                              - lhsY[2][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
                              - lhsY[2][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
                              - lhsY[2][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
                              - lhsY[2][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
  lhsY[3][3][BB][jsize][i][k] = lhsY[3][3][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
                              - lhsY[3][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
                              - lhsY[3][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
                              - lhsY[3][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
                              - lhsY[3][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
  lhsY[4][3][BB][jsize][i][k] = lhsY[4][3][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
                              - lhsY[4][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
                              - lhsY[4][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
                              - lhsY[4][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
                              - lhsY[4][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
  lhsY[0][4][BB][jsize][i][k] = lhsY[0][4][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
                              - lhsY[0][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
                              - lhsY[0][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
                              - lhsY[0][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
                              - lhsY[0][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];
  lhsY[1][4][BB][jsize][i][k] = lhsY[1][4][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
                              - lhsY[1][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
                              - lhsY[1][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
                              - lhsY[1][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
                              - lhsY[1][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];
  lhsY[2][4][BB][jsize][i][k] = lhsY[2][4][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
                              - lhsY[2][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
                              - lhsY[2][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
                              - lhsY[2][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
                              - lhsY[2][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];
  lhsY[3][4][BB][jsize][i][k] = lhsY[3][4][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
                              - lhsY[3][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
                              - lhsY[3][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
                              - lhsY[3][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
                              - lhsY[3][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];
  lhsY[4][4][BB][jsize][i][k] = lhsY[4][4][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
                              - lhsY[4][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
                              - lhsY[4][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
                              - lhsY[4][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
                              - lhsY[4][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];

	}
  }
      
    for (i = 1; i <= gp02; i++) { 
  for (k = 1; k <= gp22; k++) {
	
  pivot = 1.00/lhsY[0][0][BB][jsize][i][k];
  lhsY[0][1][BB][jsize][i][k] = lhsY[0][1][BB][jsize][i][k]*pivot;
  lhsY[0][2][BB][jsize][i][k] = lhsY[0][2][BB][jsize][i][k]*pivot;
  lhsY[0][3][BB][jsize][i][k] = lhsY[0][3][BB][jsize][i][k]*pivot;
  lhsY[0][4][BB][jsize][i][k] = lhsY[0][4][BB][jsize][i][k]*pivot;
  rhs[0][k][jsize][i]   = rhs[0][k][jsize][i]  *pivot;

  coeff = lhsY[1][0][BB][jsize][i][k];
  lhsY[1][1][BB][jsize][i][k]= lhsY[1][1][BB][jsize][i][k] - coeff*lhsY[0][1][BB][jsize][i][k];
  lhsY[1][2][BB][jsize][i][k]= lhsY[1][2][BB][jsize][i][k] - coeff*lhsY[0][2][BB][jsize][i][k];
  lhsY[1][3][BB][jsize][i][k]= lhsY[1][3][BB][jsize][i][k] - coeff*lhsY[0][3][BB][jsize][i][k];
  lhsY[1][4][BB][jsize][i][k]= lhsY[1][4][BB][jsize][i][k] - coeff*lhsY[0][4][BB][jsize][i][k];
  rhs[1][k][jsize][i]   = rhs[1][k][jsize][i]   - coeff*rhs[0][k][jsize][i];

  coeff = lhsY[2][0][BB][jsize][i][k];
  lhsY[2][1][BB][jsize][i][k]= lhsY[2][1][BB][jsize][i][k] - coeff*lhsY[0][1][BB][jsize][i][k];
  lhsY[2][2][BB][jsize][i][k]= lhsY[2][2][BB][jsize][i][k] - coeff*lhsY[0][2][BB][jsize][i][k];
  lhsY[2][3][BB][jsize][i][k]= lhsY[2][3][BB][jsize][i][k] - coeff*lhsY[0][3][BB][jsize][i][k];
  lhsY[2][4][BB][jsize][i][k]= lhsY[2][4][BB][jsize][i][k] - coeff*lhsY[0][4][BB][jsize][i][k];
  rhs[2][k][jsize][i]   = rhs[2][k][jsize][i]   - coeff*rhs[0][k][jsize][i];

  coeff = lhsY[3][0][BB][jsize][i][k];
  lhsY[3][1][BB][jsize][i][k]= lhsY[3][1][BB][jsize][i][k] - coeff*lhsY[0][1][BB][jsize][i][k];
  lhsY[3][2][BB][jsize][i][k]= lhsY[3][2][BB][jsize][i][k] - coeff*lhsY[0][2][BB][jsize][i][k];
  lhsY[3][3][BB][jsize][i][k]= lhsY[3][3][BB][jsize][i][k] - coeff*lhsY[0][3][BB][jsize][i][k];
  lhsY[3][4][BB][jsize][i][k]= lhsY[3][4][BB][jsize][i][k] - coeff*lhsY[0][4][BB][jsize][i][k];
  rhs[3][k][jsize][i]   = rhs[3][k][jsize][i]   - coeff*rhs[0][k][jsize][i];

  coeff = lhsY[4][0][BB][jsize][i][k];
  lhsY[4][1][BB][jsize][i][k]= lhsY[4][1][BB][jsize][i][k] - coeff*lhsY[0][1][BB][jsize][i][k];
  lhsY[4][2][BB][jsize][i][k]= lhsY[4][2][BB][jsize][i][k] - coeff*lhsY[0][2][BB][jsize][i][k];
  lhsY[4][3][BB][jsize][i][k]= lhsY[4][3][BB][jsize][i][k] - coeff*lhsY[0][3][BB][jsize][i][k];
  lhsY[4][4][BB][jsize][i][k]= lhsY[4][4][BB][jsize][i][k] - coeff*lhsY[0][4][BB][jsize][i][k];
  rhs[4][k][jsize][i]   = rhs[4][k][jsize][i]   - coeff*rhs[0][k][jsize][i];

  pivot = 1.00/lhsY[1][1][BB][jsize][i][k];
  lhsY[1][2][BB][jsize][i][k] = lhsY[1][2][BB][jsize][i][k]*pivot;
  lhsY[1][3][BB][jsize][i][k] = lhsY[1][3][BB][jsize][i][k]*pivot;
  lhsY[1][4][BB][jsize][i][k] = lhsY[1][4][BB][jsize][i][k]*pivot;
  rhs[1][k][jsize][i]   = rhs[1][k][jsize][i]  *pivot;

  coeff = lhsY[0][1][BB][jsize][i][k];
  lhsY[0][2][BB][jsize][i][k]= lhsY[0][2][BB][jsize][i][k] - coeff*lhsY[1][2][BB][jsize][i][k];
  lhsY[0][3][BB][jsize][i][k]= lhsY[0][3][BB][jsize][i][k] - coeff*lhsY[1][3][BB][jsize][i][k];
  lhsY[0][4][BB][jsize][i][k]= lhsY[0][4][BB][jsize][i][k] - coeff*lhsY[1][4][BB][jsize][i][k];
  rhs[0][k][jsize][i]   = rhs[0][k][jsize][i]   - coeff*rhs[1][k][jsize][i];

  coeff = lhsY[2][1][BB][jsize][i][k];
  lhsY[2][2][BB][jsize][i][k]= lhsY[2][2][BB][jsize][i][k] - coeff*lhsY[1][2][BB][jsize][i][k];
  lhsY[2][3][BB][jsize][i][k]= lhsY[2][3][BB][jsize][i][k] - coeff*lhsY[1][3][BB][jsize][i][k];
  lhsY[2][4][BB][jsize][i][k]= lhsY[2][4][BB][jsize][i][k] - coeff*lhsY[1][4][BB][jsize][i][k];
  rhs[2][k][jsize][i]   = rhs[2][k][jsize][i]   - coeff*rhs[1][k][jsize][i];

  coeff = lhsY[3][1][BB][jsize][i][k];
  lhsY[3][2][BB][jsize][i][k]= lhsY[3][2][BB][jsize][i][k] - coeff*lhsY[1][2][BB][jsize][i][k];
  lhsY[3][3][BB][jsize][i][k]= lhsY[3][3][BB][jsize][i][k] - coeff*lhsY[1][3][BB][jsize][i][k];
  lhsY[3][4][BB][jsize][i][k]= lhsY[3][4][BB][jsize][i][k] - coeff*lhsY[1][4][BB][jsize][i][k];
  rhs[3][k][jsize][i]   = rhs[3][k][jsize][i]   - coeff*rhs[1][k][jsize][i];

  coeff = lhsY[4][1][BB][jsize][i][k];
  lhsY[4][2][BB][jsize][i][k]= lhsY[4][2][BB][jsize][i][k] - coeff*lhsY[1][2][BB][jsize][i][k];
  lhsY[4][3][BB][jsize][i][k]= lhsY[4][3][BB][jsize][i][k] - coeff*lhsY[1][3][BB][jsize][i][k];
  lhsY[4][4][BB][jsize][i][k]= lhsY[4][4][BB][jsize][i][k] - coeff*lhsY[1][4][BB][jsize][i][k];
  rhs[4][k][jsize][i]   = rhs[4][k][jsize][i]   - coeff*rhs[1][k][jsize][i];

  pivot = 1.00/lhsY[2][2][BB][jsize][i][k];
  lhsY[2][3][BB][jsize][i][k] = lhsY[2][3][BB][jsize][i][k]*pivot;
  lhsY[2][4][BB][jsize][i][k] = lhsY[2][4][BB][jsize][i][k]*pivot;
  rhs[2][k][jsize][i]   = rhs[2][k][jsize][i]  *pivot;

  coeff = lhsY[0][2][BB][jsize][i][k];
  lhsY[0][3][BB][jsize][i][k]= lhsY[0][3][BB][jsize][i][k] - coeff*lhsY[2][3][BB][jsize][i][k];
  lhsY[0][4][BB][jsize][i][k]= lhsY[0][4][BB][jsize][i][k] - coeff*lhsY[2][4][BB][jsize][i][k];
  rhs[0][k][jsize][i]   = rhs[0][k][jsize][i]   - coeff*rhs[2][k][jsize][i];

  coeff = lhsY[1][2][BB][jsize][i][k];
  lhsY[1][3][BB][jsize][i][k]= lhsY[1][3][BB][jsize][i][k] - coeff*lhsY[2][3][BB][jsize][i][k];
  lhsY[1][4][BB][jsize][i][k]= lhsY[1][4][BB][jsize][i][k] - coeff*lhsY[2][4][BB][jsize][i][k];
  rhs[1][k][jsize][i]   = rhs[1][k][jsize][i]   - coeff*rhs[2][k][jsize][i];

  coeff = lhsY[3][2][BB][jsize][i][k];
  lhsY[3][3][BB][jsize][i][k]= lhsY[3][3][BB][jsize][i][k] - coeff*lhsY[2][3][BB][jsize][i][k];
  lhsY[3][4][BB][jsize][i][k]= lhsY[3][4][BB][jsize][i][k] - coeff*lhsY[2][4][BB][jsize][i][k];
  rhs[3][k][jsize][i]   = rhs[3][k][jsize][i]   - coeff*rhs[2][k][jsize][i];

  coeff = lhsY[4][2][BB][jsize][i][k];
  lhsY[4][3][BB][jsize][i][k]= lhsY[4][3][BB][jsize][i][k] - coeff*lhsY[2][3][BB][jsize][i][k];
  lhsY[4][4][BB][jsize][i][k]= lhsY[4][4][BB][jsize][i][k] - coeff*lhsY[2][4][BB][jsize][i][k];
  rhs[4][k][jsize][i]   = rhs[4][k][jsize][i]   - coeff*rhs[2][k][jsize][i];

  pivot = 1.00/lhsY[3][3][BB][jsize][i][k];
  lhsY[3][4][BB][jsize][i][k] = lhsY[3][4][BB][jsize][i][k]*pivot;
  rhs[3][k][jsize][i]   = rhs[3][k][jsize][i]  *pivot;

  coeff = lhsY[0][3][BB][jsize][i][k];
  lhsY[0][4][BB][jsize][i][k]= lhsY[0][4][BB][jsize][i][k] - coeff*lhsY[3][4][BB][jsize][i][k];
  rhs[0][k][jsize][i]   = rhs[0][k][jsize][i]   - coeff*rhs[3][k][jsize][i];

  coeff = lhsY[1][3][BB][jsize][i][k];
  lhsY[1][4][BB][jsize][i][k]= lhsY[1][4][BB][jsize][i][k] - coeff*lhsY[3][4][BB][jsize][i][k];
  rhs[1][k][jsize][i]   = rhs[1][k][jsize][i]   - coeff*rhs[3][k][jsize][i];

  coeff = lhsY[2][3][BB][jsize][i][k];
  lhsY[2][4][BB][jsize][i][k]= lhsY[2][4][BB][jsize][i][k] - coeff*lhsY[3][4][BB][jsize][i][k];
  rhs[2][k][jsize][i]   = rhs[2][k][jsize][i]   - coeff*rhs[3][k][jsize][i];

  coeff = lhsY[4][3][BB][jsize][i][k];
  lhsY[4][4][BB][jsize][i][k]= lhsY[4][4][BB][jsize][i][k] - coeff*lhsY[3][4][BB][jsize][i][k];
  rhs[4][k][jsize][i]   = rhs[4][k][jsize][i]   - coeff*rhs[3][k][jsize][i];

  pivot = 1.00/lhsY[4][4][BB][jsize][i][k];
  rhs[4][k][jsize][i]   = rhs[4][k][jsize][i]  *pivot;

  coeff = lhsY[0][4][BB][jsize][i][k];
  rhs[0][k][jsize][i]   = rhs[0][k][jsize][i]   - coeff*rhs[4][k][jsize][i];

  coeff = lhsY[1][4][BB][jsize][i][k];
  rhs[1][k][jsize][i]   = rhs[1][k][jsize][i]   - coeff*rhs[4][k][jsize][i];

  coeff = lhsY[2][4][BB][jsize][i][k];
  rhs[2][k][jsize][i]   = rhs[2][k][jsize][i]   - coeff*rhs[4][k][jsize][i];

  coeff = lhsY[3][4][BB][jsize][i][k];
  rhs[3][k][jsize][i]   = rhs[3][k][jsize][i]   - coeff*rhs[4][k][jsize][i];

	}
  }
      
      for (j = jsize-1; j >= 0; j--) {
  for (k = 1; k <= gp22; k++) {
    for (i = 1; i <= gp02; i++) {
        
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsY[0][0][CC][j][i][k]*rhs[0][k][j+1][i];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsY[0][1][CC][j][i][k]*rhs[1][k][j+1][i];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsY[0][2][CC][j][i][k]*rhs[2][k][j+1][i];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsY[0][3][CC][j][i][k]*rhs[3][k][j+1][i];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsY[0][4][CC][j][i][k]*rhs[4][k][j+1][i];
            
			rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsY[1][0][CC][j][i][k]*rhs[0][k][j+1][i];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsY[1][1][CC][j][i][k]*rhs[1][k][j+1][i];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsY[1][2][CC][j][i][k]*rhs[2][k][j+1][i];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsY[1][3][CC][j][i][k]*rhs[3][k][j+1][i];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsY[1][4][CC][j][i][k]*rhs[4][k][j+1][i];
			
			rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsY[2][0][CC][j][i][k]*rhs[0][k][j+1][i];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsY[2][1][CC][j][i][k]*rhs[1][k][j+1][i];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsY[2][2][CC][j][i][k]*rhs[2][k][j+1][i];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsY[2][3][CC][j][i][k]*rhs[3][k][j+1][i];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsY[2][4][CC][j][i][k]*rhs[4][k][j+1][i];
			
			rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsY[3][0][CC][j][i][k]*rhs[0][k][j+1][i];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsY[3][1][CC][j][i][k]*rhs[1][k][j+1][i];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsY[3][2][CC][j][i][k]*rhs[2][k][j+1][i];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsY[3][3][CC][j][i][k]*rhs[3][k][j+1][i];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsY[3][4][CC][j][i][k]*rhs[4][k][j+1][i];
			
			rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsY[4][0][CC][j][i][k]*rhs[0][k][j+1][i];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsY[4][1][CC][j][i][k]*rhs[1][k][j+1][i];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsY[4][2][CC][j][i][k]*rhs[2][k][j+1][i];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsY[4][3][CC][j][i][k]*rhs[3][k][j+1][i];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsY[4][4][CC][j][i][k]*rhs[4][k][j+1][i];
      }
    }
  }
}

void z_solve()
{
  int i, j, k, m, n, ksize, z;
  double pivot, coeff;
  int gp12, gp02;
  double fjacZ[5][5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1];
  double njacZ[5][5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1];
  double lhsZ[5][5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1];
  double temp1, temp2, temp3;

  gp12 = grid_points[1]-2;
  gp02 = grid_points[0]-2;

  ksize = grid_points[2]-1;

      for (k = 0; k <= ksize; k++) {
    for (i = 1; i <= gp02; i++) {
  for (j = 1; j <= gp12; j++) {
        temp1 = 1.0 / u[0][k][j][i];
        temp2 = temp1 * temp1;
        temp3 = temp1 * temp2;

        fjacZ[0][0][k][i][j] = 0.0;
        fjacZ[0][1][k][i][j] = 0.0;
        fjacZ[0][2][k][i][j] = 0.0;
        fjacZ[0][3][k][i][j] = 1.0;
        fjacZ[0][4][k][i][j] = 0.0;

        fjacZ[1][0][k][i][j] = - ( u[1][k][j][i]*u[3][k][j][i] ) * temp2;
        fjacZ[1][1][k][i][j] = u[3][k][j][i] * temp1;
        fjacZ[1][2][k][i][j] = 0.0;
        fjacZ[1][3][k][i][j] = u[1][k][j][i] * temp1;
        fjacZ[1][4][k][i][j] = 0.0;

        fjacZ[2][0][k][i][j] = - ( u[2][k][j][i]*u[3][k][j][i] ) * temp2;
        fjacZ[2][1][k][i][j] = 0.0;
        fjacZ[2][2][k][i][j] = u[3][k][j][i] * temp1;
        fjacZ[2][3][k][i][j] = u[2][k][j][i] * temp1;
        fjacZ[2][4][k][i][j] = 0.0;

        fjacZ[3][0][k][i][j] = - (u[3][k][j][i]*u[3][k][j][i] * temp2 ) 
          + c2 * qs[k][j][i];
        fjacZ[3][1][k][i][j] = - c2 *  u[1][k][j][i] * temp1;
        fjacZ[3][2][k][i][j] = - c2 *  u[2][k][j][i] * temp1;
        fjacZ[3][3][k][i][j] = ( 2.0 - c2 ) *  u[3][k][j][i] * temp1;
        fjacZ[3][4][k][i][j] = c2;

        fjacZ[4][0][k][i][j] = ( c2 * 2.0 * square[k][j][i] - c1 * u[4][k][j][i] )
          * u[3][k][j][i] * temp2;
        fjacZ[4][1][k][i][j] = - c2 * ( u[1][k][j][i]*u[3][k][j][i] ) * temp2;
        fjacZ[4][2][k][i][j] = - c2 * ( u[2][k][j][i]*u[3][k][j][i] ) * temp2;
        fjacZ[4][3][k][i][j] = c1 * ( u[4][k][j][i] * temp1 )
          - c2 * ( qs[k][j][i] + u[3][k][j][i]*u[3][k][j][i] * temp2 );
        fjacZ[4][4][k][i][j] = c1 * u[3][k][j][i] * temp1;

        njacZ[0][0][k][i][j] = 0.0;
        njacZ[0][1][k][i][j] = 0.0;
        njacZ[0][2][k][i][j] = 0.0;
        njacZ[0][3][k][i][j] = 0.0;
        njacZ[0][4][k][i][j] = 0.0;

        njacZ[1][0][k][i][j] = - c3c4 * temp2 * u[1][k][j][i];
        njacZ[1][1][k][i][j] =   c3c4 * temp1;
        njacZ[1][2][k][i][j] =   0.0;
        njacZ[1][3][k][i][j] =   0.0;
        njacZ[1][4][k][i][j] =   0.0;

        njacZ[2][0][k][i][j] = - c3c4 * temp2 * u[2][k][j][i];
        njacZ[2][1][k][i][j] =   0.0;
        njacZ[2][2][k][i][j] =   c3c4 * temp1;
        njacZ[2][3][k][i][j] =   0.0;
        njacZ[2][4][k][i][j] =   0.0;

        njacZ[3][0][k][i][j] = - con43 * c3c4 * temp2 * u[3][k][j][i];
        njacZ[3][1][k][i][j] =   0.0;
        njacZ[3][2][k][i][j] =   0.0;
        njacZ[3][3][k][i][j] =   con43 * c3 * c4 * temp1;
        njacZ[3][4][k][i][j] =   0.0;

        njacZ[4][0][k][i][j] = - (  c3c4
            - c1345 ) * temp3 * (u[1][k][j][i]*u[1][k][j][i])
          - ( c3c4 - c1345 ) * temp3 * (u[2][k][j][i]*u[2][k][j][i])
          - ( con43 * c3c4
              - c1345 ) * temp3 * (u[3][k][j][i]*u[3][k][j][i])
          - c1345 * temp2 * u[4][k][j][i];

        njacZ[4][1][k][i][j] = (  c3c4 - c1345 ) * temp2 * u[1][k][j][i];
        njacZ[4][2][k][i][j] = (  c3c4 - c1345 ) * temp2 * u[2][k][j][i];
        njacZ[4][3][k][i][j] = ( con43 * c3c4
            - c1345 ) * temp2 * u[3][k][j][i];
        njacZ[4][4][k][i][j] = ( c1345 )* temp1;
      }
	}
  }
      
    for (i = 1; i <= gp02; i++) {
  for (j = 1; j <= gp12; j++) {
  		for (n = 0; n < 5; n++) {
    		for (m = 0; m < 5; m++) {
      			lhsZ[m][n][0][0][i][j] = 0.0;
      			lhsZ[m][n][1][0][i][j] = 0.0;
      			lhsZ[m][n][2][0][i][j] = 0.0;
      			lhsZ[m][n][0][ksize][i][j] = 0.0;
      			lhsZ[m][n][1][ksize][i][j] = 0.0;
      			lhsZ[m][n][2][ksize][i][j] = 0.0;
    		}
  		}
	}
  }

    for (i = 1; i <= gp02; i++) {
  for (j = 1; j <= gp12; j++) {
    		lhsZ[0][0][1][0][i][j] = 1.0;
    		lhsZ[0][0][1][ksize][i][j] = 1.0;
    		lhsZ[1][1][1][0][i][j] = 1.0;
    		lhsZ[1][1][1][ksize][i][j] = 1.0;
    		lhsZ[2][2][1][0][i][j] = 1.0;
    		lhsZ[2][2][1][ksize][i][j] = 1.0;
    		lhsZ[3][3][1][0][i][j] = 1.0;
    		lhsZ[3][3][1][ksize][i][j] = 1.0;
    		lhsZ[4][4][1][0][i][j] = 1.0;
    		lhsZ[4][4][1][ksize][i][j] = 1.0;
	}
  }

      for (k = 1; k <= ksize-1; k++) {
    for (i = 1; i <= gp02; i++) {
  for (j = 1; j <= gp12; j++) {
        temp1 = dt * tz1;
        temp2 = dt * tz2;

        lhsZ[0][0][AA][k][i][j] = - temp2 * fjacZ[0][0][k-1][i][j]
          - temp1 * njacZ[0][0][k-1][i][j]
          - temp1 * dz1; 
        lhsZ[0][1][AA][k][i][j] = - temp2 * fjacZ[0][1][k-1][i][j]
          - temp1 * njacZ[0][1][k-1][i][j];
        lhsZ[0][2][AA][k][i][j] = - temp2 * fjacZ[0][2][k-1][i][j]
          - temp1 * njacZ[0][2][k-1][i][j];
        lhsZ[0][3][AA][k][i][j] = - temp2 * fjacZ[0][3][k-1][i][j]
          - temp1 * njacZ[0][3][k-1][i][j];
        lhsZ[0][4][AA][k][i][j] = - temp2 * fjacZ[0][4][k-1][i][j]
          - temp1 * njacZ[0][4][k-1][i][j];

        lhsZ[1][0][AA][k][i][j] = - temp2 * fjacZ[1][0][k-1][i][j]
          - temp1 * njacZ[1][0][k-1][i][j];
        lhsZ[1][1][AA][k][i][j] = - temp2 * fjacZ[1][1][k-1][i][j]
          - temp1 * njacZ[1][1][k-1][i][j]
          - temp1 * dz2;
        lhsZ[1][2][AA][k][i][j] = - temp2 * fjacZ[1][2][k-1][i][j]
          - temp1 * njacZ[1][2][k-1][i][j];
        lhsZ[1][3][AA][k][i][j] = - temp2 * fjacZ[1][3][k-1][i][j]
          - temp1 * njacZ[1][3][k-1][i][j];
        lhsZ[1][4][AA][k][i][j] = - temp2 * fjacZ[1][4][k-1][i][j]
          - temp1 * njacZ[1][4][k-1][i][j];

        lhsZ[2][0][AA][k][i][j] = - temp2 * fjacZ[2][0][k-1][i][j]
          - temp1 * njacZ[2][0][k-1][i][j];
        lhsZ[2][1][AA][k][i][j] = - temp2 * fjacZ[2][1][k-1][i][j]
          - temp1 * njacZ[2][1][k-1][i][j];
        lhsZ[2][2][AA][k][i][j] = - temp2 * fjacZ[2][2][k-1][i][j]
          - temp1 * njacZ[2][2][k-1][i][j]
          - temp1 * dz3;
        lhsZ[2][3][AA][k][i][j] = - temp2 * fjacZ[2][3][k-1][i][j]
          - temp1 * njacZ[2][3][k-1][i][j];
        lhsZ[2][4][AA][k][i][j] = - temp2 * fjacZ[2][4][k-1][i][j]
          - temp1 * njacZ[2][4][k-1][i][j];

        lhsZ[3][0][AA][k][i][j] = - temp2 * fjacZ[3][0][k-1][i][j]
          - temp1 * njacZ[3][0][k-1][i][j];
        lhsZ[3][1][AA][k][i][j] = - temp2 * fjacZ[3][1][k-1][i][j]
          - temp1 * njacZ[3][1][k-1][i][j];
        lhsZ[3][2][AA][k][i][j] = - temp2 * fjacZ[3][2][k-1][i][j]
          - temp1 * njacZ[3][2][k-1][i][j];
        lhsZ[3][3][AA][k][i][j] = - temp2 * fjacZ[3][3][k-1][i][j]
          - temp1 * njacZ[3][3][k-1][i][j]
          - temp1 * dz4;
        lhsZ[3][4][AA][k][i][j] = - temp2 * fjacZ[3][4][k-1][i][j]
          - temp1 * njacZ[3][4][k-1][i][j];

        lhsZ[4][0][AA][k][i][j] = - temp2 * fjacZ[4][0][k-1][i][j]
          - temp1 * njacZ[4][0][k-1][i][j];
        lhsZ[4][1][AA][k][i][j] = - temp2 * fjacZ[4][1][k-1][i][j]
          - temp1 * njacZ[4][1][k-1][i][j];
        lhsZ[4][2][AA][k][i][j] = - temp2 * fjacZ[4][2][k-1][i][j]
          - temp1 * njacZ[4][2][k-1][i][j];
        lhsZ[4][3][AA][k][i][j] = - temp2 * fjacZ[4][3][k-1][i][j]
          - temp1 * njacZ[4][3][k-1][i][j];
        lhsZ[4][4][AA][k][i][j] = - temp2 * fjacZ[4][4][k-1][i][j]
          - temp1 * njacZ[4][4][k-1][i][j]
          - temp1 * dz5;

        lhsZ[0][0][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[0][0][k][i][j]
          + temp1 * 2.0 * dz1;
        lhsZ[0][1][BB][k][i][j] = temp1 * 2.0 * njacZ[0][1][k][i][j];
        lhsZ[0][2][BB][k][i][j] = temp1 * 2.0 * njacZ[0][2][k][i][j];
        lhsZ[0][3][BB][k][i][j] = temp1 * 2.0 * njacZ[0][3][k][i][j];
        lhsZ[0][4][BB][k][i][j] = temp1 * 2.0 * njacZ[0][4][k][i][j];

        lhsZ[1][0][BB][k][i][j] = temp1 * 2.0 * njacZ[1][0][k][i][j];
        lhsZ[1][1][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[1][1][k][i][j]
          + temp1 * 2.0 * dz2;
        lhsZ[1][2][BB][k][i][j] = temp1 * 2.0 * njacZ[1][2][k][i][j];
        lhsZ[1][3][BB][k][i][j] = temp1 * 2.0 * njacZ[1][3][k][i][j];
        lhsZ[1][4][BB][k][i][j] = temp1 * 2.0 * njacZ[1][4][k][i][j];

        lhsZ[2][0][BB][k][i][j] = temp1 * 2.0 * njacZ[2][0][k][i][j];
        lhsZ[2][1][BB][k][i][j] = temp1 * 2.0 * njacZ[2][1][k][i][j];
        lhsZ[2][2][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[2][2][k][i][j]
          + temp1 * 2.0 * dz3;
        lhsZ[2][3][BB][k][i][j] = temp1 * 2.0 * njacZ[2][3][k][i][j];
        lhsZ[2][4][BB][k][i][j] = temp1 * 2.0 * njacZ[2][4][k][i][j];

        lhsZ[3][0][BB][k][i][j] = temp1 * 2.0 * njacZ[3][0][k][i][j];
        lhsZ[3][1][BB][k][i][j] = temp1 * 2.0 * njacZ[3][1][k][i][j];
        lhsZ[3][2][BB][k][i][j] = temp1 * 2.0 * njacZ[3][2][k][i][j];
        lhsZ[3][3][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[3][3][k][i][j]
          + temp1 * 2.0 * dz4;
        lhsZ[3][4][BB][k][i][j] = temp1 * 2.0 * njacZ[3][4][k][i][j];

        lhsZ[4][0][BB][k][i][j] = temp1 * 2.0 * njacZ[4][0][k][i][j];
        lhsZ[4][1][BB][k][i][j] = temp1 * 2.0 * njacZ[4][1][k][i][j];
        lhsZ[4][2][BB][k][i][j] = temp1 * 2.0 * njacZ[4][2][k][i][j];
        lhsZ[4][3][BB][k][i][j] = temp1 * 2.0 * njacZ[4][3][k][i][j];
        lhsZ[4][4][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[4][4][k][i][j] 
          + temp1 * 2.0 * dz5;

        lhsZ[0][0][CC][k][i][j] =  temp2 * fjacZ[0][0][k+1][i][j]
          - temp1 * njacZ[0][0][k+1][i][j]
          - temp1 * dz1;
        lhsZ[0][1][CC][k][i][j] =  temp2 * fjacZ[0][1][k+1][i][j]
          - temp1 * njacZ[0][1][k+1][i][j];
        lhsZ[0][2][CC][k][i][j] =  temp2 * fjacZ[0][2][k+1][i][j]
          - temp1 * njacZ[0][2][k+1][i][j];
        lhsZ[0][3][CC][k][i][j] =  temp2 * fjacZ[0][3][k+1][i][j]
          - temp1 * njacZ[0][3][k+1][i][j];
        lhsZ[0][4][CC][k][i][j] =  temp2 * fjacZ[0][4][k+1][i][j]
          - temp1 * njacZ[0][4][k+1][i][j];

        lhsZ[1][0][CC][k][i][j] =  temp2 * fjacZ[1][0][k+1][i][j]
          - temp1 * njacZ[1][0][k+1][i][j];
        lhsZ[1][1][CC][k][i][j] =  temp2 * fjacZ[1][1][k+1][i][j]
          - temp1 * njacZ[1][1][k+1][i][j]
          - temp1 * dz2;
        lhsZ[1][2][CC][k][i][j] =  temp2 * fjacZ[1][2][k+1][i][j]
          - temp1 * njacZ[1][2][k+1][i][j];
        lhsZ[1][3][CC][k][i][j] =  temp2 * fjacZ[1][3][k+1][i][j]
          - temp1 * njacZ[1][3][k+1][i][j];
        lhsZ[1][4][CC][k][i][j] =  temp2 * fjacZ[1][4][k+1][i][j]
          - temp1 * njacZ[1][4][k+1][i][j];

        lhsZ[2][0][CC][k][i][j] =  temp2 * fjacZ[2][0][k+1][i][j]
          - temp1 * njacZ[2][0][k+1][i][j];
        lhsZ[2][1][CC][k][i][j] =  temp2 * fjacZ[2][1][k+1][i][j]
          - temp1 * njacZ[2][1][k+1][i][j];
        lhsZ[2][2][CC][k][i][j] =  temp2 * fjacZ[2][2][k+1][i][j]
          - temp1 * njacZ[2][2][k+1][i][j]
          - temp1 * dz3;
        lhsZ[2][3][CC][k][i][j] =  temp2 * fjacZ[2][3][k+1][i][j]
          - temp1 * njacZ[2][3][k+1][i][j];
        lhsZ[2][4][CC][k][i][j] =  temp2 * fjacZ[2][4][k+1][i][j]
          - temp1 * njacZ[2][4][k+1][i][j];

        lhsZ[3][0][CC][k][i][j] =  temp2 * fjacZ[3][0][k+1][i][j]
          - temp1 * njacZ[3][0][k+1][i][j];
        lhsZ[3][1][CC][k][i][j] =  temp2 * fjacZ[3][1][k+1][i][j]
          - temp1 * njacZ[3][1][k+1][i][j];
        lhsZ[3][2][CC][k][i][j] =  temp2 * fjacZ[3][2][k+1][i][j]
          - temp1 * njacZ[3][2][k+1][i][j];
        lhsZ[3][3][CC][k][i][j] =  temp2 * fjacZ[3][3][k+1][i][j]
          - temp1 * njacZ[3][3][k+1][i][j]
          - temp1 * dz4;
        lhsZ[3][4][CC][k][i][j] =  temp2 * fjacZ[3][4][k+1][i][j]
          - temp1 * njacZ[3][4][k+1][i][j];

        lhsZ[4][0][CC][k][i][j] =  temp2 * fjacZ[4][0][k+1][i][j]
          - temp1 * njacZ[4][0][k+1][i][j];
        lhsZ[4][1][CC][k][i][j] =  temp2 * fjacZ[4][1][k+1][i][j]
          - temp1 * njacZ[4][1][k+1][i][j];
        lhsZ[4][2][CC][k][i][j] =  temp2 * fjacZ[4][2][k+1][i][j]
          - temp1 * njacZ[4][2][k+1][i][j];
        lhsZ[4][3][CC][k][i][j] =  temp2 * fjacZ[4][3][k+1][i][j]
          - temp1 * njacZ[4][3][k+1][i][j];
        lhsZ[4][4][CC][k][i][j] =  temp2 * fjacZ[4][4][k+1][i][j]
          - temp1 * njacZ[4][4][k+1][i][j]
          - temp1 * dz5;
      }
	}
  }
      
    for (i = 1; i <= gp02; i++) {
  for (j = 1; j <= gp12; j++) {

  pivot = 1.00/lhsZ[0][0][BB][0][i][j];
  lhsZ[0][1][BB][0][i][j] = lhsZ[0][1][BB][0][i][j]*pivot;
  lhsZ[0][2][BB][0][i][j] = lhsZ[0][2][BB][0][i][j]*pivot;
  lhsZ[0][3][BB][0][i][j] = lhsZ[0][3][BB][0][i][j]*pivot;
  lhsZ[0][4][BB][0][i][j] = lhsZ[0][4][BB][0][i][j]*pivot;
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j]*pivot;
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j]*pivot;
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j]*pivot;
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j]*pivot;
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j]*pivot;
  rhs[0][0][j][i]   = rhs[0][0][j][i]  *pivot;

  coeff = lhsZ[1][0][BB][0][i][j];
  lhsZ[1][1][BB][0][i][j]= lhsZ[1][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
  lhsZ[1][2][BB][0][i][j]= lhsZ[1][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
  lhsZ[1][3][BB][0][i][j]= lhsZ[1][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
  lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
  rhs[1][0][j][i]   = rhs[1][0][j][i]   - coeff*rhs[0][0][j][i];

  coeff = lhsZ[2][0][BB][0][i][j];
  lhsZ[2][1][BB][0][i][j]= lhsZ[2][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
  lhsZ[2][2][BB][0][i][j]= lhsZ[2][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
  lhsZ[2][3][BB][0][i][j]= lhsZ[2][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
  lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
  rhs[2][0][j][i]   = rhs[2][0][j][i]   - coeff*rhs[0][0][j][i];

  coeff = lhsZ[3][0][BB][0][i][j];
  lhsZ[3][1][BB][0][i][j]= lhsZ[3][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
  lhsZ[3][2][BB][0][i][j]= lhsZ[3][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
  lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
  lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
  rhs[3][0][j][i]   = rhs[3][0][j][i]   - coeff*rhs[0][0][j][i];

  coeff = lhsZ[4][0][BB][0][i][j];
  lhsZ[4][1][BB][0][i][j]= lhsZ[4][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
  lhsZ[4][2][BB][0][i][j]= lhsZ[4][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
  lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
  lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
  rhs[4][0][j][i]   = rhs[4][0][j][i]   - coeff*rhs[0][0][j][i];

  pivot = 1.00/lhsZ[1][1][BB][0][i][j];
  lhsZ[1][2][BB][0][i][j] = lhsZ[1][2][BB][0][i][j]*pivot;
  lhsZ[1][3][BB][0][i][j] = lhsZ[1][3][BB][0][i][j]*pivot;
  lhsZ[1][4][BB][0][i][j] = lhsZ[1][4][BB][0][i][j]*pivot;
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j]*pivot;
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j]*pivot;
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j]*pivot;
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j]*pivot;
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j]*pivot;
  rhs[1][0][j][i]   = rhs[1][0][j][i]  *pivot;

  coeff = lhsZ[0][1][BB][0][i][j];
  lhsZ[0][2][BB][0][i][j]= lhsZ[0][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
  lhsZ[0][3][BB][0][i][j]= lhsZ[0][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
  lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
  rhs[0][0][j][i]   = rhs[0][0][j][i]   - coeff*rhs[1][0][j][i];

  coeff = lhsZ[2][1][BB][0][i][j];
  lhsZ[2][2][BB][0][i][j]= lhsZ[2][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
  lhsZ[2][3][BB][0][i][j]= lhsZ[2][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
  lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
  rhs[2][0][j][i]   = rhs[2][0][j][i]   - coeff*rhs[1][0][j][i];

  coeff = lhsZ[3][1][BB][0][i][j];
  lhsZ[3][2][BB][0][i][j]= lhsZ[3][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
  lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
  lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
  rhs[3][0][j][i]   = rhs[3][0][j][i]   - coeff*rhs[1][0][j][i];

  coeff = lhsZ[4][1][BB][0][i][j];
  lhsZ[4][2][BB][0][i][j]= lhsZ[4][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
  lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
  lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
  rhs[4][0][j][i]   = rhs[4][0][j][i]   - coeff*rhs[1][0][j][i];

  pivot = 1.00/lhsZ[2][2][BB][0][i][j];
  lhsZ[2][3][BB][0][i][j] = lhsZ[2][3][BB][0][i][j]*pivot;
  lhsZ[2][4][BB][0][i][j] = lhsZ[2][4][BB][0][i][j]*pivot;
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j]*pivot;
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j]*pivot;
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j]*pivot;
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j]*pivot;
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j]*pivot;
  rhs[2][0][j][i]   = rhs[2][0][j][i]  *pivot;

  coeff = lhsZ[0][2][BB][0][i][j];
  lhsZ[0][3][BB][0][i][j]= lhsZ[0][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
  lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
  rhs[0][0][j][i]   = rhs[0][0][j][i]   - coeff*rhs[2][0][j][i];

  coeff = lhsZ[1][2][BB][0][i][j];
  lhsZ[1][3][BB][0][i][j]= lhsZ[1][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
  lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
  rhs[1][0][j][i]   = rhs[1][0][j][i]   - coeff*rhs[2][0][j][i];

  coeff = lhsZ[3][2][BB][0][i][j];
  lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
  lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
  rhs[3][0][j][i]   = rhs[3][0][j][i]   - coeff*rhs[2][0][j][i];

  coeff = lhsZ[4][2][BB][0][i][j];
  lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
  lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
  rhs[4][0][j][i]   = rhs[4][0][j][i]   - coeff*rhs[2][0][j][i];

  pivot = 1.00/lhsZ[3][3][BB][0][i][j];
  lhsZ[3][4][BB][0][i][j] = lhsZ[3][4][BB][0][i][j]*pivot;
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j]*pivot;
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j]*pivot;
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j]*pivot;
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j]*pivot;
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j]*pivot;
  rhs[3][0][j][i]   = rhs[3][0][j][i]  *pivot;

  coeff = lhsZ[0][3][BB][0][i][j];
  lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
  rhs[0][0][j][i]   = rhs[0][0][j][i]   - coeff*rhs[3][0][j][i];

  coeff = lhsZ[1][3][BB][0][i][j];
  lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
  rhs[1][0][j][i]   = rhs[1][0][j][i]   - coeff*rhs[3][0][j][i];

  coeff = lhsZ[2][3][BB][0][i][j];
  lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
  rhs[2][0][j][i]   = rhs[2][0][j][i]   - coeff*rhs[3][0][j][i];

  coeff = lhsZ[4][3][BB][0][i][j];
  lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
  rhs[4][0][j][i]   = rhs[4][0][j][i]   - coeff*rhs[3][0][j][i];

  pivot = 1.00/lhsZ[4][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j]*pivot;
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j]*pivot;
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j]*pivot;
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j]*pivot;
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j]*pivot;
  rhs[4][0][j][i]   = rhs[4][0][j][i]  *pivot;

  coeff = lhsZ[0][4][BB][0][i][j];
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
  rhs[0][0][j][i]   = rhs[0][0][j][i]   - coeff*rhs[4][0][j][i];

  coeff = lhsZ[1][4][BB][0][i][j];
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
  rhs[1][0][j][i]   = rhs[1][0][j][i]   - coeff*rhs[4][0][j][i];

  coeff = lhsZ[2][4][BB][0][i][j];
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
  rhs[2][0][j][i]   = rhs[2][0][j][i]   - coeff*rhs[4][0][j][i];

  coeff = lhsZ[3][4][BB][0][i][j];
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
  rhs[3][0][j][i]   = rhs[3][0][j][i]   - coeff*rhs[4][0][j][i];

	}
  }
      
    for (i = 1; i <= gp02; i++) {
  for (j = 1; j <= gp12; j++) {
      for (k = 1; k <= ksize-1; k++) {
        
  rhs[0][k][j][i] = rhs[0][k][j][i] - lhsZ[0][0][AA][k][i][j]*rhs[0][k-1][j][i]
                    - lhsZ[0][1][AA][k][i][j]*rhs[1][k-1][j][i]
                    - lhsZ[0][2][AA][k][i][j]*rhs[2][k-1][j][i]
                    - lhsZ[0][3][AA][k][i][j]*rhs[3][k-1][j][i]
                    - lhsZ[0][4][AA][k][i][j]*rhs[4][k-1][j][i];
  rhs[1][k][j][i] = rhs[1][k][j][i] - lhsZ[1][0][AA][k][i][j]*rhs[0][k-1][j][i]
                    - lhsZ[1][1][AA][k][i][j]*rhs[1][k-1][j][i]
                    - lhsZ[1][2][AA][k][i][j]*rhs[2][k-1][j][i]
                    - lhsZ[1][3][AA][k][i][j]*rhs[3][k-1][j][i]
                    - lhsZ[1][4][AA][k][i][j]*rhs[4][k-1][j][i];
  rhs[2][k][j][i] = rhs[2][k][j][i] - lhsZ[2][0][AA][k][i][j]*rhs[0][k-1][j][i]
                    - lhsZ[2][1][AA][k][i][j]*rhs[1][k-1][j][i]
                    - lhsZ[2][2][AA][k][i][j]*rhs[2][k-1][j][i]
                    - lhsZ[2][3][AA][k][i][j]*rhs[3][k-1][j][i]
                    - lhsZ[2][4][AA][k][i][j]*rhs[4][k-1][j][i];
  rhs[3][k][j][i] = rhs[3][k][j][i] - lhsZ[3][0][AA][k][i][j]*rhs[0][k-1][j][i]
                    - lhsZ[3][1][AA][k][i][j]*rhs[1][k-1][j][i]
                    - lhsZ[3][2][AA][k][i][j]*rhs[2][k-1][j][i]
                    - lhsZ[3][3][AA][k][i][j]*rhs[3][k-1][j][i]
                    - lhsZ[3][4][AA][k][i][j]*rhs[4][k-1][j][i];
  rhs[4][k][j][i] = rhs[4][k][j][i] - lhsZ[4][0][AA][k][i][j]*rhs[0][k-1][j][i]
                    - lhsZ[4][1][AA][k][i][j]*rhs[1][k-1][j][i]
                    - lhsZ[4][2][AA][k][i][j]*rhs[2][k-1][j][i]
                    - lhsZ[4][3][AA][k][i][j]*rhs[3][k-1][j][i]
                    - lhsZ[4][4][AA][k][i][j]*rhs[4][k-1][j][i];

  lhsZ[0][0][BB][k][i][j] = lhsZ[0][0][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[1][0][BB][k][i][j] = lhsZ[1][0][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[2][0][BB][k][i][j] = lhsZ[2][0][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[3][0][BB][k][i][j] = lhsZ[3][0][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[4][0][BB][k][i][j] = lhsZ[4][0][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[0][1][BB][k][i][j] = lhsZ[0][1][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[1][1][BB][k][i][j] = lhsZ[1][1][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[2][1][BB][k][i][j] = lhsZ[2][1][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[3][1][BB][k][i][j] = lhsZ[3][1][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[4][1][BB][k][i][j] = lhsZ[4][1][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[0][2][BB][k][i][j] = lhsZ[0][2][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[1][2][BB][k][i][j] = lhsZ[1][2][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[2][2][BB][k][i][j] = lhsZ[2][2][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[3][2][BB][k][i][j] = lhsZ[3][2][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[4][2][BB][k][i][j] = lhsZ[4][2][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[0][3][BB][k][i][j] = lhsZ[0][3][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[1][3][BB][k][i][j] = lhsZ[1][3][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[2][3][BB][k][i][j] = lhsZ[2][3][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[3][3][BB][k][i][j] = lhsZ[3][3][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[4][3][BB][k][i][j] = lhsZ[4][3][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[0][4][BB][k][i][j] = lhsZ[0][4][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
  lhsZ[1][4][BB][k][i][j] = lhsZ[1][4][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
  lhsZ[2][4][BB][k][i][j] = lhsZ[2][4][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
  lhsZ[3][4][BB][k][i][j] = lhsZ[3][4][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
  lhsZ[4][4][BB][k][i][j] = lhsZ[4][4][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];

  pivot = 1.00/lhsZ[0][0][BB][k][i][j];
  lhsZ[0][1][BB][k][i][j] = lhsZ[0][1][BB][k][i][j]*pivot;
  lhsZ[0][2][BB][k][i][j] = lhsZ[0][2][BB][k][i][j]*pivot;
  lhsZ[0][3][BB][k][i][j] = lhsZ[0][3][BB][k][i][j]*pivot;
  lhsZ[0][4][BB][k][i][j] = lhsZ[0][4][BB][k][i][j]*pivot;
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j]*pivot;
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j]*pivot;
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j]*pivot;
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j]*pivot;
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j]*pivot;
  rhs[0][k][j][i]   = rhs[0][k][j][i]  *pivot;

  coeff = lhsZ[1][0][BB][k][i][j];
  lhsZ[1][1][BB][k][i][j]= lhsZ[1][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
  lhsZ[1][2][BB][k][i][j]= lhsZ[1][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
  lhsZ[1][3][BB][k][i][j]= lhsZ[1][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
  lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsZ[2][0][BB][k][i][j];
  lhsZ[2][1][BB][k][i][j]= lhsZ[2][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
  lhsZ[2][2][BB][k][i][j]= lhsZ[2][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
  lhsZ[2][3][BB][k][i][j]= lhsZ[2][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
  lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsZ[3][0][BB][k][i][j];
  lhsZ[3][1][BB][k][i][j]= lhsZ[3][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
  lhsZ[3][2][BB][k][i][j]= lhsZ[3][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
  lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
  lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[0][k][j][i];

  coeff = lhsZ[4][0][BB][k][i][j];
  lhsZ[4][1][BB][k][i][j]= lhsZ[4][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
  lhsZ[4][2][BB][k][i][j]= lhsZ[4][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
  lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
  lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[0][k][j][i];

  pivot = 1.00/lhsZ[1][1][BB][k][i][j];
  lhsZ[1][2][BB][k][i][j] = lhsZ[1][2][BB][k][i][j]*pivot;
  lhsZ[1][3][BB][k][i][j] = lhsZ[1][3][BB][k][i][j]*pivot;
  lhsZ[1][4][BB][k][i][j] = lhsZ[1][4][BB][k][i][j]*pivot;
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j]*pivot;
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j]*pivot;
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j]*pivot;
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j]*pivot;
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j]*pivot;
  rhs[1][k][j][i]   = rhs[1][k][j][i]  *pivot;

  coeff = lhsZ[0][1][BB][k][i][j];
  lhsZ[0][2][BB][k][i][j]= lhsZ[0][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
  lhsZ[0][3][BB][k][i][j]= lhsZ[0][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
  lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsZ[2][1][BB][k][i][j];
  lhsZ[2][2][BB][k][i][j]= lhsZ[2][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
  lhsZ[2][3][BB][k][i][j]= lhsZ[2][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
  lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsZ[3][1][BB][k][i][j];
  lhsZ[3][2][BB][k][i][j]= lhsZ[3][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
  lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
  lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[1][k][j][i];

  coeff = lhsZ[4][1][BB][k][i][j];
  lhsZ[4][2][BB][k][i][j]= lhsZ[4][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
  lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
  lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[1][k][j][i];

  pivot = 1.00/lhsZ[2][2][BB][k][i][j];
  lhsZ[2][3][BB][k][i][j] = lhsZ[2][3][BB][k][i][j]*pivot;
  lhsZ[2][4][BB][k][i][j] = lhsZ[2][4][BB][k][i][j]*pivot;
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j]*pivot;
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j]*pivot;
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j]*pivot;
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j]*pivot;
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j]*pivot;
  rhs[2][k][j][i]   = rhs[2][k][j][i]  *pivot;

  coeff = lhsZ[0][2][BB][k][i][j];
  lhsZ[0][3][BB][k][i][j]= lhsZ[0][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
  lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsZ[1][2][BB][k][i][j];
  lhsZ[1][3][BB][k][i][j]= lhsZ[1][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
  lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsZ[3][2][BB][k][i][j];
  lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
  lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[2][k][j][i];

  coeff = lhsZ[4][2][BB][k][i][j];
  lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
  lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[2][k][j][i];

  pivot = 1.00/lhsZ[3][3][BB][k][i][j];
  lhsZ[3][4][BB][k][i][j] = lhsZ[3][4][BB][k][i][j]*pivot;
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j]*pivot;
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j]*pivot;
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j]*pivot;
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j]*pivot;
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j]*pivot;
  rhs[3][k][j][i]   = rhs[3][k][j][i]  *pivot;

  coeff = lhsZ[0][3][BB][k][i][j];
  lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsZ[1][3][BB][k][i][j];
  lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsZ[2][3][BB][k][i][j];
  lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[3][k][j][i];

  coeff = lhsZ[4][3][BB][k][i][j];
  lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
  rhs[4][k][j][i]   = rhs[4][k][j][i]   - coeff*rhs[3][k][j][i];

  pivot = 1.00/lhsZ[4][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j]*pivot;
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j]*pivot;
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j]*pivot;
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j]*pivot;
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j]*pivot;
  rhs[4][k][j][i]   = rhs[4][k][j][i]  *pivot;

  coeff = lhsZ[0][4][BB][k][i][j];
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
  rhs[0][k][j][i]   = rhs[0][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsZ[1][4][BB][k][i][j];
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
  rhs[1][k][j][i]   = rhs[1][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsZ[2][4][BB][k][i][j];
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
  rhs[2][k][j][i]   = rhs[2][k][j][i]   - coeff*rhs[4][k][j][i];

  coeff = lhsZ[3][4][BB][k][i][j];
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
  rhs[3][k][j][i]   = rhs[3][k][j][i]   - coeff*rhs[4][k][j][i];

      }
	}
  }
      
    for (i = 1; i <= gp02; i++) {
  for (j = 1; j <= gp12; j++) {
	
  rhs[0][ksize][j][i] = rhs[0][ksize][j][i] - lhsZ[0][0][AA][ksize][i][j]*rhs[0][ksize-1][j][i]
                    - lhsZ[0][1][AA][ksize][i][j]*rhs[1][ksize-1][j][i]
                    - lhsZ[0][2][AA][ksize][i][j]*rhs[2][ksize-1][j][i]
                    - lhsZ[0][3][AA][ksize][i][j]*rhs[3][ksize-1][j][i]
                    - lhsZ[0][4][AA][ksize][i][j]*rhs[4][ksize-1][j][i];
  rhs[1][ksize][j][i] = rhs[1][ksize][j][i] - lhsZ[1][0][AA][ksize][i][j]*rhs[0][ksize-1][j][i]
                    - lhsZ[1][1][AA][ksize][i][j]*rhs[1][ksize-1][j][i]
                    - lhsZ[1][2][AA][ksize][i][j]*rhs[2][ksize-1][j][i]
                    - lhsZ[1][3][AA][ksize][i][j]*rhs[3][ksize-1][j][i]
                    - lhsZ[1][4][AA][ksize][i][j]*rhs[4][ksize-1][j][i];
  rhs[2][ksize][j][i] = rhs[2][ksize][j][i] - lhsZ[2][0][AA][ksize][i][j]*rhs[0][ksize-1][j][i]
                    - lhsZ[2][1][AA][ksize][i][j]*rhs[1][ksize-1][j][i]
                    - lhsZ[2][2][AA][ksize][i][j]*rhs[2][ksize-1][j][i]
                    - lhsZ[2][3][AA][ksize][i][j]*rhs[3][ksize-1][j][i]
                    - lhsZ[2][4][AA][ksize][i][j]*rhs[4][ksize-1][j][i];
  rhs[3][ksize][j][i] = rhs[3][ksize][j][i] - lhsZ[3][0][AA][ksize][i][j]*rhs[0][ksize-1][j][i]
                    - lhsZ[3][1][AA][ksize][i][j]*rhs[1][ksize-1][j][i]
                    - lhsZ[3][2][AA][ksize][i][j]*rhs[2][ksize-1][j][i]
                    - lhsZ[3][3][AA][ksize][i][j]*rhs[3][ksize-1][j][i]
                    - lhsZ[3][4][AA][ksize][i][j]*rhs[4][ksize-1][j][i];
  rhs[4][ksize][j][i] = rhs[4][ksize][j][i] - lhsZ[4][0][AA][ksize][i][j]*rhs[0][ksize-1][j][i]
                    - lhsZ[4][1][AA][ksize][i][j]*rhs[1][ksize-1][j][i]
                    - lhsZ[4][2][AA][ksize][i][j]*rhs[2][ksize-1][j][i]
                    - lhsZ[4][3][AA][ksize][i][j]*rhs[3][ksize-1][j][i]
                    - lhsZ[4][4][AA][ksize][i][j]*rhs[4][ksize-1][j][i];
	}
  }
      
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
	
  lhsZ[0][0][BB][ksize][i][j] = lhsZ[0][0][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[1][0][BB][ksize][i][j] = lhsZ[1][0][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[2][0][BB][ksize][i][j] = lhsZ[2][0][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[3][0][BB][ksize][i][j] = lhsZ[3][0][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[4][0][BB][ksize][i][j] = lhsZ[4][0][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[0][1][BB][ksize][i][j] = lhsZ[0][1][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[1][1][BB][ksize][i][j] = lhsZ[1][1][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[2][1][BB][ksize][i][j] = lhsZ[2][1][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[3][1][BB][ksize][i][j] = lhsZ[3][1][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[4][1][BB][ksize][i][j] = lhsZ[4][1][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[0][2][BB][ksize][i][j] = lhsZ[0][2][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[1][2][BB][ksize][i][j] = lhsZ[1][2][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[2][2][BB][ksize][i][j] = lhsZ[2][2][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[3][2][BB][ksize][i][j] = lhsZ[3][2][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[4][2][BB][ksize][i][j] = lhsZ[4][2][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[0][3][BB][ksize][i][j] = lhsZ[0][3][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[1][3][BB][ksize][i][j] = lhsZ[1][3][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[2][3][BB][ksize][i][j] = lhsZ[2][3][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[3][3][BB][ksize][i][j] = lhsZ[3][3][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[4][3][BB][ksize][i][j] = lhsZ[4][3][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[0][4][BB][ksize][i][j] = lhsZ[0][4][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
  lhsZ[1][4][BB][ksize][i][j] = lhsZ[1][4][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
  lhsZ[2][4][BB][ksize][i][j] = lhsZ[2][4][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
  lhsZ[3][4][BB][ksize][i][j] = lhsZ[3][4][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
  lhsZ[4][4][BB][ksize][i][j] = lhsZ[4][4][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];

	}
  }
      
    for (i = 1; i <= gp02; i++) {
  for (j = 1; j <= gp12; j++) {
	
  pivot = 1.00/lhsZ[0][0][BB][ksize][i][j];
  lhsZ[0][1][BB][ksize][i][j] = lhsZ[0][1][BB][ksize][i][j]*pivot;
  lhsZ[0][2][BB][ksize][i][j] = lhsZ[0][2][BB][ksize][i][j]*pivot;
  lhsZ[0][3][BB][ksize][i][j] = lhsZ[0][3][BB][ksize][i][j]*pivot;
  lhsZ[0][4][BB][ksize][i][j] = lhsZ[0][4][BB][ksize][i][j]*pivot;
  rhs[0][ksize][j][i]   = rhs[0][ksize][j][i]  *pivot;

  coeff = lhsZ[1][0][BB][ksize][i][j];
  lhsZ[1][1][BB][ksize][i][j]= lhsZ[1][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
  lhsZ[1][2][BB][ksize][i][j]= lhsZ[1][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
  lhsZ[1][3][BB][ksize][i][j]= lhsZ[1][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
  lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
  rhs[1][ksize][j][i]   = rhs[1][ksize][j][i]   - coeff*rhs[0][ksize][j][i];

  coeff = lhsZ[2][0][BB][ksize][i][j];
  lhsZ[2][1][BB][ksize][i][j]= lhsZ[2][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
  lhsZ[2][2][BB][ksize][i][j]= lhsZ[2][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
  lhsZ[2][3][BB][ksize][i][j]= lhsZ[2][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
  lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
  rhs[2][ksize][j][i]   = rhs[2][ksize][j][i]   - coeff*rhs[0][ksize][j][i];

  coeff = lhsZ[3][0][BB][ksize][i][j];
  lhsZ[3][1][BB][ksize][i][j]= lhsZ[3][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
  lhsZ[3][2][BB][ksize][i][j]= lhsZ[3][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
  lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
  lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
  rhs[3][ksize][j][i]   = rhs[3][ksize][j][i]   - coeff*rhs[0][ksize][j][i];

  coeff = lhsZ[4][0][BB][ksize][i][j];
  lhsZ[4][1][BB][ksize][i][j]= lhsZ[4][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
  lhsZ[4][2][BB][ksize][i][j]= lhsZ[4][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
  lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
  lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
  rhs[4][ksize][j][i]   = rhs[4][ksize][j][i]   - coeff*rhs[0][ksize][j][i];

  pivot = 1.00/lhsZ[1][1][BB][ksize][i][j];
  lhsZ[1][2][BB][ksize][i][j] = lhsZ[1][2][BB][ksize][i][j]*pivot;
  lhsZ[1][3][BB][ksize][i][j] = lhsZ[1][3][BB][ksize][i][j]*pivot;
  lhsZ[1][4][BB][ksize][i][j] = lhsZ[1][4][BB][ksize][i][j]*pivot;
  rhs[1][ksize][j][i]   = rhs[1][ksize][j][i]  *pivot;

  coeff = lhsZ[0][1][BB][ksize][i][j];
  lhsZ[0][2][BB][ksize][i][j]= lhsZ[0][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
  lhsZ[0][3][BB][ksize][i][j]= lhsZ[0][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
  lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
  rhs[0][ksize][j][i]   = rhs[0][ksize][j][i]   - coeff*rhs[1][ksize][j][i];

  coeff = lhsZ[2][1][BB][ksize][i][j];
  lhsZ[2][2][BB][ksize][i][j]= lhsZ[2][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
  lhsZ[2][3][BB][ksize][i][j]= lhsZ[2][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
  lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
  rhs[2][ksize][j][i]   = rhs[2][ksize][j][i]   - coeff*rhs[1][ksize][j][i];

  coeff = lhsZ[3][1][BB][ksize][i][j];
  lhsZ[3][2][BB][ksize][i][j]= lhsZ[3][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
  lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
  lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
  rhs[3][ksize][j][i]   = rhs[3][ksize][j][i]   - coeff*rhs[1][ksize][j][i];

  coeff = lhsZ[4][1][BB][ksize][i][j];
  lhsZ[4][2][BB][ksize][i][j]= lhsZ[4][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
  lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
  lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
  rhs[4][ksize][j][i]   = rhs[4][ksize][j][i]   - coeff*rhs[1][ksize][j][i];

  pivot = 1.00/lhsZ[2][2][BB][ksize][i][j];
  lhsZ[2][3][BB][ksize][i][j] = lhsZ[2][3][BB][ksize][i][j]*pivot;
  lhsZ[2][4][BB][ksize][i][j] = lhsZ[2][4][BB][ksize][i][j]*pivot;
  rhs[2][ksize][j][i]   = rhs[2][ksize][j][i]  *pivot;

  coeff = lhsZ[0][2][BB][ksize][i][j];
  lhsZ[0][3][BB][ksize][i][j]= lhsZ[0][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
  lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
  rhs[0][ksize][j][i]   = rhs[0][ksize][j][i]   - coeff*rhs[2][ksize][j][i];

  coeff = lhsZ[1][2][BB][ksize][i][j];
  lhsZ[1][3][BB][ksize][i][j]= lhsZ[1][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
  lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
  rhs[1][ksize][j][i]   = rhs[1][ksize][j][i]   - coeff*rhs[2][ksize][j][i];

  coeff = lhsZ[3][2][BB][ksize][i][j];
  lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
  lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
  rhs[3][ksize][j][i]   = rhs[3][ksize][j][i]   - coeff*rhs[2][ksize][j][i];

  coeff = lhsZ[4][2][BB][ksize][i][j];
  lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
  lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
  rhs[4][ksize][j][i]   = rhs[4][ksize][j][i]   - coeff*rhs[2][ksize][j][i];

  pivot = 1.00/lhsZ[3][3][BB][ksize][i][j];
  lhsZ[3][4][BB][ksize][i][j] = lhsZ[3][4][BB][ksize][i][j]*pivot;
  rhs[3][ksize][j][i]   = rhs[3][ksize][j][i]  *pivot;

  coeff = lhsZ[0][3][BB][ksize][i][j];
  lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
  rhs[0][ksize][j][i]   = rhs[0][ksize][j][i]   - coeff*rhs[3][ksize][j][i];

  coeff = lhsZ[1][3][BB][ksize][i][j];
  lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
  rhs[1][ksize][j][i]   = rhs[1][ksize][j][i]   - coeff*rhs[3][ksize][j][i];

  coeff = lhsZ[2][3][BB][ksize][i][j];
  lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
  rhs[2][ksize][j][i]   = rhs[2][ksize][j][i]   - coeff*rhs[3][ksize][j][i];

  coeff = lhsZ[4][3][BB][ksize][i][j];
  lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
  rhs[4][ksize][j][i]   = rhs[4][ksize][j][i]   - coeff*rhs[3][ksize][j][i];

  pivot = 1.00/lhsZ[4][4][BB][ksize][i][j];
  rhs[4][ksize][j][i]   = rhs[4][ksize][j][i]  *pivot;

  coeff = lhsZ[0][4][BB][ksize][i][j];
  rhs[0][ksize][j][i]   = rhs[0][ksize][j][i]   - coeff*rhs[4][ksize][j][i];

  coeff = lhsZ[1][4][BB][ksize][i][j];
  rhs[1][ksize][j][i]   = rhs[1][ksize][j][i]   - coeff*rhs[4][ksize][j][i];

  coeff = lhsZ[2][4][BB][ksize][i][j];
  rhs[2][ksize][j][i]   = rhs[2][ksize][j][i]   - coeff*rhs[4][ksize][j][i];

  coeff = lhsZ[3][4][BB][ksize][i][j];
  rhs[3][ksize][j][i]   = rhs[3][ksize][j][i]   - coeff*rhs[4][ksize][j][i];

	}
  }
      
      for (k = ksize-1; k >= 0; k--) {
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
        
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsZ[0][0][CC][k][i][j]*rhs[0][k+1][j][i];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsZ[0][1][CC][k][i][j]*rhs[1][k+1][j][i];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsZ[0][2][CC][k][i][j]*rhs[2][k+1][j][i];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsZ[0][3][CC][k][i][j]*rhs[3][k+1][j][i];
            rhs[0][k][j][i] = rhs[0][k][j][i] 
              - lhsZ[0][4][CC][k][i][j]*rhs[4][k+1][j][i];

            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsZ[1][0][CC][k][i][j]*rhs[0][k+1][j][i];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsZ[1][1][CC][k][i][j]*rhs[1][k+1][j][i];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsZ[1][2][CC][k][i][j]*rhs[2][k+1][j][i];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsZ[1][3][CC][k][i][j]*rhs[3][k+1][j][i];
            rhs[1][k][j][i] = rhs[1][k][j][i] 
              - lhsZ[1][4][CC][k][i][j]*rhs[4][k+1][j][i];
            
			rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsZ[2][0][CC][k][i][j]*rhs[0][k+1][j][i];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsZ[2][1][CC][k][i][j]*rhs[1][k+1][j][i];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsZ[2][2][CC][k][i][j]*rhs[2][k+1][j][i];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsZ[2][3][CC][k][i][j]*rhs[3][k+1][j][i];
            rhs[2][k][j][i] = rhs[2][k][j][i] 
              - lhsZ[2][4][CC][k][i][j]*rhs[4][k+1][j][i];
			
			rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsZ[3][0][CC][k][i][j]*rhs[0][k+1][j][i];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsZ[3][1][CC][k][i][j]*rhs[1][k+1][j][i];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsZ[3][2][CC][k][i][j]*rhs[2][k+1][j][i];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsZ[3][3][CC][k][i][j]*rhs[3][k+1][j][i];
            rhs[3][k][j][i] = rhs[3][k][j][i] 
              - lhsZ[3][4][CC][k][i][j]*rhs[4][k+1][j][i];
			
			rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsZ[4][0][CC][k][i][j]*rhs[0][k+1][j][i];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsZ[4][1][CC][k][i][j]*rhs[1][k+1][j][i];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsZ[4][2][CC][k][i][j]*rhs[2][k+1][j][i];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsZ[4][3][CC][k][i][j]*rhs[3][k+1][j][i];
            rhs[4][k][j][i] = rhs[4][k][j][i] 
              - lhsZ[4][4][CC][k][i][j]*rhs[4][k+1][j][i];
	  
      }
    }
  }
}

void adi()
{
  compute_rhs();

  x_solve();

  y_solve();

  z_solve();

  add();
}