
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "applu.incl"
#include "timers.h"
#include "print_results.h"

double dxi, deta, dzeta;
double tx1, tx2, tx3;
double ty1, ty2, ty3;
double tz1, tz2, tz3;
int nx, ny, nz;
int nx0, ny0, nz0;
int ist, iend;
int jst, jend;
int ii1, ii2;
int ji1, ji2;
int ki1, ki2;

double dx1, dx2, dx3, dx4, dx5;
double dy1, dy2, dy3, dy4, dy5;
double dz1, dz2, dz3, dz4, dz5;
double dssp;

double u[5][ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double rsd[5][ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double frct[5][ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double flux_G[5][ISIZ3][ISIZ2][ISIZ1];
double qs   [ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double rho_i[ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];

int ipr, inorm;

double dt, omega, tolrsd[5], rsdnm[5], errnm[5], frc, ttotal;
int itmax, invert;

double a[5][5][ISIZ1*ISIZ2];
double b[5][5][ISIZ1*ISIZ2];
double c[5][5][ISIZ1*ISIZ2];
double d[5][5][ISIZ1*ISIZ2];

int np[ISIZ1+ISIZ2+ISIZ3-5];
int indxp[ISIZ1+ISIZ2+ISIZ3-5][ISIZ1*ISIZ2*3/4];
int jndxp[ISIZ1+ISIZ2+ISIZ3-5][ISIZ1*ISIZ2*3/4];
double tmat[5][5][ISIZ1*ISIZ2*3/4];
double tv[5][ISIZ1*ISIZ2*3/4];
double utmp_G[6][ISIZ2][ISIZ1][ISIZ3];
double rtmp_G[5][ISIZ2][ISIZ1][ISIZ3];

double ce[5][13];

double maxtime;
logical timeron;

int main(int argc, char *argv[])
{
  char Class;
  logical verified;
  double mflops;

  double t, tmax, trecs[t_last+1];
  int i;
  char *t_names[t_last+1];

  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[t_total] = "total";
    t_names[t_rhsx] = "rhsx";
    t_names[t_rhsy] = "rhsy";
    t_names[t_rhsz] = "rhsz";
    t_names[t_rhs] = "rhs";
    t_names[t_jacld] = "jacld";
    t_names[t_blts] = "blts";
    t_names[t_jacu] = "jacu";
    t_names[t_buts] = "buts";
    t_names[t_add] = "add";
    t_names[t_l2norm] = "l2norm";
    fclose(fp);
  } else {
    timeron = false;
  }

  read_input();
  
  domain();

  setcoeff();

{
  
  setbv();

  setiv();
  
  erhs();

  ssor(1);

  setbv();
  setiv();

  ssor(itmax);

  error();

  pintgr();
}
  
  verify ( rsdnm, errnm, frc, &Class, &verified );
  mflops = (double)itmax * (1984.77 * (double)nx0
      * (double)ny0
      * (double)nz0
      - 10923.3 * pow(((double)(nx0+ny0+nz0)/3.0), 2.0) 
      + 27770.9 * (double)(nx0+ny0+nz0)/3.0
      - 144010.0)
    / (maxtime*1000000.0);

  print_results("LU", Class, nx0,
                ny0, nz0, itmax,
                maxtime, mflops, "          floating point", verified, 
                NPBVERSION, COMPILETIME, CS1, CS2, CS3, CS4, CS5, CS6, 
                "(none)");

  if (timeron) {
    for (i = 1; i <= t_last; i++) {
      trecs[i] = timer_read(i);
    }
    tmax = maxtime;
    if (tmax == 0.0) tmax = 1.0;

    printf("  SECTION     Time (secs)\n");
    for (i = 1; i <= t_last; i++) {
      printf("  %-8s:%9.3f  (%6.2f%%)\n",
          t_names[i], trecs[i], trecs[i]*100./tmax);
      if (i == t_rhs) {
        t = trecs[t_rhsx] + trecs[t_rhsy] + trecs[t_rhsz];
        printf("     --> %8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
        t = trecs[i] - t;
        printf("     --> %8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
      }
    }
  }

  return 0;
}


void blts (int ldmx, int ldmy, int ldmz, int nx, int ny, int nz, int l,
    double omega)
{
  
  int i, j, k, m, n;
  double tmp, tmp1;
  int npl = np[l];

{
  for (n = 1; n <= npl; n++) {
  	j = jndxp[l][n];
	i = indxp[l][n];
	k = l - i - j;
      for (m = 0; m < 5; m++) {
        rsd[m][k][j][i] =  rsd[m][k][j][i]
          - omega * (  a[0][m][n] * rsd[0][k-1][j][i]
                     + a[1][m][n] * rsd[1][k-1][j][i]
                     + a[2][m][n] * rsd[2][k-1][j][i]
                     + a[3][m][n] * rsd[3][k-1][j][i]
                     + a[4][m][n] * rsd[4][k-1][j][i] );
      }
  }

  for (n = 1; n <= npl; n++) {
  	j = jndxp[l][n];
	i = indxp[l][n];
	k = l - i - j;
      for (m = 0; m < 5; m++) {
        tv[m][n] =  rsd[m][k][j][i]
          - omega * ( b[0][m][n] * rsd[0][k][j-1][i]
                    + c[0][m][n] * rsd[0][k][j][i-1]
                    + b[1][m][n] * rsd[1][k][j-1][i]
                    + c[1][m][n] * rsd[1][k][j][i-1]
                    + b[2][m][n] * rsd[2][k][j-1][i]
                    + c[2][m][n] * rsd[2][k][j][i-1]
                    + b[3][m][n] * rsd[3][k][j-1][i]
                    + c[3][m][n] * rsd[3][k][j][i-1]
                    + b[4][m][n] * rsd[4][k][j-1][i]
                    + c[4][m][n] * rsd[4][k][j][i-1] );
      }

      for (m = 0; m < 5; m++) {
        tmat[m][0][n] = d[0][m][n];
        tmat[m][1][n] = d[1][m][n];
        tmat[m][2][n] = d[2][m][n];
        tmat[m][3][n] = d[3][m][n];
        tmat[m][4][n] = d[4][m][n];
      }

      tmp1 = 1.0 / tmat[0][0][n];
      tmp = tmp1 * tmat[1][0][n];
      tmat[1][1][n] =  tmat[1][1][n] - tmp * tmat[0][1][n];
      tmat[1][2][n] =  tmat[1][2][n] - tmp * tmat[0][2][n];
      tmat[1][3][n] =  tmat[1][3][n] - tmp * tmat[0][3][n];
      tmat[1][4][n] =  tmat[1][4][n] - tmp * tmat[0][4][n];
      tv[1][n] = tv[1][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[2][0][n];
      tmat[2][1][n] =  tmat[2][1][n] - tmp * tmat[0][1][n];
      tmat[2][2][n] =  tmat[2][2][n] - tmp * tmat[0][2][n];
      tmat[2][3][n] =  tmat[2][3][n] - tmp * tmat[0][3][n];
      tmat[2][4][n] =  tmat[2][4][n] - tmp * tmat[0][4][n];
      tv[2][n] = tv[2][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[3][0][n];
      tmat[3][1][n] =  tmat[3][1][n] - tmp * tmat[0][1][n];
      tmat[3][2][n] =  tmat[3][2][n] - tmp * tmat[0][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[0][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[0][4][n];
      tv[3][n] = tv[3][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[4][0][n];
      tmat[4][1][n] =  tmat[4][1][n] - tmp * tmat[0][1][n];
      tmat[4][2][n] =  tmat[4][2][n] - tmp * tmat[0][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[0][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[0][4][n];
      tv[4][n] = tv[4][n] - tv[0][n] * tmp;

      tmp1 = 1.0 / tmat[1][1][n];
      tmp = tmp1 * tmat[2][1][n];
      tmat[2][2][n] =  tmat[2][2][n] - tmp * tmat[1][2][n];
      tmat[2][3][n] =  tmat[2][3][n] - tmp * tmat[1][3][n];
      tmat[2][4][n] =  tmat[2][4][n] - tmp * tmat[1][4][n];
      tv[2][n] = tv[2][n] - tv[1][n] * tmp;

      tmp = tmp1 * tmat[3][1][n];
      tmat[3][2][n] =  tmat[3][2][n] - tmp * tmat[1][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[1][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[1][4][n];
      tv[3][n] = tv[3][n] - tv[1][n] * tmp;

      tmp = tmp1 * tmat[4][1][n];
      tmat[4][2][n] =  tmat[4][2][n] - tmp * tmat[1][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[1][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[1][4][n];
      tv[4][n] = tv[4][n] - tv[1][n] * tmp;

      tmp1 = 1.0 / tmat[2][2][n];
      tmp = tmp1 * tmat[3][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[2][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[2][4][n];
      tv[3][n] = tv[3][n] - tv[2][n] * tmp;

      tmp = tmp1 * tmat[4][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[2][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[2][4][n];
      tv[4][n] = tv[4][n] - tv[2][n] * tmp;

      tmp1 = 1.0 / tmat[3][3][n];
      tmp = tmp1 * tmat[4][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[3][4][n];
      tv[4][n] = tv[4][n] - tv[3][n] * tmp;

      rsd[4][k][j][i] = tv[4][n] / tmat[4][4][n];

      tv[3][n] = tv[3][n] 
        - tmat[3][4][n] * rsd[4][k][j][i];
      rsd[3][k][j][i] = tv[3][n] / tmat[3][3][n];

      tv[2][n] = tv[2][n]
        - tmat[2][3][n] * rsd[3][k][j][i]
        - tmat[2][4][n] * rsd[4][k][j][i];
      rsd[2][k][j][i] = tv[2][n] / tmat[2][2][n];

      tv[1][n] = tv[1][n]
        - tmat[1][2][n] * rsd[2][k][j][i]
        - tmat[1][3][n] * rsd[3][k][j][i]
        - tmat[1][4][n] * rsd[4][k][j][i];
      rsd[1][k][j][i] = tv[1][n] / tmat[1][1][n];

      tv[0][n] = tv[0][n]
        - tmat[0][1][n] * rsd[1][k][j][i]
        - tmat[0][2][n] * rsd[2][k][j][i]
        - tmat[0][3][n] * rsd[3][k][j][i]
        - tmat[0][4][n] * rsd[4][k][j][i];
      rsd[0][k][j][i] = tv[0][n] / tmat[0][0][n];
  }
 }
}

void buts(int ldmx, int ldmy, int ldmz, int nx, int ny, int nz, int l,
    double omega)
{
  
  int i, j, k, m, n;
  double tmp, tmp1;
  unsigned int npl = np[l];
	
{
  for (n = 1; n <= npl; n++) {
  	j = jndxp[l][n];
	i = indxp[l][n];
	k = l - i - j;
      for (m = 0; m < 5; m++) {
        tv[m][n] = 
          omega * (  c[0][m][n] * rsd[0][k+1][j][i]
                   + c[1][m][n] * rsd[1][k+1][j][i]
                   + c[2][m][n] * rsd[2][k+1][j][i]
                   + c[3][m][n] * rsd[3][k+1][j][i]
                   + c[4][m][n] * rsd[4][k+1][j][i] );
      }
  }

  for (n = 1; n <= npl; n++) {
  	j = jndxp[l][n];
	i = indxp[l][n];
	k = l - i - j;
      for (m = 0; m < 5; m++) {
        tv[m][n] = tv[m][n]
          + omega * ( b[0][m][n] * rsd[0][k][j+1][i]
                    + a[0][m][n] * rsd[0][k][j][i+1]
                    + b[1][m][n] * rsd[1][k][j+1][i]
                    + a[1][m][n] * rsd[1][k][j][i+1]
                    + b[2][m][n] * rsd[2][k][j+1][i]
                    + a[2][m][n] * rsd[2][k][j][i+1]
                    + b[3][m][n] * rsd[3][k][j+1][i]
                    + a[3][m][n] * rsd[3][k][j][i+1]
                    + b[4][m][n] * rsd[4][k][j+1][i]
                    + a[4][m][n] * rsd[4][k][j][i+1] );
      }

      for (m = 0; m < 5; m++) {
        tmat[m][0][n] = d[0][m][n];
        tmat[m][1][n] = d[1][m][n];
        tmat[m][2][n] = d[2][m][n];
        tmat[m][3][n] = d[3][m][n];
        tmat[m][4][n] = d[4][m][n];
      }

      tmp1 = 1.0 / tmat[0][0][n];
      tmp = tmp1 * tmat[1][0][n];
      tmat[1][1][n] =  tmat[1][1][n] - tmp * tmat[0][1][n];
      tmat[1][2][n] =  tmat[1][2][n] - tmp * tmat[0][2][n];
      tmat[1][3][n] =  tmat[1][3][n] - tmp * tmat[0][3][n];
      tmat[1][4][n] =  tmat[1][4][n] - tmp * tmat[0][4][n];
      tv[1][n] = tv[1][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[2][0][n];
      tmat[2][1][n] =  tmat[2][1][n] - tmp * tmat[0][1][n];
      tmat[2][2][n] =  tmat[2][2][n] - tmp * tmat[0][2][n];
      tmat[2][3][n] =  tmat[2][3][n] - tmp * tmat[0][3][n];
      tmat[2][4][n] =  tmat[2][4][n] - tmp * tmat[0][4][n];
      tv[2][n] = tv[2][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[3][0][n];
      tmat[3][1][n] =  tmat[3][1][n] - tmp * tmat[0][1][n];
      tmat[3][2][n] =  tmat[3][2][n] - tmp * tmat[0][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[0][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[0][4][n];
      tv[3][n] = tv[3][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[4][0][n];
      tmat[4][1][n] =  tmat[4][1][n] - tmp * tmat[0][1][n];
      tmat[4][2][n] =  tmat[4][2][n] - tmp * tmat[0][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[0][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[0][4][n];
      tv[4][n] = tv[4][n] - tv[0][n] * tmp;

      tmp1 = 1.0 / tmat[1][1][n];
      tmp = tmp1 * tmat[2][1][n];
      tmat[2][2][n] =  tmat[2][2][n] - tmp * tmat[1][2][n];
      tmat[2][3][n] =  tmat[2][3][n] - tmp * tmat[1][3][n];
      tmat[2][4][n] =  tmat[2][4][n] - tmp * tmat[1][4][n];
      tv[2][n] = tv[2][n] - tv[1][n] * tmp;

      tmp = tmp1 * tmat[3][1][n];
      tmat[3][2][n] =  tmat[3][2][n] - tmp * tmat[1][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[1][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[1][4][n];
      tv[3][n] = tv[3][n] - tv[1][n] * tmp;

      tmp = tmp1 * tmat[4][1][n];
      tmat[4][2][n] =  tmat[4][2][n] - tmp * tmat[1][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[1][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[1][4][n];
      tv[4][n] = tv[4][n] - tv[1][n] * tmp;

      tmp1 = 1.0 / tmat[2][2][n];
      tmp = tmp1 * tmat[3][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[2][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[2][4][n];
      tv[3][n] = tv[3][n] - tv[2][n] * tmp;

      tmp = tmp1 * tmat[4][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[2][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[2][4][n];
      tv[4][n] = tv[4][n] - tv[2][n] * tmp;

      tmp1 = 1.0 / tmat[3][3][n];
      tmp = tmp1 * tmat[4][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[3][4][n];
      tv[4][n] = tv[4][n] - tv[3][n] * tmp;

      tv[4][n] = tv[4][n] / tmat[4][4][n];

      tv[3][n] = tv[3][n] - tmat[3][4][n] * tv[4][n];
      tv[3][n] = tv[3][n] / tmat[3][3][n];

      tv[2][n] = tv[2][n]
        - tmat[2][3][n] * tv[3][n]
        - tmat[2][4][n] * tv[4][n];
      tv[2][n] = tv[2][n] / tmat[2][2][n];

      tv[1][n] = tv[1][n]
        - tmat[1][2][n] * tv[2][n]
        - tmat[1][3][n] * tv[3][n]
        - tmat[1][4][n] * tv[4][n];
      tv[1][n] = tv[1][n] / tmat[1][1][n];

      tv[0][n] = tv[0][n]
        - tmat[0][1][n] * tv[1][n]
        - tmat[0][2][n] * tv[2][n]
        - tmat[0][3][n] * tv[3][n]
        - tmat[0][4][n] * tv[4][n];
      tv[0][n] = tv[0][n] / tmat[0][0][n];

      rsd[0][k][j][i] = rsd[0][k][j][i] - tv[0][n];
      rsd[1][k][j][i] = rsd[1][k][j][i] - tv[1][n];
      rsd[2][k][j][i] = rsd[2][k][j][i] - tv[2][n];
      rsd[3][k][j][i] = rsd[3][k][j][i] - tv[3][n];
      rsd[4][k][j][i] = rsd[4][k][j][i] - tv[4][n];
  }
 }
}

void erhs()
{
  
  int i, j, k, m;
  double xi, eta, zeta;
  double q;
  double u21, u31, u41;
  double tmp;
  double u21i, u31i, u41i, u51i;
  double u21j, u31j, u41j, u51j;
  double u21k, u31k, u41k, u51k;
  double u21im1, u31im1, u41im1, u51im1;
  double u21jm1, u31jm1, u41jm1, u51jm1;
  double u21km1, u31km1, u41km1, u51km1;
  
{
  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][i] = 0.0;
        }
      }
    }
  }

  for (k = 0; k < nz; k++) {
    zeta = ( (double)k ) / ( nz - 1 );
    for (j = 0; j < ny; j++) {
      eta = ( (double)j ) / ( ny0 - 1 );
      for (i = 0; i < nx; i++) {
        xi = ( (double)i ) / ( nx0 - 1 );
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][i] =  ce[m][0]
            + (ce[m][1]
            + (ce[m][4]
            + (ce[m][7]
            +  ce[m][10] * xi) * xi) * xi) * xi
            + (ce[m][2]
            + (ce[m][5]
            + (ce[m][8]
            +  ce[m][11] * eta) * eta) * eta) * eta
            + (ce[m][3]
            + (ce[m][6]
            + (ce[m][9]
            +  ce[m][12] * zeta) * zeta) * zeta) * zeta;
        }
      }
    }
  }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = 0; i < nx; i++) {
        flux_G[0][k][j][i] = rsd[1][k][j][i];
        u21 = rsd[1][k][j][i] / rsd[0][k][j][i];
        q = 0.50 * (  rsd[1][k][j][i] * rsd[1][k][j][i]
                    + rsd[2][k][j][i] * rsd[2][k][j][i]
                    + rsd[3][k][j][i] * rsd[3][k][j][i] )
                 / rsd[0][k][j][i];
        flux_G[1][k][j][i] = rsd[1][k][j][i] * u21 + C2 * ( rsd[4][k][j][i] - q );
        flux_G[2][k][j][i] = rsd[2][k][j][i] * u21;
        flux_G[3][k][j][i] = rsd[3][k][j][i] * u21;
        flux_G[4][k][j][i] = ( C1 * rsd[4][k][j][i] - C2 * q ) * u21;
      }
     }
   }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][i] =  frct[m][k][j][i]
                    - tx2 * ( flux_G[m][k][j][i+1] - flux_G[m][k][j][i-1] );
        }
      }
    }
  }
      
  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = ist; i < nx; i++) {
        tmp = 1.0 / rsd[0][k][j][i];

        u21i = tmp * rsd[1][k][j][i];
        u31i = tmp * rsd[2][k][j][i];
        u41i = tmp * rsd[3][k][j][i];
        u51i = tmp * rsd[4][k][j][i];

        tmp = 1.0 / rsd[0][k][j][i-1];

        u21im1 = tmp * rsd[1][k][j][i-1];
        u31im1 = tmp * rsd[2][k][j][i-1];
        u41im1 = tmp * rsd[3][k][j][i-1];
        u51im1 = tmp * rsd[4][k][j][i-1];

        flux_G[1][k][j][i] = (4.0/3.0) * tx3 * ( u21i - u21im1 );
        flux_G[2][k][j][i] = tx3 * ( u31i - u31im1 );
        flux_G[3][k][j][i] = tx3 * ( u41i - u41im1 );
        flux_G[4][k][j][i] = 0.50 * ( 1.0 - C1*C5 )
          * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
                  - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
          + (1.0/6.0)
          * tx3 * ( u21i*u21i - u21im1*u21im1 )
          + C1 * C5 * tx3 * ( u51i - u51im1 );
      }
    }
  }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        frct[0][k][j][i] = frct[0][k][j][i]
          + dx1 * tx1 * (        rsd[0][k][j][i-1]
                         - 2.0 * rsd[0][k][j][i]
                         +       rsd[0][k][j][i+1] );
        frct[1][k][j][i] = frct[1][k][j][i]
          + tx3 * C3 * C4 * ( flux_G[1][k][j][i+1] - flux_G[1][k][j][i] )
          + dx2 * tx1 * (        rsd[1][k][j][i-1]
                         - 2.0 * rsd[1][k][j][i]
                         +       rsd[1][k][j][i+1] );
        frct[2][k][j][i] = frct[2][k][j][i]
          + tx3 * C3 * C4 * ( flux_G[2][k][j][i+1] - flux_G[2][k][j][i] )
          + dx3 * tx1 * (        rsd[2][k][j][i-1]
                         - 2.0 * rsd[2][k][j][i]
                         +       rsd[2][k][j][i+1] );
        frct[3][k][j][i] = frct[3][k][j][i]
          + tx3 * C3 * C4 * ( flux_G[3][k][j][i+1] - flux_G[3][k][j][i] )
          + dx4 * tx1 * (        rsd[3][k][j][i-1]
                         - 2.0 * rsd[3][k][j][i]
                         +       rsd[3][k][j][i+1] );
        frct[4][k][j][i] = frct[4][k][j][i]
          + tx3 * C3 * C4 * ( flux_G[4][k][j][i+1] - flux_G[4][k][j][i] )
          + dx5 * tx1 * (        rsd[4][k][j][i-1]
                         - 2.0 * rsd[4][k][j][i]
                         +       rsd[4][k][j][i+1] );
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (m = 0; m < 5; m++) {
        frct[m][k][j][1] = frct[m][k][j][1]
          - dssp * ( + 5.0 * rsd[m][k][j][1]
                     - 4.0 * rsd[m][k][j][2]
                     +       rsd[m][k][j][3] );
        frct[m][k][j][2] = frct[m][k][j][2]
          - dssp * ( - 4.0 * rsd[m][k][j][1]
                     + 6.0 * rsd[m][k][j][2]
                     - 4.0 * rsd[m][k][j][3]
                     +       rsd[m][k][j][4] );
      }
     }
   }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = 3; i < nx - 3; i++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][i] = frct[m][k][j][i]
            - dssp * (        rsd[m][k][j][i-2]
                      - 4.0 * rsd[m][k][j][i-1]
                      + 6.0 * rsd[m][k][j][i]
                      - 4.0 * rsd[m][k][j][i+1]
                      +       rsd[m][k][j][i+2] );
        }
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (m = 0; m < 5; m++) {
        frct[m][k][j][nx-3] = frct[m][k][j][nx-3]
          - dssp * (        rsd[m][k][j][nx-5]
                    - 4.0 * rsd[m][k][j][nx-4]
                    + 6.0 * rsd[m][k][j][nx-3]
                    - 4.0 * rsd[m][k][j][nx-2] );
        frct[m][k][j][nx-2] = frct[m][k][j][nx-2]
          - dssp * (        rsd[m][k][j][nx-4]
                    - 4.0 * rsd[m][k][j][nx-3]
                    + 5.0 * rsd[m][k][j][nx-2] );
      }
    }
  }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (j = 0; j < ny; j++) {
        flux_G[0][k][i][j] = rsd[2][k][j][i];
        u31 = rsd[2][k][j][i] / rsd[0][k][j][i];
        q = 0.50 * (  rsd[1][k][j][i] * rsd[1][k][j][i]
                    + rsd[2][k][j][i] * rsd[2][k][j][i]
                    + rsd[3][k][j][i] * rsd[3][k][j][i] )
                 / rsd[0][k][j][i];
        flux_G[1][k][i][j] = rsd[1][k][j][i] * u31;
        flux_G[2][k][i][j] = rsd[2][k][j][i] * u31 + C2 * ( rsd[4][k][j][i] - q );
        flux_G[3][k][i][j] = rsd[3][k][j][i] * u31;
        flux_G[4][k][i][j] = ( C1 * rsd[4][k][j][i] - C2 * q ) * u31;
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (j = jst; j <= jend; j++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][i] =  frct[m][k][j][i]
            - ty2 * ( flux_G[m][k][i][j+1] - flux_G[m][k][i][j-1] );
        }
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (j = jst; j < ny; j++) {
        tmp = 1.0 / rsd[0][k][j][i];

        u21j = tmp * rsd[1][k][j][i];
        u31j = tmp * rsd[2][k][j][i];
        u41j = tmp * rsd[3][k][j][i];
        u51j = tmp * rsd[4][k][j][i];

        tmp = 1.0 / rsd[0][k][j-1][i];

        u21jm1 = tmp * rsd[1][k][j-1][i];
        u31jm1 = tmp * rsd[2][k][j-1][i];
        u41jm1 = tmp * rsd[3][k][j-1][i];
        u51jm1 = tmp * rsd[4][k][j-1][i];

        flux_G[1][k][i][j] = ty3 * ( u21j - u21jm1 );
        flux_G[2][k][i][j] = (4.0/3.0) * ty3 * ( u31j - u31jm1 );
        flux_G[3][k][i][j] = ty3 * ( u41j - u41jm1 );
        flux_G[4][k][i][j] = 0.50 * ( 1.0 - C1*C5 )
          * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
                  - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
          + (1.0/6.0)
          * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
          + C1 * C5 * ty3 * ( u51j - u51jm1 );
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (j = jst; j <= jend; j++) {
        frct[0][k][j][i] = frct[0][k][j][i]
          + dy1 * ty1 * (        rsd[0][k][j-1][i]
                         - 2.0 * rsd[0][k][j][i]
                         +       rsd[0][k][j+1][i] );
        frct[1][k][j][i] = frct[1][k][j][i]
          + ty3 * C3 * C4 * ( flux_G[1][k][i][j+1] - flux_G[1][k][i][j] )
          + dy2 * ty1 * (        rsd[1][k][j-1][i]
                         - 2.0 * rsd[1][k][j][i]
                         +       rsd[1][k][j+1][i] );
        frct[2][k][j][i] = frct[2][k][j][i]
          + ty3 * C3 * C4 * ( flux_G[2][k][i][j+1] - flux_G[2][k][i][j] )
          + dy3 * ty1 * (        rsd[2][k][j-1][i]
                         - 2.0 * rsd[2][k][j][i]
                         +       rsd[2][k][j+1][i] );
        frct[3][k][j][i] = frct[3][k][j][i]
          + ty3 * C3 * C4 * ( flux_G[3][k][i][j+1] - flux_G[3][k][i][j] )
          + dy4 * ty1 * (        rsd[3][k][j-1][i]
                         - 2.0 * rsd[3][k][j][i]
                         +       rsd[3][k][j+1][i] );
        frct[4][k][j][i] = frct[4][k][j][i]
          + ty3 * C3 * C4 * ( flux_G[4][k][i][j+1] - flux_G[4][k][i][j] )
          + dy5 * ty1 * (        rsd[4][k][j-1][i]
                         - 2.0 * rsd[4][k][j][i]
                         +       rsd[4][k][j+1][i] );
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (m = 0; m < 5; m++) {
        frct[m][k][1][i] = frct[m][k][1][i]
          - dssp * ( + 5.0 * rsd[m][k][1][i]
                     - 4.0 * rsd[m][k][2][i]
                     +       rsd[m][k][3][i] );
        frct[m][k][2][i] = frct[m][k][2][i]
          - dssp * ( - 4.0 * rsd[m][k][1][i]
                     + 6.0 * rsd[m][k][2][i]
                     - 4.0 * rsd[m][k][3][i]
                     +       rsd[m][k][4][i] );
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (j = 3; j < ny - 3; j++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][i] = frct[m][k][j][i]
            - dssp * (        rsd[m][k][j-2][i]
                      - 4.0 * rsd[m][k][j-1][i]
                      + 6.0 * rsd[m][k][j][i]
                      - 4.0 * rsd[m][k][j+1][i]
                      +       rsd[m][k][j+2][i] );
        }
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (m = 0; m < 5; m++) {
        frct[m][k][ny-3][i] = frct[m][k][ny-3][i]
          - dssp * (        rsd[m][k][ny-5][i]
                    - 4.0 * rsd[m][k][ny-4][i]
                    + 6.0 * rsd[m][k][ny-3][i]
                    - 4.0 * rsd[m][k][ny-2][i] );
        frct[m][k][ny-2][i] = frct[m][k][ny-2][i]
          - dssp * (        rsd[m][k][ny-4][i]
                    - 4.0 * rsd[m][k][ny-3][i]
                    + 5.0 * rsd[m][k][ny-2][i] );
      }
    }
  }

  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 0; k < nz; k++) {
        flux_G[0][j][i][k] = rsd[3][k][j][i];
        u41 = rsd[3][k][j][i] / rsd[0][k][j][i];
        q = 0.50 * (  rsd[1][k][j][i] * rsd[1][k][j][i]
                    + rsd[2][k][j][i] * rsd[2][k][j][i]
                    + rsd[3][k][j][i] * rsd[3][k][j][i] )
                 / rsd[0][k][j][i];
        flux_G[1][j][i][k] = rsd[1][k][j][i] * u41;
        flux_G[2][j][i][k] = rsd[2][k][j][i] * u41; 
        flux_G[3][j][i][k] = rsd[3][k][j][i] * u41 + C2 * ( rsd[4][k][j][i] - q );
        flux_G[4][j][i][k] = ( C1 * rsd[4][k][j][i] - C2 * q ) * u41;
      }
     }
    }

  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 1; k < nz - 1; k++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][i] =  frct[m][k][j][i]
            - tz2 * ( flux_G[m][j][i][k+1] - flux_G[m][j][i][k-1] );
        }
      }
     }
   }

  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 1; k < nz; k++) {
        tmp = 1.0 / rsd[0][k][j][i];

        u21k = tmp * rsd[1][k][j][i];
        u31k = tmp * rsd[2][k][j][i];
        u41k = tmp * rsd[3][k][j][i];
        u51k = tmp * rsd[4][k][j][i];

        tmp = 1.0 / rsd[0][k-1][j][i];

        u21km1 = tmp * rsd[1][k-1][j][i];
        u31km1 = tmp * rsd[2][k-1][j][i];
        u41km1 = tmp * rsd[3][k-1][j][i];
        u51km1 = tmp * rsd[4][k-1][j][i];

        flux_G[1][j][i][k] = tz3 * ( u21k - u21km1 );
        flux_G[2][j][i][k] = tz3 * ( u31k - u31km1 );
        flux_G[3][j][i][k] = (4.0/3.0) * tz3 * ( u41k - u41km1 );
        flux_G[4][j][i][k] = 0.50 * ( 1.0 - C1*C5 )
          * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
                  - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
          + (1.0/6.0)
          * tz3 * ( u41k*u41k - u41km1*u41km1 )
          + C1 * C5 * tz3 * ( u51k - u51km1 );
      }
     }
    }

  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 1; k < nz - 1; k++) {
        frct[0][k][j][i] = frct[0][k][j][i]
          + dz1 * tz1 * (        rsd[0][k+1][j][i]
                         - 2.0 * rsd[0][k][j][i]
                         +       rsd[0][k-1][j][i] );
        frct[1][k][j][i] = frct[1][k][j][i]
          + tz3 * C3 * C4 * ( flux_G[1][j][i][k+1] - flux_G[1][j][i][k] )
          + dz2 * tz1 * (        rsd[1][k+1][j][i]
                         - 2.0 * rsd[1][k][j][i]
                         +       rsd[1][k-1][j][i] );
        frct[2][k][j][i] = frct[2][k][j][i]
          + tz3 * C3 * C4 * ( flux_G[2][j][i][k+1] - flux_G[2][j][i][k] )
          + dz3 * tz1 * (        rsd[2][k+1][j][i]
                         - 2.0 * rsd[2][k][j][i]
                         +       rsd[2][k-1][j][i] );
        frct[3][k][j][i] = frct[3][k][j][i]
          + tz3 * C3 * C4 * ( flux_G[3][j][i][k+1] - flux_G[3][j][i][k] )
          + dz4 * tz1 * (        rsd[3][k+1][j][i]
                         - 2.0 * rsd[3][k][j][i]
                         +       rsd[3][k-1][j][i] );
        frct[4][k][j][i] = frct[4][k][j][i]
          + tz3 * C3 * C4 * ( flux_G[4][j][i][k+1] - flux_G[4][j][i][k] )
          + dz5 * tz1 * (        rsd[4][k+1][j][i]
                         - 2.0 * rsd[4][k][j][i]
                         +       rsd[4][k-1][j][i] );
      }
     }
    }

  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (m = 0; m < 5; m++) {
        frct[m][1][j][i] = frct[m][1][j][i]
          - dssp * ( + 5.0 * rsd[m][1][j][i]
                     - 4.0 * rsd[m][2][j][i]
                     +       rsd[m][3][j][i] );
        frct[m][2][j][i] = frct[m][2][j][i]
          - dssp * ( - 4.0 * rsd[m][1][j][i]
                     + 6.0 * rsd[m][2][j][i]
                     - 4.0 * rsd[m][3][j][i]
                     +       rsd[m][4][j][i] );
      }
     }
    }

  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 3; k < nz - 3; k++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][i] = frct[m][k][j][i]
            - dssp * (        rsd[m][k-2][j][i]
                      - 4.0 * rsd[m][k-1][j][i]
                      + 6.0 * rsd[m][k][j][i]
                      - 4.0 * rsd[m][k+1][j][i]
                      +       rsd[m][k+2][j][i] );
        }
      }
     }
    }

  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (m = 0; m < 5; m++) {
        frct[m][nz-3][j][i] = frct[m][nz-3][j][i]
          - dssp * (        rsd[m][nz-5][j][i]
                    - 4.0 * rsd[m][nz-4][j][i]
                    + 6.0 * rsd[m][nz-3][j][i]
                    - 4.0 * rsd[m][nz-2][j][i] );
        frct[m][nz-2][j][i] = frct[m][nz-2][j][i]
          - dssp * (        rsd[m][nz-4][j][i]
                    - 4.0 * rsd[m][nz-3][j][i]
                    + 5.0 * rsd[m][nz-2][j][i] );
      }
    }
  }
 }
}

void jacld(int l)
{
  
  int i, j, k, n;
  double r43;
  double c1345;
  double c34;
  double tmp1, tmp2, tmp3;
  int npl = np[l];

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;
{
  for (n = 1; n <= npl; n++) {
  	j = jndxp[l][n];
	i = indxp[l][n];
	k = l - i - j;
      
      tmp1 = rho_i[k][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      d[0][0][n] =  1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
      d[1][0][n] =  0.0;
      d[2][0][n] =  0.0;
      d[3][0][n] =  0.0;
      d[4][0][n] =  0.0;

      d[0][1][n] = -dt * 2.0
        * ( tx1 * r43 + ty1 + tz1 ) * c34 * tmp2 * u[1][k][j][i];
      d[1][1][n] =  1.0
        + dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 )
        + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
      d[2][1][n] = 0.0;
      d[3][1][n] = 0.0;
      d[4][1][n] = 0.0;

      d[0][2][n] = -dt * 2.0 
        * ( tx1 + ty1 * r43 + tz1 ) * c34 * tmp2 * u[2][k][j][i];
      d[1][2][n] = 0.0;
      d[2][2][n] = 1.0
        + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 )
        + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
      d[3][2][n] = 0.0;
      d[4][2][n] = 0.0;

      d[0][3][n] = -dt * 2.0
        * ( tx1 + ty1 + tz1 * r43 ) * c34 * tmp2 * u[3][k][j][i];
      d[1][3][n] = 0.0;
      d[2][3][n] = 0.0;
      d[3][3][n] = 1.0
        + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 )
        + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
      d[4][3][n] = 0.0;

      d[0][4][n] = -dt * 2.0
        * ( ( ( tx1 * ( r43*c34 - c1345 )
                + ty1 * ( c34 - c1345 )
                + tz1 * ( c34 - c1345 ) ) * ( u[1][k][j][i]*u[1][k][j][i] )
              + ( tx1 * ( c34 - c1345 )
                + ty1 * ( r43*c34 - c1345 )
                + tz1 * ( c34 - c1345 ) ) * ( u[2][k][j][i]*u[2][k][j][i] )
              + ( tx1 * ( c34 - c1345 )
                + ty1 * ( c34 - c1345 )
                + tz1 * ( r43*c34 - c1345 ) ) * (u[3][k][j][i]*u[3][k][j][i])
            ) * tmp3
            + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u[4][k][j][i] );

      d[1][4][n] = dt * 2.0 * tmp2 * u[1][k][j][i]
        * ( tx1 * ( r43*c34 - c1345 )
          + ty1 * (     c34 - c1345 )
          + tz1 * (     c34 - c1345 ) );
      d[2][4][n] = dt * 2.0 * tmp2 * u[2][k][j][i]
        * ( tx1 * ( c34 - c1345 )
          + ty1 * ( r43*c34 -c1345 )
          + tz1 * ( c34 - c1345 ) );
      d[3][4][n] = dt * 2.0 * tmp2 * u[3][k][j][i]
        * ( tx1 * ( c34 - c1345 )
          + ty1 * ( c34 - c1345 )
          + tz1 * ( r43*c34 - c1345 ) );
      d[4][4][n] = 1.0
        + dt * 2.0 * ( tx1  + ty1 + tz1 ) * c1345 * tmp1
        + dt * 2.0 * ( tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 );

      tmp1 = rho_i[k-1][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      a[0][0][n] = - dt * tz1 * dz1;
      a[1][0][n] =   0.0;
      a[2][0][n] =   0.0;
      a[3][0][n] = - dt * tz2;
      a[4][0][n] =   0.0;

      a[0][1][n] = - dt * tz2
        * ( - ( u[1][k-1][j][i]*u[3][k-1][j][i] ) * tmp2 )
        - dt * tz1 * ( - c34 * tmp2 * u[1][k-1][j][i] );
      a[1][1][n] = - dt * tz2 * ( u[3][k-1][j][i] * tmp1 )
        - dt * tz1 * c34 * tmp1
        - dt * tz1 * dz2;
      a[2][1][n] = 0.0;
      a[3][1][n] = - dt * tz2 * ( u[1][k-1][j][i] * tmp1 );
      a[4][1][n] = 0.0;

      a[0][2][n] = - dt * tz2
        * ( - ( u[2][k-1][j][i]*u[3][k-1][j][i] ) * tmp2 )
        - dt * tz1 * ( - c34 * tmp2 * u[2][k-1][j][i] );
      a[1][2][n] = 0.0;
      a[2][2][n] = - dt * tz2 * ( u[3][k-1][j][i] * tmp1 )
        - dt * tz1 * ( c34 * tmp1 )
        - dt * tz1 * dz3;
      a[3][2][n] = - dt * tz2 * ( u[2][k-1][j][i] * tmp1 );
      a[4][2][n] = 0.0;

      a[0][3][n] = - dt * tz2
        * ( - ( u[3][k-1][j][i] * tmp1 ) * ( u[3][k-1][j][i] * tmp1 )
            + C2 * qs[k-1][j][i] * tmp1 )
        - dt * tz1 * ( - r43 * c34 * tmp2 * u[3][k-1][j][i] );
      a[1][3][n] = - dt * tz2
        * ( - C2 * ( u[1][k-1][j][i] * tmp1 ) );
      a[2][3][n] = - dt * tz2
        * ( - C2 * ( u[2][k-1][j][i] * tmp1 ) );
      a[3][3][n] = - dt * tz2 * ( 2.0 - C2 )
        * ( u[3][k-1][j][i] * tmp1 )
        - dt * tz1 * ( r43 * c34 * tmp1 )
        - dt * tz1 * dz4;
      a[4][3][n] = - dt * tz2 * C2;

      a[0][4][n] = - dt * tz2
        * ( ( C2 * 2.0 * qs[k-1][j][i] - C1 * u[4][k-1][j][i] )
            * u[3][k-1][j][i] * tmp2 )
        - dt * tz1
        * ( - ( c34 - c1345 ) * tmp3 * (u[1][k-1][j][i]*u[1][k-1][j][i])
            - ( c34 - c1345 ) * tmp3 * (u[2][k-1][j][i]*u[2][k-1][j][i])
            - ( r43*c34 - c1345 )* tmp3 * (u[3][k-1][j][i]*u[3][k-1][j][i])
            - c1345 * tmp2 * u[4][k-1][j][i] );
      a[1][4][n] = - dt * tz2
        * ( - C2 * ( u[1][k-1][j][i]*u[3][k-1][j][i] ) * tmp2 )
        - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[1][k-1][j][i];
      a[2][4][n] = - dt * tz2
        * ( - C2 * ( u[2][k-1][j][i]*u[3][k-1][j][i] ) * tmp2 )
        - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[2][k-1][j][i];
      a[3][4][n] = - dt * tz2
        * ( C1 * ( u[4][k-1][j][i] * tmp1 )
          - C2 * ( qs[k-1][j][i] * tmp1
                 + u[3][k-1][j][i]*u[3][k-1][j][i] * tmp2 ) )
        - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u[3][k-1][j][i];
      a[4][4][n] = - dt * tz2
        * ( C1 * ( u[3][k-1][j][i] * tmp1 ) )
        - dt * tz1 * c1345 * tmp1
        - dt * tz1 * dz5;

      tmp1 = rho_i[k][j-1][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      b[0][0][n] = - dt * ty1 * dy1;
      b[1][0][n] =   0.0;
      b[2][0][n] = - dt * ty2;
      b[3][0][n] =   0.0;
      b[4][0][n] =   0.0;

      b[0][1][n] = - dt * ty2
        * ( - ( u[1][k][j-1][i]*u[2][k][j-1][i] ) * tmp2 )
        - dt * ty1 * ( - c34 * tmp2 * u[1][k][j-1][i] );
      b[1][1][n] = - dt * ty2 * ( u[2][k][j-1][i] * tmp1 )
        - dt * ty1 * ( c34 * tmp1 )
        - dt * ty1 * dy2;
      b[2][1][n] = - dt * ty2 * ( u[1][k][j-1][i] * tmp1 );
      b[3][1][n] = 0.0;
      b[4][1][n] = 0.0;

      b[0][2][n] = - dt * ty2
        * ( - ( u[2][k][j-1][i] * tmp1 ) * ( u[2][k][j-1][i] * tmp1 )
            + C2 * ( qs[k][j-1][i] * tmp1 ) )
        - dt * ty1 * ( - r43 * c34 * tmp2 * u[2][k][j-1][i] );
      b[1][2][n] = - dt * ty2
        * ( - C2 * ( u[1][k][j-1][i] * tmp1 ) );
      b[2][2][n] = - dt * ty2 * ( (2.0 - C2) * (u[2][k][j-1][i] * tmp1) )
        - dt * ty1 * ( r43 * c34 * tmp1 )
        - dt * ty1 * dy3;
      b[3][2][n] = - dt * ty2 * ( - C2 * ( u[3][k][j-1][i] * tmp1 ) );
      b[4][2][n] = - dt * ty2 * C2;

      b[0][3][n] = - dt * ty2
        * ( - ( u[2][k][j-1][i]*u[3][k][j-1][i] ) * tmp2 )
        - dt * ty1 * ( - c34 * tmp2 * u[3][k][j-1][i] );
      b[1][3][n] = 0.0;
      b[2][3][n] = - dt * ty2 * ( u[3][k][j-1][i] * tmp1 );
      b[3][3][n] = - dt * ty2 * ( u[2][k][j-1][i] * tmp1 )
        - dt * ty1 * ( c34 * tmp1 )
        - dt * ty1 * dy4;
      b[4][3][n] = 0.0;

      b[0][4][n] = - dt * ty2
        * ( ( C2 * 2.0 * qs[k][j-1][i] - C1 * u[4][k][j-1][i] )
            * ( u[2][k][j-1][i] * tmp2 ) )
        - dt * ty1
        * ( - (     c34 - c1345 )*tmp3*(u[1][k][j-1][i]*u[1][k][j-1][i])
            - ( r43*c34 - c1345 )*tmp3*(u[2][k][j-1][i]*u[2][k][j-1][i])
            - (     c34 - c1345 )*tmp3*(u[3][k][j-1][i]*u[3][k][j-1][i])
            - c1345*tmp2*u[4][k][j-1][i] );
      b[1][4][n] = - dt * ty2
        * ( - C2 * ( u[1][k][j-1][i]*u[2][k][j-1][i] ) * tmp2 )
        - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[1][k][j-1][i];
      b[2][4][n] = - dt * ty2
        * ( C1 * ( u[4][k][j-1][i] * tmp1 )
          - C2 * ( qs[k][j-1][i] * tmp1
                 + u[2][k][j-1][i]*u[2][k][j-1][i] * tmp2 ) )
        - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u[2][k][j-1][i];
      b[3][4][n] = - dt * ty2
        * ( - C2 * ( u[2][k][j-1][i]*u[3][k][j-1][i] ) * tmp2 )
        - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[3][k][j-1][i];
      b[4][4][n] = - dt * ty2
        * ( C1 * ( u[2][k][j-1][i] * tmp1 ) )
        - dt * ty1 * c1345 * tmp1
        - dt * ty1 * dy5;

      tmp1 = rho_i[k][j][i-1];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      c[0][0][n] = - dt * tx1 * dx1;
      c[1][0][n] = - dt * tx2;
      c[2][0][n] =   0.0;
      c[3][0][n] =   0.0;
      c[4][0][n] =   0.0;

      c[0][1][n] = - dt * tx2
        * ( - ( u[1][k][j][i-1] * tmp1 ) * ( u[1][k][j][i-1] * tmp1 )
            + C2 * qs[k][j][i-1] * tmp1 )
        - dt * tx1 * ( - r43 * c34 * tmp2 * u[1][k][j][i-1] );
      c[1][1][n] = - dt * tx2
        * ( ( 2.0 - C2 ) * ( u[1][k][j][i-1] * tmp1 ) )
        - dt * tx1 * ( r43 * c34 * tmp1 )
        - dt * tx1 * dx2;
      c[2][1][n] = - dt * tx2
        * ( - C2 * ( u[2][k][j][i-1] * tmp1 ) );
      c[3][1][n] = - dt * tx2
        * ( - C2 * ( u[3][k][j][i-1] * tmp1 ) );
      c[4][1][n] = - dt * tx2 * C2;

      c[0][2][n] = - dt * tx2
        * ( - ( u[1][k][j][i-1] * u[2][k][j][i-1] ) * tmp2 )
        - dt * tx1 * ( - c34 * tmp2 * u[2][k][j][i-1] );
      c[1][2][n] = - dt * tx2 * ( u[2][k][j][i-1] * tmp1 );
      c[2][2][n] = - dt * tx2 * ( u[1][k][j][i-1] * tmp1 )
        - dt * tx1 * ( c34 * tmp1 )
        - dt * tx1 * dx3;
      c[3][2][n] = 0.0;
      c[4][2][n] = 0.0;

      c[0][3][n] = - dt * tx2
        * ( - ( u[1][k][j][i-1]*u[3][k][j][i-1] ) * tmp2 )
        - dt * tx1 * ( - c34 * tmp2 * u[3][k][j][i-1] );
      c[1][3][n] = - dt * tx2 * ( u[3][k][j][i-1] * tmp1 );
      c[2][3][n] = 0.0;
      c[3][3][n] = - dt * tx2 * ( u[1][k][j][i-1] * tmp1 )
        - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
      c[4][3][n] = 0.0;

      c[0][4][n] = - dt * tx2
        * ( ( C2 * 2.0 * qs[k][j][i-1] - C1 * u[4][k][j][i-1] )
            * u[1][k][j][i-1] * tmp2 )
        - dt * tx1
        * ( - ( r43*c34 - c1345 ) * tmp3 * ( u[1][k][j][i-1]*u[1][k][j][i-1] )
            - (     c34 - c1345 ) * tmp3 * ( u[2][k][j][i-1]*u[2][k][j][i-1] )
            - (     c34 - c1345 ) * tmp3 * ( u[3][k][j][i-1]*u[3][k][j][i-1] )
            - c1345 * tmp2 * u[4][k][j][i-1] );
      c[1][4][n] = - dt * tx2
        * ( C1 * ( u[4][k][j][i-1] * tmp1 )
          - C2 * ( u[1][k][j][i-1]*u[1][k][j][i-1] * tmp2
                 + qs[k][j][i-1] * tmp1 ) )
        - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u[1][k][j][i-1];
      c[2][4][n] = - dt * tx2
        * ( - C2 * ( u[2][k][j][i-1]*u[1][k][j][i-1] ) * tmp2 )
        - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[2][k][j][i-1];
      c[3][4][n] = - dt * tx2
        * ( - C2 * ( u[3][k][j][i-1]*u[1][k][j][i-1] ) * tmp2 )
        - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[3][k][j][i-1];
      c[4][4][n] = - dt * tx2
        * ( C1 * ( u[1][k][j][i-1] * tmp1 ) )
        - dt * tx1 * c1345 * tmp1
        - dt * tx1 * dx5;
  }
 }
}

void jacu(int l)
{
  
  int i, j, k, n;
  double r43;
  double c1345;
  double c34;
  double tmp1, tmp2, tmp3;
  int npl = np[l];

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

{
  for (n = 1; n <= npl; n++) {
  	j = jndxp[l][n];
	i = indxp[l][n];
	k = l - i - j;
      
      tmp1 = rho_i[k][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      d[0][0][n] = 1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
      d[1][0][n] = 0.0;
      d[2][0][n] = 0.0;
      d[3][0][n] = 0.0;
      d[4][0][n] = 0.0;

      d[0][1][n] =  dt * 2.0
        * ( - tx1 * r43 - ty1 - tz1 )
        * ( c34 * tmp2 * u[1][k][j][i] );
      d[1][1][n] =  1.0
        + dt * 2.0 * c34 * tmp1 
        * (  tx1 * r43 + ty1 + tz1 )
        + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
      d[2][1][n] = 0.0;
      d[3][1][n] = 0.0;
      d[4][1][n] = 0.0;

      d[0][2][n] = dt * 2.0
        * ( - tx1 - ty1 * r43 - tz1 )
        * ( c34 * tmp2 * u[2][k][j][i] );
      d[1][2][n] = 0.0;
      d[2][2][n] = 1.0
        + dt * 2.0 * c34 * tmp1
        * (  tx1 + ty1 * r43 + tz1 )
        + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
      d[3][2][n] = 0.0;
      d[4][2][n] = 0.0;

      d[0][3][n] = dt * 2.0
        * ( - tx1 - ty1 - tz1 * r43 )
        * ( c34 * tmp2 * u[3][k][j][i] );
      d[1][3][n] = 0.0;
      d[2][3][n] = 0.0;
      d[3][3][n] = 1.0
        + dt * 2.0 * c34 * tmp1
        * (  tx1 + ty1 + tz1 * r43 )
        + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
      d[4][3][n] = 0.0;

      d[0][4][n] = -dt * 2.0
        * ( ( ( tx1 * ( r43*c34 - c1345 )
                + ty1 * ( c34 - c1345 )
                + tz1 * ( c34 - c1345 ) ) * ( u[1][k][j][i]*u[1][k][j][i] )
              + ( tx1 * ( c34 - c1345 )
                + ty1 * ( r43*c34 - c1345 )
                + tz1 * ( c34 - c1345 ) ) * ( u[2][k][j][i]*u[2][k][j][i] )
              + ( tx1 * ( c34 - c1345 )
                + ty1 * ( c34 - c1345 )
                + tz1 * ( r43*c34 - c1345 ) ) * (u[3][k][j][i]*u[3][k][j][i])
            ) * tmp3
            + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u[4][k][j][i] );

      d[1][4][n] = dt * 2.0
        * ( tx1 * ( r43*c34 - c1345 )
          + ty1 * (     c34 - c1345 )
          + tz1 * (     c34 - c1345 ) ) * tmp2 * u[1][k][j][i];
      d[2][4][n] = dt * 2.0
        * ( tx1 * ( c34 - c1345 )
          + ty1 * ( r43*c34 -c1345 )
          + tz1 * ( c34 - c1345 ) ) * tmp2 * u[2][k][j][i];
      d[3][4][n] = dt * 2.0
        * ( tx1 * ( c34 - c1345 )
          + ty1 * ( c34 - c1345 )
          + tz1 * ( r43*c34 - c1345 ) ) * tmp2 * u[3][k][j][i];
      d[4][4][n] = 1.0
        + dt * 2.0 * ( tx1 + ty1 + tz1 ) * c1345 * tmp1
        + dt * 2.0 * ( tx1 * dx5 + ty1 * dy5 + tz1 * dz5 );

      tmp1 = rho_i[k][j][i+1];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      a[0][0][n] = - dt * tx1 * dx1;
      a[1][0][n] =   dt * tx2;
      a[2][0][n] =   0.0;
      a[3][0][n] =   0.0;
      a[4][0][n] =   0.0;

      a[0][1][n] =  dt * tx2
        * ( - ( u[1][k][j][i+1] * tmp1 ) * ( u[1][k][j][i+1] * tmp1 )
            + C2 * qs[k][j][i+1] * tmp1 )
        - dt * tx1 * ( - r43 * c34 * tmp2 * u[1][k][j][i+1] );
      a[1][1][n] =  dt * tx2
        * ( ( 2.0 - C2 ) * ( u[1][k][j][i+1] * tmp1 ) )
        - dt * tx1 * ( r43 * c34 * tmp1 )
        - dt * tx1 * dx2;
      a[2][1][n] =  dt * tx2
        * ( - C2 * ( u[2][k][j][i+1] * tmp1 ) );
      a[3][1][n] =  dt * tx2
        * ( - C2 * ( u[3][k][j][i+1] * tmp1 ) );
      a[4][1][n] =  dt * tx2 * C2 ;

      a[0][2][n] =  dt * tx2
        * ( - ( u[1][k][j][i+1] * u[2][k][j][i+1] ) * tmp2 )
        - dt * tx1 * ( - c34 * tmp2 * u[2][k][j][i+1] );
      a[1][2][n] =  dt * tx2 * ( u[2][k][j][i+1] * tmp1 );
      a[2][2][n] =  dt * tx2 * ( u[1][k][j][i+1] * tmp1 )
        - dt * tx1 * ( c34 * tmp1 )
        - dt * tx1 * dx3;
      a[3][2][n] = 0.0;
      a[4][2][n] = 0.0;

      a[0][3][n] = dt * tx2
        * ( - ( u[1][k][j][i+1]*u[3][k][j][i+1] ) * tmp2 )
        - dt * tx1 * ( - c34 * tmp2 * u[3][k][j][i+1] );
      a[1][3][n] = dt * tx2 * ( u[3][k][j][i+1] * tmp1 );
      a[2][3][n] = 0.0;
      a[3][3][n] = dt * tx2 * ( u[1][k][j][i+1] * tmp1 )
        - dt * tx1 * ( c34 * tmp1 )
        - dt * tx1 * dx4;
      a[4][3][n] = 0.0;

      a[0][4][n] = dt * tx2
        * ( ( C2 * 2.0 * qs[k][j][i+1]
            - C1 * u[4][k][j][i+1] )
        * ( u[1][k][j][i+1] * tmp2 ) )
        - dt * tx1
        * ( - ( r43*c34 - c1345 ) * tmp3 * ( u[1][k][j][i+1]*u[1][k][j][i+1] )
            - (     c34 - c1345 ) * tmp3 * ( u[2][k][j][i+1]*u[2][k][j][i+1] )
            - (     c34 - c1345 ) * tmp3 * ( u[3][k][j][i+1]*u[3][k][j][i+1] )
            - c1345 * tmp2 * u[4][k][j][i+1] );
      a[1][4][n] = dt * tx2
        * ( C1 * ( u[4][k][j][i+1] * tmp1 )
            - C2
            * ( u[1][k][j][i+1]*u[1][k][j][i+1] * tmp2
              + qs[k][j][i+1] * tmp1 ) )
        - dt * tx1
        * ( r43*c34 - c1345 ) * tmp2 * u[1][k][j][i+1];
      a[2][4][n] = dt * tx2
        * ( - C2 * ( u[2][k][j][i+1]*u[1][k][j][i+1] ) * tmp2 )
        - dt * tx1
        * (  c34 - c1345 ) * tmp2 * u[2][k][j][i+1];
      a[3][4][n] = dt * tx2
        * ( - C2 * ( u[3][k][j][i+1]*u[1][k][j][i+1] ) * tmp2 )
        - dt * tx1
        * (  c34 - c1345 ) * tmp2 * u[3][k][j][i+1];
      a[4][4][n] = dt * tx2
        * ( C1 * ( u[1][k][j][i+1] * tmp1 ) )
        - dt * tx1 * c1345 * tmp1
        - dt * tx1 * dx5;

      tmp1 = rho_i[k][j+1][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      b[0][0][n] = - dt * ty1 * dy1;
      b[1][0][n] =   0.0;
      b[2][0][n] =  dt * ty2;
      b[3][0][n] =   0.0;
      b[4][0][n] =   0.0;

      b[0][1][n] =  dt * ty2
        * ( - ( u[1][k][j+1][i]*u[2][k][j+1][i] ) * tmp2 )
        - dt * ty1 * ( - c34 * tmp2 * u[1][k][j+1][i] );
      b[1][1][n] =  dt * ty2 * ( u[2][k][j+1][i] * tmp1 )
        - dt * ty1 * ( c34 * tmp1 )
        - dt * ty1 * dy2;
      b[2][1][n] =  dt * ty2 * ( u[1][k][j+1][i] * tmp1 );
      b[3][1][n] = 0.0;
      b[4][1][n] = 0.0;

      b[0][2][n] =  dt * ty2
        * ( - ( u[2][k][j+1][i] * tmp1 ) * ( u[2][k][j+1][i] * tmp1 )
            + C2 * ( qs[k][j+1][i] * tmp1 ) )
        - dt * ty1 * ( - r43 * c34 * tmp2 * u[2][k][j+1][i] );
      b[1][2][n] =  dt * ty2
        * ( - C2 * ( u[1][k][j+1][i] * tmp1 ) );
      b[2][2][n] =  dt * ty2 * ( ( 2.0 - C2 )
          * ( u[2][k][j+1][i] * tmp1 ) )
        - dt * ty1 * ( r43 * c34 * tmp1 )
        - dt * ty1 * dy3;
      b[3][2][n] =  dt * ty2
        * ( - C2 * ( u[3][k][j+1][i] * tmp1 ) );
      b[4][2][n] =  dt * ty2 * C2;

      b[0][3][n] =  dt * ty2
        * ( - ( u[2][k][j+1][i]*u[3][k][j+1][i] ) * tmp2 )
        - dt * ty1 * ( - c34 * tmp2 * u[3][k][j+1][i] );
      b[1][3][n] = 0.0;
      b[2][3][n] =  dt * ty2 * ( u[3][k][j+1][i] * tmp1 );
      b[3][3][n] =  dt * ty2 * ( u[2][k][j+1][i] * tmp1 )
        - dt * ty1 * ( c34 * tmp1 )
        - dt * ty1 * dy4;
      b[4][3][n] = 0.0;

      b[0][4][n] =  dt * ty2
        * ( ( C2 * 2.0 * qs[k][j+1][i]
            - C1 * u[4][k][j+1][i] )
        * ( u[2][k][j+1][i] * tmp2 ) )
        - dt * ty1
        * ( - (     c34 - c1345 )*tmp3*(u[1][k][j+1][i]*u[1][k][j+1][i])
            - ( r43*c34 - c1345 )*tmp3*(u[2][k][j+1][i]*u[2][k][j+1][i])
            - (     c34 - c1345 )*tmp3*(u[3][k][j+1][i]*u[3][k][j+1][i])
            - c1345*tmp2*u[4][k][j+1][i] );
      b[1][4][n] =  dt * ty2
        * ( - C2 * ( u[1][k][j+1][i]*u[2][k][j+1][i] ) * tmp2 )
        - dt * ty1
        * ( c34 - c1345 ) * tmp2 * u[1][k][j+1][i];
      b[2][4][n] =  dt * ty2
        * ( C1 * ( u[4][k][j+1][i] * tmp1 )
            - C2 
            * ( qs[k][j+1][i] * tmp1
              + u[2][k][j+1][i]*u[2][k][j+1][i] * tmp2 ) )
        - dt * ty1
        * ( r43*c34 - c1345 ) * tmp2 * u[2][k][j+1][i];
      b[3][4][n] =  dt * ty2
        * ( - C2 * ( u[2][k][j+1][i]*u[3][k][j+1][i] ) * tmp2 )
        - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[3][k][j+1][i];
      b[4][4][n] =  dt * ty2
        * ( C1 * ( u[2][k][j+1][i] * tmp1 ) )
        - dt * ty1 * c1345 * tmp1
        - dt * ty1 * dy5;

      tmp1 = rho_i[k+1][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      c[0][0][n] = - dt * tz1 * dz1;
      c[1][0][n] =   0.0;
      c[2][0][n] =   0.0;
      c[3][0][n] = dt * tz2;
      c[4][0][n] =   0.0;

      c[0][1][n] = dt * tz2
        * ( - ( u[1][k+1][j][i]*u[3][k+1][j][i] ) * tmp2 )
        - dt * tz1 * ( - c34 * tmp2 * u[1][k+1][j][i] );
      c[1][1][n] = dt * tz2 * ( u[3][k+1][j][i] * tmp1 )
        - dt * tz1 * c34 * tmp1
        - dt * tz1 * dz2;
      c[2][1][n] = 0.0;
      c[3][1][n] = dt * tz2 * ( u[1][k+1][j][i] * tmp1 );
      c[4][1][n] = 0.0;

      c[0][2][n] = dt * tz2
        * ( - ( u[2][k+1][j][i]*u[3][k+1][j][i] ) * tmp2 )
        - dt * tz1 * ( - c34 * tmp2 * u[2][k+1][j][i] );
      c[1][2][n] = 0.0;
      c[2][2][n] = dt * tz2 * ( u[3][k+1][j][i] * tmp1 )
        - dt * tz1 * ( c34 * tmp1 )
        - dt * tz1 * dz3;
      c[3][2][n] = dt * tz2 * ( u[2][k+1][j][i] * tmp1 );
      c[4][2][n] = 0.0;

      c[0][3][n] = dt * tz2
        * ( - ( u[3][k+1][j][i] * tmp1 ) * ( u[3][k+1][j][i] * tmp1 )
            + C2 * ( qs[k+1][j][i] * tmp1 ) )
        - dt * tz1 * ( - r43 * c34 * tmp2 * u[3][k+1][j][i] );
      c[1][3][n] = dt * tz2
        * ( - C2 * ( u[1][k+1][j][i] * tmp1 ) );
      c[2][3][n] = dt * tz2
        * ( - C2 * ( u[2][k+1][j][i] * tmp1 ) );
      c[3][3][n] = dt * tz2 * ( 2.0 - C2 )
        * ( u[3][k+1][j][i] * tmp1 )
        - dt * tz1 * ( r43 * c34 * tmp1 )
        - dt * tz1 * dz4;
      c[4][3][n] = dt * tz2 * C2;

      c[0][4][n] = dt * tz2
        * ( ( C2 * 2.0 * qs[k+1][j][i]
            - C1 * u[4][k+1][j][i] )
                 * ( u[3][k+1][j][i] * tmp2 ) )
        - dt * tz1
        * ( - ( c34 - c1345 ) * tmp3 * (u[1][k+1][j][i]*u[1][k+1][j][i])
            - ( c34 - c1345 ) * tmp3 * (u[2][k+1][j][i]*u[2][k+1][j][i])
            - ( r43*c34 - c1345 )* tmp3 * (u[3][k+1][j][i]*u[3][k+1][j][i])
            - c1345 * tmp2 * u[4][k+1][j][i] );
      c[1][4][n] = dt * tz2
        * ( - C2 * ( u[1][k+1][j][i]*u[3][k+1][j][i] ) * tmp2 )
        - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[1][k+1][j][i];
      c[2][4][n] = dt * tz2
        * ( - C2 * ( u[2][k+1][j][i]*u[3][k+1][j][i] ) * tmp2 )
        - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[2][k+1][j][i];
      c[3][4][n] = dt * tz2
        * ( C1 * ( u[4][k+1][j][i] * tmp1 )
            - C2
            * ( qs[k+1][j][i] * tmp1
              + u[3][k+1][j][i]*u[3][k+1][j][i] * tmp2 ) )
        - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u[3][k+1][j][i];
      c[4][4][n] = dt * tz2
        * ( C1 * ( u[3][k+1][j][i] * tmp1 ) )
        - dt * tz1 * c1345 * tmp1
        - dt * tz1 * dz5;
  }
 }
}

void l2norm (int ldx, int ldy, int ldz, int nx0, int ny0, int nz0,
     int ist, int iend, int jst, int jend)
{
  
  int i, j, k, m;
  double rsdnm0, rsdnm1, rsdnm2, rsdnm3, rsdnm4;
  
  rsdnm0 = (double)0.0;
  rsdnm1 = (double)0.0;
  rsdnm2 = (double)0.0;
  rsdnm3 = (double)0.0;
  rsdnm4 = (double)0.0;

{
  for (m = 0; m < 5; m++) {
    rsdnm[m] = 0.0;
  }

#ifdef __PGIC__
  for (k = 1; k < nz0-1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
			rsdnm0 = rsdnm0 + rsd[0][k][j][i] * rsd[0][k][j][i];
			rsdnm1 = rsdnm1 + rsd[1][k][j][i] * rsd[1][k][j][i];
			rsdnm2 = rsdnm2 + rsd[2][k][j][i] * rsd[2][k][j][i];
			rsdnm3 = rsdnm3 + rsd[3][k][j][i] * rsd[3][k][j][i];
			rsdnm4 = rsdnm4 + rsd[4][k][j][i] * rsd[4][k][j][i];
      }
    }
  }
#else
  for (k = 1; k < nz0-1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
			rsdnm0 = rsdnm0 + rsd[0][k][j][i] * rsd[0][k][j][i];
			rsdnm1 = rsdnm1 + rsd[1][k][j][i] * rsd[1][k][j][i];
			rsdnm2 = rsdnm2 + rsd[2][k][j][i] * rsd[2][k][j][i];
			rsdnm3 = rsdnm3 + rsd[3][k][j][i] * rsd[3][k][j][i];
			rsdnm4 = rsdnm4 + rsd[4][k][j][i] * rsd[4][k][j][i];
      }
    }
  }
#endif
  
  {
    rsdnm[0] = rsdnm0;
    rsdnm[1] = rsdnm1;
    rsdnm[2] = rsdnm2;
    rsdnm[3] = rsdnm3;
    rsdnm[4] = rsdnm4;
  }

  for (m = 0; m < 5; m++) {
    rsdnm[m] = sqrt ( rsdnm[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
  }
 }
}

void rhs()
{
  
  int i, j, k, m;
  double q;
  double tmp;
  double u21, u31, u41;
  double u21i, u31i, u41i, u51i;
  double u21j, u31j, u41j, u51j;
  double u21k, u31k, u41k, u51k;
  double u21im1, u31im1, u41im1, u51im1;
  double u21jm1, u31jm1, u41jm1, u51jm1;
  double u21km1, u31km1, u41km1, u51km1;
  unsigned num_workers3 = 0;
  unsigned num_workers2 = 0;

{
  if (timeron) timer_start(t_rhs);
  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][i] = - frct[m][k][j][i];
        }
        tmp = 1.0 / u[0][k][j][i];
        rho_i[k][j][i] = tmp;
        qs[k][j][i] = 0.50 * (  u[1][k][j][i] * u[1][k][j][i]
                              + u[2][k][j][i] * u[2][k][j][i]
                              + u[3][k][j][i] * u[3][k][j][i] )
                           * tmp;
      }
    }
  }

  if (timeron) timer_start(t_rhsx);
  if(((jend-jst+1))<32)
	num_workers3 = (jend-jst+1);
  else
	num_workers3 = 16;
  
  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = 0; i < nx; i++) {
        flux_G[0][k][j][i] = u[1][k][j][i];
        u21 = u[1][k][j][i] * rho_i[k][j][i];

        q = qs[k][j][i];

        flux_G[1][k][j][i] = u[1][k][j][i] * u21 + C2 * ( u[4][k][j][i] - q );
        flux_G[2][k][j][i] = u[2][k][j][i] * u21;
        flux_G[3][k][j][i] = u[3][k][j][i] * u21;
        flux_G[4][k][j][i] = ( C1 * u[4][k][j][i] - C2 * q ) * u21;
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][i] =  rsd[m][k][j][i]
            - tx2 * ( flux_G[m][k][j][i+1] - flux_G[m][k][j][i-1] );
        }
      }
    }
  }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = ist; i < nx; i++) {
        tmp = rho_i[k][j][i];

        u21i = tmp * u[1][k][j][i];
        u31i = tmp * u[2][k][j][i];
        u41i = tmp * u[3][k][j][i];
        u51i = tmp * u[4][k][j][i];

        tmp = rho_i[k][j][i-1];

        u21im1 = tmp * u[1][k][j][i-1];
        u31im1 = tmp * u[2][k][j][i-1];
        u41im1 = tmp * u[3][k][j][i-1];
        u51im1 = tmp * u[4][k][j][i-1];

        flux_G[1][k][j][i] = (4.0/3.0) * tx3 * (u21i-u21im1);
        flux_G[2][k][j][i] = tx3 * ( u31i - u31im1 );
        flux_G[3][k][j][i] = tx3 * ( u41i - u41im1 );
        flux_G[4][k][j][i] = 0.50 * ( 1.0 - C1*C5 )
          * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
                  - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
          + (1.0/6.0)
          * tx3 * ( u21i*u21i - u21im1*u21im1 )
          + C1 * C5 * tx3 * ( u51i - u51im1 );
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        rsd[0][k][j][i] = rsd[0][k][j][i]
          + dx1 * tx1 * (        u[0][k][j][i-1]
                         - 2.0 * u[0][k][j][i]
                         +       u[0][k][j][i+1] );
        rsd[1][k][j][i] = rsd[1][k][j][i]
          + tx3 * C3 * C4 * ( flux_G[1][k][j][i+1] - flux_G[1][k][j][i] )
          + dx2 * tx1 * (        u[1][k][j][i-1]
                         - 2.0 * u[1][k][j][i]
                         +       u[1][k][j][i+1] );
        rsd[2][k][j][i] = rsd[2][k][j][i]
          + tx3 * C3 * C4 * ( flux_G[2][k][j][i+1] - flux_G[2][k][j][i] )
          + dx3 * tx1 * (        u[2][k][j][i-1]
                         - 2.0 * u[2][k][j][i]
                         +       u[2][k][j][i+1] );
        rsd[3][k][j][i] = rsd[3][k][j][i]
          + tx3 * C3 * C4 * ( flux_G[3][k][j][i+1] - flux_G[3][k][j][i] )
          + dx4 * tx1 * (        u[3][k][j][i-1]
                         - 2.0 * u[3][k][j][i]
                         +       u[3][k][j][i+1] );
        rsd[4][k][j][i] = rsd[4][k][j][i]
          + tx3 * C3 * C4 * ( flux_G[4][k][j][i+1] - flux_G[4][k][j][i] )
          + dx5 * tx1 * (        u[4][k][j][i-1]
                         - 2.0 * u[4][k][j][i]
                         +       u[4][k][j][i+1] );
       }
     }
   }

  if(((jend-jst+1)/32)<32)
     num_workers2 = (jend-jst+1)/32;
  else
     num_workers2 = 16;
  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (m = 0; m < 5; m++) {
        rsd[m][k][j][1] = rsd[m][k][j][1]
          - dssp * ( + 5.0 * u[m][k][j][1]
                     - 4.0 * u[m][k][j][2]
                     +       u[m][k][j][3] );
        rsd[m][k][j][2] = rsd[m][k][j][2]
          - dssp * ( - 4.0 * u[m][k][j][1]
                     + 6.0 * u[m][k][j][2]
                     - 4.0 * u[m][k][j][3]
                     +       u[m][k][j][4] );
      }
    }
  }
  
  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (i = 3; i < nx - 3; i++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][i] = rsd[m][k][j][i]
            - dssp * (         u[m][k][j][i-2]
                       - 4.0 * u[m][k][j][i-1]
                       + 6.0 * u[m][k][j][i]
                       - 4.0 * u[m][k][j][i+1]
                       +       u[m][k][j][i+2] );
        }
      }
    }
  }

  for (k = 1; k < nz - 1; k++) {
    for (j = jst; j <= jend; j++) {
      for (m = 0; m < 5; m++) {
        rsd[m][k][j][nx-3] = rsd[m][k][j][nx-3]
          - dssp * (         u[m][k][j][nx-5]
                     - 4.0 * u[m][k][j][nx-4]
                     + 6.0 * u[m][k][j][nx-3]
                     - 4.0 * u[m][k][j][nx-2] );
        rsd[m][k][j][nx-2] = rsd[m][k][j][nx-2]
          - dssp * (         u[m][k][j][nx-4]
                     - 4.0 * u[m][k][j][nx-3]
                     + 5.0 * u[m][k][j][nx-2] );
      }

    }
  }
  if (timeron) timer_stop(t_rhsx);

  if (timeron) timer_start(t_rhsy);
  
  if(((jend-jst+1))<32)
     num_workers3 = (iend-ist+1);
  else
     num_workers3 = 16;
  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (j = 0; j < ny; j++) {
        flux_G[0][k][i][j] = u[2][k][j][i];
        u31 = u[2][k][j][i] * rho_i[k][j][i];

        q = qs[k][j][i];

        flux_G[1][k][i][j] = u[1][k][j][i] * u31;
        flux_G[2][k][i][j] = u[2][k][j][i] * u31 + C2 * (u[4][k][j][i]-q);
        flux_G[3][k][i][j] = u[3][k][j][i] * u31;
        flux_G[4][k][i][j] = ( C1 * u[4][k][j][i] - C2 * q ) * u31;
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (j = jst; j <= jend; j++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][i] =  rsd[m][k][j][i]
            - ty2 * ( flux_G[m][k][i][j+1] - flux_G[m][k][i][j-1] );
        }
      }
     }
   }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (j = jst; j < ny; j++) {
        tmp = rho_i[k][j][i];

        u21j = tmp * u[1][k][j][i];
        u31j = tmp * u[2][k][j][i];
        u41j = tmp * u[3][k][j][i];
        u51j = tmp * u[4][k][j][i];

        tmp = rho_i[k][j-1][i];
        u21jm1 = tmp * u[1][k][j-1][i];
        u31jm1 = tmp * u[2][k][j-1][i];
        u41jm1 = tmp * u[3][k][j-1][i];
        u51jm1 = tmp * u[4][k][j-1][i];

        flux_G[1][k][i][j] = ty3 * ( u21j - u21jm1 );
        flux_G[2][k][i][j] = (4.0/3.0) * ty3 * (u31j-u31jm1);
        flux_G[3][k][i][j] = ty3 * ( u41j - u41jm1 );
        flux_G[4][k][i][j] = 0.50 * ( 1.0 - C1*C5 )
          * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
                  - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
          + (1.0/6.0)
          * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
          + C1 * C5 * ty3 * ( u51j - u51jm1 );
      }
     }
    }

  for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
      for (j = jst; j <= jend; j++) {
        rsd[0][k][j][i] = rsd[0][k][j][i]
          + dy1 * ty1 * (         u[0][k][j-1][i]
                          - 2.0 * u[0][k][j][i]
                          +       u[0][k][j+1][i] );

        rsd[1][k][j][i] = rsd[1][k][j][i]
          + ty3 * C3 * C4 * ( flux_G[1][k][i][j+1] - flux_G[1][k][i][j] )
          + dy2 * ty1 * (         u[1][k][j-1][i]
                          - 2.0 * u[1][k][j][i]
                          +       u[1][k][j+1][i] );

        rsd[2][k][j][i] = rsd[2][k][j][i]
          + ty3 * C3 * C4 * ( flux_G[2][k][i][j+1] - flux_G[2][k][i][j] )
          + dy3 * ty1 * (         u[2][k][j-1][i]
                          - 2.0 * u[2][k][j][i]
                          +       u[2][k][j+1][i] );

        rsd[3][k][j][i] = rsd[3][k][j][i]
          + ty3 * C3 * C4 * ( flux_G[3][k][i][j+1] - flux_G[3][k][i][j] )
          + dy4 * ty1 * (         u[3][k][j-1][i]
                          - 2.0 * u[3][k][j][i]
                          +       u[3][k][j+1][i] );

        rsd[4][k][j][i] = rsd[4][k][j][i]
          + ty3 * C3 * C4 * ( flux_G[4][k][i][j+1] - flux_G[4][k][i][j] )
          + dy5 * ty1 * (         u[4][k][j-1][i]
                          - 2.0 * u[4][k][j][i]
                          +       u[4][k][j+1][i] );
      }
    }
  }

  if(((jend-jst+1)/32)<32)
     num_workers2 = (iend-ist+1)/32;
  else
     num_workers2 = 16;
  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[m][k][1][i] = rsd[m][k][1][i]
          - dssp * ( + 5.0 * u[m][k][1][i]
                     - 4.0 * u[m][k][2][i]
                     +       u[m][k][3][i] );
        rsd[m][k][2][i] = rsd[m][k][2][i]
          - dssp * ( - 4.0 * u[m][k][1][i]
                     + 6.0 * u[m][k][2][i]
                     - 4.0 * u[m][k][3][i]
                     +       u[m][k][4][i] );
      }
    }
  }

  unsigned int num_workers4 = 0;
  if((ny-6)<8)
  	num_workers4 = ny-6;
  else
  	num_workers4 = 4;
  	
  for (k = 1; k < nz - 1; k++) {
    for (j = 3; j < ny - 3; j++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][i] = rsd[m][k][j][i]
            - dssp * (         u[m][k][j-2][i]
                       - 4.0 * u[m][k][j-1][i]
                       + 6.0 * u[m][k][j][i]
                       - 4.0 * u[m][k][j+1][i]
                       +       u[m][k][j+2][i] );
        }
      }
    }
  }

  for (k = 1; k < nz - 1; k++) {
    for (i = ist; i <= iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[m][k][ny-3][i] = rsd[m][k][ny-3][i]
          - dssp * (         u[m][k][ny-5][i]
                     - 4.0 * u[m][k][ny-4][i]
                     + 6.0 * u[m][k][ny-3][i]
                     - 4.0 * u[m][k][ny-2][i] );
        rsd[m][k][ny-2][i] = rsd[m][k][ny-2][i]
          - dssp * (         u[m][k][ny-4][i]
                     - 4.0 * u[m][k][ny-3][i]
                     + 5.0 * u[m][k][ny-2][i] );
      }
    }

  }
  if (timeron) timer_stop(t_rhsy);

  if (timeron) timer_start(t_rhsz);
  
  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 0; k < nz; k++) {
        utmp_G[0][j][i][k] = u[0][k][j][i];
        utmp_G[1][j][i][k] = u[1][k][j][i];
        utmp_G[2][j][i][k] = u[2][k][j][i];
        utmp_G[3][j][i][k] = u[3][k][j][i];
        utmp_G[4][j][i][k] = u[4][k][j][i];
        utmp_G[5][j][i][k] = rho_i[k][j][i];
      }
    }
  }
  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 0; k < nz; k++) {
        flux_G[0][j][i][k] = utmp_G[3][j][i][k];
        u41 = utmp_G[3][j][i][k] * utmp_G[5][j][i][k];

        q = qs[k][j][i];

        flux_G[1][j][i][k] = utmp_G[1][j][i][k] * u41;
        flux_G[2][j][i][k] = utmp_G[2][j][i][k] * u41;
        flux_G[3][j][i][k] = utmp_G[3][j][i][k] * u41 + C2 * (utmp_G[4][j][i][k]-q);
        flux_G[4][j][i][k] = ( C1 * utmp_G[4][j][i][k] - C2 * q ) * u41;
      }
     }
    }

  for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 1; k < nz - 1; k++) {
        for (m = 0; m < 5; m++) {
          rtmp_G[m][j][i][k] =  rsd[m][k][j][i]
            - tz2 * ( flux_G[m][j][i][k+1] - flux_G[m][j][i][k-1] );
        }
      }
     }
    }

   for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 1; k < nz; k++) {
        tmp = utmp_G[5][j][i][k];

        u21k = tmp * utmp_G[1][j][i][k];
        u31k = tmp * utmp_G[2][j][i][k];
        u41k = tmp * utmp_G[3][j][i][k];
        u51k = tmp * utmp_G[4][j][i][k];

        tmp = utmp_G[5][j][i][k-1];

        u21km1 = tmp * utmp_G[1][j][i][k-1];
        u31km1 = tmp * utmp_G[2][j][i][k-1];
        u41km1 = tmp * utmp_G[3][j][i][k-1];
        u51km1 = tmp * utmp_G[4][j][i][k-1];

        flux_G[1][j][i][k] = tz3 * ( u21k - u21km1 );
        flux_G[2][j][i][k] = tz3 * ( u31k - u31km1 );
        flux_G[3][j][i][k] = (4.0/3.0) * tz3 * (u41k-u41km1);
        flux_G[4][j][i][k] = 0.50 * ( 1.0 - C1*C5 )
          * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
                  - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
          + (1.0/6.0)
          * tz3 * ( u41k*u41k - u41km1*u41km1 )
          + C1 * C5 * tz3 * ( u51k - u51km1 );
      }
     }
    }

   for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 1; k < nz - 1; k++) {
        rtmp_G[0][j][i][k] = rtmp_G[0][j][i][k]
          + dz1 * tz1 * (         utmp_G[0][j][i][k-1]
                          - 2.0 * utmp_G[0][j][i][k]
                          +       utmp_G[0][j][i][k+1] );
        rtmp_G[1][j][i][k] = rtmp_G[1][j][i][k]
          + tz3 * C3 * C4 * ( flux_G[1][j][i][k+1] - flux_G[1][j][i][k] )
          + dz2 * tz1 * (         utmp_G[1][j][i][k-1]
                          - 2.0 * utmp_G[1][j][i][k]
                          +       utmp_G[1][j][i][k+1] );
        rtmp_G[2][j][i][k] = rtmp_G[2][j][i][k]
          + tz3 * C3 * C4 * ( flux_G[2][j][i][k+1] - flux_G[2][j][i][k] )
          + dz3 * tz1 * (         utmp_G[2][j][i][k-1]
                          - 2.0 * utmp_G[2][j][i][k]
                          +       utmp_G[2][j][i][k+1] );
        rtmp_G[3][j][i][k] = rtmp_G[3][j][i][k]
          + tz3 * C3 * C4 * ( flux_G[3][j][i][k+1] - flux_G[3][j][i][k] )
          + dz4 * tz1 * (         utmp_G[3][j][i][k-1]
                          - 2.0 * utmp_G[3][j][i][k]
                          +       utmp_G[3][j][i][k+1] );
        rtmp_G[4][j][i][k] = rtmp_G[4][j][i][k]
          + tz3 * C3 * C4 * ( flux_G[4][j][i][k+1] - flux_G[4][j][i][k] )
          + dz5 * tz1 * (         utmp_G[4][j][i][k-1]
                          - 2.0 * utmp_G[4][j][i][k]
                          +       utmp_G[4][j][i][k+1] );
      }
     }
    }

   for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[m][1][j][i] = rtmp_G[m][j][i][1]
          - dssp * ( + 5.0 * utmp_G[m][j][i][1]
                     - 4.0 * utmp_G[m][j][i][2]
                     +       utmp_G[m][j][i][3] );
        rsd[m][2][j][i] = rtmp_G[m][j][i][2]
          - dssp * ( - 4.0 * utmp_G[m][j][i][1]
                     + 6.0 * utmp_G[m][j][i][2]
                     - 4.0 * utmp_G[m][j][i][3]
                     +       utmp_G[m][j][i][4] );
      }
     }
    }

   for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (k = 3; k < nz - 3; k++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][i] = rtmp_G[m][j][i][k]
            - dssp * (         utmp_G[m][j][i][k-2]
                       - 4.0 * utmp_G[m][j][i][k-1]
                       + 6.0 * utmp_G[m][j][i][k]
                       - 4.0 * utmp_G[m][j][i][k+1]
                       +       utmp_G[m][j][i][k+2] );
        }
      }
     }
    }

   for (j = jst; j <= jend; j++) {
    for (i = ist; i <= iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[m][nz-3][j][i] = rtmp_G[m][j][i][nz-3]
          - dssp * (         utmp_G[m][j][i][nz-5]
                     - 4.0 * utmp_G[m][j][i][nz-4]
                     + 6.0 * utmp_G[m][j][i][nz-3]
                     - 4.0 * utmp_G[m][j][i][nz-2] );
        rsd[m][nz-2][j][i] = rtmp_G[m][j][i][nz-2]
          - dssp * (         utmp_G[m][j][i][nz-4]
                     - 4.0 * utmp_G[m][j][i][nz-3]
                     + 5.0 * utmp_G[m][j][i][nz-2] );
      }
    }
  }
  if (timeron) timer_stop(t_rhsz);
  if (timeron) timer_stop(t_rhs);
}
}

void ssor(int niter)
{
  
  int i, j, k, m, n;
  int istep;
  double tmp;
  double delunm[5];

  int l, lst, lend;
  unsigned int num_workers = 0, num_gangs = 0;

  tmp = 1.0 / ( omega * ( 2.0 - omega ) );
  lst = ist + jst + 1;
  lend = iend + jend + nz -2;

  num_gangs = (ISIZ1*ISIZ2)/512;
  for (l = 0; l < ISIZ1*ISIZ2; l++) {
      for (n = 0; n < 5; n++) {
        for (m = 0; m < 5; m++) {
          a[n][m][l] = 0.0;
          b[n][m][l] = 0.0;
          c[n][m][l] = 0.0;
          d[n][m][l] = 0.0;
        }
      }
  }
  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }

  rhs();

  l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
          ist, iend, jst, jend);

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }

	calcnp(lst, lend);
  timer_start(1);
  
  for (istep = 1; istep <= niter; istep++) {
    
    if ((istep % 20) == 0 || istep == itmax || istep == 1) {
      if (niter > 1) printf(" Time step %4d\n", istep);
    }

    if (timeron) timer_start(t_rhs);
    if((jend-jst+1)<32)
    	num_workers = jend-jst+1;
    else
    	num_workers = 16;
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = ist; i <= iend; i++) {
          for (m = 0; m < 5; m++) {
            rsd[m][k][j][i] = dt * rsd[m][k][j][i];
          }
        }
      }
    }
    if (timeron) timer_stop(t_rhs);

	for(l = lst; l <= lend; l++)
	{
		jacld(l);

		blts(ISIZ1, ISIZ2, ISIZ3, 
			 nx, ny, nz, l,
			 omega); 
	}

	for(l = lend; l >= lst; l--)
	{
		
		jacu(l);

		buts(ISIZ1, ISIZ2, ISIZ3, 
			 nx, ny, nz, l,
			 omega);
	}

    if (timeron) timer_start(t_add);
    for (k = 1; k < nz-1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = ist; i <= iend; i++) {
          
          {
            u[0][k][j][i] = u[0][k][j][i] + tmp * rsd[0][k][j][i];
	    u[1][k][j][i] = u[1][k][j][i] + tmp * rsd[1][k][j][i];
	    u[2][k][j][i] = u[2][k][j][i] + tmp * rsd[2][k][j][i];
	    u[3][k][j][i] = u[3][k][j][i] + tmp * rsd[3][k][j][i];
	    u[4][k][j][i] = u[4][k][j][i] + tmp * rsd[4][k][j][i];
          }
        }
      }
    }
    if (timeron) timer_stop(t_add);

    if ( (istep % inorm) == 0 ) {
      if (timeron) timer_start(t_l2norm);
	  
      if (timeron) timer_stop(t_l2norm);
      
    }
 
    rhs();
 
    if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {
      if (timeron) timer_start(t_l2norm);
      l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
              ist, iend, jst, jend);
      if (timeron) timer_stop(t_l2norm);
      
    }

    if ( ( rsdnm[0] < tolrsd[0] ) && ( rsdnm[1] < tolrsd[1] ) &&
         ( rsdnm[2] < tolrsd[2] ) && ( rsdnm[3] < tolrsd[3] ) &&
         ( rsdnm[4] < tolrsd[4] ) ) {
      
      printf(" \n convergence was achieved after %4d pseudo-time steps\n",
          istep);
      
      break;
    }
  }

  timer_stop(1);
  maxtime = timer_read(1);
}