#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <chrono>
#include "gate.h"

#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400

#define SQ(x) ((x)*(x))

typedef double nRarray[DATAYSIZE][DATAXSIZE];

// Tune OpenMP team geometry for NVIDIA SM 8.9 (RTX 4070 Laptop) GPUs.
static constexpr int OMP_TEAM_SIZE = 256;  // 8 warps per team balances latency and register pressure.
static constexpr int DOMAIN_POINTS = DATAXSIZE * DATAYSIZE * DATAZSIZE;
static constexpr int INTERIOR_POINTS =
    (DATAXSIZE - 2) * (DATAYSIZE - 2) * (DATAZSIZE - 2);
static constexpr int TEAMS_DOMAIN =
    (DOMAIN_POINTS + OMP_TEAM_SIZE - 1) / OMP_TEAM_SIZE;
static constexpr int TEAMS_INTERIOR =
    (INTERIOR_POINTS + OMP_TEAM_SIZE - 1) / OMP_TEAM_SIZE;

#ifdef VERIFY
#include <string.h>
#include "reference.h"
#endif

#pragma omp declare target

double dFphi(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}

double GradientX(double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x+1][y][z] - phi[x-1][y][z]) / (2.0*dx);
}

double GradientY(double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y+1][z] - phi[x][y-1][z]) / (2.0*dy);
}

double GradientZ(double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y][z+1] - phi[x][y][z-1]) / (2.0*dz);
}

double Divergence(double phix[][DATAYSIZE][DATAXSIZE],
                  double phiy[][DATAYSIZE][DATAXSIZE],
                  double phiz[][DATAYSIZE][DATAXSIZE],
                  double dx, double dy, double dz, int x, int y, int z)
{
  return GradientX(phix,dx,dy,dz,x,y,z) +
         GradientY(phiy,dx,dy,dz,x,y,z) +
         GradientZ(phiz,dx,dy,dz,x,y,z);
}

double Laplacian(double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  double phixx = (phi[x+1][y][z] + phi[x-1][y][z] - 2.0 * phi[x][y][z]) / SQ(dx);
  double phiyy = (phi[x][y+1][z] + phi[x][y-1][z] - 2.0 * phi[x][y][z]) / SQ(dy);
  double phizz = (phi[x][y][z+1] + phi[x][y][z-1] - 2.0 * phi[x][y][z]) / SQ(dz);
  return phixx + phiyy + phizz;
}

double An(double phix, double phiy, double phiz, double epsilon)
{
  if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
    return ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon))*
           ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
           ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
  }
  else
  {
    return (1.0-((5.0/3.0)*epsilon));
  }
}

double Wn(double phix, double phiy, double phiz, double epsilon, double W0)
{
  return (W0*An(phix,phiy,phiz,epsilon));
}

double taun(double phix, double phiy, double phiz, double epsilon, double tau0)
{
  return tau0 * SQ(An(phix,phiy,phiz,epsilon));
}

double dFunc(double l, double m, double n)
{
  if (l != 0.0 || m != 0.0 || n != 0.0){
    return (((l*l*l*(SQ(m)+SQ(n)))-(l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
            ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
  }
  else
  {
    return 0.0;
  }
}

#pragma omp end declare target

void calculateForce(double phi[][DATAYSIZE][DATAXSIZE],
                    double Fx[][DATAYSIZE][DATAXSIZE],
                    double Fy[][DATAYSIZE][DATAXSIZE],
                    double Fz[][DATAYSIZE][DATAXSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0)
{
  // Precompute invariant coefficients to reduce repeated divisions device-side.
  // Reuse invariant geometry factors across the interior update.
  // Reuse invariant geometry factors across the interior update.
  const double inv2dx = 0.5 / dx;
  const double inv2dy = 0.5 / dy;
  const double inv2dz = 0.5 / dz;
  const double anis_base = 1.0 - 3.0 * epsilon;
  const double anis_zero = 1.0 - (5.0 / 3.0) * epsilon;
  const double epsilon_four = 4.0 * epsilon;
  const double force_scale = 16.0 * W0 * epsilon;

  // Offload force calculation stencil across the 3-D lattice.
#pragma omp target teams distribute parallel for collapse(3) \
        num_teams(TEAMS_DOMAIN) thread_limit(OMP_TEAM_SIZE) \
        map(present: phi[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         Fx[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         Fy[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         Fz[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE])
    for (int ix = 0; ix < DATAXSIZE; ix++) {
      for (int iy = 0; iy < DATAYSIZE; iy++) {
        for (int iz = 0; iz < DATAZSIZE; iz++) {

        if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) &&
            (iz < (DATAZSIZE-1)) && (ix > (0)) &&
            (iy > (0)) && (iz > (0))) {

          const double phi_xp = phi[ix+1][iy][iz];
          const double phi_xm = phi[ix-1][iy][iz];
          const double phi_yp = phi[ix][iy+1][iz];
          const double phi_ym = phi[ix][iy-1][iz];
          const double phi_zp = phi[ix][iy][iz+1];
          const double phi_zm = phi[ix][iy][iz-1];

          const double phix = (phi_xp - phi_xm) * inv2dx;
          const double phiy = (phi_yp - phi_ym) * inv2dy;
          const double phiz = (phi_zp - phi_zm) * inv2dz;

          const double phix2 = phix * phix;
          const double phiy2 = phiy * phiy;
          const double phiz2 = phiz * phiz;
          const double grad2 = phix2 + phiy2 + phiz2;

          double an = anis_zero;
          double w = W0 * an;
          double w2 = w * w;
          double dfx = 0.0;
          double dfy = 0.0;
          double dfz = 0.0;

          if (grad2 > 0.0) {
            const double phix4 = phix2 * phix2;
            const double phiy4 = phiy2 * phiy2;
            const double phiz4 = phiz2 * phiz2;
            const double grad4 = grad2 * grad2;
            const double grad4_inv = 1.0 / grad4;
            const double phix3 = phix * phix2;
            const double phiy3 = phiy * phiy2;
            const double phiz3 = phiz * phiz2;

            an = anis_base + epsilon_four * (phix4 + phiy4 + phiz4) * grad4_inv;
            w = W0 * an;
            w2 = w * w;

            const double sum_yz = phiy2 + phiz2;
            const double sum_zx = phiz2 + phix2;
            const double sum_xy = phix2 + phiy2;
            const double phiy4_plus = phiy4 + phiz4;
            const double phiz4_plus = phiz4 + phix4;
            const double phix4_plus = phix4 + phiy4;

            dfx = (phix3 * sum_yz - phix * phiy4_plus) * grad4_inv;
            dfy = (phiy3 * sum_zx - phiy * phiz4_plus) * grad4_inv;
            dfz = (phiz3 * sum_xy - phiz * phix4_plus) * grad4_inv;
          }
          else
          {
            Fx[ix][iy][iz] = 0.0;
            Fy[ix][iy][iz] = 0.0;
            Fz[ix][iy][iz] = 0.0;
            continue;
          }

          const double grad_term = grad2 * w * force_scale;

          Fx[ix][iy][iz] = w2 * phix + grad_term * dfx;
          Fy[ix][iy][iz] = w2 * phiy + grad_term * dfy;
          Fz[ix][iy][iz] = w2 * phiz + grad_term * dfz;
        }
        else
        {
          Fx[ix][iy][iz] = 0.0;
          Fy[ix][iy][iz] = 0.0;
          Fz[ix][iy][iz] = 0.0;
        }
      }
    }
  }
}

void allenCahn(double phinew[][DATAYSIZE][DATAXSIZE],
               double phiold[][DATAYSIZE][DATAXSIZE],
               double uold[][DATAYSIZE][DATAXSIZE],
               double Fx[][DATAYSIZE][DATAXSIZE],
               double Fy[][DATAYSIZE][DATAXSIZE],
               double Fz[][DATAYSIZE][DATAXSIZE],
               double epsilon, double W0, double tau0, double lambda,
               double dt, double dx, double dy, double dz)
{
  const double inv2dx = 0.5 / dx;
  const double inv2dy = 0.5 / dy;
  const double inv2dz = 0.5 / dz;
  const double anis_base = 1.0 - 3.0 * epsilon;
  const double anis_zero = 1.0 - (5.0 / 3.0) * epsilon;
  const double epsilon_four = 4.0 * epsilon;
  // Offload Allen-Cahn update for interior voxels.
#pragma omp target teams distribute parallel for collapse(3) \
        num_teams(TEAMS_INTERIOR) thread_limit(OMP_TEAM_SIZE) \
        map(present: phinew[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         phiold[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         uold[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         Fx[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         Fy[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         Fz[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE])
    for (int ix = 1; ix < DATAXSIZE-1; ix++) {
      for (int iy = 1; iy < DATAYSIZE-1; iy++) {
        for (int iz = 1; iz < DATAZSIZE-1; iz++) {

          const double phi_xp = phiold[ix+1][iy][iz];
          const double phi_xm = phiold[ix-1][iy][iz];
          const double phi_yp = phiold[ix][iy+1][iz];
          const double phi_ym = phiold[ix][iy-1][iz];
          const double phi_zp = phiold[ix][iy][iz+1];
          const double phi_zm = phiold[ix][iy][iz-1];

          const double phix = (phi_xp - phi_xm) * inv2dx;
          const double phiy = (phi_yp - phi_ym) * inv2dy;
          const double phiz = (phi_zp - phi_zm) * inv2dz;

          const double phix2 = phix * phix;
          const double phiy2 = phiy * phiy;
          const double phiz2 = phiz * phiz;
          const double grad2 = phix2 + phiy2 + phiz2;

          double an = anis_zero;
          if (grad2 > 0.0) {
            const double phix4 = phix2 * phix2;
            const double phiy4 = phiy2 * phiy2;
            const double phiz4 = phiz2 * phiz2;
            const double grad4 = grad2 * grad2;
            an = anis_base + epsilon_four * (phix4 + phiy4 + phiz4) / grad4;
          }

          const double tau_val = tau0 * an * an;
          const double inv_tau = dt / tau_val;

          const double fxp = Fx[ix+1][iy][iz];
          const double fxm = Fx[ix-1][iy][iz];
          const double fyp = Fy[ix][iy+1][iz];
          const double fym = Fy[ix][iy-1][iz];
          const double fzp = Fz[ix][iy][iz+1];
          const double fzm = Fz[ix][iy][iz-1];

          const double divergence =
              (fxp - fxm) * inv2dx +
              (fyp - fym) * inv2dy +
              (fzp - fzm) * inv2dz;

          const double phi_c = phiold[ix][iy][iz];
          const double u_c = uold[ix][iy][iz];
          const double phi_sq = phi_c * phi_c;
          const double one_minus_phi_sq = 1.0 - phi_sq;
          const double reaction =
              -phi_c * one_minus_phi_sq +
              lambda * u_c * one_minus_phi_sq * one_minus_phi_sq;

          phinew[ix][iy][iz] = phi_c + inv_tau * (divergence - reaction);
        }
      }
    }
}

void boundaryConditionsPhi(double phinew[][DATAYSIZE][DATAXSIZE])
{
#pragma omp target teams distribute parallel for collapse(3) \
        num_teams(TEAMS_DOMAIN) thread_limit(OMP_TEAM_SIZE) \
        map(present: phinew[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE])
    for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {

        if (ix == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (ix == DATAXSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iy == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iy == DATAYSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iz == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iz == DATAZSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
      }
    }
  }
}

void thermalEquation(double unew[][DATAYSIZE][DATAXSIZE],
                     double uold[][DATAYSIZE][DATAXSIZE],
                     double phinew[][DATAYSIZE][DATAXSIZE],
                     double phiold[][DATAYSIZE][DATAXSIZE],
                     double D, double dt, double dx, double dy, double dz)
{
  // Cache diffusion coefficients to limit per-voxel arithmetic.
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const double inv_dz2 = 1.0 / (dz * dz);
  const double diffusion_scale = dt * D;
  // Offload thermal diffusion step for interior voxels.
#pragma omp target teams distribute parallel for collapse(3) \
        num_teams(TEAMS_INTERIOR) thread_limit(OMP_TEAM_SIZE) \
        map(present: unew[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         uold[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         phinew[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         phiold[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE])
    for (int ix = 1; ix < DATAXSIZE-1; ix++) {
      for (int iy = 1; iy < DATAYSIZE-1; iy++) {
        for (int iz = 1; iz < DATAZSIZE-1; iz++) {

          const double u_c = uold[ix][iy][iz];
          const double u_xp = uold[ix+1][iy][iz];
          const double u_xm = uold[ix-1][iy][iz];
          const double u_yp = uold[ix][iy+1][iz];
          const double u_ym = uold[ix][iy-1][iz];
          const double u_zp = uold[ix][iy][iz+1];
          const double u_zm = uold[ix][iy][iz-1];

          const double lap =
              (u_xp + u_xm - 2.0 * u_c) * inv_dx2 +
              (u_yp + u_ym - 2.0 * u_c) * inv_dy2 +
              (u_zp + u_zm - 2.0 * u_c) * inv_dz2;

          const double phase_delta = 0.5 * (phinew[ix][iy][iz] - phiold[ix][iy][iz]);

          unew[ix][iy][iz] = u_c + phase_delta + diffusion_scale * lap;
        }
      }
    }
}

void boundaryConditionsU(double unew[][DATAYSIZE][DATAXSIZE], double delta)
{
#pragma omp target teams distribute parallel for collapse(3) \
        num_teams(TEAMS_DOMAIN) thread_limit(OMP_TEAM_SIZE) \
        map(present: unew[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE])
    for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {

        if (ix == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (ix == DATAXSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iy == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iy == DATAYSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iz == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iz == DATAZSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
      }
    }
  }
}

void swapGrid(double cnew[][DATAYSIZE][DATAXSIZE],
              double cold[][DATAYSIZE][DATAXSIZE])
{
#pragma omp target teams distribute parallel for collapse(3) \
        num_teams(TEAMS_DOMAIN) thread_limit(OMP_TEAM_SIZE) \
        map(present: cnew[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                         cold[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE])
    for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {
        double tmp = cnew[ix][iy][iz];
        cnew[ix][iy][iz] = cold[ix][iy][iz];
        cold[ix][iy][iz] = tmp;
      }
    }
  }
}

void initializationPhi(double phi[][DATAYSIZE][DATAXSIZE], double r0)
{
    for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {
        double r = std::sqrt(SQ(ix-0.5*DATAXSIZE) + SQ(iy-0.5*DATAYSIZE) + SQ(iz-0.5*DATAZSIZE));
        if (r < r0){
          phi[ix][iy][iz] = 1.0;
        }
        else
        {
          phi[ix][iy][iz] = -1.0;
        }
      }
    }
  }
}

void initializationU(double u[][DATAYSIZE][DATAXSIZE], double r0, double delta)
{
    for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {
        double r = std::sqrt(SQ(ix-0.5*DATAXSIZE) + SQ(iy-0.5*DATAYSIZE) + SQ(iz-0.5*DATAZSIZE));
        if (r < r0) {
          u[ix][iy][iz] = 0.0;
        }
        else
        {
          u[ix][iy][iz] = -delta * (1.0 - std::exp(-(r-r0)));
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  const int num_steps = atoi(argv[1]);

  const double dx = 0.4;
  const double dy = 0.4;
  const double dz = 0.4;
  const double dt = 0.01;
  const double delta = 0.8;
  const double r0 = 5.0;
  const double epsilon = 0.07;
  const double W0 = 1.0;
  const double beta0 = 0.0;
  const double D = 2.0;
  const double d0 = 0.5;
  const double a1 = 1.25 / std::sqrt(2.0);
  const double a2 = 0.64;
  const double lambda = (W0*a1)/(d0);
  const double tau0 = ((W0*W0*W0*a1*a2)/(d0*D)) + ((W0*W0*beta0)/(d0));

  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_in_bytes = sizeof(double) * vol;

  nRarray *phi_host = (nRarray *)malloc(vol_in_bytes);
  nRarray *u_host = (nRarray *)malloc(vol_in_bytes);
  initializationPhi(phi_host,r0);
  initializationU(u_host,r0,delta);

#ifdef VERIFY
  nRarray *phi_ref = (nRarray *)malloc(vol_in_bytes);
  nRarray *u_ref = (nRarray *)malloc(vol_in_bytes);
  memcpy(phi_ref, phi_host, vol_in_bytes);
  memcpy(u_ref, u_host, vol_in_bytes);
  reference(phi_ref, u_ref, vol, num_steps);
#endif

  auto offload_start = std::chrono::steady_clock::now();

  nRarray *phiold = phi_host;
  nRarray *uold = u_host;
  nRarray *phinew = (nRarray *)malloc(vol_in_bytes);
  nRarray *unew = (nRarray *)malloc(vol_in_bytes);
  nRarray *Fx = (nRarray *)malloc(vol_in_bytes);
  nRarray *Fy = (nRarray *)malloc(vol_in_bytes);
  nRarray *Fz = (nRarray *)malloc(vol_in_bytes);

  // Keep field data resident on the GPU across the timestep loop.
#pragma omp target data \
    map(tofrom: phiold[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                 uold[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE]) \
    map(alloc: phinew[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                unew[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                Fx[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                Fy[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE], \
                Fz[0:DATAXSIZE][0:DATAYSIZE][0:DATAZSIZE])
  {
      int t = 0;

      auto start = std::chrono::steady_clock::now();

      while (t <= num_steps) {

        calculateForce(phiold, Fx, Fy, Fz,
                       dx,dy,dz,epsilon,W0,tau0);

        allenCahn(phinew, phiold, uold,
                  Fx, Fy, Fz,
                  epsilon,W0,tau0,lambda, dt,dx,dy,dz);

        boundaryConditionsPhi(phinew);

        thermalEquation(unew, uold, phinew, phiold,
                        D,dt,dx,dy,dz);

        boundaryConditionsU(unew,delta);

        swapGrid(phinew, phiold);

        swapGrid(unew, uold);

        t++;
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Total kernel execution time: %.3f (ms)\n", time * 1e-6f);
  }

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();
  printf("Offload time: %.3f (ms)\n", offload_time * 1e-6f);

  GATE_STATS_F64("phi_final", (const double*)phi_host, vol);
  GATE_STATS_F64("u_final", (const double*)u_host, vol);

#ifdef VERIFY
  bool ok = true;
  for (int idx = 0; idx < nx; idx++)
    for (int idy = 0; idy < ny; idy++)
      for (int idz = 0; idz < nz; idz++) {
        if (fabs(phi_ref[idx][idy][idz] - phi_host[idx][idy][idz]) > 1e-3) {
          ok = false; printf("phi: %lf %lf\n", phi_ref[idx][idy][idz], phi_host[idx][idy][idz]);
	}
        if (fabs(u_ref[idx][idy][idz] - u_host[idx][idy][idz]) > 1e-3) {
          ok = false; printf("u: %lf %lf\n", u_ref[idx][idy][idz], u_host[idx][idy][idz]);
        }
      }
  printf("%s\n", ok ? "PASS" : "FAIL");
  free(phi_ref);
  free(u_ref);
#endif

  free(phi_host);
  free(u_host);
  free(phinew);
  free(unew);
  free(Fx);
  free(Fy);
  free(Fz);
  return 0;
}
