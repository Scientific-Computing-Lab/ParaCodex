#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include "gate.h"

#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400

#define SQ(x) ((x) * (x))

typedef double nRarray[DATAYSIZE][DATAXSIZE];

#ifdef VERIFY
#include <cstring>
#include "reference.h"
#endif

#pragma omp begin declare target
constexpr int NX = DATAXSIZE;
constexpr int NY = DATAYSIZE;
constexpr int NZ = DATAZSIZE;
constexpr std::size_t VOL = static_cast<std::size_t>(NX) * NY * NZ;
constexpr std::size_t PLANE_STRIDE = static_cast<std::size_t>(NY) * NZ;
constexpr std::size_t ROW_STRIDE = NZ;

inline std::size_t linear_index(int ix, int iy, int iz) {
  return (static_cast<std::size_t>(ix) * NY + iy) * NZ + iz;
}

inline double anisotropy_factor(double phix, double phiy, double phiz, double epsilon) {
  const double grad_sq = phix * phix + phiy * phiy + phiz * phiz;
  if (grad_sq > 0.0) {
    const double grad_quartic = phix * phix * phix * phix +
                                phiy * phiy * phiy * phiy +
                                phiz * phiz * phiz * phiz;
    const double base = 1.0 - 3.0 * epsilon;
    const double correction = (4.0 * epsilon / base) * (grad_quartic / (grad_sq * grad_sq));
    return base * (1.0 + correction);
  }
  return 1.0 - (5.0 * epsilon) / 3.0;
}

inline double interface_width(double phix, double phiy, double phiz, double epsilon, double W0) {
  return W0 * anisotropy_factor(phix, phiy, phiz, epsilon);
}

inline double relaxation_time(double phix, double phiy, double phiz, double epsilon, double tau0) {
  const double an = anisotropy_factor(phix, phiy, phiz, epsilon);
  return tau0 * an * an;
}

inline double directional_derivative(double l, double m, double n) {
  const double l2 = l * l;
  const double m2 = m * m;
  const double n2 = n * n;
  const double denom = l2 + m2 + n2;
  if (denom > 0.0) {
    const double l3 = l * l2;
    const double numerator = l3 * (m2 + n2) - l * (m2 * m2 + n2 * n2);
    return numerator / (denom * denom);
  }
  return 0.0;
}

inline double gradient_x(const double *field, std::size_t id, double inv_2dx) {
  return (field[id + PLANE_STRIDE] - field[id - PLANE_STRIDE]) * inv_2dx;
}

inline double gradient_y(const double *field, std::size_t id, double inv_2dy) {
  return (field[id + ROW_STRIDE] - field[id - ROW_STRIDE]) * inv_2dy;
}

inline double gradient_z(const double *field, std::size_t id, double inv_2dz) {
  return (field[id + 1] - field[id - 1]) * inv_2dz;
}

inline double divergence(const double *Fx, const double *Fy, const double *Fz,
                         std::size_t id, double inv_2dx, double inv_2dy, double inv_2dz) {
  const double dFx_dx = gradient_x(Fx, id, inv_2dx);
  const double dFy_dy = gradient_y(Fy, id, inv_2dy);
  const double dFz_dz = gradient_z(Fz, id, inv_2dz);
  return dFx_dx + dFy_dy + dFz_dz;
}

inline double laplacian(const double *field, std::size_t id,
                        double inv_dx2, double inv_dy2, double inv_dz2) {
  const double second_x = (field[id + PLANE_STRIDE] + field[id - PLANE_STRIDE] - 2.0 * field[id]) * inv_dx2;
  const double second_y = (field[id + ROW_STRIDE] + field[id - ROW_STRIDE] - 2.0 * field[id]) * inv_dy2;
  const double second_z = (field[id + 1] + field[id - 1] - 2.0 * field[id]) * inv_dz2;
  return second_x + second_y + second_z;
}

inline double reaction_term(double phi, double u, double lambda) {
  const double one_minus_phi2 = 1.0 - phi * phi;
  const double one_minus_phi2_sq = one_minus_phi2 * one_minus_phi2;
  return -phi * one_minus_phi2 + lambda * u * one_minus_phi2_sq;
}
#pragma omp end declare target

void calculate_force(double *phi,
                     double *Fx, double *Fy, double *Fz,
                     double dx, double dy, double dz,
                     double epsilon, double W0) {
  const double inv_2dx = 0.5 / dx;
  const double inv_2dy = 0.5 / dy;
  const double inv_2dz = 0.5 / dz;
  const double anis_prefactor = 16.0 * W0 * epsilon;

#pragma omp target teams distribute parallel for collapse(3) \
    map(present: phi[0:VOL], Fx[0:VOL], Fy[0:VOL], Fz[0:VOL]) \
    firstprivate(inv_2dx, inv_2dy, inv_2dz, anis_prefactor, epsilon, W0)
  for (int ix = 0; ix < NX; ++ix) {
    for (int iy = 0; iy < NY; ++iy) {
      for (int iz = 0; iz < NZ; ++iz) {
        const std::size_t id = linear_index(ix, iy, iz);
        if (ix > 0 && ix < NX - 1 &&
            iy > 0 && iy < NY - 1 &&
            iz > 0 && iz < NZ - 1) {
          const double phix = gradient_x(phi, id, inv_2dx);
          const double phiy = gradient_y(phi, id, inv_2dy);
          const double phiz = gradient_z(phi, id, inv_2dz);
          const double grad_sq = phix * phix + phiy * phiy + phiz * phiz;
          const double w = interface_width(phix, phiy, phiz, epsilon, W0);
          const double w2 = w * w;
          const double anis_term = anis_prefactor * grad_sq * w;

          Fx[id] = w2 * phix + anis_term * directional_derivative(phix, phiy, phiz);
          Fy[id] = w2 * phiy + anis_term * directional_derivative(phiy, phiz, phix);
          Fz[id] = w2 * phiz + anis_term * directional_derivative(phiz, phix, phiy);
        } else {
          Fx[id] = 0.0;
          Fy[id] = 0.0;
          Fz[id] = 0.0;
        }
      }
    }
  }
}

void allen_cahn_step(double *phi_new,
                     const double *phi_old,
                     const double *u_old,
                     const double *Fx,
                     const double *Fy,
                     const double *Fz,
                     double epsilon, double tau0, double lambda,
                     double dt, double dx, double dy, double dz) {
  const double inv_2dx = 0.5 / dx;
  const double inv_2dy = 0.5 / dy;
  const double inv_2dz = 0.5 / dz;

#pragma omp target teams distribute parallel for collapse(3) \
    map(present: phi_new[0:VOL], phi_old[0:VOL], u_old[0:VOL], Fx[0:VOL], Fy[0:VOL], Fz[0:VOL]) \
    firstprivate(inv_2dx, inv_2dy, inv_2dz, epsilon, tau0, lambda, dt)
  for (int ix = 1; ix < NX - 1; ++ix) {
    for (int iy = 1; iy < NY - 1; ++iy) {
      for (int iz = 1; iz < NZ - 1; ++iz) {
        const std::size_t id = linear_index(ix, iy, iz);
        const double phix = gradient_x(phi_old, id, inv_2dx);
        const double phiy = gradient_y(phi_old, id, inv_2dy);
        const double phiz = gradient_z(phi_old, id, inv_2dz);
        const double tau = relaxation_time(phix, phiy, phiz, epsilon, tau0);
        const double divF = divergence(Fx, Fy, Fz, id, inv_2dx, inv_2dy, inv_2dz);
        const double reaction = reaction_term(phi_old[id], u_old[id], lambda);
        phi_new[id] = phi_old[id] + (dt / tau) * (divF - reaction);
      }
    }
  }
}

void apply_boundary_phi(double *phi) {
#pragma omp target teams distribute parallel for collapse(3) map(present: phi[0:VOL])
  for (int ix = 0; ix < NX; ++ix) {
    for (int iy = 0; iy < NY; ++iy) {
      for (int iz = 0; iz < NZ; ++iz) {
        if (ix == 0 || ix == NX - 1 ||
            iy == 0 || iy == NY - 1 ||
            iz == 0 || iz == NZ - 1) {
          const std::size_t id = linear_index(ix, iy, iz);
          phi[id] = -1.0;
        }
      }
    }
  }
}

void thermal_step(double *u_new,
                  const double *u_old,
                  const double *phi_new,
                  const double *phi_old,
                  double D, double dt,
                  double dx, double dy, double dz) {
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const double inv_dz2 = 1.0 / (dz * dz);

#pragma omp target teams distribute parallel for collapse(3) \
    map(present: u_new[0:VOL], u_old[0:VOL], phi_new[0:VOL], phi_old[0:VOL]) \
    firstprivate(inv_dx2, inv_dy2, inv_dz2, D, dt)
  for (int ix = 1; ix < NX - 1; ++ix) {
    for (int iy = 1; iy < NY - 1; ++iy) {
      for (int iz = 1; iz < NZ - 1; ++iz) {
        const std::size_t id = linear_index(ix, iy, iz);
        const double lap = laplacian(u_old, id, inv_dx2, inv_dy2, inv_dz2);
        u_new[id] = u_old[id] +
                    0.5 * (phi_new[id] - phi_old[id]) +
                    dt * D * lap;
      }
    }
  }
}

void apply_boundary_u(double *u, double delta) {
#pragma omp target teams distribute parallel for collapse(3) map(present: u[0:VOL]) firstprivate(delta)
  for (int ix = 0; ix < NX; ++ix) {
    for (int iy = 0; iy < NY; ++iy) {
      for (int iz = 0; iz < NZ; ++iz) {
        if (ix == 0 || ix == NX - 1 ||
            iy == 0 || iy == NY - 1 ||
            iz == 0 || iz == NZ - 1) {
          const std::size_t id = linear_index(ix, iy, iz);
          u[id] = -delta;
        }
      }
    }
  }
}

void initialize_phi(std::vector<double> &phi, double r0) {
  const double cx = 0.5 * NX;
  const double cy = 0.5 * NY;
  const double cz = 0.5 * NZ;

  for (int ix = 0; ix < NX; ++ix) {
    for (int iy = 0; iy < NY; ++iy) {
      for (int iz = 0; iz < NZ; ++iz) {
        const double dx = ix - cx;
        const double dy = iy - cy;
        const double dz = iz - cz;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        const std::size_t id = linear_index(ix, iy, iz);
        phi[id] = (r < r0) ? 1.0 : -1.0;
      }
    }
  }
}

void initialize_u(std::vector<double> &u, double r0, double delta) {
  const double cx = 0.5 * NX;
  const double cy = 0.5 * NY;
  const double cz = 0.5 * NZ;

  for (int ix = 0; ix < NX; ++ix) {
    for (int iy = 0; iy < NY; ++iy) {
      for (int iz = 0; iz < NZ; ++iz) {
        const double dx = ix - cx;
        const double dy = iy - cy;
        const double dz = iz - cz;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        const std::size_t id = linear_index(ix, iy, iz);
        if (r < r0) {
          u[id] = 0.0;
        } else {
          u[id] = -delta * (1.0 - std::exp(-(r - r0)));
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <num_steps>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const int num_steps = std::atoi(argv[1]);

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
  const double lambda = (W0 * a1) / d0;
  const double tau0 = ((W0 * W0 * W0 * a1 * a2) / (d0 * D)) + ((W0 * W0 * beta0) / d0);

  std::vector<double> phi_state[2];
  std::vector<double> u_state[2];
  phi_state[0].resize(VOL);
  phi_state[1].resize(VOL);
  u_state[0].resize(VOL);
  u_state[1].resize(VOL);

  std::vector<double> Fx(VOL, 0.0);
  std::vector<double> Fy(VOL, 0.0);
  std::vector<double> Fz(VOL, 0.0);

  initialize_phi(phi_state[0], r0);
  initialize_u(u_state[0], r0, delta);

#ifdef VERIFY
  std::vector<double> phi_ref_storage(VOL);
  std::vector<double> u_ref_storage(VOL);
  std::copy(phi_state[0].begin(), phi_state[0].end(), phi_ref_storage.begin());
  std::copy(u_state[0].begin(), u_state[0].end(), u_ref_storage.begin());
  reference(reinterpret_cast<nRarray *>(phi_ref_storage.data()),
            reinterpret_cast<nRarray *>(u_ref_storage.data()),
            VOL, num_steps);
#endif

  const auto offload_start = std::chrono::steady_clock::now();

  double *phi_buf0 = phi_state[0].data();
  double *phi_buf1 = phi_state[1].data();
  double *u_buf0 = u_state[0].data();
  double *u_buf1 = u_state[1].data();
  double *Fx_ptr = Fx.data();
  double *Fy_ptr = Fy.data();
  double *Fz_ptr = Fz.data();

  double *phi_old = phi_buf0;
  double *phi_new = phi_buf1;
  double *u_old = u_buf0;
  double *u_new = u_buf1;

  int t = 0;
  const auto kernel_start = std::chrono::steady_clock::now();

#pragma omp target data map(tofrom: phi_buf0[0:VOL], u_buf0[0:VOL]) \
    map(alloc: phi_buf1[0:VOL], u_buf1[0:VOL], Fx_ptr[0:VOL], Fy_ptr[0:VOL], Fz_ptr[0:VOL])
  {
    while (t <= num_steps) {
      calculate_force(phi_old, Fx_ptr, Fy_ptr, Fz_ptr, dx, dy, dz, epsilon, W0);
      allen_cahn_step(phi_new, phi_old, u_old, Fx_ptr, Fy_ptr, Fz_ptr,
                      epsilon, tau0, lambda, dt, dx, dy, dz);
      apply_boundary_phi(phi_new);
      thermal_step(u_new, u_old, phi_new, phi_old, D, dt, dx, dy, dz);
      apply_boundary_u(u_new, delta);

      std::swap(phi_old, phi_new);
      std::swap(u_old, u_new);
      ++t;
    }

#pragma omp target update from(phi_old[0:VOL])
#pragma omp target update from(u_old[0:VOL])
  }

  const auto kernel_end = std::chrono::steady_clock::now();
  const auto offload_end = std::chrono::steady_clock::now();

  const double kernel_time_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count() * 1e-6;
  const double offload_time_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count() * 1e-6;

  std::printf("Total kernel execution time: %.3f (ms)\n", kernel_time_ms);
  std::printf("Offload time: %.3f (ms)\n", offload_time_ms);

  GATE_STATS_F64("phi_final", static_cast<const double *>(phi_old), VOL);
  GATE_STATS_F64("u_final", static_cast<const double *>(u_old), VOL);

#ifdef VERIFY
  bool ok = true;
  const double *phi_final = phi_old;
  const double *u_final = u_old;
  for (int ix = 0; ix < NX; ++ix) {
    for (int iy = 0; iy < NY; ++iy) {
      for (int iz = 0; iz < NZ; ++iz) {
        const std::size_t id = linear_index(ix, iy, iz);
        if (std::fabs(phi_ref_storage[id] - phi_final[id]) > 1e-3) {
          ok = false;
        }
        if (std::fabs(u_ref_storage[id] - u_final[id]) > 1e-3) {
          ok = false;
        }
      }
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
#endif

  return 0;
}
