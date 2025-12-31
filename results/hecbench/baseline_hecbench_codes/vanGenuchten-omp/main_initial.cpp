#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <chrono>
#include <omp.h>

#include "reference.h"
#include "gate.h"

#pragma omp declare target
extern const double alpha;
extern const double theta_S;
extern const double theta_R;
extern const double n;

inline void vanGenuchten_point(const double ksat_val,
                               const double psi_scaled,
                               double &theta_out,
                               double &K_out,
                               double &C_out,
                               const double theta_range,
                               const double inv_theta_range,
                               const double m,
                               const double inv_m,
                               const double inv_n,
                               const double coeff_C)
{
  double theta_val = theta_S;
  double K_val = ksat_val;
  double C_val = 0.0;

  if (psi_scaled < 0.0) {
    const double abs_psi = -psi_scaled;
    const double scaled = alpha * abs_psi;
    const double scaled_pow_n = pow(scaled, n);
    const double base = 1.0 + scaled_pow_n;
    const double base_pow_m = pow(base, m);

    theta_val = theta_R + theta_range / base_pow_m;

    const double Se = (theta_val - theta_R) * inv_theta_range;
    const double Se_pow = pow(Se, inv_m);
    const double one_minus = 1.0 - pow(1.0 - Se_pow, m);
    const double sqrt_Se = sqrt(Se);

    K_val = ksat_val * sqrt_Se * one_minus * one_minus;

    const double scaled_pow_nm1 = pow(scaled, n - 1.0);
    const double base_pow = pow(base, inv_n - 2.0);
    C_val = coeff_C * scaled_pow_nm1 * base_pow;
  }

  theta_out = theta_val;
  K_out = K_val;
  C_out = C_val;
}
#pragma omp end declare target

namespace
{

void vanGenuchten_omp(const double *__restrict Ksat,
                      const double *__restrict psi,
                      double *__restrict C,
                      double *__restrict theta,
                      double *__restrict K,
                      const int size,
                      const bool use_device)
{
  if (size <= 0) {
    return;
  }

  const double lambda = n - 1.0;
  const double m = lambda / n;
  const double inv_m = 1.0 / m;
  const double inv_n = 1.0 / n;
  const double theta_range = theta_S - theta_R;
  const double inv_theta_range = 1.0 / theta_range;
  const double coeff_C = 100.0 * alpha * n * (inv_n - 1.0) * (theta_R - theta_S);

  const int team_size = 256;
  const int teams = (size + team_size - 1) / team_size;

#pragma omp target teams distribute parallel for simd if(use_device) \
    num_teams(teams > 0 ? teams : 1) thread_limit(team_size) \
    map(to: Ksat[0:size], psi[0:size]) map(from: C[0:size], theta[0:size], K[0:size])
  for (int i = 0; i < size; ++i) {
    double theta_val;
    double K_val;
    double C_val;
    const double psi_scaled = psi[i] * 100.0;
    vanGenuchten_point(Ksat[i], psi_scaled, theta_val, K_val, C_val,
                       theta_range, inv_theta_range, m, inv_m, inv_n, coeff_C);
    theta[i] = theta_val;
    K[i] = K_val;
    C[i] = C_val;
  }
}

} // namespace

int main(int argc, char *argv[])
{
  if (argc != 5) {
    std::printf("Usage: ./%s <dimX> <dimY> <dimZ> <repeat>\n", argv[0]);
    return 1;
  }

  const int dimX = std::atoi(argv[1]);
  const int dimY = std::atoi(argv[2]);
  const int dimZ = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  const int size = dimX * dimY * dimZ;

  if (size <= 0 || repeat <= 0) {
    std::fprintf(stderr, "Invalid problem dimensions or repeat count.\n");
    return 1;
  }

  double *Ksat = new double[size];
  double *psi = new double[size];
  double *C = new double[size];
  double *theta = new double[size];
  double *K = new double[size];

  double *C_ref = new double[size];
  double *theta_ref = new double[size];
  double *K_ref = new double[size];

  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 + (1.0 - 1e-6) * static_cast<double>(i) / static_cast<double>(size);
    psi[i] = -100.0 + 101.0 * static_cast<double>(i) / static_cast<double>(size);
  }

  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  const int device_count = omp_get_num_devices();
  const bool use_device = device_count > 0;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    vanGenuchten_omp(Ksat, psi, C, theta, K, size, use_device);
  }

  auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (elapsed * 1e-9) / repeat);

  GATE_STATS_F64("vanGenuchten:C", C, size);
  GATE_STATS_F64("vanGenuchten:theta", theta, size);
  GATE_STATS_F64("vanGenuchten:K", K, size);

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (std::fabs(C[i] - C_ref[i]) > 1e-3 ||
        std::fabs(theta[i] - theta_ref[i]) > 1e-3 ||
        std::fabs(K[i] - K_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  delete[] Ksat;
  delete[] psi;
  delete[] C;
  delete[] theta;
  delete[] K;
  delete[] C_ref;
  delete[] theta_ref;
  delete[] K_ref;

  return ok ? 0 : 1;
}
