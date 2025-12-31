#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "../../../gate_sdk/gate.h"
#include "reference.h"

void vanGenuchten(
  const double *__restrict Ksat,
  const double *__restrict psi,
        double *__restrict C,
        double *__restrict theta,
        double *__restrict K,
  const int size)
{
  const double alpha_local   = alpha;
  const double theta_S_local = theta_S;
  const double theta_R_local = theta_R;
  const double n_local       = n;
  // Precompute invariants once to reduce expensive device-side pow evaluations.
  const double theta_diff = theta_S_local - theta_R_local;
  const double inv_theta_diff = 1.0 / theta_diff;
  const double inv_n = 1.0 / n_local;
  const double m_local = 1.0 - inv_n;
  const double inv_m = 1.0 / m_local;
  const double c_prefactor = 100.0 * alpha_local * n_local * (inv_n - 1.0) * (theta_R_local - theta_S_local);
  // GPU concurrency tuning: RTX 4060 Laptop GPU offers 24 SMs; ensure 256-thread teams and at least 4 teams per SM.
  const int threads_per_team = 256;
  const int min_teams = 24 * 4;
  int team_count = (size + threads_per_team - 1) / threads_per_team;
  if (team_count < min_teams) {
    team_count = min_teams;
  }

  // Offload computation while reusing persistent device allocations from the surrounding target data region.
#pragma omp target teams distribute parallel for \
    num_teams(team_count) thread_limit(threads_per_team) \
    map(present: Ksat[0:size], psi[0:size], C[0:size], theta[0:size], K[0:size]) \
    firstprivate(alpha_local, theta_S_local, theta_R_local, n_local, theta_diff, inv_theta_diff, inv_n, m_local, inv_m, c_prefactor)
  for (int i = 0; i < size; i++) {
    const double psi_scaled = psi[i] * 100.0;
    const double Ksat_val = Ksat[i];

    double theta_val;
    double K_val;
    double C_val;

    if (psi_scaled < 0.0) {
      const double psi_mag = -psi_scaled;
      const double alpha_psi = alpha_local * psi_mag;
      const double alpha_pow_n = std::pow(alpha_psi, n_local);
      const double base = 1.0 + alpha_pow_n;
      const double denom = std::pow(base, m_local);

      theta_val = theta_R_local + theta_diff / denom;

      const double Se = (theta_val - theta_R_local) * inv_theta_diff;
      const double powSeInvM = std::pow(Se, inv_m);
      const double one_minus_pow = 1.0 - powSeInvM;
      const double flow_term = 1.0 - std::pow(one_minus_pow, m_local);
      const double sqrtSe = std::sqrt(Se);
      K_val = Ksat_val * sqrtSe * flow_term * flow_term;

      const double alpha_pow_n_minus1 = alpha_pow_n / alpha_psi;
      const double pow_term = 1.0 / (base * denom);
      C_val = c_prefactor * alpha_pow_n_minus1 * pow_term;
    } else {
      theta_val = theta_S_local;
      K_val = Ksat_val;
      C_val = 0.0;
    }

    theta[i] = theta_val;
    K[i] = K_val;
    C[i] = C_val;
  }
}

int main(int argc, char* argv[])
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

  double *Ksat, *psi, *C, *theta, *K;
  double *C_ref, *theta_ref, *K_ref;

  Ksat = new double[size];
  psi = new double[size];
  C = new double[size];
  theta = new double[size];
  K = new double[size];

  C_ref = new double[size];
  theta_ref = new double[size];
  K_ref = new double[size];

  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 + (1.0 - 1e-6) * static_cast<double>(i) / static_cast<double>(size);
    psi[i] = -100.0 + 101.0 * static_cast<double>(i) / static_cast<double>(size);
  }

  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  int mismatch_count = 0;
#pragma omp target data \
    map(to: Ksat[0:size], psi[0:size], C_ref[0:size], theta_ref[0:size], K_ref[0:size]) \
    map(alloc: C[0:size], theta[0:size], K[0:size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      vanGenuchten(Ksat, psi, C, theta, K, size);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9) / repeat);

    // Pull results back once for statistics and subsequent validation.
#pragma omp target update from(C[0:size], theta[0:size], K[0:size])

    GATE_STATS_F64("C", C, size);
    GATE_STATS_F64("theta", theta, size);
    GATE_STATS_F64("K", K, size);

    const int validate_threads_per_team = 256;
    const int validate_min_teams = 24 * 4;
    int validate_team_count = (size + validate_threads_per_team - 1) / validate_threads_per_team;
    if (validate_team_count < validate_min_teams) {
      validate_team_count = validate_min_teams;
    }

    mismatch_count = 0;
    // Validate results on the GPU using data that remains resident on the device.
#pragma omp target teams distribute parallel for \
    num_teams(validate_team_count) thread_limit(validate_threads_per_team) \
    map(present: C[0:size], theta[0:size], K[0:size], C_ref[0:size], theta_ref[0:size], K_ref[0:size]) \
    reduction(+: mismatch_count)
    for (int i = 0; i < size; i++) {
      if (std::fabs(C[i] - C_ref[i]) > 1e-3 ||
          std::fabs(theta[i] - theta_ref[i]) > 1e-3 ||
          std::fabs(K[i] - K_ref[i]) > 1e-3) {
        mismatch_count += 1;
      }
    }
  }

  bool ok = (mismatch_count == 0);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  delete[] Ksat;
  delete[] psi;
  delete[] C;
  delete[] theta;
  delete[] K;
  delete[] C_ref;
  delete[] theta_ref;
  delete[] K_ref;

  return 0;
}
