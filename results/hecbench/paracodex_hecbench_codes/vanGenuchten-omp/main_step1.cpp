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

  // Offload the element-wise update to the GPU and explicitly manage data mappings.
#pragma omp target teams distribute parallel for \
    map(to: Ksat[0:size], psi[0:size]) \
    map(from: C[0:size], theta[0:size], K[0:size]) \
    firstprivate(alpha_local, theta_S_local, theta_R_local, n_local)
  for (int i = 0; i < size; i++) {
    double Se, _theta, _psi, lambda, m, t;

    lambda = n_local - 1.0;
    m = lambda / n_local;

    _psi = psi[i] * 100.0;
    if (_psi < 0.0) {
      _theta = (theta_S_local - theta_R_local) / std::pow(1.0 + std::pow((alpha_local * (-_psi)), n_local), m) + theta_R_local;
    } else {
      _theta = theta_S_local;
    }

    theta[i] = _theta;

    Se = (_theta - theta_R_local) / (theta_S_local - theta_R_local);

    t = 1.0 - std::pow(1.0 - std::pow(Se, 1.0 / m), m);
    K[i] = Ksat[i] * std::sqrt(Se) * t * t;

    if (_psi < 0.0) {
      C[i] = 100.0 * alpha_local * n_local * (1.0 / n_local - 1.0) * std::pow(alpha_local * std::abs(_psi), n_local - 1.0)
        * (theta_R_local - theta_S_local) * std::pow(std::pow(alpha_local * std::abs(_psi), n_local) + 1.0, 1.0 / n_local - 2.0);
    } else {
      C[i] = 0.0;
    }
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

  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      vanGenuchten(Ksat, psi, C, theta, K, size);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9) / repeat);
  }

  GATE_STATS_F64("C", C, size);
  GATE_STATS_F64("theta", theta, size);
  GATE_STATS_F64("K", K, size);

  int mismatch_count = 0;
  // Validate results on the GPU while reducing a mismatch counter.
#pragma omp target teams distribute parallel for \
    map(to: C[0:size], theta[0:size], K[0:size], C_ref[0:size], theta_ref[0:size], K_ref[0:size]) \
    reduction(+: mismatch_count)
  for (int i = 0; i < size; i++) {
    if (std::fabs(C[i] - C_ref[i]) > 1e-3 ||
        std::fabs(theta[i] - theta_ref[i]) > 1e-3 ||
        std::fabs(K[i] - K_ref[i]) > 1e-3) {
      mismatch_count += 1;
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
