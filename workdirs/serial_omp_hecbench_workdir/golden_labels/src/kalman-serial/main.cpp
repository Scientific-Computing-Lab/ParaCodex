


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>




template <int n>
void Mv_l(const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = sum;
  }
}

template <int n>
void Mv_l(double alpha, const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = alpha * sum;
  }
}



template <int n, bool aT = false, bool bT = false>
void MM_l(const double* A, const double* B, double* out)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        double Aik = aT ? A[k + i * n] : A[i + k * n];
        double Bkj = bT ? B[j + k * n] : B[k + j * n];
        sum += Aik * Bkj;
      }
      out[i + j * n] = sum;
    }
  }
}



template <int rd>
void kalman(
  const double*__restrict ys,
  int nobs,
  const double*__restrict T,
  const double*__restrict Z,
  const double*__restrict RQR,
  const double*__restrict P,
  const double*__restrict alpha,
  bool intercept,
  const double*__restrict d_mu,
  int batch_size,
  double*__restrict vs,
  double*__restrict Fs,
  double*__restrict sum_logFs,
  int n_diff,
  int fc_steps = 0,
  double*__restrict d_fc = nullptr,
  bool conf_int = false,
  double* d_F_fc = nullptr)
{
    for (int bid = 0; bid < batch_size; bid++) {
    constexpr int rd2 = rd * rd;
    double l_RQR[rd2];
    double l_T[rd2];
    double l_Z[rd];
    double l_P[rd2];
    double l_alpha[rd];
    double l_K[rd];
    double l_tmp[rd2];
    double l_TP[rd2];

    

    int b_rd_offset  = bid * rd;
    int b_rd2_offset = bid * rd2;
    for (int i = 0; i < rd2; i++) {
      l_RQR[i] = RQR[b_rd2_offset + i];
      l_T[i]   = T[b_rd2_offset + i];
      l_P[i]   = P[b_rd2_offset + i];
    }
    for (int i = 0; i < rd; i++) {
      if (n_diff > 0) l_Z[i] = Z[b_rd_offset + i];
      l_alpha[i] = alpha[b_rd_offset + i];
    }

    double b_sum_logFs = 0.0;
    const double* b_ys = ys + bid * nobs;
    double* b_vs       = vs + bid * nobs; 
    double* b_Fs       = Fs + bid * nobs;

    double mu = intercept ? d_mu[bid] : 0.0;

    for (int it = 0; it < nobs; it++) {
      

      double vs_it = b_ys[it];
      if (n_diff == 0)
        vs_it -= l_alpha[0];
      else {
        for (int i = 0; i < rd; i++) {
          vs_it -= l_alpha[i] * l_Z[i];
        }
      }
      b_vs[it] = vs_it;

      

      double _Fs;
      if (n_diff == 0)
        _Fs = l_P[0];
      else {
        _Fs = 0.0;
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
          }
        }
      }
      b_Fs[it] = _Fs;
      if (it >= n_diff) b_sum_logFs += log(_Fs);

      

      

      MM_l<rd>(l_T, l_P, l_TP);
      

      double _1_Fs = 1.0 / _Fs;
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_K[i] = _1_Fs * l_TP[i];
        }
      } else
        Mv_l<rd>(_1_Fs, l_TP, l_Z, l_K);

      

      

      Mv_l<rd>(l_T, l_alpha, l_tmp);
      

      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i] + l_K[i] * vs_it;
      }
      

      l_alpha[n_diff] += mu;

      

      

      for (int i = 0; i < rd2; i++) {
        l_tmp[i] = l_T[i];
      }
      

      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_tmp[i] -= l_K[i];
        }
      } else {
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            l_tmp[j * rd + i] -= l_K[i] * l_Z[j];
          }
        }
      }

      

      

      MM_l<rd, false, true>(l_TP, l_tmp, l_P);
      

      for (int i = 0; i < rd2; i++) {
        l_P[i] += l_RQR[i];
      }
    }
    sum_logFs[bid] = b_sum_logFs;

    

    double* b_fc   = fc_steps ? d_fc + bid * fc_steps : nullptr;
    double* b_F_fc = conf_int ? d_F_fc + bid * fc_steps : nullptr;
    for (int it = 0; it < fc_steps; it++) {
      if (n_diff == 0)
        b_fc[it] = l_alpha[0];
      else {
        double pred = 0.0;
        for (int i = 0; i < rd; i++) {
          pred += l_alpha[i] * l_Z[i];
        }
        b_fc[it] = pred;
      }

      

      Mv_l<rd>(l_T, l_alpha, l_tmp);
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i];
      }
      l_alpha[n_diff] += mu;

      if (conf_int) {
        if (n_diff == 0)
          b_F_fc[it] = l_P[0];
        else {
          double _Fs = 0.0;
          for (int i = 0; i < rd; i++) {
            for (int j = 0; j < rd; j++) {
              _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
            }
          }
          b_F_fc[it] = _Fs;
        }

        

        

        MM_l<rd>(l_T, l_P, l_TP);
        

        MM_l<rd, false, true>(l_TP, l_T, l_P);
        

        for (int i = 0; i < rd2; i++) {
          l_P[i] += l_RQR[i];
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <#series> <#observations> <forcast steps> <repeat>\n", argv[0]);
    return 1;
  }
  
  const int nseries = atoi(argv[1]); 
  const int nobs = atoi(argv[2]);
  const int fc_steps = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int rd = 8;
  const int rd2 = rd * rd;
  const int batch_size = nseries;

  const int rd2_word = nseries * rd2;
  const int rd_word = nseries * rd;
  const int nobs_word = nseries * nobs;
  const int ns_word = nseries;
  const int fc_word = fc_steps * nseries;

  const int rd2_size = rd2_word * sizeof(double);
  const int rd_size = rd_word * sizeof(double);
  const int nobs_size = nobs_word * sizeof(double);
  const int ns_size = ns_word * sizeof(double);
  const int fc_size = fc_word * sizeof(double);

  int i;
  srand(123);
  double *RQR = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    RQR[i] = (double)rand() / (double)RAND_MAX;

  double *T = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    T[i] = (double)rand() / (double)RAND_MAX;

  double *P = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    P[i] = (double)rand() / (double)RAND_MAX;

  double *Z = (double*) malloc (rd_size);
  for (i = 0; i < rd * nseries; i++)
    Z[i] = (double)rand() / (double)RAND_MAX;

  double *alpha = (double*) malloc (rd_size);
  for (i = 0; i < rd * nseries; i++)
    alpha[i] = (double)rand() / (double)RAND_MAX;

  double *ys = (double*) malloc (nobs_size);
  for (i = 0; i < nobs * nseries; i++)
    ys[i] = (double)rand() / (double)RAND_MAX;

  double *mu = (double*) malloc (ns_size);
  for (i = 0; i < nseries; i++)
    mu[i] = (double)rand() / (double)RAND_MAX;

  double *vs = (double*) malloc (nobs_size);

  double *Fs = (double*) malloc (nobs_size);

  double *sum_logFs = (double*) malloc (ns_size);

  double *fc = (double*) malloc (fc_size);

  double *F_fc = (double*) malloc (fc_size);

    {
    for (int n_diff = 0; n_diff < rd; n_diff++) {

      auto start = std::chrono::steady_clock::now();

      for (i = 0; i < repeat; i++)
        kalman<rd> (
          ys,
          nobs,
          T,
          Z,
          RQR,
          P,
          alpha,
          true, 

          mu,
          batch_size,
          vs,
          Fs,
          sum_logFs,
          n_diff,
          fc_steps,
          fc,
          true, 

          F_fc );

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time (n_diff = %d): %f (s)\n", n_diff, (time * 1e-9f) / repeat);
    }
  }

  double sum = 0.0;
  for (i = 0; i < fc_steps * nseries - 1; i++)
    sum += (fabs(F_fc[i+1]) - fabs(F_fc[i])) / (fabs(F_fc[i+1]) + fabs(F_fc[i]));
  printf("Checksum: %lf\n", sum);

  free(fc);
  free(F_fc);
  free(sum_logFs);
  free(mu);
  free(Fs);
  free(vs);
  free(ys);
  free(alpha);
  free(Z);
  free(P);
  free(T);
  free(RQR);
  return 0;
}