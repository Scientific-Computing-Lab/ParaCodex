































#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "snap.h"
#include "utils.cpp"

#if REFDATA_TWOJ == 14
#include "refdata_2J14_W.h"
#elif REFDATA_TWOJ == 8
#include "refdata_2J8_W.h"
#elif REFDATA_TWOJ == 4
#include "refdata_2J4_W.h"
#else
#include "refdata_2J2_W.h"
#endif


int nsteps = 1; 


int main(int argc, char* argv[])
{
  options(argc, argv);

  const int switch_flag = 1;     


  

  double elapsed_ui = 0.0, 
         elapsed_yi = 0.0, 
         elapsed_duidrj = 0.0,
         elapsed_deidrj = 0.0;

  const int ninside = refdata.ninside;
  const int ncoeff = refdata.ncoeff;
  const int nlocal = refdata.nlocal;
  const int nghost = refdata.nghost;
  const int ntotal = nlocal + nghost;
  const int twojmax = refdata.twojmax;
  const double rcutfac = refdata.rcutfac;

  const double wself = 1.0;
  const int num_atoms = nlocal; 
  const int num_nbor = ninside; 

  

  double* coeffi = (double*) malloc (sizeof(double) * (ncoeff+1));

  for (int icoeff = 0; icoeff < ncoeff + 1; icoeff++)
    coeffi[icoeff] = refdata.coeff[icoeff];

  double* beta = coeffi + 1;

  

  const int jdim = twojmax + 1;

  


  int *idxcg_block = (int*) malloc(sizeof(int) * jdim * jdim * jdim);

  int idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxcg_block[j1 + j2 *jdim + jdim*jdim*j] = idxcg_count;
        for (int m1 = 0; m1 <= j1; m1++)
          for (int m2 = 0; m2 <= j2; m2++)
            idxcg_count++;
      }
  const int idxcg_max = idxcg_count;

  

  

  


  int* idxu_block = (int*) malloc (sizeof(int) * jdim);
  int idxu_count = 0;

  for (int j = 0; j <= twojmax; j++) {
    idxu_block[j] = idxu_count;
    for (int mb = 0; mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++)
        idxu_count++;
  }
  const int idxu_max = idxu_count;

  

  

  


  

  int* ulist_parity = (int*) malloc (sizeof(int) * idxu_max);
  idxu_count = 0;
  for (int j = 0; j <= twojmax; j++) {
    int mbpar = 1;
    for (int mb = 0; mb <= j; mb++) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ma++) {
        ulist_parity[idxu_count] = mapar;
        mapar = -mapar;
        idxu_count++;
      }
      mbpar = -mbpar;
    }
  }

  

  

  

  


  int* idxdu_block = (int*) malloc (sizeof(int) * jdim);
  int idxdu_count = 0;

  for (int j = 0; j <= twojmax; j++) {
    idxdu_block[j] = idxdu_count;
    for (int mb = 0; 2 * mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++)
        idxdu_count++;
  }
  const int idxdu_max = idxdu_count;

  


  int idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1)
          idxb_count++;

  const int idxb_max = idxb_count;
  SNA_BINDICES* idxb = (SNA_BINDICES*) malloc (sizeof(SNA_BINDICES) * idxb_max);

  idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1) {
          idxb[idxb_count].j1 = j1;
          idxb[idxb_count].j2 = j2;
          idxb[idxb_count].j = j;
          idxb_count++;
        }

  


  int* idxb_block = (int*) malloc (sizeof(int) * jdim * jdim * jdim);
  idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        if (j < j1)
          continue;
        idxb_block[j1*jdim*jdim+j2*jdim+j] = idxb_count;
        idxb_count++;
      }


  


  int idxz_count = 0;

  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        for (int mb = 0; 2 * mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++)
            idxz_count++;

  const int idxz_max = idxz_count;
  

  int* idxz = (int*) malloc (sizeof(int) * idxz_max * 9);

  

  double* idxzbeta = (double*) malloc (sizeof(double) * idxz_max);

  

  int* idxz_block = (int*) malloc (sizeof(int) * jdim * jdim * jdim);

  idxz_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxz_block[j1*jdim*jdim+j2*jdim+j] = idxz_count;

        

        

        

        


        double betaj;
        if (j >= j1) {
          const int jjb = idxb_block[j1*jdim*jdim+j2*jdim+j];
          if (j1 == j) {
            if (j2 == j) {
              betaj = 3 * beta[jjb];
            }
            else {
              betaj = 2 * beta[jjb];
            }
          } else {
            betaj = beta[jjb];
          }
        } else if (j >= j2) {
          const int jjb = idxb_block[j*jdim*jdim+j2*jdim+j1];
          if (j2 == j) {
            betaj = 2 * beta[jjb] * (j1 + 1) / (j + 1.0);
          }
          else {
            betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
          }
        } else {
          const int jjb = idxb_block[j2*jdim*jdim+j*jdim+j1];
          betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
        }

        for (int mb = 0; 2 * mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++) {

            idxz[IDXZ_INDEX(idxz_count, 0)] = j1;
            idxz[IDXZ_INDEX(idxz_count, 1)] = j2;
            idxz[IDXZ_INDEX(idxz_count, 2)] = j;

            int ma1min = MAX(0, (2 * ma - j - j2 + j1) / 2);
            idxz[IDXZ_INDEX(idxz_count, 3)] = ma1min;
            idxz[IDXZ_INDEX(idxz_count, 4)] = (2 * ma - j - (2 * ma1min - j1) + j2) / 2;
            idxz[IDXZ_INDEX(idxz_count, 5)] =
              MIN(j1, (2 * ma - j + j2 + j1) / 2) - ma1min + 1;

            int mb1min = MAX(0, (2 * mb - j - j2 + j1) / 2);
            idxz[IDXZ_INDEX(idxz_count, 6)] = mb1min;
            idxz[IDXZ_INDEX(idxz_count, 7)] = (2 * mb - j - (2 * mb1min - j1) + j2) / 2;
            idxz[IDXZ_INDEX(idxz_count, 8)] =
              MIN(j1, (2 * mb - j + j2 + j1) / 2) - mb1min + 1;

            idxzbeta[idxz_count] = betaj;

            idxz_count++;
          }
      }
  



  if (compute_ncoeff(twojmax) != ncoeff) {
    printf("ERROR: ncoeff from SNA does not match reference data\n");
    exit(1);
  }

  


  double *rij    = (double*) malloc(sizeof(double) * (num_atoms * num_nbor * 3));
  double *inside = (double*) malloc(sizeof(double) * (num_atoms * num_nbor));
  double *wj     = (double*) malloc(sizeof(double) * (num_atoms * num_nbor));
  double *rcutij = (double*) malloc(sizeof(double) * (num_atoms * num_nbor));

  const int jdimpq = twojmax + 2;
  double* rootpqarray = (double*) malloc(sizeof(double) * jdimpq * jdimpq);
  double* cglist = (double*) malloc (sizeof(double) * idxcg_max);
  double* dedr = (double*) malloc (sizeof(double) * num_atoms * num_nbor * 3); 

  COMPLEX* ulist = (COMPLEX*) malloc (sizeof(COMPLEX) * num_atoms * num_nbor * idxu_max); 
  COMPLEX* ylist = (COMPLEX*) malloc (sizeof(COMPLEX) * num_atoms * idxdu_max);
  COMPLEX* ulisttot = (COMPLEX*) malloc (sizeof(COMPLEX) * num_atoms * idxu_max);
  COMPLEX* dulist = (COMPLEX*) malloc (sizeof(COMPLEX) * num_atoms * num_nbor * 3 * idxdu_max);

  

  for (int p = 1; p <= twojmax; p++)
    for (int q = 1; q <= twojmax; q++)
      rootpqarray[ROOTPQ_INDEX(p, q)] = sqrt(static_cast<double>(p) / q);

  

  double sum, dcg, sfaccg;
  int m, aa2, bb2, cc2;
  int ifac;

  idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        for (int m1 = 0; m1 <= j1; m1++) {
          aa2 = 2 * m1 - j1;

          for (int m2 = 0; m2 <= j2; m2++) {

            


            bb2 = 2 * m2 - j2;
            m = (aa2 + bb2 + j) / 2;

            if (m < 0 || m > j) {
              cglist[idxcg_count] = 0.0;
              idxcg_count++;
              continue;
            }

            sum = 0.0;

            for (int z = MAX(0, MAX(-(j - j2 + aa2) / 2, -(j - j1 - bb2) / 2));
                z <=
                MIN((j1 + j2 - j) / 2, MIN((j1 - aa2) / 2, (j2 + bb2) / 2));
                z++) {
              ifac = z % 2 ? -1 : 1;
              sum += ifac / (factorial(z) * factorial((j1 + j2 - j) / 2 - z) *
                  factorial((j1 - aa2) / 2 - z) *
                  factorial((j2 + bb2) / 2 - z) *
                  factorial((j - j2 + aa2) / 2 + z) *
                  factorial((j - j1 - bb2) / 2 + z));
            }

            cc2 = 2 * m - j;
            dcg = deltacg(j1, j2, j);
            sfaccg = sqrt(
                factorial((j1 + aa2) / 2) * factorial((j1 - aa2) / 2) *
                factorial((j2 + bb2) / 2) * factorial((j2 - bb2) / 2) *
                factorial((j + cc2) / 2) * factorial((j - cc2) / 2) * (j + 1));

            cglist[idxcg_count] = sum * dcg * sfaccg;
            idxcg_count++;
          }
        }
      }

  double* f = (double*) malloc (sizeof(double) * ntotal * 3);

  

  double sumsqferr = 0.0;

#if defined(OPENMP_TARGET)
{

  


  auto begin = myclock::now();
  for (int istep = 0; istep < nsteps; istep++) {

    time_point<system_clock> start, end;
    duration<double> elapsed;

    for (int j = 0; j < ntotal * 3; j++) {
      f[j] = 0.0;
    }

    int jt = 0, jjt = 0;
    for (int natom = 0; natom < num_atoms; natom++) {
      for (int nbor = 0; nbor < num_nbor; nbor++) {
        rij[ULIST_INDEX(natom, nbor, 0)] = refdata.rij[jt++];
        rij[ULIST_INDEX(natom, nbor, 1)] = refdata.rij[jt++];
        rij[ULIST_INDEX(natom, nbor, 2)] = refdata.rij[jt++];
        inside[INDEX_2D(natom, nbor)] = refdata.jlist[jjt++];
        wj[INDEX_2D(natom, nbor)] = 1.0;
        rcutij[INDEX_2D(natom, nbor)] = rcutfac;
      }
    }

#if defined(OPENMP_TARGET)
#endif

    

    start = system_clock::now();
    

    

    

    

    

    


#if defined(OPENMP_TARGET)
#else
#endif
    for (int i = 0; i < num_atoms * idxu_max; ++i)
      ulisttot[i] = { 0.0, 0.0 };

#if (OPENMP_TARGET)
#else
#endif
    for (int natom = 0; natom < num_atoms; natom++) {
      for (int j = 0; j <= twojmax; j++) {
        int jju = idxu_block[j];
        for (int ma = 0; ma <= j; ma++) {
          ulisttot[INDEX_2D(natom, jju)] = { wself, 0.0 };
          jju += j + 2;
        }
      }
    }

#if (OPENMP_TARGET)
#else
#endif
    for (int nbor = 0; nbor < num_nbor; nbor++) {
      for (int natom = 0; natom < num_atoms; natom++) {
        double x = rij[ULIST_INDEX(natom, nbor, 0)];
        double y = rij[ULIST_INDEX(natom, nbor, 1)];
        double z = rij[ULIST_INDEX(natom, nbor, 2)];
        double rsq = x * x + y * y + z * z;
        double r = sqrt(rsq);

        double theta0 = (r - rmin0) * rfac0 * MY_PI / (rcutij[INDEX_2D(natom, nbor)] - rmin0);
        double z0 = r / tan(theta0);

        double rootpq;
        int jju, jjup;

        


        double r0inv = 1.0 / sqrt(r * r + z0 * z0);
        double a_r = r0inv * z0;
        double a_i = -r0inv * z;
        double b_r = r0inv * y;
        double b_i = -r0inv * x;

        double sfac;

        sfac = compute_sfac(r, rcutij[INDEX_2D(natom, nbor)], switch_flag);
        sfac *= wj[INDEX_2D(natom, nbor)];

        

        


        

        


        

        


        

        

        ulist[ULIST_INDEX(natom, nbor, 0)].re = 1.0;
        ulist[ULIST_INDEX(natom, nbor, 0)].im = 0.0;

        

        jju = 1;
        for (int j = 1; j <= twojmax; j++) {
          int deljju = j + 1;
          for (int mb = 0; 2 * mb <= j; mb++) {
            ulist[ULIST_INDEX(natom, nbor, jju)].re = 0.0;
            ulist[ULIST_INDEX(natom, nbor, jju)].im = 0.0;
            jju += deljju;
          }
          int ncolhalf = deljju / 2;
          jju += deljju * ncolhalf;
        }

        jju = 1;
        jjup = 0;
        for (int j = 1; j <= twojmax; j++) {
          int deljju = j + 1;
          int deljjup = j;
          int mb_max = (j + 1) / 2;
          int ma_max = j;
          int m_max = ma_max * mb_max;

          

          for (int m_iter = 0; m_iter < m_max; ++m_iter) {
            int mb = m_iter / ma_max;
            int ma = m_iter % ma_max;
            double up_r = ulist[ULIST_INDEX(natom, nbor, jjup)].re;
            double up_i = ulist[ULIST_INDEX(natom, nbor, jjup)].im;

            rootpq = rootpqarray[ROOTPQ_INDEX(j - ma, j - mb)];
            ulist[ULIST_INDEX(natom, nbor, jju)].re += rootpq * (a_r * up_r + a_i * up_i);
            ulist[ULIST_INDEX(natom, nbor, jju)].im += rootpq * (a_r * up_i - a_i * up_r);

            rootpq = rootpqarray[ROOTPQ_INDEX(ma + 1, j - mb)];
            ulist[ULIST_INDEX(natom, nbor, jju+1)].re = -rootpq * (b_r * up_r + b_i * up_i);
            ulist[ULIST_INDEX(natom, nbor, jju+1)].im = -rootpq * (b_r * up_i - b_i * up_r);

            


            if (2 * (mb + 1) == j) {
              rootpq = rootpqarray[ROOTPQ_INDEX(j - ma, mb + 1)];
              ulist[ULIST_INDEX(natom, nbor, jju+deljju)].re += rootpq * (b_r * up_r - b_i * up_i);
              ulist[ULIST_INDEX(natom, nbor, jju+deljju)].im += rootpq * (b_r * up_i + b_i * up_r);

              rootpq = rootpqarray[ROOTPQ_INDEX(ma + 1, mb + 1)];
              ulist[ULIST_INDEX(natom, nbor, jju+deljju+1)].re = rootpq * (a_r * up_r - a_i * up_i);
              ulist[ULIST_INDEX(natom, nbor, jju+deljju+1)].im = rootpq * (a_r * up_i + a_i * up_r);
            }

            jju++;
            jjup++;

            if (ma == ma_max - 1)
              jju++;
          }

          

          

          

          

          int jjui = idxu_block[j];
          int jjuip = jjui + (j + 1) * (j + 1) - 1;
          for (int mb = 0; 2 * mb < j; mb++) {
            for (int ma = 0; ma <= j; ma++) {
              ulist[ULIST_INDEX(natom, nbor, jjuip)].re = ulist_parity[jjui] * ulist[ULIST_INDEX(natom, nbor, jjui)].re;
              ulist[ULIST_INDEX(natom, nbor, jjuip)].im = ulist_parity[jjui] * -ulist[ULIST_INDEX(natom, nbor, jjui)].im;
              jjui++;
              jjuip--;
            }
          }

          

          

          if (j % 2 == 0)
            jju += deljju;
          int ncolhalf = deljju / 2;
          jju += deljju * ncolhalf;
          int ncolhalfp = deljjup / 2;
          jjup += deljjup * ncolhalfp;
        }


        sfac = compute_sfac(r, rcutij[INDEX_2D(natom, nbor)], switch_flag);
        sfac *= wj[INDEX_2D(natom, nbor)];

        for (int j = 0; j <= twojmax; j++) {
          int jju = idxu_block[j];
          for (int mb = 0; mb <= j; mb++)
            for (int ma = 0; ma <= j; ma++) {
              ulisttot[INDEX_2D(natom, jju)].re += sfac * ulist[ULIST_INDEX(natom, nbor, jju)].re;
              ulisttot[INDEX_2D(natom, jju)].im += sfac * ulist[ULIST_INDEX(natom, nbor, jju)].im;

              jju++;
            }
        }
      }
    }

    end = system_clock::now();
    elapsed = end - start;
    elapsed_ui += elapsed.count();

    start = system_clock::now();

    


    

#if defined(OPENMP_TARGET)
#else
#endif
    for (int i = 0; i < num_atoms * idxdu_max; i++)
      ylist[i] = { 0.0, 0.0 };

#if defined(OPENMP_TARGET)
#else
#endif
    for (int jjz = 0; jjz < idxz_max; jjz++)
      for (int natom = 0; natom < num_atoms; natom++)
          {
            const int j1 = idxz[IDXZ_INDEX(jjz, 0)];
            const int j2 = idxz[IDXZ_INDEX(jjz, 1)];
            const int j = idxz[IDXZ_INDEX(jjz, 2)];
            const int ma1min = idxz[IDXZ_INDEX(jjz, 3)];
            const int ma2max = idxz[IDXZ_INDEX(jjz, 4)];
            const int na = idxz[IDXZ_INDEX(jjz, 5)];
            const int mb1min = idxz[IDXZ_INDEX(jjz, 6)];
            const int mb2max = idxz[IDXZ_INDEX(jjz, 7)];
            const int nb = idxz[IDXZ_INDEX(jjz, 8)];

            const double betaj = idxzbeta[jjz];

            

            const double* cgblock = cglist + idxcg_block[j1 + jdim*j2 + jdim*jdim*j];

            int mb = (2 * (mb1min + mb2max) - j1 - j2 + j) / 2;
            int ma = (2 * (ma1min + ma2max) - j1 - j2 + j) / 2;
            const int jjdu = idxdu_block[j] + (j + 1) * mb + ma;

            int jju1 = idxu_block[j1] + (j1 + 1) * mb1min;
            int jju2 = idxu_block[j2] + (j2 + 1) * mb2max;
            int icgb = mb1min * (j2 + 1) + mb2max;

            double ztmp_r = 0.0;
            double ztmp_i = 0.0;

            

            

            


            for (int ib = 0; ib < nb; ib++) {

              double suma1_r = 0.0;
              double suma1_i = 0.0;

              int ma1 = ma1min;
              int ma2 = ma2max;
              int icga = ma1min * (j2 + 1) + ma2max;

              

              

              


              for (int ia = 0; ia < na; ia++) {
                suma1_r +=
                  cgblock[icga] *
                  (ulisttot[INDEX_2D(natom, jju1 + ma1)].re * ulisttot[INDEX_2D(natom, jju2 + ma2)].re -
                   ulisttot[INDEX_2D(natom, jju1 + ma1)].im * ulisttot[INDEX_2D(natom, jju2 + ma2)].im);

                suma1_i +=
                  cgblock[icga] *
                  (ulisttot[INDEX_2D(natom, jju1 + ma1)].re * ulisttot[INDEX_2D(natom, jju2 + ma2)].im +
                   ulisttot[INDEX_2D(natom, jju1 + ma1)].im * ulisttot[INDEX_2D(natom, jju2 + ma2)].re);

                ma1++;
                ma2--;
                icga += j2;
              } 


              ztmp_r += cgblock[icgb] * suma1_r;
              ztmp_i += cgblock[icgb] * suma1_i;
              jju1 += j1 + 1;
              jju2 -= j2 + 1;
              icgb += j2;
            } 


            


            ylist[INDEX_2D(natom, jjdu)].re += betaj * ztmp_r;
            ylist[INDEX_2D(natom, jjdu)].im += betaj * ztmp_i;

          } 


    end = system_clock::now();
    elapsed = end - start;
    elapsed_yi += elapsed.count();

    

    start = system_clock::now();
#if defined(OPENMP_TARGET)
#else
#endif
    for (int nbor = 0; nbor < num_nbor; nbor++) {
      for (int natom = 0; natom < num_atoms; natom++) {
        double wj_in = wj[INDEX_2D(natom, nbor)];
        double rcut = rcutij[INDEX_2D(natom, nbor)];

        double x = rij[ULIST_INDEX(natom, nbor, 0)];
        double y = rij[ULIST_INDEX(natom, nbor, 1)];
        double z = rij[ULIST_INDEX(natom, nbor, 2)];
        double rsq = x * x + y * y + z * z;
        double r = sqrt(rsq);
        double rscale0 = rfac0 * MY_PI / (rcut - rmin0);
        double theta0 = (r - rmin0) * rscale0;
        double cs = cos(theta0);
        double sn = sin(theta0);
        double z0 = r * cs / sn;
        double dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;

        compute_duarray(natom, nbor, num_atoms, num_nbor, twojmax, 
                        idxdu_max, jdimpq, switch_flag,
                        x, y, z, z0, r, dz0dr, wj_in, rcut,
                        rootpqarray, ulist, dulist);
      }
    }

    end = system_clock::now();
    elapsed = end - start;
    elapsed_duidrj += elapsed.count();

    start = system_clock::now();
    

#if (OPENMP_TARGET)
#else
#endif
    for (int nbor = 0; nbor < num_nbor; nbor++) {
      for (int natom = 0; natom < num_atoms; natom++) {
        for (int k = 0; k < 3; k++)
          dedr[ULIST_INDEX(natom, nbor, k)] = 0.0;

        for (int j = 0; j <= twojmax; j++) {
          int jjdu = idxdu_block[j];

          for (int mb = 0; 2 * mb < j; mb++)
            for (int ma = 0; ma <= j; ma++) {

              double jjjmambyarray_r = ylist[INDEX_2D(natom, jjdu)].re;
              double jjjmambyarray_i = ylist[INDEX_2D(natom, jjdu)].im;

              for (int k = 0; k < 3; k++)
                dedr[ULIST_INDEX(natom, nbor, k)] +=
                  dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * jjjmambyarray_r +
                  dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * jjjmambyarray_i;
              jjdu++;
            } 


          


          if (j % 2 == 0) {

            int mb = j / 2;
            for (int ma = 0; ma < mb; ma++) {
              double jjjmambyarray_r = ylist[INDEX_2D(natom, jjdu)].re;
              double jjjmambyarray_i = ylist[INDEX_2D(natom, jjdu)].im;

              for (int k = 0; k < 3; k++)
                dedr[ULIST_INDEX(natom, nbor, k)] +=
                  dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * jjjmambyarray_r +
                  dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * jjjmambyarray_i;
              jjdu++;
            }

            double jjjmambyarray_r = ylist[INDEX_2D(natom, jjdu)].re;
            double jjjmambyarray_i = ylist[INDEX_2D(natom, jjdu)].im;

            for (int k = 0; k < 3; k++)
              dedr[ULIST_INDEX(natom, nbor, k)] +=
                (dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * jjjmambyarray_r +
                 dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * jjjmambyarray_i) *
                0.5;
            jjdu++;

          } 


        } 


        for (int k = 0; k < 3; k++)
          dedr[ULIST_INDEX(natom, nbor, k)] *= 2.0;

      } 

    }   


#if defined(OPENMP_TARGET)
#endif
    end = system_clock::now();
    elapsed = end - start;
    elapsed_deidrj += elapsed.count();

    

    

    for (int natom = 0; natom < num_atoms; natom++) {
      for (int nbor = 0; nbor < num_nbor; nbor++) {
        int j = inside[INDEX_2D(natom, nbor)];
        f[F_INDEX(natom, 0)] += dedr[ULIST_INDEX(natom, nbor, 0)];
        f[F_INDEX(natom, 1)] += dedr[ULIST_INDEX(natom, nbor, 1)];
        f[F_INDEX(natom, 2)] += dedr[ULIST_INDEX(natom, nbor, 2)];
        f[F_INDEX(j, 0)] -= dedr[ULIST_INDEX(natom, nbor, 0)];
        f[F_INDEX(j, 1)] -= dedr[ULIST_INDEX(natom, nbor, 1)];
        f[F_INDEX(j, 2)] -= dedr[ULIST_INDEX(natom, nbor, 2)];

      } 

    }   

    

    jt = 0;
    for (int j = 0; j < ntotal; j++) {
      double ferrx = f[F_INDEX(j, 0)] - refdata.fj[jt++];
      double ferry = f[F_INDEX(j, 1)] - refdata.fj[jt++];
      double ferrz = f[F_INDEX(j, 2)] - refdata.fj[jt++];
      sumsqferr += ferrx * ferrx + ferry * ferry + ferrz * ferrz;
    }

  }
  auto stop = myclock::now();
  myduration elapsed = stop - begin;
  double duration = elapsed.count(); 

  printf("-----------------------\n");
  printf("Summary of TestSNAP run\n");
  printf("-----------------------\n");
  printf("natoms = %d \n", nlocal);
  printf("nghostatoms = %d \n", nghost);
  printf("nsteps = %d \n", nsteps);
  printf("nneighs = %d \n", ninside);
  printf("twojmax = %d \n", twojmax);
  printf("duration = %g [sec]\n", duration);

  

  double ktime = elapsed_ui + elapsed_yi + elapsed_duidrj + elapsed_deidrj;
  printf("step time = %g [msec/step]\n", 1000.0 * duration / nsteps);
  printf("\n Individual kernel timings for each step\n");
  printf("   compute_ui = %g [msec/step]\n", 1000.0 * elapsed_ui / nsteps);
  printf("   compute_yi = %g [msec/step]\n", 1000.0 * elapsed_yi / nsteps);
  printf("   compute_duidrj = %g [msec/step]\n", 1000.0 * elapsed_duidrj / nsteps);
  printf("   compute_deidrj = %g [msec/step]\n", 1000.0 * elapsed_deidrj / nsteps);
  printf("   Total kernel time = %g [msec/step]\n", 1000.0 * ktime / nsteps);
  printf("   Percentage of step time = %g%%\n\n", ktime / duration * 100.0);
  printf("grind time = %g [msec/atom-step]\n", 1000.0 * duration / (nlocal * nsteps));
  printf("RMS |Fj| deviation %g [eV/A]\n", sqrt(sumsqferr / (ntotal * nsteps)));

#if defined(OPENMP_TARGET)
}
#endif


  free(coeffi);
  free(idxcg_block);
  free(idxu_block);
  free(ulist_parity);
  free(idxdu_block);
  free(idxb);
  free(idxb_block);
  free(idxz);
  free(idxzbeta);
  free(idxz_block);
  free(rij);
  free(inside);
  free(wj);
  free(rcutij);
  free(rootpqarray);
  free(cglist);
  free(dedr);
  free(ulist);
  free(ylist);
  free(ulisttot);
  free(dulist);
  free(f);

  return 0;
}