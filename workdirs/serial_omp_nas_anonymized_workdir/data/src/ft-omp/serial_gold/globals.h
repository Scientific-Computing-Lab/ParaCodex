#ifndef GLOBALS_H
#define GLOBALS_H

#include "npbparams.h"

#define FFTBLOCKPAD   33
#define FFTBLOCK      32

#define T_total       1
#define T_setup       2
#define T_fft         3
#define T_evolve      4
#define T_checksum    5 
#define T_fftx        6
#define T_ffty        7
#define T_fftz        8
#define T_max         8

#define SEED          314159265.0
#define A             1220703125.0
#define PI            3.141592653589793238
#define ALPHA         1.0e-6

#define dcmplx(r,i)       (dcomplex){r, i}
#define dcmplx_add(a,b)   (dcomplex){(a).real+(b).real, (a).imag+(b).imag}
#define dcmplx_sub(a,b)   (dcomplex){(a).real-(b).real, (a).imag-(b).imag}
#define dcmplx_mul(a,b)   (dcomplex){((a).real*(b).real)-((a).imag*(b).imag),\
                                     ((a).real*(b).imag)+((a).imag*(b).real)}
#define dcmplx_mul2(a,b)  (dcomplex){(a).real*(b), (a).imag*(b)}

#define dcmplx_div2(a,b)  (dcomplex){(a).real/(b), (a).imag/(b)}
#define dcmplx_abs(x)     sqrt(((x).real*(x).real) + ((x).imag*(x).imag))

#define dconjg(x)         (dcomplex){(x).real, -1.0*(x).imag}

typedef enum { false, true } logical;
typedef struct { 
  double real;
  double imag;
} dcomplex;

extern double u_real[NXP];
extern double u_imag[NXP];
extern double u1_real[NTOTALP];
extern double u1_imag[NTOTALP];
extern double u0_real[NTOTALP];
extern double u0_imag[NTOTALP];
extern double twiddle[NTOTALP];
extern double gty1_real[MAXDIM][MAXDIM][MAXDIM];
extern double gty1_imag[MAXDIM][MAXDIM][MAXDIM];
extern double gty2_real[MAXDIM][MAXDIM][MAXDIM];
extern double gty2_imag[MAXDIM][MAXDIM][MAXDIM];
extern dcomplex sums[NITER_DEFAULT+1];
extern int dims[3];

extern int fftblock, fftblockpad;
extern logical timers_enabled;
extern logical debug;
extern inline dcomplex dcmplx_div(dcomplex z1, dcomplex z2);

void compute_initial_conditions(int d1, int d2, int d3);
void evolve(int d1, int d2, int d3);
void compute_indexmap(int d1, int d2, int d3);
void fft(int dir);
void fft_init(int n);
void checksum(int i, int d1, int d2, int d3);
void cffts1_pos(int is, int d1, int d2, int d3);
void cffts2_neg(int is, int d1, int d2, int d3);
void cffts2_pos(int is, int d1, int d2, int d3);
void cffts1_neg(int is, int d1, int d2, int d3);
void cffts3_pos(int is, int d1, int d2, int d3);
void cffts3_neg(int is, int d1, int d2, int d3);

#endif

