


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include "util.h"

int main ( int argc, char **argv )
{
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]); 
    return 1;
  }

  const int repeat = atoi(argv[1]);
  double a = 2.0;
  double b = 1.0;
  int dim = 2;
  double err;
  double h = 0.001;
  int N = 1000;
  int n_inside;
  int ni;
  int nj;
  double rth;
  int seed = 123456789;

  printf ( "\n" );

  printf ( "\n" );
  printf ( "FEYNMAN_KAC_2D:\n" );
  printf ( "\n" );
  printf ( "  Program parameters:\n" );
  printf ( "\n" );
  printf ( "  The calculation takes place inside a 2D ellipse.\n" );
  printf ( "  A rectangular grid of points will be defined.\n" );
  printf ( "  The solution will be estimated for those grid points\n" );
  printf ( "  that lie inside the ellipse.\n" );
  printf ( "\n" );
  printf ( "  Each solution will be estimated by computing %d trajectories\n", N );
  printf ( "  from the point to the boundary.\n" );
  printf ( "\n" );
  printf ( "    (X/A)^2 + (Y/B)^2 = 1\n" );
  printf ( "\n" );
  printf ( "  The ellipse parameters A, B are set to:\n" );
  printf ( "\n" );
  printf ( "    A = %f\n", a );
  printf ( "    B = %f\n", b );
  printf ( "  Stepsize H = %6.4f\n", h );

  

  rth = sqrt ( ( double ) dim * h );

  

  nj = 128;
  ni = 1 + i4_ceiling ( a / b ) * ( nj - 1 );

  printf ( "\n" );
  printf ( "  X coordinate marked by %d points\n", ni );
  printf ( "  Y coordinate marked by %d points\n", nj );

  err = 0.0;
  n_inside = 0;

    {
    long time = 0;
    for (int i = 0; i < repeat; i++) {
            
      auto start = std::chrono::steady_clock::now();
            for (int j = 0; j < nj; j++) {
        for (int i = 0; i < ni; i++) {
          double x = ( ( double ) ( nj - j     ) * ( - a )
                     + ( double ) (      j - 1 ) *     a )
                     / ( double ) ( nj     - 1 );

          double y = ( ( double ) ( ni - i     ) * ( - b )
                     + ( double ) (      i - 1 ) *     b ) 
                     / ( double ) ( ni     - 1 );

          double dx;
          double dy;
          double us;
          double ut;
          double vh;
          double vs;
          double x1;
          double x2;
          double w;
          double w_exact;
          double we;
          double wt;
          double chk = pow ( x / a, 2.0 ) + pow ( y / b, 2.0 );

          if ( 1.0 < chk )
          {
            w_exact = 1.0;
            wt = 1.0;
          }
          else {
            n_inside++;
            w_exact = exp ( pow ( x / a, 2.0 ) + pow ( y / b, 2.0 ) - 1.0 );
            wt = 0.0;
            for ( int k = 0; k < N; k++ )
            {
              x1 = x;
              x2 = y;
              w = 1.0;  
              chk = 0.0;
              while ( chk < 1.0 )
              {
                ut = r8_uniform_01 ( &seed );
                if ( ut < 1.0 / 2.0 )
                {
                  us = r8_uniform_01 ( &seed ) - 0.5;
                  if ( us < 0.0)
                    dx = - rth;
                  else
                    dx = rth;
                } 
                else
                {
                  dx = 0.0;
                }

                ut = r8_uniform_01 ( &seed );
                if ( ut < 1.0 / 2.0 )
                {
                  us = r8_uniform_01 ( &seed ) - 0.5;
                  if ( us < 0.0 )
                    dy = - rth;
                  else
                    dy = rth;
                }
                else
                {
                  dy = 0.0;
                }
                vs = potential ( a, b, x1, x2 );
                x1 = x1 + dx;
                x2 = x2 + dy;

                vh = potential ( a, b, x1, x2 );

                we = ( 1.0 - h * vs ) * w;
                w = w - 0.5 * h * ( vh * we + vs * w ); 

                chk = pow ( x1 / a, 2.0 ) + pow ( x2 / b, 2.0 );
              }
              wt += w;
            }
            wt /= ( double ) ( N ); 
            err += pow ( w_exact - wt, 2.0 );
          }
        }
      }
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    printf("Average kernel time: %lf (s)\n", time * 1e-9 / repeat);
  }

  err = sqrt ( err / ( double ) ( n_inside ) );
  printf ( "\n" );
  printf ( "  RMS absolute error in solution = %e\n", err );
  printf ( "\n" );
  printf ( "FEYNMAN_KAC_2D:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}