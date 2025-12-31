/*******************************************************
 * C file translated from WHIRL Thu Oct  2 15:26:29 2014
 *******************************************************/

/* Include file-level type and variable decls */
#include <alloca.h>
#include"wtime.h"
#include<time.h>
#include<sys/time.h>
/* File-level vars and routines */
extern void wtime_(double *);


extern void wtime_(
  double * t)
{
  
  static int _w2c_sec2123 = -1;
  struct timeval tv;
  
  gettimeofday(&tv, (struct timezone *) 0ULL);
  if(_w2c_sec2123 < 0)
  {
    _w2c_sec2123 = (tv).tv_sec;
  }
  * t = (double)(((tv).tv_sec - (long long) _w2c_sec2123)) + ((double)((tv).tv_usec) * 9.99999999999999954748e-07);
  return;
} /* wtime_ */


// Code was translated using: /mnt/lbosm1/home/yonif/NPB-fornow/NPB-paper/openacc-npb-saturator-transformed-intel_migration/CG/common/intel-application-migration-tool-for-openacc-to-openmp/src/intel-application-migration-tool-for-openacc-to-openmp -overwrite-input -suppress-openacc openacc-npb-saturator-transformed-intel_migration/CG/common/wtime.w2c.c
