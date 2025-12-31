#include "wtime.h"
#include <time.h>
#ifndef DOS
#include <sys/time.h>
#endif

void wtime(double *t)
{
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, (void *)0);
  if (sec < 0) sec = tv.tv_sec;
  *t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

    

// Code was translated using: /mnt/lbosm1/home/yonif/NPB-fornow/NPB-paper/openacc-npb-saturator-transformed-intel_migration/FT/FT/intel-application-migration-tool-for-openacc-to-openmp/src/intel-application-migration-tool-for-openacc-to-openmp -overwrite-input -suppress-openacc openacc-npb-saturator-transformed-intel_migration/FT/FT/wtime.c
