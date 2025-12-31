


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include "parameters.h"
#include "kernels.h"

int main(int argc, char *argv[])
{
  if (argc != 2) {fprintf(stderr, "usage: %s trace_length\n", argv[0]); exit(-1);}
  int length = atoi(argv[1]);

  assert(sizeof(unsigned short) == 2);
  assert(0 < length);
  assert((FSMSIZE & (FSMSIZE - 1)) == 0);
  assert((TABSIZE & (TABSIZE - 1)) == 0);
  assert((0 < FSMSIZE) && (FSMSIZE <= 256));
  assert((0 < TABSIZE) && (TABSIZE <= 32768));
  assert(0 < POPCNT);
  assert((0 < POPSIZE) && (POPSIZE <= 1024));
  assert(0 < CUTOFF);

  int i, j, d, s, bit, pc, misses, besthits, generations;
  unsigned short *data;
  unsigned char state[TABSIZE], fsm[FSMSIZE * 2];
  int best[FSMSIZE * 2 + 3], trans[FSMSIZE][2];
  double runtime;
  struct timeval starttime, endtime;

  data = (unsigned short*) malloc (sizeof(unsigned short) * length);

  srand(123);
  for (int i = 0; i < length; i++) data[i] = rand();

  printf("%d\t#kernel execution times\n", REPEAT);
  printf("%d\t#fsm size\n", FSMSIZE);
  printf("%d\t#entries\n", length);
  printf("%d\t#tab size\n", TABSIZE);
  printf("%d\t#blocks\n", POPCNT);
  printf("%d\t#threads\n", POPSIZE);
  printf("%d\t#cutoff\n", CUTOFF);

  unsigned int *rndstate = (unsigned int*) malloc (POPCNT * POPSIZE * sizeof(unsigned int));
  unsigned char *bfsm = (unsigned char*) malloc (POPCNT * FSMSIZE * 2 * sizeof(unsigned char));
  unsigned char *same = (unsigned char*) malloc (POPCNT * sizeof(unsigned char));
  int *smax = (int*) malloc (POPCNT * sizeof(int));
  int *sbest = (int*) malloc (POPCNT * sizeof(int));
  int *oldmax = (int*) malloc (POPCNT * sizeof(int));

    {
    gettimeofday(&starttime, NULL);

    for (int i = 0; i < REPEAT; i++) {
            for (int i = 0; i < FSMSIZE * 2 + 3; i++) best[i] = 0;
      FSMKernel(length, data, best, rndstate, bfsm, same, smax, sbest, oldmax);
      MaxKernel(best, bfsm);
    }

    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec + endtime.tv_usec / 1000000.0 - starttime.tv_sec - starttime.tv_usec / 1000000.0;
    printf("%.6lf\t#runtime [s]\n", runtime / REPEAT);
  }

  besthits = best[1];
  generations = best[2];

  printf("%.6lf\t#throughput [Gtr/s]\n", 0.000000001 * POPSIZE * generations * length / (runtime / REPEAT));

  

  for (i = 0; i < FSMSIZE; i++) {
    fsm[i * 2 + 0] = i - 1;
    fsm[i * 2 + 1] = i + 1;
  }
  fsm[0] = 0;
  fsm[(FSMSIZE - 1) * 2 + 1] = FSMSIZE - 1;
  memset(state, 0, TABSIZE);
  misses = 0;
  for (i = 0; i < length; i++) {
    d = (int)data[i];
    pc = (d >> 1) & (TABSIZE - 1);
    bit = d & 1;
    s = (int)state[pc];
    misses += bit ^ (((s + s) / FSMSIZE) & 1);
    state[pc] = fsm[s + s + bit];
  }
  printf("%d\t#sudcnt hits\n", length-misses);
  printf("%d\t#GAfsm hits\n", besthits);

  printf("%.3lf%%\t#sudcnt hits\n", 100.0 * (length - misses) / length);
  printf("%.3lf%%\t#GAfsm hits\n\n", 100.0 * besthits / length);

  

  for (i = 0; i < FSMSIZE; i++) {
    for (j = 0; j < 2; j++) {
      trans[i][j] = 0;
    }
  }
  for (i = 0; i < FSMSIZE * 2; i++) {
    fsm[i] = best[i + 3];
  }
  memset(state, 0, TABSIZE);
  misses = 0;
  for (i = 0; i < length; i++) {
    d = (int)data[i];
    pc = (d >> 1) & (TABSIZE - 1);
    bit = d & 1;
    s = (int)state[pc];
    trans[s][bit]++;
    misses += bit ^ (s & 1);
    state[pc] = (unsigned char)fsm[s + s + bit];
  }

  bool ok = ((length - misses) == besthits);
  printf("%s\n", ok ? "PASS" : "FAIL");
  
#ifdef DEBUG
  

  for (bit = 0; bit < 2; bit++) {
    for (s = 0; s < FSMSIZE; s++) {
      d = fsm[s + s + bit];
      printf("%c%d %c%d %d\n", (s & 1) ? 'P' : 'N', s / 2, (d & 1) ? 'P' : 'N', d / 2, ((bit * 2) - 1) * trans[s][bit]);
    }
  }
#endif

  free(data);
  free(rndstate);
  free(bfsm);
  free(same);
  free(smax);
  free(sbest);
  free(oldmax);
  return 0;
}