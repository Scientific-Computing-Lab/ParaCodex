



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <chrono>













#define tilesize 128
#define dist(a, b) int(sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))
#define swap(a, b) {float tmp = a;  a = b;  b = tmp;}

float LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}








static int best_thread_count(int cities)
{
  int max, best, threads, smem, blocks, thr, perf, bthr;

  max = cities - 2;
  if (max > 256) max = 256;
  best = 0;
  bthr = 4;
  for (threads = 1; threads <= max; threads++) {
    smem = sizeof(int) * threads + 2 * sizeof(float) * tilesize + sizeof(int) * tilesize;
    blocks = (16384 * 2) / smem;
    if (blocks > 16) blocks = 16;
    thr = (threads + 31) / 32 * 32;
    while (blocks * thr > 2048) blocks--;
    perf = threads * blocks;
    if (perf > best) {
      best = perf;
      bthr = threads;
    }
  }

  return bthr;
}

int main(int argc, char *argv[])
{
  printf("2-opt TSP OpenMP target offloading GPU code v2.3\n");
  printf("Copyright (c) 2014-2020, Texas State University. All rights reserved.\n");

  if (argc != 4) {
    fprintf(stderr, "\narguments: <input_file> <restart_count> <repeat>\n");
    exit(-1);
  }

  FILE *f = fopen(argv[1], "rt");
  if (f == NULL) {fprintf(stderr, "could not open file %s\n", argv[1]);  exit(-1);}

  int restarts = atoi(argv[2]);
  if (restarts < 1) {fprintf(stderr, "restart_count is too small: %d\n", restarts); exit(-1);}

  int repeat = atoi(argv[3]);

  

  

  

  int ch, in1;
  float in2, in3;
  char str[256];

  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
  fscanf(f, "%s\n", str);

  int cities = atoi(str);
  if (cities < 100) {
    fprintf(stderr, "the problem size must be at least 100 for this version of the code\n");
    fclose(f);
    exit(-1);
  } 

  ch = getc(f); 
  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  fscanf(f, "%s\n", str);
  if (strcmp(str, "NODE_COORD_SECTION") != 0) {
    fprintf(stderr, "wrong file format\n");
    fclose(f);
    exit(-1);
  }

  float *posx = (float *)malloc(sizeof(float) * cities);
  if (posx == NULL) fprintf(stderr, "cannot allocate posx\n");
  float *posy = (float *)malloc(sizeof(float) * cities);
  if (posy == NULL) fprintf(stderr, "cannot allocate posy\n");

  int cnt = 0;
  while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {
    posx[cnt] = in2;
    posy[cnt] = in3;
    cnt++;
    if (cnt > cities) fprintf(stderr, "input too long\n");
    if (cnt != in1) fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);
  }
  if (cnt != cities) fprintf(stderr, "read %d instead of %d cities\n", cnt, cities);

  fscanf(f, "%s", str);
  if (strcmp(str, "EOF") != 0) fprintf(stderr, "didn't see 'EOF' at end of file\n");

  fclose(f);

  printf("configuration: %d cities, %d restarts, %s input\n", cities, restarts, argv[1]);

  

  

  

  int climbs[1] = {0};
  int best[1] = {INT_MAX};

  const int glob_size = restarts * ((3 * cities + 2 + 31) / 32 * 32);
  int* glob = (int*) malloc (sizeof(int) * glob_size); 

    {

  int threads = best_thread_count(cities);
  printf("number of threads per team: %d\n", threads);

  double ktime = 0.0;

  for (int i = 0; i < repeat; i++) {
        
    auto kstart = std::chrono::steady_clock::now();

        {
      float px_s[tilesize];
      float py_s[tilesize];
      float bf_s[tilesize];
      float buf_s[128];
            {
        const int lid = omp_get_thread_num();
        const int bid = omp_get_team_num();
        const int dim = omp_get_num_threads();

        int *buf = &glob[bid * ((3 * cities + 2 + 31) / 32 * 32)];
        float *px = (float *)(&buf[cities]);
        float *py = &px[cities + 1];

        for (int i = lid; i < cities; i += dim) px[i] = posx[i];
        for (int i = lid; i < cities; i += dim) py[i] = posy[i];
        
        if (lid == 0) {  

          unsigned int seed = bid;
          for (unsigned int i = 1; i < cities; i++) {
            int j = (int)(LCG_random(&seed) * (cities - 1)) + 1;
            swap(px[i], px[j]);
            swap(py[i], py[j]);
          }
          px[cities] = px[0];
          py[cities] = py[0];
        }
        
        int minchange;
        do {
          for (int i = lid; i < cities; i += dim) buf[i] = -dist(i, i + 1);
          
          minchange = 0;
          int mini = 1;
          int minj = 0;
          for (int ii = 0; ii < cities - 2; ii += dim) {
            int i = ii + lid;
            float pxi0, pyi0, pxi1, pyi1, pxj1, pyj1;
            if (i < cities - 2) {
              minchange -= buf[i];
              pxi0 = px[i];
              pyi0 = py[i];
              pxi1 = px[i + 1];
              pyi1 = py[i + 1];
              pxj1 = px[cities];
              pyj1 = py[cities];
            }
            for (int jj = cities - 1; jj >= ii + 2; jj -= tilesize) {
              int bound = jj - tilesize + 1;
              for (int k = lid; k < tilesize; k += dim) {
                if (k + bound >= ii + 2) {
                  px_s[k] = px[k + bound];
                  py_s[k] = py[k + bound];
                  bf_s[k] = buf[k + bound];
                }
              }
              
              int lower = bound;
              if (lower < i + 2) lower = i + 2;
              for (int j = jj; j >= lower; j--) {
                int jm = j - bound;
                float pxj0 = px_s[jm];
                float pyj0 = py_s[jm];
                int change = bf_s[jm]
                  + int(sqrtf((pxi0 - pxj0) * (pxi0 - pxj0) + (pyi0 - pyj0) * (pyi0 - pyj0)))
                  + int(sqrtf((pxi1 - pxj1) * (pxi1 - pxj1) + (pyi1 - pyj1) * (pyi1 - pyj1)));
                pxj1 = pxj0;
                pyj1 = pyj0;
                if (minchange > change) {
                  minchange = change;
                  mini = i;
                  minj = j;
                }
              }
                          }

            if (i < cities - 2) {
              minchange += buf[i];
            }
          }
          
          int change = buf_s[lid] = minchange;
          if (lid == 0) {
                        climbs[0]++;
          }
          
          int j = dim;
          do {
            int k = (j + 1) / 2;
            if ((lid + k) < j) {
              int tmp = buf_s[lid + k];
              if (change > tmp) change = tmp;
              buf_s[lid] = change;
            }
            j = k;
                      } while (j > 1);

          if (minchange == buf_s[0]) {
            buf_s[1] = lid;  

          }
          
          if (lid == buf_s[1]) {
            buf_s[2] = mini + 1;
            buf_s[3] = minj;
          }
          
          minchange = buf_s[0];
          mini = buf_s[2];
          int sum = buf_s[3] + mini;
          for (int i = lid; (i + i) < sum; i += dim) {
            if (mini <= i) {
              int j = sum - i;
              swap(px[i], px[j]);
              swap(py[i], py[j]);
            }
          }
                  } while (minchange < 0);

        int term = 0;
        for (int i = lid; i < cities; i += dim) {
          term += dist(i, i + 1);
        }
        buf_s[lid] = term;
        
        int j = dim;
        do {
          int k = (j + 1) / 2;
          if ((lid + k) < j) {
            term += buf_s[lid + k];
          }
                    if ((lid + k) < j) {
            buf_s[lid] = term;
          }
          j = k;
                  } while (j > 1);

        if (lid == 0) {
          int t;
                    {
            t = best[0];
            best[0] = (term < best[0]) ? term : best[0];
          }
        }
      }
    }

    auto kend = std::chrono::steady_clock::now();
    if (i > 0)
      ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  }

    
  long long moves = 1LL * climbs[0] * (cities - 2) * (cities - 1) / 2;

  printf("Average kernel time: %.4f s\n", ktime * 1e-9f / repeat);
  printf("%.3f Gmoves/s\n", moves * repeat / ktime);
  printf("Best found tour length is %d with %d climbers\n", best[0], climbs[0]);

  

  if (best[0] < 38000 && best[0] >= 35002)
    printf("PASS\n");
  else
    printf("FAIL\n");

  }

  free(posx);
  free(posy);
  free(glob);
  return 0;
}