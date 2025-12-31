

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>
#include "read_data.h"
#include <chrono>

static void check(int a, int b, const char *s)
{
  if (a != b) printf("Error: %s %d %d\n", s, a, b);
}

typedef struct {
  int h, e;
} eh_t;

float extend2(struct extend2_dat *d)
{
  eh_t *eh = NULL; 

  char *qp = NULL; 

  posix_memalign((void**)&eh, 64, (d->qlen+1) * 8);
  posix_memalign((void**)&qp, 64, d->qlen * d->m);
  memset(eh, 0, (d->qlen+1) * 8);

  int d_qle, 
      d_tle, 
      d_gtle, 
      d_gscore, 
      d_max_off, 
      d_score;

  const int qlen = d->qlen;
  const int tlen = d->tlen;
  const int m = d->m;
  const int o_del = d->o_del; 
  const int e_del = d->e_del; 
  const int o_ins = d->o_ins; 
  const int e_ins = d->e_ins; 
  int w = d->w;
  const int end_bonus = d->end_bonus;
  const int zdrop = d->zdrop;
  const int h0 = d->h0;

  unsigned char *query = d->query;
  unsigned char *target = d->target;
  char* mat = d->mat;

  auto start = std::chrono::steady_clock::now();

    {
    int oe_del = o_del + e_del;
    int oe_ins = o_ins + e_ins; 
    int i, j, k;
    int beg, end;
    int max, max_i, max_j, max_ins, max_del, max_ie;
    int gscore;
    int max_off;

    

    for (k = i = 0; k < m; ++k) {
      char *p = mat + k * m;
      for (j = 0; j < qlen; ++j)
        qp[i++] = p[query[j]];
    }

    

    eh[0].h = h0; 
    eh[1].h = h0 > oe_ins? h0 - oe_ins : 0;

    for (j = 2; j <= qlen && eh[j-1].h > e_ins; ++j)
      eh[j].h = eh[j-1].h - e_ins;

    

    k = m * m;
    for (i = 0, max = 0; i < k; ++i) 

      max = max > mat[i]? max : mat[i];
    max_ins = (int)((float)(qlen * max + end_bonus - o_ins) / e_ins + 1.f);
    max_ins = max_ins > 1? max_ins : 1;
    w = w < max_ins? w : max_ins;
    max_del = (int)((float)(qlen * max + end_bonus - o_del) / e_del + 1.f);
    max_del = max_del > 1? max_del : 1;
    w = w < max_del? w : max_del; 

    

    max = h0, max_i = max_j = -1; max_ie = -1, gscore = -1;
    max_off = 0;
    beg = 0, end = qlen;
    for (i = 0; i < tlen; ++i) {
      int t, f = 0, h1, m = 0, mj = -1;
      char *q = qp + target[i] * qlen;

      

      if (beg < i - w) beg = i - w;
      if (end > i + w + 1) end = i + w + 1;
      if (end > qlen) end = qlen;

      

      if (beg == 0) {
        h1 = h0 - (o_del + e_del * (i + 1));
        if (h1 < 0) h1 = 0;
      } 
      else 
        h1 = 0;

      for (j = beg; j < end; ++j) {
        

        

        

        

        

        eh_t *p = eh+j;
        int h, M = p->h, e = p->e; 

        p->h = h1;          

        M = M? M + q[j] : 0;

        h = M > e? M : e;   

        h = h > f? h : f;
        h1 = h;             

        mj = m > h? mj : j; 

        m = m > h? m : h;   

        t = M - oe_del;
        t = t > 0? t : 0;
        e -= e_del;
        e = e > t? e : t;   

        p->e = e;           

        t = M - oe_ins;
        t = t > 0? t : 0;
        f -= e_ins;
        f = f > t? f : t;   

      }
      eh[end].h = h1; eh[end].e = 0;
      if (j == qlen) {
        max_ie = gscore > h1? max_ie : i;
        gscore = gscore > h1? gscore : h1;
      }
      if (m == 0) break;
      if (m > max) {
        max = m, max_i = i, max_j = mj;
        max_off = max_off > abs(mj - i)? max_off : abs(mj - i);
      } else if (zdrop > 0) {
        if (i - max_i > mj - max_j) {
          if (max - m - ((i - max_i) - (mj - max_j)) * e_del > zdrop) break;
        } else {
          if (max - m - ((mj - max_j) - (i - max_i)) * e_ins > zdrop) break;
        }
      }
      

      for (j = beg; j < end && eh[j].h == 0 && eh[j].e == 0; ++j);
      beg = j;
      for (j = end; j >= beg && eh[j].h == 0 && eh[j].e == 0; --j);
      end = j + 2 < qlen? j + 2 : qlen;
      

    }
    d_qle = max_j + 1;
    d_tle = max_i + 1;
    d_gtle = max_ie + 1;
    d_gscore = gscore;
    d_max_off = max_off;
    d_score = max;
  }

  auto stop = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

  check(d->qle, d_qle, "qle");
  check(d->tle, d_tle, "tle");
  check(d->gtle, d_gtle, "gtle");
  check(d->gscore, d_gscore, "gscore");
  check(d->max_off, d_max_off, "max_off");
  check(d->score, d_score, "score");

  free(eh);
  free(qp);

#ifdef VERBOSE
  printf("device: qle=%d, tle=%d, gtle=%d, gscore=%d, max_off=%d, score=%d\n",
      d_qle, d_tle, d_gtle, d_gscore, d_max_off, d_score);
#endif

  return time;
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  struct extend2_dat d;

  

  const char* files[] = {
#include "filelist.txt"
  };

  float time = 0.f;
  for (int f = 0; f < repeat; f++) {
    read_data(files[f%17], &d);
    time += extend2(&d);
  }
  printf("Average offload time %f (us)\n", (time * 1e-3f) / repeat);
  return 0;
}