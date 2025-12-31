



#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include "graph.h"

static const int BPI = 32;  

static const int MSB = 1 << (BPI - 1);
static const int Mask = (1 << (BPI / 2)) - 1;

#ifdef OMP_TARGET
#endif



#define popcount(x)  __builtin_popcount(x)
#define clz(x)  __builtin_clz(x)



static unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}
#ifdef OMP_TARGET
#endif

static int init(
  const int nodes,
  const int edges,
  const int* const __restrict nidx,
  const int* const __restrict nlist,
  int* const __restrict nlist2,
  int* const __restrict posscol,
  int* const __restrict posscol2,
  int* const __restrict color,
  int* const __restrict wl,
       const int threads)
{
  int wlsize = 0;
  int maxrange = -1;
#ifdef OMP_TARGET
  #else
  #endif
  for (int v = 0; v < nodes; v++) {
    int active;
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    const int degv = end - beg;
    const bool cond = (degv >= BPI);
    int pos = beg;
    if (cond) {
      int tmp;
            tmp = wlsize++;
      wl[tmp] = v;
      for (int i = beg; i < end; i++) {
        const int nei = nlist[i];
        const int degn = nidx[nei + 1] - nidx[nei];
        if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
          nlist2[pos] = nei;
          pos++;
        }
      }
    } else {
      active = 0;
      for (int i = beg; i < end; i++) {
        const int nei = nlist[i];
        const int degn = nidx[nei + 1] - nidx[nei];
        if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
          active |= (unsigned int)MSB >> (i - beg);
          pos++;
        }
      }
    }
    const int range = pos - beg;
    maxrange = std::max(maxrange, range);  

    color[v] = (cond || (range == 0)) ? (range << (BPI / 2)) : active;
    posscol[v] = (range >= BPI) ? -1 : (MSB >> range);
  }
  if (maxrange >= Mask) {printf("too many active neighbors\n"); exit(-1);}

#ifdef OMP_TARGET
  #else
  #endif
  for (int i = 0; i < edges / BPI + 1; i++) posscol2[i] = -1;
  return wlsize;
}


void runLarge(
  const int* const __restrict nidx,
  const int* const __restrict nlist,
  int* const __restrict posscol,
  volatile int* const __restrict posscol2,
  volatile int* const __restrict color,
  const int* const __restrict wl,
  const int wlsize,
  const int threads)
{
  if (wlsize != 0) {
    bool again;
#ifdef OMP_TARGET
    #else
    #endif
    do {
      again = false;
            for (int w = 0; w < wlsize; w++) {
        bool shortcut = true;
        bool done = true;
        const int v = wl[w];
        int data;  

                data = color[v];
        const int range = data >> (BPI / 2);
        if (range > 0) {
          const int beg = nidx[v];
          int pcol = posscol[v];
          const int mincol = data & Mask;
          const int maxcol = mincol + range;
          const int end = beg + maxcol;
          const int offs = beg / BPI;
          for (int i = beg; i < end; i++) {
            const int nei = nlist[i];
            int neidata;  

                        neidata = color[nei];
            const int neirange = neidata >> (BPI / 2);
            if (neirange == 0) {
              const int neicol = neidata;
              if (neicol < BPI) {
                pcol &= ~((unsigned int)MSB >> neicol);
              } else {
                if ((mincol <= neicol) && (neicol < maxcol)) {
                  int pc;  

                                    pc = posscol2[offs + neicol / BPI];
                  if ((pc << (neicol % BPI)) < 0) {
                                        posscol2[offs + neicol / BPI] &= ~((unsigned int)MSB >> (neicol % BPI));
                  }
                }
              }
            } else {
              done = false;
              const int neimincol = neidata & Mask;
              const int neimaxcol = neimincol + neirange;
              if ((neimincol <= mincol) && (neimaxcol >= mincol)) shortcut = false;
            }
          }
          int val = pcol;
          int mc = 0;
          if (pcol == 0) {
            const int offs = beg / BPI;
            mc = std::max(1, mincol / BPI) - 1;
            do {
              mc++;
                            val = posscol2[offs + mc];
            } while (val == 0);
          }
          int newmincol = mc * BPI + clz(val);
          if (mincol != newmincol) shortcut = false;
          if (shortcut || done) {
            pcol = (newmincol < BPI) ? ((unsigned int)MSB >> newmincol) : 0;
          } else {
            const int maxcol = mincol + range;
            const int range = maxcol - newmincol;
            newmincol = (range << (BPI / 2)) | newmincol;
            again = true;
          }
          posscol[v] = pcol;
                    color[v] = newmincol;
        }
      }
    } while (again);
  }
}


void runSmall(
  const int nodes,
  const int* const __restrict nidx,
  const int* const __restrict nlist,
  volatile int* const __restrict posscol,
  int* const __restrict color,
  const int threads)
{
  bool again;
#ifdef OMP_TARGET
  #else
  #endif
  do {
    again = false;
        for (int v = 0; v < nodes; v++) {
      int pcol;
            pcol = posscol[v];
      if (popcount(pcol) > 1) {
        const int beg = nidx[v];
        int active = color[v];
        int allnei = 0;
        int keep = active;
        do {
          const int old = active;
          active &= active - 1;
          const int curr = old ^ active;
          const int i = beg + __builtin_clz(curr);
          const int nei = nlist[i];
          int neipcol;
                    neipcol = posscol[nei];
          allnei |= neipcol;
          if ((pcol & neipcol) == 0) {
            pcol &= pcol - 1;
            keep ^= curr;
          } else if (popcount(neipcol) == 1) {
            pcol ^= neipcol;
            keep ^= curr;
          }
        } while (active != 0);
        if (keep != 0) {
          const int best = (unsigned int)MSB >> __builtin_clz(pcol);
          if ((best & ~allnei) != 0) {
            pcol = best;
            keep = 0;
          }
        }
        again |= keep;
        if (keep == 0) keep = __builtin_clz(pcol);
        color[v] = keep;
                posscol[v] = pcol;
      }
    }
  } while (again);
}

int main(int argc, char* argv[])
{
  printf("ECL-GC OpenMP v1.2 (%s)\n", __FILE__);
  printf("Copyright 2020 Texas State University\n\n");

  if (argc != 4) {printf("USAGE: %s <input_file_name> <thread_count> <repeat>\n\n", argv[0]);  exit(-1);}
  if (BPI != sizeof(int) * 8) {printf("ERROR: bits per int size must be %ld\n\n", sizeof(int) * 8);  exit(-1);}
  const int threads = atoi(argv[2]);
  if (threads < 1) {fprintf(stderr, "ERROR: thread_limit must be at least 1\n"); exit(-1);}

  const int repeat = atoi(argv[3]);

  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  int* const color = new int [g.nodes];
  int* const nlist2 = new int [g.edges];
  int* const posscol = new int [g.nodes];
  int* const posscol2 = new int [g.edges / BPI + 1];
  int* const wl = new int [g.nodes];

  double runtime;
  const int* nindex = g.nindex;
  const int* nlist = g.nlist;
  
#ifdef OMP_TARGET
    {

    auto start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < repeat; n++) {
      const int wlsize = init(g.nodes, g.edges, nindex, nlist, nlist2, posscol, posscol2, color, wl, threads);
      runLarge(nindex, nlist2, posscol, posscol2, color, wl, wlsize, threads);
      runSmall(g.nodes, nindex, nlist, posscol, color, threads);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    runtime = elapsed_seconds.count() / repeat;

#ifdef OMP_TARGET
  }
#endif

  printf("average runtime (%d runs):    %.6f s\n", repeat, runtime);
  printf("throughput: %.6f Mnodes/s\n", g.nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", g.edges * 0.000001 / runtime);

  bool ok = true;
  for (int v = 0; v < g.nodes; v++) {
    if (color[v] < 0) {
       printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n",
              v, g.nindex[v + 1] - g.nindex[v]);
       ok = false;
       break;
    }
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      if (color[g.nlist[i]] == color[v]) {
        printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n",
               color[v], v, g.nlist[i]);
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  const int vals = 16;
  int c[vals];
  for (int i = 0; i < vals; i++) c[i] = 0;
  int cols = -1;
  for (int v = 0; v < g.nodes; v++) {
    cols = std::max(cols, color[v]);
    if (color[v] < vals) c[color[v]]++;
  }
  cols++;
  printf("Number of distinct colors used: %d\n", cols);

  int sum = 0;
  for (int i = 0; i < std::min(vals, cols); i++) {
    sum += c[i];
    printf("color %2d: %10d (%5.1f%%)\n", i, c[i], 100.0 * sum / g.nodes);
  }

  delete [] color;
  delete [] nlist2;
  delete [] posscol;
  delete [] posscol2;
  delete [] wl;
  freeECLgraph(g);
  return 0;
}