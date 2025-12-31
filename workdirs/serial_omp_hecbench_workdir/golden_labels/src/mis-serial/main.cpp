



#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "graph.h"

static const int ThreadsPerBlock = 256;

typedef unsigned char stattype;
static const stattype in = 0xfe;
static const stattype out = 0;






unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

void computeMIS(
    const int repeat,
    const int nodes,
    const int edges,
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    volatile stattype* const __restrict nstat)
{
    {
  const int blocks = 24;

  auto start = std::chrono::high_resolution_clock::now();

  const float avg = (float)edges / nodes;
  const float scaledavg = ((in / 2) - 1) * avg;

  for (int n = 0; n < 100; n++) {
        for (int i = 0; i < nodes; i++) {
      stattype val = in;
      const int degree = nidx[i + 1] - nidx[i];
      if (degree > 0) {
        float x = degree - (hash(i) * 0.00000000023283064365386962890625f);
        int res = int(scaledavg / (avg + x));
        val = (res + res) | 1;
      }
      nstat[i] = val;
    }
    
        {
            {
        const int from = omp_get_thread_num() + omp_get_team_num() * ThreadsPerBlock;
        const int incr = omp_get_num_teams() * ThreadsPerBlock;

        int missing;
        do {
          missing = 0;
          for (int v = from; v < nodes; v += incr) {
            const stattype nv = nstat[v];
            if (nv & 1) {
              int i = nidx[v];
              while ((i < nidx[v + 1]) && ((nv > nstat[nlist[i]]) || ((nv == nstat[nlist[i]]) && (v > nlist[i])))) {
                i++;
              }
              if (i < nidx[v + 1]) {
                missing = 1;
              } else {
                for (int i = nidx[v]; i < nidx[v + 1]; i++) {
                  nstat[nlist[i]] = out;
                }
                nstat[v] = in;
              }
            }
          }
        } while (missing != 0);
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = (float)elapsed_seconds.count() / repeat;
  printf("compute time: %.6f s\n", runtime);
  printf("throughput: %.6f Mnodes/s\n", nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", edges * 0.000001 / runtime);

  }
}

int main(int argc, char* argv[])
{
  printf("ECL-MIS v1.3 (%s)\n", __FILE__);
  printf("Copyright 2017-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "USAGE: %s <input_file_name> <repeat>\n\n", argv[0]);
    exit(-1);
  }

  ECLgraph g = readECLgraph(argv[1]);
  printf("configuration: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);
  printf("average degree: %.2f edges per node\n", 1.0 * g.edges / g.nodes);

  stattype* nstatus = (stattype*)malloc(g.nodes * sizeof(nstatus[0]));

  if (nstatus == NULL) {
    fprintf(stderr, "ERROR: could not allocate nstatus\n\n");
  }
  else {
    const int repeat = atoi(argv[2]);

    computeMIS(repeat, g.nodes, g.edges, g.nindex, g.nlist, nstatus);

    


    for (int v = 0; v < g.nodes; v++) {
      if ((nstatus[v] != in) && (nstatus[v] != out)) {
        fprintf(stderr, "ERROR: found unprocessed node in graph\n\n");
        break;
      }
      if (nstatus[v] == in) {
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          if (nstatus[g.nlist[i]] == in) {
            fprintf(stderr, "ERROR: found adjacent nodes in MIS\n\n");
            break;
          }
        }
      } else {
        int flag = 0;
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          if (nstatus[g.nlist[i]] == in) {
            flag = 1;
          }
        }
        if (flag == 0) {
          fprintf(stderr, "ERROR: set is not maximal\n\n");
          break;
        }
      }
    }
  }

  freeECLgraph(g);
  if (nstatus != NULL) free(nstatus);
  return 0;
}