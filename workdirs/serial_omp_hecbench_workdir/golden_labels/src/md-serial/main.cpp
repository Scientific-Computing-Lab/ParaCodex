#include <cassert>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <list>
#include <iostream>
#include "MD.h"
#include "reference.h"
#include "utils.h"

void md (
    const POSVECTYPE* __restrict position,
    FORCEVECTYPE* __restrict force,
    const int* __restrict neighborList, 
    const int nAtom,
    const int maxNeighbors, 
    const FPTYPE lj1_t,
    const FPTYPE lj2_t,
    const FPTYPE cutsq_t )
{
    for (uint idx = 0; idx < nAtom; idx++) {
    POSVECTYPE ipos = position[idx];
    FORCEVECTYPE f = zero;

    int j = 0;
    while (j < maxNeighbors)
    {
      int jidx = neighborList[j*nAtom + idx];

      

      POSVECTYPE jpos = position[jidx];

      

      FPTYPE delx = ipos.x - jpos.x;
      FPTYPE dely = ipos.y - jpos.y;
      FPTYPE delz = ipos.z - jpos.z;
      FPTYPE r2inv = delx*delx + dely*dely + delz*delz;

      

      if (r2inv > 0 && r2inv < cutsq_t)
      {
        r2inv = (FPTYPE)1.0 / r2inv;
        FPTYPE r6inv = r2inv * r2inv * r2inv;
        FPTYPE forceC = r2inv*r6inv*(lj1_t*r6inv - lj2_t);

        f.x += delx * forceC;
        f.y += dely * forceC;
        f.z += delz * forceC;
      }
      j++;
    }
    force[idx] = f;
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("usage: %s <class size> <iteration>", argv[0]);
    return 1;
  }

  

  int sizeClass = atoi(argv[1]);
  int iteration = atoi(argv[2]);
  const int probSizes[] = { 12288, 24576, 36864, 73728 };
  assert(sizeClass >= 0 && sizeClass < 4);
  assert(iteration >= 0);

  int nAtom = probSizes[sizeClass];

  

  POSVECTYPE* position = (POSVECTYPE*) malloc(nAtom * sizeof(POSVECTYPE));
  FORCEVECTYPE* force = (FORCEVECTYPE*) malloc(nAtom * sizeof(FORCEVECTYPE));
  int *neighborList = (int*) malloc(maxNeighbors * nAtom * sizeof(int));

  std::cout << "Initializing test problem (this can take several minutes for large problems).\n";

  

  srand(123);

  

  

  

  for (int i = 0; i < nAtom; i++)
  {
    position[i].x = rand() % domainEdge;
    position[i].y = rand() % domainEdge;
    position[i].z = rand() % domainEdge;
  }

  std::cout << "Finished.\n";
  int totalPairs = buildNeighborList<FPTYPE, POSVECTYPE>(nAtom, position, neighborList);
  std::cout << totalPairs << " of " << nAtom*maxNeighbors
            << " pairs within cutoff distance = "
            << 100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %\n";

    {
    

    md(position,
       force,
       neighborList,
       nAtom,
       maxNeighbors,
       lj1,
       lj2,
       cutsq);

    
    std::cout << "Performing Correctness Check (may take several minutes)\n";

    checkResults<FPTYPE, FORCEVECTYPE, POSVECTYPE>(force, position, neighborList, nAtom);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iteration; i++) {
      md(position,
         force,
         neighborList,
         nAtom,
         maxNeighbors,
         lj1,
         lj2,
         cutsq);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / iteration << " (s)\n";
  }

  free(position);
  free(force);
  free(neighborList);

  return 0;
}