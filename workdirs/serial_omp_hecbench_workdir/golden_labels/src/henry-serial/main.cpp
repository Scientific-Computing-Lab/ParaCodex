#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

#define NUMTHREADS 256  








struct StructureAtom {
  

  double x;
  double y;
  double z;
  

  double epsilon;  

  

  double sigma;  

};




const double T = 298.0; 



const double R = 8.314; 



double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; 

  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}









double compute(double x, double y, double z,
    const StructureAtom * __restrict__ structureAtoms,
    double natoms, double L) 
{
  

  

  

  

  

  double E = 0.0;  


  

  for (int i = 0; i < natoms; i++) {
    

    double dx = x - structureAtoms[i].x;
    double dy = y - structureAtoms[i].y;
    double dz = z - structureAtoms[i].z;

    

    const double boxupper = 0.5 * L;
    const double boxlower = -boxupper;

    dx = (dx >  boxupper) ? dx-L : dx;
    dx = (dx >  boxupper) ? dx-L : dx;
    dy = (dy >  boxupper) ? dy-L : dy;
    dy = (dy <= boxlower) ? dy-L : dy;
    dz = (dz <= boxlower) ? dz-L : dz;
    dz = (dz <= boxlower) ? dz-L : dz;

    

    double rinv = 1.0 / sqrt(dx*dx + dy*dy + dz*dz);

    

    

    double sig_ovr_r = rinv * structureAtoms[i].sigma;
    double sig_ovr_r6 = pow(sig_ovr_r, 6.0);
    double sig_ovr_r12 = sig_ovr_r6 * sig_ovr_r6;
    E += 4.0 * structureAtoms[i].epsilon * (sig_ovr_r12 - sig_ovr_r6);
  }
  return exp(-E / (R * T));  

}


int main(int argc, char *argv[]) {
  

  if (argc != 3) {
    printf("Usage: ./%s <material file> <ninsertions>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  

  StructureAtom *structureAtoms;  

  

  std::ifstream materialfile(argv[1]);
  if (materialfile.fail()) {
    printf("Failed to import file %s.\n", argv[1]);
    exit(EXIT_FAILURE);
  }

  const int ncycles = atoi(argv[2]);  


  

  


  

  std::map<std::string, double> epsilons;
  epsilons["Zn"] = 96.152688;
  epsilons["O"] = 66.884614;
  epsilons["C"] = 88.480032;
  epsilons["H"] = 57.276566;

  

  std::map<std::string, double> sigmas;
  sigmas["Zn"] = 3.095775;
  sigmas["O"] = 3.424075;
  sigmas["C"] = 3.580425;
  sigmas["H"] = 3.150565;

  

  std::string line;
  getline(materialfile, line);
  std::istringstream istream(line);

  double L;  

  istream >> L;
  printf("L = %f\n", L);

  

  getline(materialfile, line);

  

  getline(materialfile, line);
  int natoms;  

  istream.str(line);
  istream.clear();
  istream >> natoms;
  printf("%d atoms\n", natoms);

  

  getline(materialfile, line);

  

  structureAtoms = (StructureAtom *) malloc(natoms * sizeof(StructureAtom));

  

  for (int i = 0; i < natoms; i++) {
    

    getline(materialfile, line);
    istream.str(line);
    istream.clear();

    int atomno;
    double xf, yf, zf;  

    std::string element;

    istream >> atomno >> element >> xf >> yf >> zf;

    

    structureAtoms[i].x = L * xf;
    structureAtoms[i].y = L * yf;
    structureAtoms[i].z = L * zf;

    

    structureAtoms[i].epsilon = epsilons[element];
    structureAtoms[i].sigma = sigmas[element];
  }

  

  const int nBlocks = 1024;
  const int insertionsPerCycle = nBlocks * NUMTHREADS;
  const int ninsertions = ncycles * insertionsPerCycle;  

  double * boltzmannFactors = (double*) malloc (insertionsPerCycle * sizeof(double));

    {
    

    

    

    double total_time = 0.0;

    double KH = 0.0;  

    for (int cycle = 0; cycle < ncycles; cycle++) {

      auto start = std::chrono::steady_clock::now();

      

      

      

            for (int id = 0; id < insertionsPerCycle; id++) {

        

        uint64_t seed = id;

        

        double x = L * LCG_random_double(&seed);
        double y = L * LCG_random_double(&seed);
        double z = L * LCG_random_double(&seed);

        

        boltzmannFactors[id] = compute(x, y, z, structureAtoms, natoms, L);
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      total_time += time;

      
      

      for(int i = 0; i < insertionsPerCycle; i++)
        KH += boltzmannFactors[i];
    }

    

    KH = KH / ninsertions;  
    KH = KH / (R * T);  

    printf("Used %d blocks with %d thread each\n", nBlocks, NUMTHREADS);
    printf("Henry constant = %e mol/(m3 - Pa)\n", KH);
    printf("Number of actual insertions: %d\n", ninsertions);
    printf("Number of times we called the device kernel: %d\n", ncycles);
    printf("Average kernel execution time %f (s)\n", (total_time * 1e-9) / ncycles);
  }

  free(structureAtoms);
  free(boltzmannFactors);
  return EXIT_SUCCESS;
}