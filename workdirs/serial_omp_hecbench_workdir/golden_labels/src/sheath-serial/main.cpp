








#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>




#define EPS_0 8.85418782e-12 

#define K 1.38065e-23        

#define ME 9.10938215e-31    

#define QE 1.602176565e-19   

#define AMU 1.660538921e-27  

#define EV_TO_K 11604.52     




#define PLASMA_DEN 1e16      

#define NUM_IONS 500000      

#define NUM_ELECTRONS 500000 

#define DX 1e-4              

#define NC 100               

#define NUM_TS 1000          

#define DT 1e-11             

#define ELECTRON_TEMP 3.0    

#define ION_TEMP 1.0         




#define X0 0           

#define XL NC* DX      

#define XMAX (X0 + XL) 


const int THREADS_PER_BLOCK = 256;


struct Domain
{
  const int ni      = NC + 1; 

  const double x0   = X0;
  const double dx   = DX;
  const double xl   = XL;
  const double xmax = XMAX;

  

  double* phi; 

  double* ef;  

  double* rho; 


  float* ndi; 

  float* nde; 

};



struct Particle
{
  double x;   

  double v;   

  bool alive; 

};



struct Species
{
  double mass;   

  double charge; 

  double spwt;   


  int np;             

  int np_alloc;       

  Particle* part;     

};



double rnd();
double SampleVel(double v_th);
void ScatterSpecies(Species* species, Particle* particles, float* den, double &time);
void ComputeRho(Species* ions, Species* electrons);
bool SolvePotential(double* phi, double* rho);
bool SolvePotentialDirect(double* phi, double* rho);
void ComputeEF(double* phi, double* ef);
void PushSpecies(Species* species, Particle* particles, double* ef);
void RewindSpecies(Species* species, Particle* particles, double* ef);
void AddParticle(Species* species, double x, double v);
double XtoL(double pos);
double gather(double lc, const double* field);
void scatter(double lc, float value, float* field);

void WriteResults(int ts);



Domain domain;

FILE* file_res;



int main(int argc, char* argv[])
{
  int p;
  int ts; 

  double sp_time = 0.0; 


  domain.phi = new double[domain.ni]; 

  domain.rho = new double[domain.ni]; 

  domain.ef  = new double[domain.ni]; 

  domain.nde = new float[domain.ni];  

  domain.ndi = new float[domain.ni];  


  

  double* phi = domain.phi;
  double* rho = domain.rho;
  double* ef  = domain.ef;
  float* nde  = domain.nde;
  float* ndi  = domain.ndi;

  

  memset(phi, 0, sizeof(double) * domain.ni);

  

  Species ions;
  Species electrons;

  

  ions.mass     = 16 * AMU;
  ions.charge   = QE;
  ions.spwt     = PLASMA_DEN * domain.xl / NUM_IONS;
  ions.np       = 0;
  ions.np_alloc = NUM_IONS;
  ions.part     = new Particle[NUM_IONS];

  electrons.mass     = ME; 

  electrons.charge   = -QE;
  electrons.spwt     = PLASMA_DEN * domain.xl / NUM_ELECTRONS;
  electrons.np       = 0;
  electrons.np_alloc = NUM_ELECTRONS;
  electrons.part     = new Particle[NUM_ELECTRONS];

  Particle *ions_part = ions.part;
  Particle *electrons_part = electrons.part;

  

  srand(123);

  

  double delta_ions = domain.xl / NUM_IONS;
  double v_thi      = sqrt(2 * K * ION_TEMP * EV_TO_K / ions.mass);
  for (p = 0; p < NUM_IONS; p++)
  {
    double x = domain.x0 + p * delta_ions;
    double v = SampleVel(v_thi);
    AddParticle(&ions, x, v);
  }

  

  double delta_electrons = domain.xl / NUM_ELECTRONS;
  double v_the           = sqrt(2 * K * ELECTRON_TEMP * EV_TO_K / electrons.mass);
  for (p = 0; p < NUM_ELECTRONS; p++)
  {
    double x = domain.x0 + p * delta_electrons;
    double v = SampleVel(v_the);
    AddParticle(&electrons, x, v);
  }

    {

  

  ScatterSpecies(&ions, ions_part, ndi, sp_time);
  ScatterSpecies(&electrons, electrons_part, nde, sp_time);

  

  ComputeRho(&ions, &electrons);

  SolvePotential(phi, rho);

  ComputeEF(phi, ef);

  RewindSpecies(&ions, ions_part, ef);
  RewindSpecies(&electrons, electrons_part, ef);

  

  file_res = fopen("result.dat", "w");
  fprintf(file_res, "VARIABLES = x nde ndi rho phi ef\n");
  WriteResults(0);

  auto start = std::chrono::steady_clock::now();

  

  for (ts = 1; ts <= NUM_TS; ts++)
  {
    

    ScatterSpecies(&ions, ions_part, ndi, sp_time);
    ScatterSpecies(&electrons, electrons_part, nde, sp_time);

    ComputeRho(&ions, &electrons);
    SolvePotential(phi, rho);
    ComputeEF(phi, ef);

    

    PushSpecies(&electrons, electrons_part, ef);
    PushSpecies(&ions, ions_part, ef);

    

    if (ts % 25 == 0)
    {
      

      double max_phi = abs(phi[0]);
      for (int i = 0; i < domain.ni; i++)
        if (abs(phi[i]) > max_phi)
          max_phi = abs(phi[i]);

      printf("TS:%i\tnp_i:%d\tnp_e:%d\tdphi:%.3g\n", ts, ions.np, electrons.np,
          max_phi - phi[0]);
    }

    

    if (ts % 1000 == 0)
      WriteResults(ts);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  fclose(file_res);

  

  delete phi;
  delete rho;
  delete ef;
  delete nde;
  delete ndi;

  

  delete ions.part;
  delete electrons.part;

  printf("Total kernel execution time (scatter particles) : %.3g (s)\n", sp_time * 1e-9f),
  printf("Total time for %d time steps: %.3g (s)\n", NUM_TS, time * 1e-9f);
  printf("Time per time step: %.3g (ms)\n", (time * 1e-6f) / NUM_TS);

  } 


  return 0;
}





double rnd()
{
  return rand() / (double)RAND_MAX;
}



double SampleVel(double v_th)
{
  const int M = 12;
  double sum  = 0;
  for (int i = 0; i < M; i++)
    sum += rnd();

  return sqrt(0.5) * v_th * (sum - M / 2.0) / sqrt(M / 12.0);
}




void ScatterSpecies(Species* species, Particle* particles, float* den, double &time)
{
  

  int nodes = domain.ni;

    for (int p = 0; p < nodes; p++) {
    den[p] = 0;
  }

  int size = species->np_alloc;

  auto start = std::chrono::steady_clock::now();

  

    for (long p = 0; p < size; p++)
  if (particles[p].alive)
  {
    double lc = XtoL(particles[p].x);
    scatter(lc, 1.f, den);
  }

  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  

  
  

  for (int i = 0; i < domain.ni; i++)
    den[i] *= species->spwt / domain.dx;

  

  den[0] *= 2.0;
  den[domain.ni - 1] *= 2.0;
}



void AddParticle(Species* species, double x, double v)
{
  

  if (species->np > species->np_alloc - 1)
  {
    printf("Too many particles!\n");
    exit(-1);
  }

  

  species->part[species->np].x     = x;
  species->part[species->np].v     = v;
  species->part[species->np].alive = true;

  

  species->np++;
}



void ComputeRho(Species* ions, Species* electrons)
{
  double* rho = domain.rho;

  for (int i = 0; i < domain.ni; i++)
    rho[i] = ions->charge * domain.ndi[i] + electrons->charge * domain.nde[i];
}



bool SolvePotentialDirect(double* x, double* rho)
{
  

  int ni     = domain.ni;
  double dx2 = domain.dx * domain.dx;
  int i;
  double* a = new double[ni];
  double* b = new double[ni];
  double* c = new double[ni];

  

  for (i = 1; i < ni - 1; i++)
  {
    a[i] = 1;
    b[i] = -2;
    c[i] = 1;
  }

  

  a[0]      = 0;
  b[0]      = 1;
  c[0]      = 0;
  a[ni - 1] = 0;
  b[ni - 1] = 1;
  c[ni - 1] = 0;

  

  for (i = 1; i < domain.ni - 1; i++)
    x[i] = -rho[i] * dx2 / EPS_0;

  x[0]      = 0;
  x[ni - 1] = 0;

  

  c[0] /= b[0]; 

  x[0] /= b[0]; 

  for (i = 1; i < ni; i++)
  {
    double id = (b[i] - c[i - 1] * a[i]); 

    c[i] /= id;                           

    x[i] = (x[i] - x[i - 1] * a[i]) / id;
  }

  

  for (i = ni - 2; i >= 0; i--)
    x[i] = x[i] - c[i] * x[i + 1];

  return true;
}



bool SolvePotential(double* phi, double* rho)
{
  double L2;
  double dx2 = domain.dx * domain.dx; 


  

  phi[0] = phi[domain.ni - 1] = 0;

  

  for (int solver_it = 0; solver_it < 40000; solver_it++)
  {
    

    for (int i = 1; i < domain.ni - 1; i++)
    {
      

      double g = 0.5 * (phi[i - 1] + phi[i + 1] + dx2 * rho[i] / EPS_0);
      phi[i]   = phi[i] + 1.4 * (g - phi[i]);
    }

    

    if (solver_it % 25 == 0)
    {
      double sum = 0;
      for (int i = 1; i < domain.ni - 1; i++)
      {
        double R = -rho[i] / EPS_0 - (phi[i - 1] - 2 * phi[i] + phi[i + 1]) / dx2;
        sum += R * R;
      }
      L2 = sqrt(sum) / domain.ni;
      if (L2 < 1e-4)
      {
        return true;
      }
    }
  }
  printf("Gauss-Seidel solver failed to converge, L2=%.3g!\n", L2);
  return false;
}



void ComputeEF(double* phi, double* ef)
{
  for (int i = 1; i < domain.ni - 1; i++)
    ef[i] = -(phi[i + 1] - phi[i - 1]) / (2 * domain.dx); 


  

  ef[0]             = -(phi[1] - phi[0]) / domain.dx;
  ef[domain.ni - 1] = -(phi[domain.ni - 1] - phi[domain.ni - 2]) / domain.dx;

  

  }



void PushSpecies(Species* species, Particle* particles, double* ef)
{
  

  double qm = species->charge / species->mass;

  int size = species->np_alloc;

  

    for (long p = 0; p < size; p++)
    if (particles[p].alive)
    {
      

      Particle* part = &particles[p];

      

      double lc = XtoL(part->x);

      

      double part_ef = gather(lc, ef);

      

      part->v += DT * qm * part_ef;

      

      part->x += DT * part->v;

      

      if (part->x < X0 || part->x >= XMAX)
        part->alive = false;
    }
}



void RewindSpecies(Species* species, Particle* particles, double* ef)
{
  

  double qm = species->charge / species->mass;

  int size = species->np_alloc;

  

    for (long p = 0; p < size; p++)
    if (particles[p].alive)
    {
      

      Particle* part = &particles[p];

      

      double lc = XtoL(part->x);

      

      double part_ef = gather(lc, ef);

      

      part->v -= 0.5 * DT * qm * part_ef;
    }
}




double XtoL(double pos)
{
  double li = (pos - 0) / DX;
  return li;
}



void scatter(double lc, float value, float* field)
{
  int i    = (int)lc;
  float di = lc - i;
  field[i] += value * (1 - di);
  field[i + 1] += value * (di);
}



double gather(double lc, const double* field)
{
  int i     = (int)lc;
  double di = lc - i;

  

  double val = field[i] * (1 - di) + field[i + 1] * (di);
  return val;
}




void WriteResults(int ts)
{
  fprintf(file_res, "ZONE I=%d T=ZONE_%06d\n", domain.ni, ts);
  for (int i = 0; i < domain.ni; i++)
  {
    fprintf(file_res, "%g %g %g %g %g %g\n", i * domain.dx, domain.nde[i], domain.ndi[i],
        domain.rho[i], domain.phi[i], domain.ef[i]);
  }

  fflush(file_res);
}