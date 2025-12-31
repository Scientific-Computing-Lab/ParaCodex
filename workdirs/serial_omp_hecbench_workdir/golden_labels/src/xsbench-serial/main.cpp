#include "XSbench_header.h"
























unsigned long long
run_event_based_simulation(Inputs in, SimulationData SD, int mype)
{
  if(mype==0) printf("Beginning event based simulation on the host for verification...\n");

  int * verification = (int *) malloc(in.lookups * sizeof(int));

  

  

  

  

  

  if( SD.length_unionized_energy_array == 0 )
  {
    SD.length_unionized_energy_array = 1;
    SD.unionized_energy_array = (double *) malloc(sizeof(double));
  }

  if( SD.length_index_grid == 0 )
  {
    SD.length_index_grid = 1;
    SD.index_grid = (int *) malloc(sizeof(int));
  }

    for( int i = 0; i < in.lookups; i++ )
  {
    

    uint64_t seed = STARTING_SEED;

    

    seed = fast_forward_LCG(seed, 2*i);

    

    double p_energy = LCG_random_double(&seed);
    int mat         = pick_mat(&seed);

    

    


    double macro_xs_vector[5] = {0};

    

    calculate_macro_xs(
        p_energy,        

        mat,             

        in.n_isotopes,   

        in.n_gridpoints, 

        SD.num_nucs,     

        SD.concs,        

        SD.unionized_energy_array, 

        SD.index_grid,   

        SD.nuclide_grid, 

        SD.mats,         

        macro_xs_vector, 

        in.grid_type,    

        in.hash_bins,    

        SD.max_num_nucs  

    );

    

    

    

    

    

    

    

    

    double max = -1.0;
    int max_idx = 0;
    for(int j = 0; j < 5; j++ )
    {
      if( macro_xs_vector[j] > max )
      {
        max = macro_xs_vector[j];
        max_idx = j;
      }
    }
    verification[i] = max_idx+1;
  }

  

  unsigned long long verification_scalar = 0;
  for( int i = 0; i < in.lookups; i++ )
    verification_scalar += verification[i];

  if( SD.length_unionized_energy_array == 0 ) free(SD.unionized_energy_array);
  if( SD.length_index_grid == 0 ) free(SD.index_grid);
  free(verification);

  return verification_scalar;
}

unsigned long long
run_event_based_simulation(Inputs in, SimulationData SD,
                           int mype, double *kernel_time)
{

  

  

  

  

  

  

  

  

  

  

  

  

  

  

  

  

  

  


  if( mype == 0)
    printf("Beginning event based simulation...\n");

  if( mype == 0 )
     printf("Allocating an additional %.1lf MB of memory for verification arrays...\n",
            in.lookups * sizeof(int) /1024.0/1024.0);

  if( SD.length_unionized_energy_array == 0 )
  {
    SD.length_unionized_energy_array = 1;
    SD.unionized_energy_array = (double *) malloc(sizeof(double));
  }

  if( SD.length_index_grid == 0 )
  {
    SD.length_index_grid = 1;
    SD.index_grid = (int *) malloc(sizeof(int));
  }

  

  

  

  int *verification = (int *) malloc(in.lookups * sizeof(int));

  const int SD_max_num_nucs = SD.max_num_nucs;
  const int *SD_num_nucs = SD.num_nucs;
  const double *SD_concs = SD.concs;
  const int *SD_mats = SD.mats;
  const NuclideGridPoint *SD_nuclide_grid  = SD.nuclide_grid;
  const double *SD_unionized_energy_array = SD.unionized_energy_array;
  const    int *SD_index_grid = SD.index_grid;


    {

    double kstart = get_time();

    for (int n = 0; n < in.kernel_repeat; n++) {

            for( int i = 0; i < in.lookups; i++ )
      {
        

        uint64_t seed = STARTING_SEED;

        

        seed = fast_forward_LCG(seed, 2*i);

        

        double p_energy = LCG_random_double(&seed);
        int mat         = pick_mat(&seed);

        

        


        double macro_xs_vector[5] = {0};

        

        calculate_macro_xs(
            p_energy,        

            mat,             

            in.n_isotopes,   

            in.n_gridpoints, 

            SD_num_nucs,     

            SD_concs,        

            SD_unionized_energy_array, 

            SD_index_grid,   

            SD_nuclide_grid, 

            SD_mats,         

            macro_xs_vector, 

            in.grid_type,    

            in.hash_bins,    

            SD_max_num_nucs  

        );

        

        

        

        

        

        

        

        

        double max = -1.0;
        int max_idx = 0;
        for(int j = 0; j < 5; j++ )
        {
          if( macro_xs_vector[j] > max )
          {
            max = macro_xs_vector[j];
            max_idx = j;
          }
        }
        verification[i] = max_idx+1;
      }
    }

    double kstop = get_time();
    *kernel_time = (kstop - kstart) / in.kernel_repeat;

  } 


  

  unsigned long long verification_scalar = 0;
  for( int i = 0; i < in.lookups; i++ )
    verification_scalar += verification[i];

  if( SD.length_unionized_energy_array == 0 ) free(SD.unionized_energy_array);
  if( SD.length_index_grid == 0 ) free(SD.index_grid);
  free(verification);

  return verification_scalar;
}






template <class T>
long grid_search( long n, double quarry, T A)
{
  long lowerLimit = 0;
  long upperLimit = n-1;
  long examinationPoint;
  long length = upperLimit - lowerLimit;

  while( length > 1 )
  {
    examinationPoint = lowerLimit + ( length / 2 );

    if( A[examinationPoint] > quarry )
      upperLimit = examinationPoint;
    else
      lowerLimit = examinationPoint;

    length = upperLimit - lowerLimit;
  }

  return lowerLimit;
}



template <class Double_Type, class Int_Type, class NGP_Type>
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
    long n_gridpoints,
    Double_Type  egrid, Int_Type  index_data,
    NGP_Type  nuclide_grids,
    long idx, double *  xs_vector, int grid_type, int hash_bins ){
  

  double f;
  NuclideGridPoint low, high;
  long low_idx, high_idx;

  

  

  if( grid_type == NUCLIDE )
  {
    

    long offset = nuc * n_gridpoints;
    idx = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids, offset, offset + n_gridpoints-1);

    

    

    if( idx == n_gridpoints - 1 )
      low_idx = idx - 1;
    else
      low_idx = idx;
  }
  else if( grid_type == UNIONIZED) 

  {
    

    

    if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
      low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1;
    else
    {
      low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc];
    }
  }
  else 

  {
    

    int u_low = index_data[idx * n_isotopes + nuc];

    

    int u_high;
    if( idx == hash_bins - 1 )
      u_high = n_gridpoints - 1;
    else
      u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;

    

    

    

    double e_low  = nuclide_grids[nuc*n_gridpoints + u_low].energy;
    double e_high = nuclide_grids[nuc*n_gridpoints + u_high].energy;
    long lower;
    if( p_energy <= e_low )
      lower = nuc*n_gridpoints;
    else if( p_energy >= e_high )
      lower = nuc*n_gridpoints + n_gridpoints - 1;
    else
    {
      long offset = nuc*n_gridpoints;
      lower = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids, offset+u_low, offset+u_high);
    }

    if( (lower % n_gridpoints) == n_gridpoints - 1 )
      low_idx = lower - 1;
    else
      low_idx = lower;
  }

  high_idx = low_idx + 1;
  low = nuclide_grids[low_idx];
  high = nuclide_grids[high_idx];

  

  f = (high.energy - p_energy) / (high.energy - low.energy);

  

  xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);

  

  xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);

  

  xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);

  

  xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);

  

  xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);
}



template <class Double_Type, class Int_Type, class NGP_Type, class E_GRID_TYPE, class INDEX_TYPE>
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
    long n_gridpoints, Int_Type  num_nucs,
    Double_Type  concs,
    E_GRID_TYPE  egrid, INDEX_TYPE  index_data,
    NGP_Type  nuclide_grids,
    Int_Type  mats,
    double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
  int p_nuc; 

  long idx = -1;
  double conc; 


  

  for( int k = 0; k < 5; k++ )
    macro_xs_vector[k] = 0;

  

  

  

  

  

  if( grid_type == UNIONIZED )
    idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);
  else if( grid_type == HASH )
  {
    double du = 1.0 / hash_bins;
    idx = p_energy / du;
  }

  

  

  

  

  

  

  

  

  

  

  for( int j = 0; j < num_nucs[mat]; j++ )
  {
    double xs_vector[5];
    p_nuc = mats[mat*max_num_nucs + j];
    conc = concs[mat*max_num_nucs + j];
    calculate_micro_xs( p_energy, p_nuc, n_isotopes,
        n_gridpoints, egrid, index_data,
        nuclide_grids, idx, xs_vector, grid_type, hash_bins );
    for( int k = 0; k < 5; k++ )
      macro_xs_vector[k] += xs_vector[k] * conc;
  }
}



int pick_mat( unsigned long * seed )
{
  

  

  

  


  

  


  double dist[12];
  dist[0]  = 0.140;  

  dist[1]  = 0.052;  

  dist[2]  = 0.275;  

  dist[3]  = 0.134;  

  dist[4]  = 0.154;  

  dist[5]  = 0.064;  

  dist[6]  = 0.066;  

  dist[7]  = 0.055;  

  dist[8]  = 0.008;  

  dist[9]  = 0.015;  

  dist[10] = 0.025;  

  dist[11] = 0.013;  


  double roll = LCG_random_double(seed);

  

  for( int i = 0; i < 12; i++ )
  {
    double running = 0;
    for( int j = i; j > 0; j-- )
      running += dist[j];
    if( roll < running )
      return i;
  }

  return 0;
}

double LCG_random_double(uint64_t * seed)
{
  

  const uint64_t m = 9223372036854775808ULL; 

  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
  

  const uint64_t m = 9223372036854775808ULL; 

  uint64_t a = 2806196910506780709ULL;
  uint64_t c = 1ULL;

  n = n % m;

  uint64_t a_new = 1;
  uint64_t c_new = 0;

  while(n > 0)
  {
    if(n & 1)
    {
      a_new *= a;
      c_new = c_new * a + c;
    }
    c *= (a + 1);
    a *= a;

    n >>= 1;
  }

  return (a_new * seed + c_new) % m;
}

// Simple get_time implementation
double get_time(void)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Basic main function for serial version
int main(int argc, char* argv[])
{
  printf("XSBench Serial Version\n");
  printf("This is a placeholder implementation.\n");
  printf("The complete xsbench implementation appears to be missing.\n");
  return 0;
}
