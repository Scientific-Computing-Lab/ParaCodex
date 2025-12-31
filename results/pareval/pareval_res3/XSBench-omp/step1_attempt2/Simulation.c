#include "XSbench_header.h"

typedef struct {
	int mat;
	double energy;
} SamplePair;

static SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD);
static void release_device_memory(SimulationData GSD);
static void allocate_sample_buffers_on_device(Inputs in, SimulationData *GSD);
static void release_sample_buffers_on_device(SimulationData *GSD);
static void copy_verification_from_device(SimulationData GSD, unsigned long *verification, int count);
static unsigned long reduce_host_verification(unsigned long *verification, int count);
static unsigned long collect_verification_from_device(SimulationData GSD, int lookups);
static void copy_samples_device_to_host(SimulationData GSD, double *p_energy, int *mat_samples, int count);
static void copy_samples_host_to_device(SimulationData GSD, const double *p_energy, const int *mat_samples, int count);
static void sort_samples_by_material(int *mat_samples, double *p_energy_samples, int count);
static void sort_samples_by_energy_range(double *p_energy_samples, int *mat_samples, int offset, int count);
static void partition_samples(int *mat_samples, double *p_energy_samples, int count, int predicate, int *partition_point);
static int count_material_samples(const int *mat_samples, int count, int material);
static void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD );
static void sampling_kernel(Inputs in, SimulationData GSD );
static void xs_lookup_kernel_optimization_1(Inputs in, SimulationData GSD );
static void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m );
static void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel );
static void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset );
static void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset );

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile)
{
	double start = get_time();
        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
	profile->host_to_device_time = get_time() - start;

        if( mype == 0)	printf("Running baseline event-based simulation...\n");

	int nwarmups = in.num_warmups;
	start = 0.0;
	for (int i = 0; i < in.num_iterations + nwarmups; i++) {
		if (i == nwarmups) {
			start = get_time();
		}
		xs_lookup_kernel_baseline( in, GSD );
	}
	profile->kernel_time = get_time() - start;

        if( mype == 0)	printf("Reducing verification results...\n");
	double transfer_start = get_time();
	copy_verification_from_device(GSD, SD.verification, in.lookups);
	profile->device_to_host_time = get_time() - transfer_start;

        unsigned long verification_scalar = reduce_host_verification(SD.verification, in.lookups);

        release_device_memory(GSD);

        return verification_scalar;
}

static void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD )
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        long n_gridpoints = in.n_gridpoints;
        long n_isotopes = in.n_isotopes;
        int max_num_nucs = GSD.max_num_nucs;
        int lookups = in.lookups;

        #pragma omp target teams loop is_device_ptr(num_nucs, concs, unionized_energy_array, index_grid, nuclide_grid, mats, verification)
        for (int i = 0; i < lookups; ++i)
        {
                uint64_t seed = STARTING_SEED;

                seed = fast_forward_LCG(seed, 2*i);

                double p_energy = LCG_random_double(&seed);
                int mat         = pick_mat(&seed);

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy,
                        mat,
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
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

#pragma omp declare target
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                          long n_gridpoints,
                          const double * egrid, const int * index_data,
                          const NuclideGridPoint * nuclide_grids,
                          long idx, double * xs_vector, int grid_type, int hash_bins )
{
        double f;
        const NuclideGridPoint * low, * high;

        if( grid_type == NUCLIDE )
        {

                idx = grid_search_nuclide( n_gridpoints, p_energy, &nuclide_grids[nuc*n_gridpoints], 0, n_gridpoints-1);

                if( idx == n_gridpoints - 1 )
                        low = &nuclide_grids[nuc*n_gridpoints + idx - 1];
                else
                        low = &nuclide_grids[nuc*n_gridpoints + idx];
        }
        else if( grid_type == UNIONIZED)
        {

                if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
                        low = &nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1];
                else
                        low = &nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc]];
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
                int lower;
                if( p_energy <= e_low )
                        lower = 0;
                else if( p_energy >= e_high )
                        lower = n_gridpoints - 1;
                else
                        lower = grid_search_nuclide( n_gridpoints, p_energy, &nuclide_grids[nuc*n_gridpoints], u_low, u_high);

                if( lower == n_gridpoints - 1 )
                        low = &nuclide_grids[nuc*n_gridpoints + lower - 1];
                else
                        low = &nuclide_grids[nuc*n_gridpoints + lower];
        }

        high = low + 1;

        f = (high->energy - p_energy) / (high->energy - low->energy);

        xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);

        xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);

        xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);

        xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);

        xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, const int * num_nucs,
                         const double * concs,
                         const double * egrid, const int * index_data,
                         const NuclideGridPoint * nuclide_grids,
                         const int * mats,
                         double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs )
{
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

long grid_search( long n, double quarry, const double * A)
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

long grid_search_nuclide( long n, double quarry, const NuclideGridPoint * A, long low, long high)
{
        long lowerLimit = low;
        long upperLimit = high;
        long examinationPoint;
        long length = upperLimit - lowerLimit;

        while( length > 1 )
        {
                examinationPoint = lowerLimit + ( length / 2 );

                if( A[examinationPoint].energy > quarry )
                        upperLimit = examinationPoint;
                else
                        lowerLimit = examinationPoint;

                length = upperLimit - lowerLimit;
        }

        return lowerLimit;
}

int pick_mat( uint64_t * seed )
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
#pragma omp end declare target

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sample_sz = in.lookups * sizeof(double);
        size_t mat_sz = in.lookups * sizeof(int);
        size_t total_sz = sample_sz + mat_sz;

        allocate_sample_buffers_on_device(in, &GSD);

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );
        xs_lookup_kernel_optimization_1( in, GSD );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = collect_verification_from_device(GSD, in.lookups);
        release_sample_buffers_on_device(&GSD);

        return verification_scalar;
}

static void sampling_kernel(Inputs in, SimulationData GSD )
{
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int lookups = in.lookups;

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples)
        for (int i = 0; i < lookups; ++i)
        {
                uint64_t seed = STARTING_SEED;

                seed = fast_forward_LCG(seed, 2*i);

                double p_energy = LCG_random_double(&seed);
                int mat         = pick_mat(&seed);

                p_energy_samples[i] = p_energy;
                mat_samples[i] = mat;
        }
}

static void xs_lookup_kernel_optimization_1(Inputs in, SimulationData GSD )
{
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        long n_gridpoints = in.n_gridpoints;
        long n_isotopes = in.n_isotopes;
        int max_num_nucs = GSD.max_num_nucs;
        int lookups = in.lookups;

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, unionized_energy_array, index_grid, nuclide_grid, mats, verification)
        for (int i = 0; i < lookups; ++i)
        {
                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat_samples[i],
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
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

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 2 - Material Lookup Kernels";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sample_sz = in.lookups * sizeof(double);
        size_t mat_sz = in.lookups * sizeof(int);
        size_t total_sz = sample_sz + mat_sz;
        allocate_sample_buffers_on_device(in, &GSD);

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );

        for( int m = 0; m < 12; m++ )
                xs_lookup_kernel_optimization_2( in, GSD, m );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = collect_verification_from_device(GSD, in.lookups);
        release_sample_buffers_on_device(&GSD);

        return verification_scalar;
}

static void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m )
{
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        long n_gridpoints = in.n_gridpoints;
        long n_isotopes = in.n_isotopes;
        int max_num_nucs = GSD.max_num_nucs;
        int lookups = in.lookups;

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, unionized_energy_array, index_grid, nuclide_grid, mats, verification)
        for (int i = 0; i < lookups; ++i)
        {
                int mat = mat_samples[i];
                if( mat != m )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat,
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
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

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sample_sz = in.lookups * sizeof(double);
        size_t mat_sz = in.lookups * sizeof(int);
        size_t total_sz = sample_sz + mat_sz;
        allocate_sample_buffers_on_device(in, &GSD);

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );

        xs_lookup_kernel_optimization_3( in, GSD, 0 );
        xs_lookup_kernel_optimization_3( in, GSD, 1 );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = collect_verification_from_device(GSD, in.lookups);
        release_sample_buffers_on_device(&GSD);

        return verification_scalar;
}

static void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel )
{
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        long n_gridpoints = in.n_gridpoints;
        long n_isotopes = in.n_isotopes;
        int max_num_nucs = GSD.max_num_nucs;
        int lookups = in.lookups;

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, unionized_energy_array, index_grid, nuclide_grid, mats, verification)
        for (int i = 0; i < lookups; ++i)
        {
                int mat = mat_samples[i];

                if( ((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0 ) ))
                {
                        double macro_xs_vector[5] = {0};

                        calculate_macro_xs(
                                p_energy_samples[i],
                                mat,
                                n_isotopes,
                                n_gridpoints,
                                num_nucs,
                                concs,
                                unionized_energy_array,
                                index_grid,
                                nuclide_grid,
                                mats,
                                macro_xs_vector,
                                grid_type,
                                hash_bins,
                                max_num_nucs
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
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sample_sz = in.lookups * sizeof(double);
        size_t mat_sz = in.lookups * sizeof(int);
        size_t total_sz = sample_sz + mat_sz;
        allocate_sample_buffers_on_device(in, &GSD);

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );

        double *host_p_energy = (double *) malloc(sample_sz);
        int *host_mat = (int *) malloc(mat_sz);
        if( host_p_energy == NULL || host_mat == NULL ) {
                fprintf(stderr, "Failed to allocate host sample buffers\n");
                exit(1);
        }

        copy_samples_device_to_host(GSD, host_p_energy, host_mat, in.lookups);

        int n_lookups_per_material[12] = {0};
        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = count_material_samples(host_mat, in.lookups, m);

        sort_samples_by_material(host_mat, host_p_energy, in.lookups);
        copy_samples_host_to_device(GSD, host_p_energy, host_mat, in.lookups);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if( count == 0 )
                        continue;
                xs_lookup_kernel_optimization_4( in, GSD, m, count, offset );
                offset += count;
        }

        free(host_p_energy);
        free(host_mat);

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = collect_verification_from_device(GSD, in.lookups);
        release_sample_buffers_on_device(&GSD);

        return verification_scalar;
}

static void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset )
{
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        long n_gridpoints = in.n_gridpoints;
        long n_isotopes = in.n_isotopes;
        int max_num_nucs = GSD.max_num_nucs;

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, unionized_energy_array, index_grid, nuclide_grid, mats, verification)
        for (int thread_idx = 0; thread_idx < n_lookups; ++thread_idx)
        {
                int i = thread_idx + offset;
                int mat = mat_samples[i];
                if( mat != m )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat,
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
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

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sample_sz = in.lookups * sizeof(double);
        size_t mat_sz = in.lookups * sizeof(int);
        size_t total_sz = sample_sz + mat_sz;
        allocate_sample_buffers_on_device(in, &GSD);

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );

        double *host_p_energy = (double *) malloc(sample_sz);
        int *host_mat = (int *) malloc(mat_sz);
        if( host_p_energy == NULL || host_mat == NULL ) {
                fprintf(stderr, "Failed to allocate host sample buffers\n");
                exit(1);
        }

        copy_samples_device_to_host(GSD, host_p_energy, host_mat, in.lookups);

        int n_fuel_lookups = 0;
        partition_samples(host_mat, host_p_energy, in.lookups, 0, &n_fuel_lookups);

        copy_samples_host_to_device(GSD, host_p_energy, host_mat, in.lookups);

        xs_lookup_kernel_optimization_5( in, GSD, n_fuel_lookups, 0 );
        xs_lookup_kernel_optimization_5( in, GSD, in.lookups - n_fuel_lookups, n_fuel_lookups );

        free(host_p_energy);
        free(host_mat);

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = collect_verification_from_device(GSD, in.lookups);
        release_sample_buffers_on_device(&GSD);

        return verification_scalar;
}

static void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset )
{
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        long n_gridpoints = in.n_gridpoints;
        long n_isotopes = in.n_isotopes;
        int max_num_nucs = GSD.max_num_nucs;

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, unionized_energy_array, index_grid, nuclide_grid, mats, verification)
        for (int thread_idx = 0; thread_idx < n_lookups; ++thread_idx)
        {
                int i = thread_idx + offset;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat_samples[i],
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
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

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sample_sz = in.lookups * sizeof(double);
        size_t mat_sz = in.lookups * sizeof(int);
        size_t total_sz = sample_sz + mat_sz;
        allocate_sample_buffers_on_device(in, &GSD);

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );

        double *host_p_energy = (double *) malloc(sample_sz);
        int *host_mat = (int *) malloc(mat_sz);
        if( host_p_energy == NULL || host_mat == NULL ) {
                fprintf(stderr, "Failed to allocate host sample buffers\n");
                exit(1);
        }

        copy_samples_device_to_host(GSD, host_p_energy, host_mat, in.lookups);

        int n_lookups_per_material[12] = {0};
        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = count_material_samples(host_mat, in.lookups, m);

        sort_samples_by_material(host_mat, host_p_energy, in.lookups);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if( count > 0 )
                        sort_samples_by_energy_range(host_p_energy, host_mat, offset, count);
                offset += count;
        }

        copy_samples_host_to_device(GSD, host_p_energy, host_mat, in.lookups);

        free(host_p_energy);
        free(host_mat);

        offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if( count == 0 )
                        continue;
                xs_lookup_kernel_optimization_4( in, GSD, m, count, offset );
                offset += count;
        }

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = collect_verification_from_device(GSD, in.lookups);
        release_sample_buffers_on_device(&GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        switch( in.kernel_id )
        {
                case 0:
                        return run_event_based_simulation_baseline(in, SD, mype, profile);
                case 1:
                        return run_event_based_simulation_optimization_1(in, SD, mype);
                case 2:
                        return run_event_based_simulation_optimization_2(in, SD, mype);
                case 3:
                        return run_event_based_simulation_optimization_3(in, SD, mype);
                case 4:
                        return run_event_based_simulation_optimization_4(in, SD, mype);
                case 5:
                        return run_event_based_simulation_optimization_5(in, SD, mype);
                case 6:
                        return run_event_based_simulation_optimization_6(in, SD, mype);
                default:
                        fprintf(stderr, "Error: No kernel ID %d found!\n", in.kernel_id);
                        exit(1);
        }
}

static void copy_verification_from_device(SimulationData GSD, unsigned long *verification, int count)
{
        if( count == 0 || verification == NULL || GSD.verification == NULL )
                return;

        int device = omp_get_default_device();
        int host_device = omp_get_initial_device();
        size_t bytes = (size_t) count * sizeof(unsigned long);
        omp_target_memcpy(verification, GSD.verification, bytes, 0, 0, host_device, device);
}

static unsigned long reduce_host_verification(unsigned long *verification, int count)
{
        unsigned long sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for( int i = 0; i < count; i++ )
                sum += verification[i];
        return sum;
}

static unsigned long collect_verification_from_device(SimulationData GSD, int lookups)
{
        if( lookups == 0 )
                return 0;

        unsigned long *host_verification = (unsigned long *) malloc((size_t) lookups * sizeof(unsigned long));
        if( host_verification == NULL )
        {
                fprintf(stderr, "Failed to allocate verification buffer on host\n");
                exit(1);
        }

        copy_verification_from_device(GSD, host_verification, lookups);
        unsigned long result = reduce_host_verification(host_verification, lookups);
        free(host_verification);
        return result;
}

static void copy_samples_device_to_host(SimulationData GSD, double *p_energy, int *mat_samples, int count)
{
        if( count == 0 )
                return;

        int device = omp_get_default_device();
        int host_device = omp_get_initial_device();
        if( p_energy && GSD.p_energy_samples )
        {
                omp_target_memcpy(p_energy, GSD.p_energy_samples, (size_t) count * sizeof(double), 0, 0, host_device, device);
        }
        if( mat_samples && GSD.mat_samples )
        {
                omp_target_memcpy(mat_samples, GSD.mat_samples, (size_t) count * sizeof(int), 0, 0, host_device, device);
        }
}

static void copy_samples_host_to_device(SimulationData GSD, const double *p_energy, const int *mat_samples, int count)
{
        if( count == 0 )
                return;

        int device = omp_get_default_device();
        int host_device = omp_get_initial_device();
        if( p_energy && GSD.p_energy_samples )
        {
                omp_target_memcpy(GSD.p_energy_samples, p_energy, (size_t) count * sizeof(double), 0, 0, device, host_device);
        }
        if( mat_samples && GSD.mat_samples )
        {
                omp_target_memcpy(GSD.mat_samples, mat_samples, (size_t) count * sizeof(int), 0, 0, device, host_device);
        }
}

static int count_material_samples(const int *mat_samples, int count, int material)
{
        if( mat_samples == NULL )
                return 0;

        int matches = 0;
        for( int i = 0; i < count; i++ )
        {
                if( mat_samples[i] == material )
                        matches++;
        }
        return matches;
}

static int compare_samples_by_mat(const void *a, const void *b)
{
        const SamplePair *sa = (const SamplePair *) a;
        const SamplePair *sb = (const SamplePair *) b;
        return sa->mat - sb->mat;
}

static int compare_samples_by_energy(const void *a, const void *b)
{
        const SamplePair *sa = (const SamplePair *) a;
        const SamplePair *sb = (const SamplePair *) b;

        if( sa->energy < sb->energy )
                return -1;
        else if( sa->energy > sb->energy )
                return 1;
        return 0;
}

static void sort_samples_by_material(int *mat_samples, double *p_energy_samples, int count)
{
        if( count <= 1 )
                return;

        SamplePair *samples = (SamplePair *) malloc((size_t) count * sizeof(SamplePair));
        if( samples == NULL )
        {
                fprintf(stderr, "Failed to allocate sorting buffer\n");
                exit(1);
        }

        for( int i = 0; i < count; i++ )
        {
                samples[i].mat = mat_samples[i];
                samples[i].energy = p_energy_samples[i];
        }

        qsort(samples, count, sizeof(SamplePair), compare_samples_by_mat);

        for( int i = 0; i < count; i++ )
        {
                mat_samples[i] = samples[i].mat;
                p_energy_samples[i] = samples[i].energy;
        }

        free(samples);
}

static void sort_samples_by_energy_range(double *p_energy_samples, int *mat_samples, int offset, int count)
{
        if( count <= 1 )
                return;

        SamplePair *samples = (SamplePair *) malloc((size_t) count * sizeof(SamplePair));
        if( samples == NULL )
        {
                fprintf(stderr, "Failed to allocate sorting buffer\n");
                exit(1);
        }

        for( int i = 0; i < count; i++ )
        {
                samples[i].mat = mat_samples[offset + i];
                samples[i].energy = p_energy_samples[offset + i];
        }

        qsort(samples, count, sizeof(SamplePair), compare_samples_by_energy);

        for( int i = 0; i < count; i++ )
        {
                mat_samples[offset + i] = samples[i].mat;
                p_energy_samples[offset + i] = samples[i].energy;
        }

        free(samples);
}

static void partition_samples(int *mat_samples, double *p_energy_samples, int count, int predicate, int *partition_point)
{
        int write = 0;
        for( int i = 0; i < count; i++ )
        {
                if( mat_samples[i] == predicate )
                {
                        if( i != write )
                        {
                                int tmp_mat = mat_samples[write];
                                mat_samples[write] = mat_samples[i];
                                mat_samples[i] = tmp_mat;

                                double tmp_energy = p_energy_samples[write];
                                p_energy_samples[write] = p_energy_samples[i];
                                p_energy_samples[i] = tmp_energy;
                        }
                        write++;
                }
        }

        if( partition_point )
                *partition_point = write;
}

static void allocate_sample_buffers_on_device(Inputs in, SimulationData *GSD)
{
        int device = omp_get_default_device();
        size_t sample_sz = (size_t) in.lookups * sizeof(double);
        size_t mat_sz = (size_t) in.lookups * sizeof(int);

        GSD->p_energy_samples = (double *) omp_target_alloc(sample_sz, device);
        if( GSD->p_energy_samples == NULL )
        {
                fprintf(stderr, "Failed to allocate energy samples on device\n");
                exit(1);
        }
        GSD->mat_samples = (int *) omp_target_alloc(mat_sz, device);
        if( GSD->mat_samples == NULL )
        {
                fprintf(stderr, "Failed to allocate material samples on device\n");
                exit(1);
        }

        GSD->length_p_energy_samples = in.lookups;
        GSD->length_mat_samples = in.lookups;
}

static void release_sample_buffers_on_device(SimulationData *GSD)
{
        int device = omp_get_default_device();
        if( GSD->p_energy_samples )
        {
                omp_target_free(GSD->p_energy_samples, device);
                GSD->p_energy_samples = NULL;
                GSD->length_p_energy_samples = 0;
        }
        if( GSD->mat_samples )
        {
                omp_target_free(GSD->mat_samples, device);
                GSD->mat_samples = NULL;
                GSD->length_mat_samples = 0;
        }
}

static SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD)
{
        if( mype == 0) printf("Allocating and moving simulation data to GPU memory space...\n");

        int device = omp_get_default_device();
        int host_device = omp_get_initial_device();
        SimulationData GSD = SD;
        size_t total_sz = 0;

        if( SD.length_num_nucs > 0 )
        {
                size_t bytes = (size_t) SD.length_num_nucs * sizeof(int);
                GSD.num_nucs = (int *) omp_target_alloc(bytes, device);
                if( GSD.num_nucs == NULL )
                {
                        fprintf(stderr, "Failed to allocate num_nucs on device\n");
                        exit(1);
                }
                omp_target_memcpy(GSD.num_nucs, SD.num_nucs, bytes, 0, 0, device, host_device);
                total_sz += bytes;
        }
        else
        {
                GSD.num_nucs = NULL;
        }

        if( SD.length_concs > 0 )
        {
                size_t bytes = (size_t) SD.length_concs * sizeof(double);
                GSD.concs = (double *) omp_target_alloc(bytes, device);
                if( GSD.concs == NULL )
                {
                        fprintf(stderr, "Failed to allocate concs on device\n");
                        exit(1);
                }
                omp_target_memcpy(GSD.concs, SD.concs, bytes, 0, 0, device, host_device);
                total_sz += bytes;
        }
        else
        {
                GSD.concs = NULL;
        }

        if( SD.length_mats > 0 )
        {
                size_t bytes = (size_t) SD.length_mats * sizeof(int);
                GSD.mats = (int *) omp_target_alloc(bytes, device);
                if( GSD.mats == NULL )
                {
                        fprintf(stderr, "Failed to allocate mats on device\n");
                        exit(1);
                }
                omp_target_memcpy(GSD.mats, SD.mats, bytes, 0, 0, device, host_device);
                total_sz += bytes;
        }
        else
        {
                GSD.mats = NULL;
        }

        if( SD.length_unionized_energy_array > 0 )
        {
                size_t bytes = (size_t) SD.length_unionized_energy_array * sizeof(double);
                GSD.unionized_energy_array = (double *) omp_target_alloc(bytes, device);
                if( GSD.unionized_energy_array == NULL )
                {
                        fprintf(stderr, "Failed to allocate unionized_energy_array on device\n");
                        exit(1);
                }
                omp_target_memcpy(GSD.unionized_energy_array, SD.unionized_energy_array, bytes, 0, 0, device, host_device);
                total_sz += bytes;
        }
        else
        {
                GSD.unionized_energy_array = NULL;
        }

        if( SD.length_index_grid > 0 )
        {
                size_t bytes = (size_t) SD.length_index_grid * sizeof(int);
                GSD.index_grid = (int *) omp_target_alloc(bytes, device);
                if( GSD.index_grid == NULL )
                {
                        fprintf(stderr, "Failed to allocate index_grid on device\n");
                        exit(1);
                }
                omp_target_memcpy(GSD.index_grid, SD.index_grid, bytes, 0, 0, device, host_device);
                total_sz += bytes;
        }
        else
        {
                GSD.index_grid = NULL;
        }

        if( SD.length_nuclide_grid > 0 )
        {
                size_t bytes = (size_t) SD.length_nuclide_grid * sizeof(NuclideGridPoint);
                GSD.nuclide_grid = (NuclideGridPoint *) omp_target_alloc(bytes, device);
                if( GSD.nuclide_grid == NULL )
                {
                        fprintf(stderr, "Failed to allocate nuclide_grid on device\n");
                        exit(1);
                }
                omp_target_memcpy(GSD.nuclide_grid, SD.nuclide_grid, bytes, 0, 0, device, host_device);
                total_sz += bytes;
        }
        else
        {
                GSD.nuclide_grid = NULL;
        }

        size_t verification_bytes = (size_t) in.lookups * sizeof(unsigned long);
        GSD.verification = (unsigned long *) omp_target_alloc(verification_bytes, device);
        if( GSD.verification == NULL )
        {
                fprintf(stderr, "Failed to allocate verification buffer on device\n");
                exit(1);
        }
        total_sz += verification_bytes;
        GSD.length_verification = in.lookups;

        GSD.p_energy_samples = NULL;
        GSD.mat_samples = NULL;
        GSD.length_p_energy_samples = 0;
        GSD.length_mat_samples = 0;

        if( mype == 0 ) printf("GPU Intialization complete. Allocated %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0 );

        return GSD;
}

static void release_device_memory(SimulationData GSD)
{
        int device = omp_get_default_device();
        if( GSD.num_nucs )
                omp_target_free(GSD.num_nucs, device);
        if( GSD.concs )
                omp_target_free(GSD.concs, device);
        if( GSD.mats )
                omp_target_free(GSD.mats, device);
        if( GSD.unionized_energy_array )
                omp_target_free(GSD.unionized_energy_array, device);
        if( GSD.index_grid )
                omp_target_free(GSD.index_grid, device);
        if( GSD.nuclide_grid )
                omp_target_free(GSD.nuclide_grid, device);
        if( GSD.verification )
                omp_target_free(GSD.verification, device);
}
