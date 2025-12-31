#include "XSbench_header.h"

static unsigned long long host_reduce_verification(const unsigned long* verification, int lookups);
static int host_count(const int* arr, int n, int value);
static void sort_key_value_int_double(int* keys, double* values, int n);
static void sort_key_value_double_int(double* keys, int* values, int n);
static int host_partition_fuel(int* mats, double* energies, int n);
static void xs_lookup_kernel_baseline(const Inputs* in, SimulationData* GSD );
static void sampling_kernel(const Inputs* in, SimulationData* GSD );
static void xs_lookup_kernel_optimization_1(const Inputs* in, SimulationData* GSD );
static void xs_lookup_kernel_optimization_2(const Inputs* in, SimulationData* GSD, int m );
static void xs_lookup_kernel_optimization_3(const Inputs* in, SimulationData* GSD, int is_fuel );
static void xs_lookup_kernel_optimization_4(const Inputs* in, SimulationData* GSD, int m, int n_lookups, int offset );
static void xs_lookup_kernel_optimization_5(const Inputs* in, SimulationData* GSD, int n_lookups, int offset );
static int compare_int_double(const void* a, const void* b);
static int compare_double_int(const void* a, const void* b);
unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype, Profile* profile);
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile);
unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if( mype == 0)	printf("Simulation Kernel:"%s"
", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...
");
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *) calloc(in.lookups, sizeof(double));
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *) malloc(sz);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.
", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...
");

        #pragma omp target data map(to: GSD.num_nucs[0:GSD.length_num_nucs],                                    GSD.concs[0:GSD.length_concs],                                    GSD.mats[0:GSD.length_mats],                                    GSD.unionized_energy_array[0:GSD.length_unionized_energy_array],                                    GSD.index_grid[0:GSD.length_index_grid],                                    GSD.nuclide_grid[0:GSD.length_nuclide_grid])                                 map(tofrom: GSD.verification[0:in.lookups])                                 map(alloc: GSD.p_energy_samples[0:in.lookups], GSD.mat_samples[0:in.lookups])
        {
                sampling_kernel(&in, &GSD);
                xs_lookup_kernel_optimization_1(&in, &GSD);
        }

        if( mype == 0)	printf("Reducing verification results...
");

        unsigned long verification_scalar = host_reduce_verification(GSD.verification, in.lookups);

        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        GSD.p_energy_samples = NULL;
        GSD.mat_samples = NULL;

        return verification_scalar;
}
unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype);

unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        switch (in.kernel_id)
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
                        fprintf(stderr, "Error: unsupported kernel_id %d for OpenMP code.\n", in.kernel_id);
                        exit(1);
        }
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile)
{
	double start = get_time();
	SimulationData GSD = SD;
	double device_start = 0.0;

	#pragma omp target data map(to: GSD.num_nucs[0:GSD.length_num_nucs], \
	                           GSD.concs[0:GSD.length_concs], \
	                           GSD.mats[0:GSD.length_mats], \
	                           GSD.unionized_energy_array[0:GSD.length_unionized_energy_array], \
	                           GSD.index_grid[0:GSD.length_index_grid], \
	                           GSD.nuclide_grid[0:GSD.length_nuclide_grid]) \
	                        map(tofrom: GSD.verification[0:in.lookups])
	{
		profile->host_to_device_time = get_time() - start;

		if( mype == 0)	printf("Running baseline event-based simulation...\n");

		int nwarmups = in.num_warmups;
		double kernel_start = 0.0;
		for (int i = 0; i < in.num_iterations + nwarmups; i++) {
			if (i == nwarmups)
				kernel_start = get_time();
			xs_lookup_kernel_baseline(&in, &GSD);
		}
		profile->kernel_time = get_time() - kernel_start;

		if( mype == 0)	printf("Reducing verification results...\n");
		device_start = get_time();
	}

	profile->device_to_host_time = get_time() - device_start;

	return host_reduce_verification(GSD.verification, in.lookups);
}

static void xs_lookup_kernel_baseline(const Inputs* in, SimulationData* GSD )
{
        const int lookups = in->lookups;
        int* num_nucs = GSD->num_nucs;
        double* concs = GSD->concs;
        double* unionized = GSD->unionized_energy_array;
        int* index_grid = GSD->index_grid;
        NuclideGridPoint* nuclide_grid = GSD->nuclide_grid;
        int* mats = GSD->mats;
        unsigned long* verification = GSD->verification;
        const int grid_type = in->grid_type;
        const int hash_bins = in->hash_bins;
        const int max_num_nucs = GSD->max_num_nucs;
        const long n_isotopes = in->n_isotopes;
        const long n_gridpoints = in->n_gridpoints;

        #pragma omp target teams loop thread_limit(256) map(to: *in) \
                is_device_ptr(verification, num_nucs, concs, unionized, index_grid, nuclide_grid, mats)
        for (int i = 0; i < lookups; i++)
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*i);

                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy,
                        mat,
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        unionized,
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
                                   const double * __restrict__ egrid, const int * __restrict__ index_data,
                                   const NuclideGridPoint * __restrict__ nuclide_grids,
                                   long idx, double * __restrict__ xs_vector, int grid_type, int hash_bins )
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
                                   long n_gridpoints, const int * __restrict__ num_nucs,
                                   const double * __restrict__ concs,
                                   const double * __restrict__ egrid, const int * __restrict__ index_data,
                                   const NuclideGridPoint * __restrict__ nuclide_grids,
                                   const int * __restrict__ mats,
                                   double * __restrict__ macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs )
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

        long grid_search( long n, double quarry, const double * __restrict__ A)
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
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *) calloc(in.lookups, sizeof(double));
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *) malloc(sz);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        #pragma omp target data map(to: GSD.num_nucs[0:GSD.length_num_nucs], \
                                   GSD.concs[0:GSD.length_concs], \
                                   GSD.mats[0:GSD.length_mats], \
                                   GSD.unionized_energy_array[0:GSD.length_unionized_energy_array], \
                                   GSD.index_grid[0:GSD.length_index_grid], \
                                   GSD.nuclide_grid[0:GSD.length_nuclide_grid]) \
                                map(tofrom: GSD.verification[0:in.lookups]) \
                                map(alloc: GSD.p_energy_samples[0:in.lookups], GSD.mat_samples[0:in.lookups])
        {
                sampling_kernel(&in, &GSD);
                xs_lookup_kernel_optimization_1(&in, &GSD);
        }

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = host_reduce_verification(GSD.verification, in.lookups);

        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        GSD.p_energy_samples = NULL;
        GSD.mat_samples = NULL;

        return verification_scalar;
}

static void sampling_kernel(const Inputs* in, SimulationData* GSD )
{
        const int lookups = in->lookups;
        double* p_energy_samples = GSD->p_energy_samples;
        int* mat_samples = GSD->mat_samples;

        #pragma omp target teams loop thread_limit(32) map(to: *in) \
                is_device_ptr(p_energy_samples, mat_samples)
        for (int i = 0; i < lookups; i++)
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*i);

                double p_energy = LCG_random_double(&seed);
                int mat         = pick_mat(&seed);

                p_energy_samples[i] = p_energy;
                mat_samples[i] = mat;
        }
}

static void xs_lookup_kernel_optimization_1(const Inputs* in, SimulationData* GSD )
{
        const int lookups = in->lookups;
        double* p_energy_samples = GSD->p_energy_samples;
        int* mat_samples = GSD->mat_samples;
        int* num_nucs = GSD->num_nucs;
        double* concs = GSD->concs;
        double* unionized = GSD->unionized_energy_array;
        int* index_grid = GSD->index_grid;
        NuclideGridPoint* nuclide_grid = GSD->nuclide_grid;
        int* mats = GSD->mats;
        unsigned long* verification = GSD->verification;
        const int grid_type = in->grid_type;
        const int hash_bins = in->hash_bins;
        const int max_num_nucs = GSD->max_num_nucs;
        const long n_isotopes = in->n_isotopes;
        const long n_gridpoints = in->n_gridpoints;

        #pragma omp target teams loop thread_limit(32) map(to: *in) \
                is_device_ptr(p_energy_samples, mat_samples, verification, num_nucs, concs, unionized, index_grid, nuclide_grid, mats)
        for (int i = 0; i < lookups; i++)
        {
                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat_samples[i],
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        unionized,
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
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *) calloc(in.lookups, sizeof(double));
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *) malloc(sz);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        enter_simulation_data(&in, &GSD);

        if (in.lookups > 0)
        {
#pragma omp target enter data map(alloc: GSD.p_energy_samples[0:in.lookups])
#pragma omp target enter data map(alloc: GSD.mat_samples[0:in.lookups])
        }

        sampling_kernel(&in, &GSD);
        for( int m = 0; m < 12; m++ )
                xs_lookup_kernel_optimization_2(&in, &GSD, m );

        if (in.lookups > 0)
        {
#pragma omp target exit data map(release: GSD.p_energy_samples[0:in.lookups])
#pragma omp target exit data map(release: GSD.mat_samples[0:in.lookups])
        }

        if( mype == 0)	printf("Reducing verification results...\n");
        exit_simulation_data(&in, &GSD);

        unsigned long verification_scalar = host_reduce_verification(GSD.verification, in.lookups);

        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        GSD.p_energy_samples = NULL;
        GSD.mat_samples = NULL;

        return verification_scalar;
}

static void xs_lookup_kernel_optimization_2(const Inputs* in, SimulationData* GSD, int m )
{
        const int lookups = in->lookups;
        double* p_energy_samples = GSD->p_energy_samples;
        int* mat_samples = GSD->mat_samples;
        int* num_nucs = GSD->num_nucs;
        double* concs = GSD->concs;
        double* unionized = GSD->unionized_energy_array;
        int* index_grid = GSD->index_grid;
        NuclideGridPoint* nuclide_grid = GSD->nuclide_grid;
        int* mats = GSD->mats;
        unsigned long* verification = GSD->verification;
        const int grid_type = in->grid_type;
        const int hash_bins = in->hash_bins;
        const int max_num_nucs = GSD->max_num_nucs;
        const long n_isotopes = in->n_isotopes;
        const long n_gridpoints = in->n_gridpoints;

        #pragma omp target teams loop thread_limit(32) map(to: *in) \
                is_device_ptr(p_energy_samples, mat_samples, verification, num_nucs, concs, unionized, index_grid, nuclide_grid, mats)
        for (int i = 0; i < lookups; i++)
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
                        unionized,
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
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *) calloc(in.lookups, sizeof(double));
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *) malloc(sz);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        enter_simulation_data(&in, &GSD);

        if (in.lookups > 0)
        {
#pragma omp target enter data map(alloc: GSD.p_energy_samples[0:in.lookups])
#pragma omp target enter data map(alloc: GSD.mat_samples[0:in.lookups])
        }

        sampling_kernel(&in, &GSD);
        xs_lookup_kernel_optimization_3(&in, &GSD, 0 );
        xs_lookup_kernel_optimization_3(&in, &GSD, 1 );

        if (in.lookups > 0)
        {
#pragma omp target exit data map(release: GSD.p_energy_samples[0:in.lookups])
#pragma omp target exit data map(release: GSD.mat_samples[0:in.lookups])
        }

        if( mype == 0)	printf("Reducing verification results...\n");
        exit_simulation_data(&in, &GSD);

        unsigned long verification_scalar = host_reduce_verification(GSD.verification, in.lookups);

        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        GSD.p_energy_samples = NULL;
        GSD.mat_samples = NULL;

        return verification_scalar;
}

static void xs_lookup_kernel_optimization_3(const Inputs* in, SimulationData* GSD, int is_fuel )
{
        const int lookups = in->lookups;
        double* p_energy_samples = GSD->p_energy_samples;
        int* mat_samples = GSD->mat_samples;
        int* num_nucs = GSD->num_nucs;
        double* concs = GSD->concs;
        double* unionized = GSD->unionized_energy_array;
        int* index_grid = GSD->index_grid;
        NuclideGridPoint* nuclide_grid = GSD->nuclide_grid;
        int* mats = GSD->mats;
        unsigned long* verification = GSD->verification;
        const int grid_type = in->grid_type;
        const int hash_bins = in->hash_bins;
        const int max_num_nucs = GSD->max_num_nucs;
        const long n_isotopes = in->n_isotopes;
        const long n_gridpoints = in->n_gridpoints;

        #pragma omp target teams loop thread_limit(32) map(to: *in) \
                is_device_ptr(p_energy_samples, mat_samples, verification, num_nucs, concs, unionized, index_grid, nuclide_grid, mats)
        for (int i = 0; i < lookups; i++)
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
                                unionized,
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
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *) calloc(in.lookups, sizeof(double));
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *) malloc(sz);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        enter_simulation_data(&in, &GSD);

        if (in.lookups > 0)
        {
#pragma omp target enter data map(alloc: GSD.p_energy_samples[0:in.lookups])
#pragma omp target enter data map(alloc: GSD.mat_samples[0:in.lookups])
        }

        sampling_kernel(&in, &GSD);

        int n_lookups_per_material[12];
        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = host_count(GSD.mat_samples, in.lookups, m);

        sort_key_value_int_double(GSD.mat_samples, GSD.p_energy_samples, in.lookups);

        if (in.lookups > 0)
        {
#pragma omp target update to(GSD.mat_samples[0:in.lookups])
#pragma omp target update to(GSD.p_energy_samples[0:in.lookups])
        }

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if (count == 0)
                        continue;
                xs_lookup_kernel_optimization_4(&in, &GSD, m, count, offset );
                offset += count;
        }

        if (in.lookups > 0)
        {
#pragma omp target exit data map(release: GSD.p_energy_samples[0:in.lookups])
#pragma omp target exit data map(release: GSD.mat_samples[0:in.lookups])
        }

        if( mype == 0)	printf("Reducing verification results...\n");
        exit_simulation_data(&in, &GSD);

        unsigned long verification_scalar = host_reduce_verification(GSD.verification, in.lookups);

        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        GSD.p_energy_samples = NULL;
        GSD.mat_samples = NULL;

        return verification_scalar;
}

static void xs_lookup_kernel_optimization_4(const Inputs* in, SimulationData* GSD, int m, int n_lookups, int offset )
{
        double* p_energy_samples = GSD->p_energy_samples;
        int* mat_samples = GSD->mat_samples;
        int* num_nucs = GSD->num_nucs;
        double* concs = GSD->concs;
        double* unionized = GSD->unionized_energy_array;
        int* index_grid = GSD->index_grid;
        NuclideGridPoint* nuclide_grid = GSD->nuclide_grid;
        int* mats = GSD->mats;
        unsigned long* verification = GSD->verification;
        const int grid_type = in->grid_type;
        const int hash_bins = in->hash_bins;
        const int max_num_nucs = GSD->max_num_nucs;
        const long n_isotopes = in->n_isotopes;
        const long n_gridpoints = in->n_gridpoints;

        #pragma omp target teams loop thread_limit(32) map(to: *in) \
                is_device_ptr(p_energy_samples, mat_samples, verification, num_nucs, concs, unionized, index_grid, nuclide_grid, mats)
        for (int idx = 0; idx < n_lookups; idx++)
        {
                int i = idx + offset;

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
                        unionized,
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
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *) calloc(in.lookups, sizeof(double));
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *) malloc(sz);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        enter_simulation_data(&in, &GSD);

        if (in.lookups > 0)
        {
#pragma omp target enter data map(alloc: GSD.p_energy_samples[0:in.lookups])
#pragma omp target enter data map(alloc: GSD.mat_samples[0:in.lookups])
        }

        sampling_kernel(&in, &GSD);

        int n_fuel_lookups = host_count(GSD.mat_samples, in.lookups, 0);
        int partition_index = host_partition_fuel(GSD.mat_samples, GSD.p_energy_samples, in.lookups);
        (void) partition_index; // for clarity; host_partition_fuel returns the first non-fuel index (should equal n_fuel_lookups)

        if (in.lookups > 0)
        {
#pragma omp target update to(GSD.mat_samples[0:in.lookups])
#pragma omp target update to(GSD.p_energy_samples[0:in.lookups])
        }

        xs_lookup_kernel_optimization_5(&in, &GSD, n_fuel_lookups, 0 );
        xs_lookup_kernel_optimization_5(&in, &GSD, in.lookups - n_fuel_lookups, n_fuel_lookups );

        if (in.lookups > 0)
        {
#pragma omp target exit data map(release: GSD.p_energy_samples[0:in.lookups])
#pragma omp target exit data map(release: GSD.mat_samples[0:in.lookups])
        }

        if( mype == 0)	printf("Reducing verification results...\n");
        exit_simulation_data(&in, &GSD);

        unsigned long verification_scalar = host_reduce_verification(GSD.verification, in.lookups);

        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        GSD.p_energy_samples = NULL;
        GSD.mat_samples = NULL;

        return verification_scalar;
}

static void xs_lookup_kernel_optimization_5(const Inputs* in, SimulationData* GSD, int n_lookups, int offset )
{
        double* p_energy_samples = GSD->p_energy_samples;
        int* mat_samples = GSD->mat_samples;
        int* num_nucs = GSD->num_nucs;
        double* concs = GSD->concs;
        double* unionized = GSD->unionized_energy_array;
        int* index_grid = GSD->index_grid;
        NuclideGridPoint* nuclide_grid = GSD->nuclide_grid;
        int* mats = GSD->mats;
        unsigned long* verification = GSD->verification;
        const int grid_type = in->grid_type;
        const int hash_bins = in->hash_bins;
        const int max_num_nucs = GSD->max_num_nucs;
        const long n_isotopes = in->n_isotopes;
        const long n_gridpoints = in->n_gridpoints;

        #pragma omp target teams loop thread_limit(32) map(to: *in) \
                is_device_ptr(p_energy_samples, mat_samples, verification, num_nucs, concs, unionized, index_grid, nuclide_grid, mats)
        for (int idx = 0; idx < n_lookups; idx++)
        {
                int i = idx + offset;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat_samples[i],
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        unionized,
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
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *) calloc(in.lookups, sizeof(double));
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *) malloc(sz);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        enter_simulation_data(&in, &GSD);

        if (in.lookups > 0)
        {
#pragma omp target enter data map(alloc: GSD.p_energy_samples[0:in.lookups])
#pragma omp target enter data map(alloc: GSD.mat_samples[0:in.lookups])
        }

        sampling_kernel(&in, &GSD);

        int n_lookups_per_material[12];
        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = host_count(GSD.mat_samples, in.lookups, m);

        sort_key_value_int_double(GSD.mat_samples, GSD.p_energy_samples, in.lookups);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if (count == 0)
                {
                        continue;
                }
                sort_key_value_double_int(GSD.p_energy_samples + offset, GSD.mat_samples + offset, count);
                offset += count;
        }

        if (in.lookups > 0)
        {
#pragma omp target update to(GSD.mat_samples[0:in.lookups])
#pragma omp target update to(GSD.p_energy_samples[0:in.lookups])
        }

        offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if (count == 0)
                {
                        continue;
                }
                xs_lookup_kernel_optimization_4(&in, &GSD, m, count, offset );
                offset += count;
        }

        if (in.lookups > 0)
        {
#pragma omp target exit data map(release: GSD.p_energy_samples[0:in.lookups])
#pragma omp target exit data map(release: GSD.mat_samples[0:in.lookups])
        }

        if( mype == 0)	printf("Reducing verification results...\n");
        exit_simulation_data(&in, &GSD);

        unsigned long verification_scalar = host_reduce_verification(GSD.verification, in.lookups);

        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        GSD.p_energy_samples = NULL;
        GSD.mat_samples = NULL;

        return verification_scalar;
}

static unsigned long long host_reduce_verification(const unsigned long* verification, int lookups)
{
        unsigned long long sum = 0;
        for (int i = 0; i < lookups; i++)
                sum += verification[i];
        return sum;
}

static int host_count(const int* arr, int n, int value)
{
        int count = 0;
        for (int i = 0; i < n; i++)
                if (arr[i] == value)
                        count++;
        return count;
}

typedef struct { int key; double value; } KeyValueIntDouble;
typedef struct { double key; int value; } KeyValueDoubleInt;

static void sort_key_value_int_double(int* keys, double* values, int n)
{
        KeyValueIntDouble* pairs = (KeyValueIntDouble*) malloc(n * sizeof(KeyValueIntDouble));
        for (int i = 0; i < n; i++)
        {
                pairs[i].key = keys[i];
                pairs[i].value = values[i];
        }

        qsort(pairs, n, sizeof(KeyValueIntDouble), compare_int_double);

        for (int i = 0; i < n; i++)
        {
                keys[i] = pairs[i].key;
                values[i] = pairs[i].value;
        }

        free(pairs);
}

static void sort_key_value_double_int(double* keys, int* values, int n)
{
        KeyValueDoubleInt* pairs = (KeyValueDoubleInt*) malloc(n * sizeof(KeyValueDoubleInt));
        for (int i = 0; i < n; i++)
        {
                pairs[i].key = keys[i];
                pairs[i].value = values[i];
        }

        qsort(pairs, n, sizeof(KeyValueDoubleInt), compare_double_int);

        for (int i = 0; i < n; i++)
        {
                keys[i] = pairs[i].key;
                values[i] = pairs[i].value;
        }

        free(pairs);
}

static int host_partition_fuel(int* mats, double* energies, int n)
{
        int left = 0;
        int right = n - 1;
        while (left <= right)
        {
                while (left <= right && mats[left] == 0)
                        left++;
                while (left <= right && mats[right] != 0)
                        right--;
                if (left < right)
                {
                        int tmp_mat = mats[left];
                        mats[left] = mats[right];
                        mats[right] = tmp_mat;

                        double tmp_energy = energies[left];
                        energies[left] = energies[right];
                        energies[right] = tmp_energy;

                        left++;
                        right--;
                }
        }
        return left;
}

static int compare_int_double(const void* a, const void* b)
{
        const KeyValueIntDouble* lhs = (const KeyValueIntDouble*) a;
        const KeyValueIntDouble* rhs = (const KeyValueIntDouble*) b;
        return lhs->key - rhs->key;
}

static int compare_double_int(const void* a, const void* b)
{
        const KeyValueDoubleInt* lhs = (const KeyValueDoubleInt*) a;
        const KeyValueDoubleInt* rhs = (const KeyValueDoubleInt*) b;
        if (lhs->key < rhs->key)
                return -1;
        if (lhs->key > rhs->key)
                return 1;
        return 0;
}
