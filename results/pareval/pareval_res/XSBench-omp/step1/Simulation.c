#include "XSbench_header.h"

// Pair helper for sorting material-energy tuples on the host.
typedef struct {
        int mat;
        double energy;
} MatSamplePair;

static inline int host_device_id(void)
{
        return omp_get_initial_device();
}

static inline int gpu_device_id(void)
{
        return omp_get_default_device();
}

static inline void copy_to_device(void *dest, const void *src, size_t size)
{
        omp_target_memcpy(dest, src, size, 0, 0, gpu_device_id(), host_device_id());
}

static inline void copy_to_host(void *dest, const void *src, size_t size)
{
        omp_target_memcpy(dest, src, size, 0, 0, host_device_id(), gpu_device_id());
}

SimulationData move_simulation_data_to_device( Inputs in, int mype, SimulationData SD );
void release_device_memory( SimulationData GSD );
static void allocate_sample_buffers( Inputs in, SimulationData * GSD );
static void free_sample_buffers( SimulationData * GSD );
static void copy_sample_buffers_to_host( const SimulationData * GSD, double * p_energy_samples, int * mat_samples, int lookups );
static void copy_sample_buffers_to_device( SimulationData * GSD, const double * p_energy_samples, const int * mat_samples, int lookups );
static unsigned long reduce_verification_from_device( SimulationData GSD, unsigned long * host_verification, int lookups );
static int count_material( const int * mat_samples, int lookups, int material );
static void sort_samples_by_material( int * mat_samples, double * p_energy_samples, int lookups );
static void sort_samples_by_energy_range( int * mat_samples, double * p_energy_samples, int offset, int len );
static void partition_samples_by_fuel( int * mat_samples, double * p_energy_samples, int lookups, int n_fuel );
static int compare_material_pairs( const void * a, const void * b );
static int compare_energy_pairs( const void * a, const void * b );

void xs_lookup_kernel_baseline( Inputs in, SimulationData GSD );
void sampling_kernel( Inputs in, SimulationData GSD );
void xs_lookup_kernel_optimization_1( Inputs in, SimulationData GSD );
void xs_lookup_kernel_optimization_2( Inputs in, SimulationData GSD, int m );
void xs_lookup_kernel_optimization_3( Inputs in, SimulationData GSD, int is_fuel );
void xs_lookup_kernel_optimization_4( Inputs in, SimulationData GSD, int m, int n_lookups, int offset );
void xs_lookup_kernel_optimization_5( Inputs in, SimulationData GSD, int n_lookups, int offset );

unsigned long long run_event_based_simulation_baseline( Inputs in, SimulationData SD, int mype, Profile * profile )
{
        double start = get_time();
        SimulationData GSD = move_simulation_data_to_device( in, mype, SD );
        profile->host_to_device_time = get_time() - start;

        if( mype == 0 )
                printf("Running baseline event-based simulation...\n");

        int total_iters = in.num_iterations + in.num_warmups;
        start = 0.0;
        for( int iter = 0; iter < total_iters; iter++ )
        {
                if( iter == in.num_warmups )
                        start = get_time();
                xs_lookup_kernel_baseline( in, GSD );
        }
        profile->kernel_time = get_time() - start;

        if( mype == 0 )
                printf("Reducing verification results...\n");
        start = get_time();
        unsigned long verification_scalar = reduce_verification_from_device( GSD, SD.verification, in.lookups );
        profile->device_to_host_time = get_time() - start;

        release_device_memory( GSD );
        return verification_scalar;
}

unsigned long long run_event_based_simulation( Inputs in, SimulationData SD, int mype, Profile * profile )
{
        return run_event_based_simulation_baseline( in, SD, mype, profile );
}

SimulationData move_simulation_data_to_device( Inputs in, int mype, SimulationData SD )
{
        if( mype == 0 )
                printf("Allocating and moving simulation data to GPU memory space...\n");

        SimulationData GSD = SD;
        int device = gpu_device_id();
        int host = host_device_id();
        size_t sz;
        size_t total_sz = 0;

        sz = GSD.length_num_nucs * sizeof(int);
        GSD.num_nucs = (int *) omp_target_alloc( sz, device );
        assert( GSD.num_nucs != NULL );
        copy_to_device( GSD.num_nucs, SD.num_nucs, sz );
        total_sz += sz;

        sz = GSD.length_concs * sizeof(double);
        GSD.concs = (double *) omp_target_alloc( sz, device );
        assert( GSD.concs != NULL );
        copy_to_device( GSD.concs, SD.concs, sz );
        total_sz += sz;

        sz = GSD.length_mats * sizeof(int);
        GSD.mats = (int *) omp_target_alloc( sz, device );
        assert( GSD.mats != NULL );
        copy_to_device( GSD.mats, SD.mats, sz );
        total_sz += sz;

        if( SD.length_unionized_energy_array > 0 )
        {
                sz = GSD.length_unionized_energy_array * sizeof(double);
                GSD.unionized_energy_array = (double *) omp_target_alloc( sz, device );
                assert( GSD.unionized_energy_array != NULL );
                copy_to_device( GSD.unionized_energy_array, SD.unionized_energy_array, sz );
                total_sz += sz;
        }
        else
        {
                GSD.unionized_energy_array = NULL;
        }

        if( SD.length_index_grid > 0 )
        {
                sz = SD.length_index_grid * sizeof(int);
                GSD.index_grid = (int *) omp_target_alloc( sz, device );
                assert( GSD.index_grid != NULL );
                copy_to_device( GSD.index_grid, SD.index_grid, sz );
                total_sz += sz;
        }
        else
        {
                GSD.index_grid = NULL;
        }

        sz = GSD.length_nuclide_grid * sizeof( NuclideGridPoint );
        GSD.nuclide_grid = (NuclideGridPoint *) omp_target_alloc( sz, device );
        assert( GSD.nuclide_grid != NULL );
        copy_to_device( GSD.nuclide_grid, SD.nuclide_grid, sz );
        total_sz += sz;

        sz = in.lookups * sizeof( unsigned long );
        GSD.verification = (unsigned long *) omp_target_alloc( sz, device );
        assert( GSD.verification != NULL );
        GSD.length_verification = in.lookups;
        total_sz += sz;

        if( mype == 0 )
                printf("GPU initialization complete. Allocated %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

        return GSD;
}

void release_device_memory( SimulationData GSD )
{
        int device = gpu_device_id();
        if( GSD.num_nucs ) omp_target_free( GSD.num_nucs, device );
        if( GSD.concs ) omp_target_free( GSD.concs, device );
        if( GSD.mats ) omp_target_free( GSD.mats, device );
        if( GSD.unionized_energy_array ) omp_target_free( GSD.unionized_energy_array, device );
        if( GSD.index_grid ) omp_target_free( GSD.index_grid, device );
        if( GSD.nuclide_grid ) omp_target_free( GSD.nuclide_grid, device );
        if( GSD.verification ) omp_target_free( GSD.verification, device );
}

static void allocate_sample_buffers( Inputs in, SimulationData * GSD )
{
        int device = gpu_device_id();
        size_t sz_double = in.lookups * sizeof( double );
        size_t sz_int = in.lookups * sizeof( int );

        GSD->p_energy_samples = (double *) omp_target_alloc( sz_double, device );
        assert( GSD->p_energy_samples != NULL );
        GSD->length_p_energy_samples = in.lookups;

        GSD->mat_samples = (int *) omp_target_alloc( sz_int, device );
        assert( GSD->mat_samples != NULL );
        GSD->length_mat_samples = in.lookups;
}

static void free_sample_buffers( SimulationData * GSD )
{
        int device = gpu_device_id();
        if( GSD->p_energy_samples )
        {
                omp_target_free( GSD->p_energy_samples, device );
                GSD->p_energy_samples = NULL;
        }
        if( GSD->mat_samples )
        {
                omp_target_free( GSD->mat_samples, device );
                GSD->mat_samples = NULL;
        }
}

static void copy_sample_buffers_to_host( const SimulationData * GSD, double * p_energy_samples, int * mat_samples, int lookups )
{
        copy_to_host( p_energy_samples, GSD->p_energy_samples, lookups * sizeof( double ) );
        copy_to_host( mat_samples, GSD->mat_samples, lookups * sizeof( int ) );
}

static void copy_sample_buffers_to_device( SimulationData * GSD, const double * p_energy_samples, const int * mat_samples, int lookups )
{
        copy_to_device( GSD->p_energy_samples, p_energy_samples, lookups * sizeof( double ) );
        copy_to_device( GSD->mat_samples, mat_samples, lookups * sizeof( int ) );
}

static unsigned long reduce_verification_from_device( SimulationData GSD, unsigned long * host_verification, int lookups )
{
        int host = host_device_id();
        int device = gpu_device_id();
        size_t sz = lookups * sizeof( unsigned long );
        unsigned long * scratch = host_verification;
        unsigned long * owned = NULL;

        if( scratch == NULL )
        {
                scratch = (unsigned long *) malloc( sz );
                assert( scratch != NULL );
                owned = scratch;
        }

        omp_target_memcpy( scratch, GSD.verification, sz, 0, 0, host, device );

        unsigned long sum = 0;
        for( int i = 0; i < lookups; i++ )
                sum += scratch[i];

        if( owned )
                free( owned );

        return sum;
}

static int count_material( const int * mat_samples, int lookups, int material )
{
        int count = 0;
        for( int i = 0; i < lookups; i++ )
                if( mat_samples[i] == material )
                        count++;
        return count;
}

static void sort_samples_by_material( int * mat_samples, double * p_energy_samples, int lookups )
{
        MatSamplePair * pairs = (MatSamplePair *) malloc( lookups * sizeof( MatSamplePair ) );
        assert( pairs != NULL );
        for( int i = 0; i < lookups; i++ )
        {
                pairs[i].mat = mat_samples[i];
                pairs[i].energy = p_energy_samples[i];
        }
        qsort( pairs, lookups, sizeof( MatSamplePair ), compare_material_pairs );
        for( int i = 0; i < lookups; i++ )
        {
                mat_samples[i] = pairs[i].mat;
                p_energy_samples[i] = pairs[i].energy;
        }
        free( pairs );
}

static void sort_samples_by_energy_range( int * mat_samples, double * p_energy_samples, int offset, int len )
{
        if( len <= 1 )
                return;
        MatSamplePair * pairs = (MatSamplePair *) malloc( len * sizeof( MatSamplePair ) );
        assert( pairs != NULL );
        for( int i = 0; i < len; i++ )
        {
                pairs[i].mat = mat_samples[offset + i];
                pairs[i].energy = p_energy_samples[offset + i];
        }
        qsort( pairs, len, sizeof( MatSamplePair ), compare_energy_pairs );
        for( int i = 0; i < len; i++ )
        {
                mat_samples[offset + i] = pairs[i].mat;
                p_energy_samples[offset + i] = pairs[i].energy;
        }
        free( pairs );
}

static void partition_samples_by_fuel( int * mat_samples, double * p_energy_samples, int lookups, int n_fuel )
{
        int * mat_copy = (int *) malloc( lookups * sizeof( int ) );
        double * energy_copy = (double *) malloc( lookups * sizeof( double ) );
        assert( mat_copy != NULL );
        assert( energy_copy != NULL );

        int fuel_idx = 0;
        int other_idx = n_fuel;
        for( int i = 0; i < lookups; i++ )
        {
                if( mat_samples[i] == 0 )
                {
                        mat_copy[fuel_idx] = mat_samples[i];
                        energy_copy[fuel_idx] = p_energy_samples[i];
                        fuel_idx++;
                }
                else
                {
                        mat_copy[other_idx] = mat_samples[i];
                        energy_copy[other_idx] = p_energy_samples[i];
                        other_idx++;
                }
        }

        memcpy( mat_samples, mat_copy, lookups * sizeof( int ) );
        memcpy( p_energy_samples, energy_copy, lookups * sizeof( double ) );

        free( mat_copy );
        free( energy_copy );
}

static int compare_material_pairs( const void * a, const void * b )
{
        const MatSamplePair * lhs = (const MatSamplePair *) a;
        const MatSamplePair * rhs = (const MatSamplePair *) b;
        if( lhs->mat < rhs->mat )
                return -1;
        if( lhs->mat > rhs->mat )
                return 1;
        return 0;
}

static int compare_energy_pairs( const void * a, const void * b )
{
        const MatSamplePair * lhs = (const MatSamplePair *) a;
        const MatSamplePair * rhs = (const MatSamplePair *) b;
        if( lhs->energy < rhs->energy )
                return -1;
        if( lhs->energy > rhs->energy )
                return 1;
        return 0;
}

#pragma omp declare target
long grid_search( long n, double quarry, const double * __restrict__ A )
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

inline long grid_search_nuclide( long n, double quarry, const NuclideGridPoint * A, long low, long high )
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

inline double LCG_random_double(uint64_t * seed)
{
        const uint64_t m = 9223372036854775808ULL;
        const uint64_t a = 2806196910506780709ULL;
        const uint64_t c = 1ULL;
        *seed = (a * (*seed) + c) % m;
        return (double) (*seed) / (double) m;
}

inline uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
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

void calculate_micro_xs( double p_energy, int nuc, long n_isotopes,
                          long n_gridpoints,
                          const double * __restrict__ egrid, const int * __restrict__ index_data,
                          const NuclideGridPoint * __restrict__ nuclide_grids,
                          long idx, double * __restrict__ xs_vector, int grid_type, int hash_bins )
{
        double f;
        const NuclideGridPoint * low;
        const NuclideGridPoint * high;

        if( grid_type == NUCLIDE )
        {
                idx = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids + nuc * n_gridpoints, 0, n_gridpoints - 1 );

                if( idx == n_gridpoints - 1 )
                        low = &nuclide_grids[nuc*n_gridpoints + idx - 1];
                else
                        low = &nuclide_grids[nuc*n_gridpoints + idx];
        }
        else if( grid_type == UNIONIZED )
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
                        lower = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids + nuc * n_gridpoints, u_low, u_high );

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
                idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid );
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
#pragma omp end declare target

void xs_lookup_kernel_baseline( Inputs in, SimulationData GSD )
{
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        double * egrid = GSD.unionized_energy_array;
        int * index_grid = GSD.index_grid;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;
        long n_isotopes = in.n_isotopes;
        long n_gridpoints = in.n_gridpoints;

#pragma omp target teams loop is_device_ptr(num_nucs, concs, egrid, index_grid, nuclide_grid, mats, verification)
        for( int idx = 0; idx < in.lookups; idx++ )
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*idx);

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
                        egrid,
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
                for( int j = 0; j < 5; j++ )
                {
                        if( macro_xs_vector[j] > max )
                        {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                verification[idx] = max_idx + 1;
        }
}

void sampling_kernel( Inputs in, SimulationData GSD )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;

#pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples)
        for( int idx = 0; idx < in.lookups; idx++ )
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*idx);

                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);

                p_energy_samples[idx] = p_energy;
                mat_samples[idx] = mat;
        }
}

void xs_lookup_kernel_optimization_1( Inputs in, SimulationData GSD )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        double * egrid = GSD.unionized_energy_array;
        int * index_grid = GSD.index_grid;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;
        long n_isotopes = in.n_isotopes;
        long n_gridpoints = in.n_gridpoints;

#pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, egrid, index_grid, nuclide_grid, mats, verification)
        for( int idx = 0; idx < in.lookups; idx++ )
        {
                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[idx],
                        mat_samples[idx],
                        n_isotopes,
                        n_gridpoints,
                        num_nucs,
                        concs,
                        egrid,
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
                for( int j = 0; j < 5; j++ )
                {
                        if( macro_xs_vector[j] > max )
                        {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                verification[idx] = max_idx + 1;
        }
}

void xs_lookup_kernel_optimization_2( Inputs in, SimulationData GSD, int m )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        double * egrid = GSD.unionized_energy_array;
        int * index_data = GSD.index_grid;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, egrid, index_data, nuclide_grid, mats, verification)
        for( int idx = 0; idx < in.lookups; idx++ )
        {
                int mat = mat_samples[idx];
                if( mat != m )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[idx],
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
                        num_nucs,
                        concs,
                        egrid,
                        index_data,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for( int j = 0; j < 5; j++ )
                {
                        if( macro_xs_vector[j] > max )
                        {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                verification[idx] = max_idx + 1;
        }
}

void xs_lookup_kernel_optimization_3( Inputs in, SimulationData GSD, int is_fuel )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        double * egrid = GSD.unionized_energy_array;
        int * index_data = GSD.index_grid;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, egrid, index_data, nuclide_grid, mats, verification)
        for( int idx = 0; idx < in.lookups; idx++ )
        {
                int mat = mat_samples[idx];
                int do_compute = ((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0));
                if( !do_compute )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[idx],
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
                        num_nucs,
                        concs,
                        egrid,
                        index_data,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for( int j = 0; j < 5; j++ )
                {
                        if( macro_xs_vector[j] > max )
                        {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                verification[idx] = max_idx + 1;
        }
}

void xs_lookup_kernel_optimization_4( Inputs in, SimulationData GSD, int m, int n_lookups, int offset )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        double * egrid = GSD.unionized_energy_array;
        int * index_data = GSD.index_grid;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, egrid, index_data, nuclide_grid, mats, verification)
        for( int idx = 0; idx < n_lookups; idx++ )
        {
                int global_idx = idx + offset;
                int mat = mat_samples[global_idx];
                if( mat != m )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[global_idx],
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
                        num_nucs,
                        concs,
                        egrid,
                        index_data,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for( int j = 0; j < 5; j++ )
                {
                        if( macro_xs_vector[j] > max )
                        {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                verification[global_idx] = max_idx + 1;
        }
}

void xs_lookup_kernel_optimization_5( Inputs in, SimulationData GSD, int n_lookups, int offset )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        double * egrid = GSD.unionized_energy_array;
        int * index_data = GSD.index_grid;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, egrid, index_data, nuclide_grid, mats, verification)
        for( int idx = 0; idx < n_lookups; idx++ )
        {
                int global_idx = idx + offset;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[global_idx],
                        mat_samples[global_idx],
                        in.n_isotopes,
                        in.n_gridpoints,
                        num_nucs,
                        concs,
                        egrid,
                        index_data,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        grid_type,
                        hash_bins,
                        max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for( int j = 0; j < 5; j++ )
                {
                        if( macro_xs_vector[j] > max )
                        {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                verification[global_idx] = max_idx + 1;
        }
}

unsigned long long run_event_based_simulation_optimization_1( Inputs in, SimulationData GSD, int mype )
{
        const char * optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if( mype == 0 )
                printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0 )
                printf("Allocating additional device data required by kernel...\n");

        allocate_sample_buffers( in, &GSD );
        if( mype == 0 )
                printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );
        xs_lookup_kernel_optimization_1( in, GSD );

        unsigned long verification_scalar = reduce_verification_from_device( GSD, NULL, in.lookups );
        free_sample_buffers( &GSD );
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_2( Inputs in, SimulationData GSD, int mype )
{
        const char * optimization_name = "Optimization 2 - Material Lookup Kernels";

        if( mype == 0 )
                printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0 )
                printf("Allocating additional device data required by kernel...\n");

        allocate_sample_buffers( in, &GSD );
        if( mype == 0 )
                printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );
        for( int m = 0; m < 12; m++ )
                xs_lookup_kernel_optimization_2( in, GSD, m );

        unsigned long verification_scalar = reduce_verification_from_device( GSD, NULL, in.lookups );
        free_sample_buffers( &GSD );
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_3( Inputs in, SimulationData GSD, int mype )
{
        const char * optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

        if( mype == 0 )
                printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0 )
                printf("Allocating additional device data required by kernel...\n");

        allocate_sample_buffers( in, &GSD );
        if( mype == 0 )
                printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );
        xs_lookup_kernel_optimization_3( in, GSD, 0 );
        xs_lookup_kernel_optimization_3( in, GSD, 1 );

        unsigned long verification_scalar = reduce_verification_from_device( GSD, NULL, in.lookups );
        free_sample_buffers( &GSD );
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_4( Inputs in, SimulationData GSD, int mype )
{
        const char * optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";

        if( mype == 0 )
                printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0 )
                printf("Allocating additional device data required by kernel...\n");

        allocate_sample_buffers( in, &GSD );
        if( mype == 0 )
                printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );

        double * host_p_energy = (double *) malloc( in.lookups * sizeof( double ) );
        int * host_mat_samples = (int *) malloc( in.lookups * sizeof( int ) );
        assert( host_p_energy != NULL );
        assert( host_mat_samples != NULL );

        copy_sample_buffers_to_host( &GSD, host_p_energy, host_mat_samples, in.lookups );

        int n_lookups_per_material[12];
        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = count_material( host_mat_samples, in.lookups, m );

        sort_samples_by_material( host_mat_samples, host_p_energy, in.lookups );
        copy_sample_buffers_to_device( &GSD, host_p_energy, host_mat_samples, in.lookups );

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int n_lookups = n_lookups_per_material[m];
                if( n_lookups == 0 )
                        continue;
                xs_lookup_kernel_optimization_4( in, GSD, m, n_lookups, offset );
                offset += n_lookups;
        }

        unsigned long verification_scalar = reduce_verification_from_device( GSD, NULL, in.lookups );
        free_sample_buffers( &GSD );
        free( host_p_energy );
        free( host_mat_samples );
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_5( Inputs in, SimulationData GSD, int mype )
{
        const char * optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

        if( mype == 0 )
                printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0 )
                printf("Allocating additional device data required by kernel...\n");

        allocate_sample_buffers( in, &GSD );
        if( mype == 0 )
                printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );

        double * host_p_energy = (double *) malloc( in.lookups * sizeof( double ) );
        int * host_mat_samples = (int *) malloc( in.lookups * sizeof( int ) );
        assert( host_p_energy != NULL );
        assert( host_mat_samples != NULL );

        copy_sample_buffers_to_host( &GSD, host_p_energy, host_mat_samples, in.lookups );

        int n_fuel_lookups = count_material( host_mat_samples, in.lookups, 0 );
        partition_samples_by_fuel( host_mat_samples, host_p_energy, in.lookups, n_fuel_lookups );
        copy_sample_buffers_to_device( &GSD, host_p_energy, host_mat_samples, in.lookups );

        xs_lookup_kernel_optimization_5( in, GSD, n_fuel_lookups, 0 );
        xs_lookup_kernel_optimization_5( in, GSD, in.lookups - n_fuel_lookups, n_fuel_lookups );

        unsigned long verification_scalar = reduce_verification_from_device( GSD, NULL, in.lookups );
        free_sample_buffers( &GSD );
        free( host_p_energy );
        free( host_mat_samples );
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_6( Inputs in, SimulationData GSD, int mype )
{
        const char * optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

        if( mype == 0 )
                printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0 )
                printf("Allocating additional device data required by kernel...\n");

        allocate_sample_buffers( in, &GSD );
        if( mype == 0 )
                printf("Beginning optimized simulation...\n");

        sampling_kernel( in, GSD );

        double * host_p_energy = (double *) malloc( in.lookups * sizeof( double ) );
        int * host_mat_samples = (int *) malloc( in.lookups * sizeof( int ) );
        assert( host_p_energy != NULL );
        assert( host_mat_samples != NULL );

        copy_sample_buffers_to_host( &GSD, host_p_energy, host_mat_samples, in.lookups );

        int n_lookups_per_material[12];
        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = count_material( host_mat_samples, in.lookups, m );

        sort_samples_by_material( host_mat_samples, host_p_energy, in.lookups );

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int n_lookups = n_lookups_per_material[m];
                sort_samples_by_energy_range( host_mat_samples, host_p_energy, offset, n_lookups );
                offset += n_lookups;
        }

        copy_sample_buffers_to_device( &GSD, host_p_energy, host_mat_samples, in.lookups );

        offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int n_lookups = n_lookups_per_material[m];
                if( n_lookups == 0 )
                        continue;
                xs_lookup_kernel_optimization_4( in, GSD, m, n_lookups, offset );
                offset += n_lookups;
        }

        unsigned long verification_scalar = reduce_verification_from_device( GSD, NULL, in.lookups );
        free_sample_buffers( &GSD );
        free( host_p_energy );
        free( host_mat_samples );
        return verification_scalar;
}
