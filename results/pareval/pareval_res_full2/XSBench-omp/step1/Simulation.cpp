#include "XSbench_header.cuh"
#include <algorithm>
#include <numeric>
#include <vector>
#include <utility>
#include <omp.h>

#pragma omp declare target

long grid_search( long n, double quarry, double * __restrict__ A)
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

long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high)
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

void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                                   long n_gridpoints,
                                   double * __restrict__ egrid, int * __restrict__ index_data,
                                   NuclideGridPoint * __restrict__ nuclide_grids,
                                   long idx, double * __restrict__ xs_vector, int grid_type, int hash_bins )
{

        double f;
        NuclideGridPoint * low, * high;

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
                                   long n_gridpoints, int * __restrict__ num_nucs,
                                   double * __restrict__ concs,
                                   double * __restrict__ egrid, int * __restrict__ index_data,
                                   NuclideGridPoint * __restrict__ nuclide_grids,
                                   int * __restrict__ mats,
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

#pragma omp end declare target

namespace {

static void sort_by_mat_key(int *keys, double *values, int n)
{
        std::vector<std::pair<int,double>> work;
        work.reserve(n);
        for( int i = 0; i < n; i++ )
                work.emplace_back(keys[i], values[i]);
        std::stable_sort(work.begin(), work.end(), [](const std::pair<int,double> &a, const std::pair<int,double> &b){
                return a.first < b.first;
        });
        for( int i = 0; i < n; i++ )
        {
                keys[i] = work[i].first;
                values[i] = work[i].second;
        }
}

static void sort_by_energy_key(double *keys, int *values, int n)
{
        std::vector<std::pair<double,int>> work;
        work.reserve(n);
        for( int i = 0; i < n; i++ )
                work.emplace_back(keys[i], values[i]);
        std::stable_sort(work.begin(), work.end(), [](const std::pair<double,int> &a, const std::pair<double,int> &b){
                return a.first < b.first;
        });
        for( int i = 0; i < n; i++ )
        {
                keys[i] = work[i].first;
                values[i] = work[i].second;
        }
}

static int partition_mat_fuel(int *mats, double *energies, int n)
{
        std::vector<std::pair<int,double>> work;
        work.reserve(n);
        for( int i = 0; i < n; i++ )
                work.emplace_back(mats[i], energies[i]);
        auto it = std::stable_partition(work.begin(), work.end(), [](const std::pair<int,double> &item){
                return item.first == 0;
        });
        int count = it - work.begin();
        for( int i = 0; i < n; i++ )
        {
                mats[i] = work[i].first;
                energies[i] = work[i].second;
        }
        return count;
}

static int count_material_occurrences(const int *mats, int n, int mat)
{
        return (int) std::count(mats, mats + n, mat);
}

template <typename... Ts>
static void mark_used(const Ts&...)
{
}

} // namespace

static void xs_lookup_kernel_baseline(const Inputs &in, SimulationData &GSD)
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        unsigned long *verification = GSD.verification;
        int max_num_nucs = GSD.max_num_nucs;
        int lookups = in.lookups;

        if( lookups <= 0 )
                return;

#pragma omp target teams loop thread_limit(256) is_device_ptr(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification)
        for( int i = 0; i < lookups; i++ )
        {
                double macro_xs_vector[5] = {0};
                uint64_t seed = STARTING_SEED;

                seed = fast_forward_LCG(seed, 2*i);

                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);

                calculate_macro_xs(
                        p_energy,
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
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
                verification[i] = max_idx + 1;
        }
}

static void sampling_kernel(const Inputs &in, SimulationData &GSD)
{
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int lookups = in.lookups;

        if( lookups <= 0 )
                return;

#pragma omp target teams loop thread_limit(32) is_device_ptr(p_energy_samples, mat_samples)
        for( int i = 0; i < lookups; i++ )
        {
                uint64_t seed = STARTING_SEED;

                seed = fast_forward_LCG(seed, 2*i);

                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);

                p_energy_samples[i] = p_energy;
                mat_samples[i] = mat;
        }
}

static void xs_lookup_kernel_optimization_1(const Inputs &in, SimulationData &GSD)
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        unsigned long *verification = GSD.verification;
        int max_num_nucs = GSD.max_num_nucs;
        int lookups = in.lookups;

        if( lookups <= 0 )
                return;

#pragma omp target teams loop thread_limit(32) is_device_ptr(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, p_energy_samples, mat_samples, verification)
        for( int i = 0; i < lookups; i++ )
        {
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(
                        p_energy_samples[i],
                        mat_samples[i],
                        in.n_isotopes,
                        in.n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
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
                verification[i] = max_idx + 1;
        }
}

static void xs_lookup_kernel_optimization_2(const Inputs &in, SimulationData &GSD, int m)
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        unsigned long *verification = GSD.verification;
        int max_num_nucs = GSD.max_num_nucs;
        int lookups = in.lookups;

        if( lookups <= 0 )
                return;

#pragma omp target teams loop thread_limit(32) is_device_ptr(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, p_energy_samples, mat_samples, verification)
        for( int i = 0; i < lookups; i++ )
        {
                int mat = mat_samples[i];
                if( mat != m )
                        continue;

                double macro_xs_vector[5] = {0};
                calculate_macro_xs(
                        p_energy_samples[i],
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
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
                verification[i] = max_idx + 1;
        }
}

static void xs_lookup_kernel_optimization_3(const Inputs &in, SimulationData &GSD, int is_fuel)
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        unsigned long *verification = GSD.verification;
        int max_num_nucs = GSD.max_num_nucs;
        int lookups = in.lookups;

        if( lookups <= 0 )
                return;

#pragma omp target teams loop thread_limit(32) is_device_ptr(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, p_energy_samples, mat_samples, verification)
        for( int i = 0; i < lookups; i++ )
        {
                int mat = mat_samples[i];
                if( ((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0)) )
                {
                        double macro_xs_vector[5] = {0};
                        calculate_macro_xs(
                                p_energy_samples[i],
                                mat,
                                in.n_isotopes,
                                in.n_gridpoints,
                                num_nucs,
                                concs,
                                unionized_energy_array,
                                index_grid,
                                nuclide_grid,
                                mats,
                                macro_xs_vector,
                                in.grid_type,
                                in.hash_bins,
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
                        verification[i] = max_idx + 1;
                }
        }
}

static void xs_lookup_kernel_optimization_4(const Inputs &in, SimulationData &GSD, int m, int n_lookups, int offset)
{
        if( n_lookups <= 0 )
                return;

        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        unsigned long *verification = GSD.verification;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop thread_limit(32) is_device_ptr(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, p_energy_samples, mat_samples, verification)
        for( int i = 0; i < n_lookups; i++ )
        {
                int global_idx = i + offset;
                if( global_idx >= in.lookups )
                        continue;

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
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
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

static void xs_lookup_kernel_optimization_5(const Inputs &in, SimulationData &GSD, int n_lookups, int offset)
{
        if( n_lookups <= 0 )
                return;

        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        unsigned long *verification = GSD.verification;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop thread_limit(32) is_device_ptr(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, p_energy_samples, mat_samples, verification)
        for( int i = 0; i < n_lookups; i++ )
        {
                int global_idx = i + offset;
                if( global_idx >= in.lookups )
                        continue;

                double macro_xs_vector[5] = {0};
                calculate_macro_xs(
                        p_energy_samples[global_idx],
                        mat_samples[global_idx],
                        in.n_isotopes,
                        in.n_gridpoints,
                        num_nucs,
                        concs,
                        unionized_energy_array,
                        index_grid,
                        nuclide_grid,
                        mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
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

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        double start = get_time();
        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        profile->host_to_device_time = get_time() - start;

        if( mype == 0) printf("Running baseline event-based simulation...\n");

        int nwarmups = in.num_warmups;
        start = 0.0;
        int total_iters = in.num_iterations + nwarmups;

        #pragma diag_push
        #pragma diag_push
        #pragma diag_push
        #pragma diag_push
        #pragma diag_push
        #pragma diag_push
        #pragma diag_push
        #pragma diag_suppress declared_but_not_referenced
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        unsigned long *verification = GSD.verification;

        int num_nucs_len = GSD.length_num_nucs;
        int concs_len = GSD.length_concs;
        int mats_len = GSD.length_mats;
        int unionized_len = GSD.length_unionized_energy_array;
        long index_len = GSD.length_index_grid;
        int nuclide_len = GSD.length_nuclide_grid;
        int verification_len = GSD.length_verification;
        #pragma diag_pop

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

#pragma omp target data \
        map(to: num_nucs[:num_nucs_len], concs[:concs_len], mats[:mats_len], unionized_energy_array[:unionized_len], index_grid[:index_len], nuclide_grid[:nuclide_len]) \
        map(from: verification[:verification_len])
        {
                for( int i = 0; i < total_iters; i++ )
                {
                        if( i == nwarmups )
                                start = get_time();
                        xs_lookup_kernel_baseline(in, GSD);
                }
        }

        profile->kernel_time = get_time() - start;
        profile->device_to_host_time = 0.0;

        if( mype == 0) printf("Reducing verification results...\n");

        unsigned long long verification_scalar = 0;
        for( int i = 0; i < in.lookups; i++ )
                verification_scalar += GSD.verification[i];

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        int lookups = in.lookups;
        std::vector<double> p_energy_samples(lookups);
        std::vector<int> mat_samples(lookups);
        GSD.p_energy_samples = p_energy_samples.data();
        GSD.mat_samples = mat_samples.data();
        GSD.length_p_energy_samples = lookups;
        GSD.length_mat_samples = lookups;

        if( mype == 0) printf("Allocated %.0lf MB of data for samples.\n", (double)lookups * (sizeof(double) + sizeof(int)) / 1024.0 / 1024.0);

        if( mype == 0) printf("Beginning optimized simulation...\n");

        #pragma diag_suppress declared_but_not_referenced
        #pragma diag_suppress declared_but_not_referenced
        #pragma diag_suppress declared_but_not_referenced
        #pragma diag_suppress declared_but_not_referenced
        #pragma diag_suppress declared_but_not_referenced
        #pragma diag_suppress declared_but_not_referenced
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        unsigned long *verification = GSD.verification;
        double *sample_energies = GSD.p_energy_samples;
        int *sample_mats = GSD.mat_samples;

        int num_nucs_len = GSD.length_num_nucs;
        int concs_len = GSD.length_concs;
        int mats_len = GSD.length_mats;
        int unionized_len = GSD.length_unionized_energy_array;
        long index_len = GSD.length_index_grid;
        int nuclide_len = GSD.length_nuclide_grid;
        int verification_len = GSD.length_verification;
        #pragma diag_pop
        #pragma diag_pop
        #pragma diag_pop
        #pragma diag_pop
        #pragma diag_pop
        #pragma diag_pop

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, sample_energies, sample_mats, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, sample_energies, sample_mats, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, sample_energies, sample_mats, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, sample_energies, sample_mats, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, sample_energies, sample_mats, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, sample_energies, sample_mats, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, sample_energies, sample_mats, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

#pragma omp target data \
        map(to: num_nucs[:num_nucs_len], concs[:concs_len], mats[:mats_len], unionized_energy_array[:unionized_len], index_grid[:index_len], nuclide_grid[:nuclide_len]) \
        map(tofrom: sample_energies[:lookups], sample_mats[:lookups]) \
        map(from: verification[:verification_len])
        {
                sampling_kernel(in, GSD);
                #pragma omp target update from(sample_energies[:lookups], sample_mats[:lookups])
                #pragma omp target update to(sample_energies[:lookups], sample_mats[:lookups])
                xs_lookup_kernel_optimization_1(in, GSD);
        }

        if( mype == 0) printf("Reducing verification results...\n");
        return std::accumulate(verification, verification + lookups, 0ULL);
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 2 - Material Lookup Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        int lookups = in.lookups;
        std::vector<double> p_energy_samples(lookups);
        std::vector<int> mat_samples(lookups);
        GSD.p_energy_samples = p_energy_samples.data();
        GSD.mat_samples = mat_samples.data();
        GSD.length_p_energy_samples = lookups;
        GSD.length_mat_samples = lookups;

        if( mype == 0) printf("Allocated %.0lf MB of data for samples.\n", (double)lookups * (sizeof(double) + sizeof(int)) / 1024.0 / 1024.0);

        if( mype == 0) printf("Beginning optimized simulation...\n");

        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        unsigned long *verification = GSD.verification;
        double *sample_energies = GSD.p_energy_samples;
        int *sample_mats = GSD.mat_samples;

        int num_nucs_len = GSD.length_num_nucs;
        int concs_len = GSD.length_concs;
        int mats_len = GSD.length_mats;
        int unionized_len = GSD.length_unionized_energy_array;
        long index_len = GSD.length_index_grid;
        int nuclide_len = GSD.length_nuclide_grid;
        int verification_len = GSD.length_verification;

#pragma omp target data \
        map(to: num_nucs[:num_nucs_len], concs[:concs_len], mats[:mats_len], unionized_energy_array[:unionized_len], index_grid[:index_len], nuclide_grid[:nuclide_len]) \
        map(tofrom: sample_energies[:lookups], sample_mats[:lookups]) \
        map(from: verification[:verification_len])
        {
                sampling_kernel(in, GSD);
                for( int m = 0; m < 12; m++ )
                        xs_lookup_kernel_optimization_2(in, GSD, m);
        }

        if( mype == 0) printf("Reducing verification results...\n");
        return std::accumulate(verification, verification + lookups, 0ULL);
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        int lookups = in.lookups;
        std::vector<double> p_energy_samples(lookups);
        std::vector<int> mat_samples(lookups);
        GSD.p_energy_samples = p_energy_samples.data();
        GSD.mat_samples = mat_samples.data();
        GSD.length_p_energy_samples = lookups;
        GSD.length_mat_samples = lookups;

        if( mype == 0) printf("Allocated %.0lf MB of data for samples.\n", (double)lookups * (sizeof(double) + sizeof(int)) / 1024.0 / 1024.0);

        if( mype == 0) printf("Beginning optimized simulation...\n");

        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        unsigned long *verification = GSD.verification;
        double *sample_energies = GSD.p_energy_samples;
        int *sample_mats = GSD.mat_samples;

        int num_nucs_len = GSD.length_num_nucs;
        int concs_len = GSD.length_concs;
        int mats_len = GSD.length_mats;
        int unionized_len = GSD.length_unionized_energy_array;
        long index_len = GSD.length_index_grid;
        int nuclide_len = GSD.length_nuclide_grid;
        int verification_len = GSD.length_verification;

#pragma omp target data \
        map(to: num_nucs[:num_nucs_len], concs[:concs_len], mats[:mats_len], unionized_energy_array[:unionized_len], index_grid[:index_len], nuclide_grid[:nuclide_len]) \
        map(tofrom: sample_energies[:lookups], sample_mats[:lookups]) \
        map(from: verification[:verification_len])
        {
                sampling_kernel(in, GSD);
                xs_lookup_kernel_optimization_3(in, GSD, 0);
                xs_lookup_kernel_optimization_3(in, GSD, 1);
        }

        if( mype == 0) printf("Reducing verification results...\n");
        return std::accumulate(verification, verification + lookups, 0ULL);
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        int lookups = in.lookups;
        std::vector<double> p_energy_samples(lookups);
        std::vector<int> mat_samples(lookups);
        GSD.p_energy_samples = p_energy_samples.data();
        GSD.mat_samples = mat_samples.data();
        GSD.length_p_energy_samples = lookups;
        GSD.length_mat_samples = lookups;

        if( mype == 0) printf("Allocated %.0lf MB of data for samples.\n", (double)lookups * (sizeof(double) + sizeof(int)) / 1024.0 / 1024.0);

        if( mype == 0) printf("Beginning optimized simulation...\n");

        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        unsigned long *verification = GSD.verification;
        double *sample_energies = GSD.p_energy_samples;
        int *sample_mats = GSD.mat_samples;

        int num_nucs_len = GSD.length_num_nucs;
        int concs_len = GSD.length_concs;
        int mats_len = GSD.length_mats;
        int unionized_len = GSD.length_unionized_energy_array;
        long index_len = GSD.length_index_grid;
        int nuclide_len = GSD.length_nuclide_grid;
        int verification_len = GSD.length_verification;

        mark_used(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, verification, sample_energies, sample_mats, num_nucs_len, concs_len, mats_len, unionized_len, index_len, nuclide_len, verification_len);

        int n_lookups_per_material[12] = {0};

#pragma omp target data \
        map(to: num_nucs[:num_nucs_len], concs[:concs_len], mats[:mats_len], unionized_energy_array[:unionized_len], index_grid[:index_len], nuclide_grid[:nuclide_len]) \
        map(tofrom: sample_energies[:lookups], sample_mats[:lookups]) \
        map(from: verification[:verification_len])
        {
                sampling_kernel(in, GSD);
                #pragma omp target update from(sample_energies[:lookups], sample_mats[:lookups])
                sort_by_mat_key(sample_mats, sample_energies, lookups);
                for( int m = 0; m < 12; m++ )
                        n_lookups_per_material[m] = count_material_occurrences(sample_mats, lookups, m);
                #pragma omp target update to(sample_energies[:lookups], sample_mats[:lookups])

                int offset = 0;
                for( int m = 0; m < 12; m++ )
                {
                        int n = n_lookups_per_material[m];
                        xs_lookup_kernel_optimization_4(in, GSD, m, n, offset);
                        offset += n;
                }
        }

        if( mype == 0) printf("Reducing verification results...\n");
        return std::accumulate(verification, verification + lookups, 0ULL);
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        int lookups = in.lookups;
        std::vector<double> p_energy_samples(lookups);
        std::vector<int> mat_samples(lookups);
        GSD.p_energy_samples = p_energy_samples.data();
        GSD.mat_samples = mat_samples.data();
        GSD.length_p_energy_samples = lookups;
        GSD.length_mat_samples = lookups;

        if( mype == 0) printf("Allocated %.0lf MB of data for samples.\n", (double)lookups * (sizeof(double) + sizeof(int)) / 1024.0 / 1024.0);

        if( mype == 0) printf("Beginning optimized simulation...\n");

        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        unsigned long *verification = GSD.verification;
        double *sample_energies = GSD.p_energy_samples;
        int *sample_mats = GSD.mat_samples;

        int num_nucs_len = GSD.length_num_nucs;
        int concs_len = GSD.length_concs;
        int mats_len = GSD.length_mats;
        int unionized_len = GSD.length_unionized_energy_array;
        long index_len = GSD.length_index_grid;
        int nuclide_len = GSD.length_nuclide_grid;
        int verification_len = GSD.length_verification;

#pragma omp target data \
        map(to: num_nucs[:num_nucs_len], concs[:concs_len], mats[:mats_len], unionized_energy_array[:unionized_len], index_grid[:index_len], nuclide_grid[:nuclide_len]) \
        map(tofrom: sample_energies[:lookups], sample_mats[:lookups]) \
        map(from: verification[:verification_len])
        {
                sampling_kernel(in, GSD);
                #pragma omp target update from(sample_energies[:lookups], sample_mats[:lookups])
                int n_fuel_lookups = partition_mat_fuel(sample_mats, sample_energies, lookups);
                #pragma omp target update to(sample_energies[:lookups], sample_mats[:lookups])

                xs_lookup_kernel_optimization_5(in, GSD, n_fuel_lookups, 0);
                xs_lookup_kernel_optimization_5(in, GSD, lookups - n_fuel_lookups, n_fuel_lookups);
        }

        if( mype == 0) printf("Reducing verification results...\n");
        return std::accumulate(verification, verification + lookups, 0ULL);
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        int lookups = in.lookups;
        std::vector<double> p_energy_samples(lookups);
        std::vector<int> mat_samples(lookups);
        GSD.p_energy_samples = p_energy_samples.data();
        GSD.mat_samples = mat_samples.data();
        GSD.length_p_energy_samples = lookups;
        GSD.length_mat_samples = lookups;

        if( mype == 0) printf("Allocated %.0lf MB of data for samples.\n", (double)lookups * (sizeof(double) + sizeof(int)) / 1024.0 / 1024.0);

        if( mype == 0) printf("Beginning optimized simulation...\n");

        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        int *mats = GSD.mats;
        double *unionized_energy_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        unsigned long *verification = GSD.verification;
        double *sample_energies = GSD.p_energy_samples;
        int *sample_mats = GSD.mat_samples;

        int num_nucs_len = GSD.length_num_nucs;
        int concs_len = GSD.length_concs;
        int mats_len = GSD.length_mats;
        int unionized_len = GSD.length_unionized_energy_array;
        long index_len = GSD.length_index_grid;
        int nuclide_len = GSD.length_nuclide_grid;
        int verification_len = GSD.length_verification;

        int n_lookups_per_material[12] = {0};

#pragma omp target data \
        map(to: num_nucs[:num_nucs_len], concs[:concs_len], mats[:mats_len], unionized_energy_array[:unionized_len], index_grid[:index_len], nuclide_grid[:nuclide_len]) \
        map(tofrom: sample_energies[:lookups], sample_mats[:lookups]) \
        map(from: verification[:verification_len])
        {
                sampling_kernel(in, GSD);
                #pragma omp target update from(sample_energies[:lookups], sample_mats[:lookups])
                sort_by_mat_key(sample_mats, sample_energies, lookups);
                for( int m = 0; m < 12; m++ )
                {
                        n_lookups_per_material[m] = count_material_occurrences(sample_mats, lookups, m);
                }
                int material_offset = 0;
                for( int m = 0; m < 12; m++ )
                {
                        int n = n_lookups_per_material[m];
                        if( n > 0 )
                                sort_by_energy_key(sample_energies + material_offset, sample_mats + material_offset, n);
                        material_offset += n;
                }
                #pragma omp target update to(sample_energies[:lookups], sample_mats[:lookups])

                int offset = 0;
                for( int m = 0; m < 12; m++ )
                {
                        int n = n_lookups_per_material[m];
                        xs_lookup_kernel_optimization_4(in, GSD, m, n, offset);
                        offset += n;
                }
        }

        if( mype == 0) printf("Reducing verification results...\n");
        return std::accumulate(verification, verification + lookups, 0ULL);
}
