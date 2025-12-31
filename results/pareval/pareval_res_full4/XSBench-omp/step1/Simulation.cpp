#include "XSbench_header.cuh"

#include <algorithm>
#include <numeric>
#include <omp.h>
#include <utility>
#include <vector>
#include <iterator>

namespace {
int target_device()
{
    return omp_get_default_device();
}

int host_device()
{
    return omp_get_initial_device();
}

void copy_from_device(void *host_ptr, const void *device_ptr, size_t bytes)
{
    omp_target_memcpy(host_ptr, device_ptr, bytes, 0, 0, host_device(), target_device());
}

void copy_to_device(void *device_ptr, const void *host_ptr, size_t bytes)
{
    omp_target_memcpy(device_ptr, host_ptr, bytes, 0, 0, target_device(), host_device());
}

void allocate_sample_buffers(Inputs in, SimulationData &GSD)
{
    size_t energy_bytes = in.lookups * sizeof(double);
    size_t mat_bytes = in.lookups * sizeof(int);
    GSD.p_energy_samples = static_cast<double *>(omp_target_alloc(energy_bytes, target_device()));
    assert(GSD.p_energy_samples != nullptr);
    GSD.length_p_energy_samples = in.lookups;
    GSD.mat_samples = static_cast<int *>(omp_target_alloc(mat_bytes, target_device()));
    assert(GSD.mat_samples != nullptr);
    GSD.length_mat_samples = in.lookups;
}

void free_sample_buffers(SimulationData &GSD)
{
    if (GSD.p_energy_samples)
        omp_target_free(GSD.p_energy_samples, target_device());
    if (GSD.mat_samples)
        omp_target_free(GSD.mat_samples, target_device());
    GSD.p_energy_samples = nullptr;
    GSD.mat_samples = nullptr;
}

void copy_samples_to_host(SimulationData &GSD, std::vector<int> &host_mats, std::vector<double> &host_energies)
{
    size_t mat_bytes = host_mats.size() * sizeof(int);
    size_t energy_bytes = host_energies.size() * sizeof(double);
    copy_from_device(host_mats.data(), GSD.mat_samples, mat_bytes);
    copy_from_device(host_energies.data(), GSD.p_energy_samples, energy_bytes);
}

void copy_samples_to_device(SimulationData &GSD, const std::vector<int> &host_mats, const std::vector<double> &host_energies)
{
    size_t mat_bytes = host_mats.size() * sizeof(int);
    size_t energy_bytes = host_energies.size() * sizeof(double);
    copy_to_device(GSD.mat_samples, host_mats.data(), mat_bytes);
    copy_to_device(GSD.p_energy_samples, host_energies.data(), energy_bytes);
}

void sort_samples_by_material(std::vector<int> &host_mats, std::vector<double> &host_energies)
{
    size_t count = host_mats.size();
    std::vector<std::pair<int, double>> zipped(count);
    for (size_t i = 0; i < count; ++i)
        zipped[i] = {host_mats[i], host_energies[i]};
    std::sort(zipped.begin(), zipped.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
    });
    for (size_t i = 0; i < count; ++i) {
        host_mats[i] = zipped[i].first;
        host_energies[i] = zipped[i].second;
    }
}

void sort_samples_by_material_then_energy(std::vector<int> &host_mats, std::vector<double> &host_energies)
{
    size_t count = host_mats.size();
    std::vector<std::pair<int, double>> zipped(count);
    for (size_t i = 0; i < count; ++i)
        zipped[i] = {host_mats[i], host_energies[i]};
    std::sort(zipped.begin(), zipped.end(), [](const auto &a, const auto &b) {
        if (a.first != b.first)
            return a.first < b.first;
        return a.second < b.second;
    });
    for (size_t i = 0; i < count; ++i) {
        host_mats[i] = zipped[i].first;
        host_energies[i] = zipped[i].second;
    }
}

unsigned long reduce_verification(SimulationData &SD, int lookups)
{
    return std::accumulate(SD.verification, SD.verification + lookups, 0ull);
}
} // namespace

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
#pragma omp end declare target

void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD )
{
        double *d_concs = GSD.concs;
        int *d_num_nucs = GSD.num_nucs;
        int *d_index_grid = GSD.index_grid;
        NuclideGridPoint *d_nuclide_grid = GSD.nuclide_grid;
        double *d_unionized_energy_array = GSD.unionized_energy_array;
        int *d_mats = GSD.mats;
        unsigned long *d_verification = GSD.verification;

        #pragma omp target teams loop thread_limit(256) is_device_ptr(d_concs, d_num_nucs, d_index_grid, d_nuclide_grid, d_unionized_energy_array, d_mats, d_verification)
        for (int i = 0; i < in.lookups; ++i)
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
                        d_num_nucs,
                        d_concs,
                        d_unionized_energy_array,
                        d_index_grid,
                        d_nuclide_grid,
                        d_mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
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
                d_verification[i] = max_idx+1;
        }
}

void sampling_kernel(Inputs in, SimulationData GSD )
{
        double *d_p_energy_samples = GSD.p_energy_samples;
        int *d_mat_samples = GSD.mat_samples;

        #pragma omp target teams loop thread_limit(32) is_device_ptr(d_p_energy_samples, d_mat_samples)
        for (int i = 0; i < in.lookups; ++i)
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                d_p_energy_samples[i] = p_energy;
                d_mat_samples[i] = mat;
        }
}

void xs_lookup_kernel_optimization_1(Inputs in, SimulationData GSD )
{
        double *d_concs = GSD.concs;
        int *d_num_nucs = GSD.num_nucs;
        int *d_index_grid = GSD.index_grid;
        NuclideGridPoint *d_nuclide_grid = GSD.nuclide_grid;
        double *d_unionized_energy_array = GSD.unionized_energy_array;
        int *d_mats = GSD.mats;
        unsigned long *d_verification = GSD.verification;
        double *d_p_energy_samples = GSD.p_energy_samples;
        int *d_mat_samples = GSD.mat_samples;

        #pragma omp target teams loop thread_limit(32) is_device_ptr(d_concs, d_num_nucs, d_index_grid, d_nuclide_grid, d_unionized_energy_array, d_mats, d_p_energy_samples, d_mat_samples, d_verification)
        for (int i = 0; i < in.lookups; ++i)
        {
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(
                        d_p_energy_samples[i],
                        d_mat_samples[i],
                        in.n_isotopes,
                        in.n_gridpoints,
                        d_num_nucs,
                        d_concs,
                        d_unionized_energy_array,
                        d_index_grid,
                        d_nuclide_grid,
                        d_mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
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
                d_verification[i] = max_idx+1;
        }
}

void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m )
{
        double *d_concs = GSD.concs;
        int *d_num_nucs = GSD.num_nucs;
        int *d_index_grid = GSD.index_grid;
        NuclideGridPoint *d_nuclide_grid = GSD.nuclide_grid;
        double *d_unionized_energy_array = GSD.unionized_energy_array;
        int *d_mats = GSD.mats;
        unsigned long *d_verification = GSD.verification;
        double *d_p_energy_samples = GSD.p_energy_samples;
        int *d_mat_samples = GSD.mat_samples;

        #pragma omp target teams loop thread_limit(32) is_device_ptr(d_concs, d_num_nucs, d_index_grid, d_nuclide_grid, d_unionized_energy_array, d_mats, d_p_energy_samples, d_mat_samples, d_verification)
        for (int i = 0; i < in.lookups; ++i)
        {
                if (d_mat_samples[i] != m)
                        continue;

                double macro_xs_vector[5] = {0};
                calculate_macro_xs(
                        d_p_energy_samples[i],
                        m,
                        in.n_isotopes,
                        in.n_gridpoints,
                        d_num_nucs,
                        d_concs,
                        d_unionized_energy_array,
                        d_index_grid,
                        d_nuclide_grid,
                        d_mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
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
                d_verification[i] = max_idx+1;
        }
}

void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel )
{
        double *d_concs = GSD.concs;
        int *d_num_nucs = GSD.num_nucs;
        int *d_index_grid = GSD.index_grid;
        NuclideGridPoint *d_nuclide_grid = GSD.nuclide_grid;
        double *d_unionized_energy_array = GSD.unionized_energy_array;
        int *d_mats = GSD.mats;
        unsigned long *d_verification = GSD.verification;
        double *d_p_energy_samples = GSD.p_energy_samples;
        int *d_mat_samples = GSD.mat_samples;

        #pragma omp target teams loop thread_limit(32) is_device_ptr(d_concs, d_num_nucs, d_index_grid, d_nuclide_grid, d_unionized_energy_array, d_mats, d_p_energy_samples, d_mat_samples, d_verification)
        for (int i = 0; i < in.lookups; ++i)
        {
                int mat = d_mat_samples[i];
                if( ((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0 )) )
                {
                        double macro_xs_vector[5] = {0};
                        calculate_macro_xs(
                                d_p_energy_samples[i],
                                mat,
                                in.n_isotopes,
                                in.n_gridpoints,
                                d_num_nucs,
                                d_concs,
                                d_unionized_energy_array,
                                d_index_grid,
                                d_nuclide_grid,
                                d_mats,
                                macro_xs_vector,
                                in.grid_type,
                                in.hash_bins,
                                GSD.max_num_nucs
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
                        d_verification[i] = max_idx+1;
                }
        }
}

void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset )
{
        double *d_concs = GSD.concs;
        int *d_num_nucs = GSD.num_nucs;
        int *d_index_grid = GSD.index_grid;
        NuclideGridPoint *d_nuclide_grid = GSD.nuclide_grid;
        double *d_unionized_energy_array = GSD.unionized_energy_array;
        int *d_mats = GSD.mats;
        unsigned long *d_verification = GSD.verification;
        double *d_p_energy_samples = GSD.p_energy_samples;
        int *d_mat_samples = GSD.mat_samples;

        #pragma omp target teams loop thread_limit(32) is_device_ptr(d_concs, d_num_nucs, d_index_grid, d_nuclide_grid, d_unionized_energy_array, d_mats, d_p_energy_samples, d_mat_samples, d_verification)
        for (int i = 0; i < n_lookups; ++i)
        {
                int idx = offset + i;
                int mat = d_mat_samples[idx];
                if( mat != m )
                        continue;

                double macro_xs_vector[5] = {0};
                calculate_macro_xs(
                        d_p_energy_samples[idx],
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
                        d_num_nucs,
                        d_concs,
                        d_unionized_energy_array,
                        d_index_grid,
                        d_nuclide_grid,
                        d_mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
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
                d_verification[idx] = max_idx+1;
        }
}

void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset )
{
        double *d_concs = GSD.concs;
        int *d_num_nucs = GSD.num_nucs;
        int *d_index_grid = GSD.index_grid;
        NuclideGridPoint *d_nuclide_grid = GSD.nuclide_grid;
        double *d_unionized_energy_array = GSD.unionized_energy_array;
        int *d_mats = GSD.mats;
        unsigned long *d_verification = GSD.verification;
        double *d_p_energy_samples = GSD.p_energy_samples;
        int *d_mat_samples = GSD.mat_samples;

        #pragma omp target teams loop thread_limit(32) is_device_ptr(d_concs, d_num_nucs, d_index_grid, d_nuclide_grid, d_unionized_energy_array, d_mats, d_p_energy_samples, d_mat_samples, d_verification)
        for (int i = 0; i < n_lookups; ++i)
        {
                int idx = offset + i;
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(
                        d_p_energy_samples[idx],
                        d_mat_samples[idx],
                        in.n_isotopes,
                        in.n_gridpoints,
                        d_num_nucs,
                        d_concs,
                        d_unionized_energy_array,
                        d_index_grid,
                        d_nuclide_grid,
                        d_mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
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
                d_verification[idx] = max_idx+1;
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
        for (int i = 0; i < in.num_iterations + nwarmups; i++) {
                if (i == nwarmups)
                        start = get_time();
                xs_lookup_kernel_baseline(in, GSD);
        }
        profile->kernel_time = get_time() - start;

        start = get_time();
        copy_from_device(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long));
        profile->device_to_host_time = get_time() - start;

        unsigned long verification_scalar = reduce_verification(SD, in.lookups);

        release_device_memory(GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        size_t sz = in.lookups * sizeof(double) + in.lookups * sizeof(int);
        allocate_sample_buffers(in, GSD);
        if( mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", sz/1024.0/1024.0);
        if( mype == 0) printf("Beginning optimized simulation...\n");

        sampling_kernel(in, GSD);
        xs_lookup_kernel_optimization_1(in, GSD);

        copy_from_device(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long));
        unsigned long verification_scalar = reduce_verification(SD, in.lookups);

        free_sample_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 2 - Material Lookup Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        size_t sz = in.lookups * sizeof(double) + in.lookups * sizeof(int);
        allocate_sample_buffers(in, GSD);
        if( mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", sz/1024.0/1024.0);
        if( mype == 0) printf("Beginning optimized simulation...\n");

        sampling_kernel(in, GSD);
        for( int m = 0; m < 12; m++ )
                xs_lookup_kernel_optimization_2(in, GSD, m);

        copy_from_device(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long));
        unsigned long verification_scalar = reduce_verification(SD, in.lookups);

        free_sample_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        size_t sz = in.lookups * sizeof(double) + in.lookups * sizeof(int);
        allocate_sample_buffers(in, GSD);
        if( mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", sz/1024.0/1024.0);
        if( mype == 0) printf("Beginning optimized simulation...\n");

        sampling_kernel(in, GSD);
        xs_lookup_kernel_optimization_3(in, GSD, 0);
        xs_lookup_kernel_optimization_3(in, GSD, 1);

        copy_from_device(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long));
        unsigned long verification_scalar = reduce_verification(SD, in.lookups);

        free_sample_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        size_t sz = in.lookups * sizeof(double) + in.lookups * sizeof(int);
        allocate_sample_buffers(in, GSD);
        if( mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", sz/1024.0/1024.0);
        if( mype == 0) printf("Beginning optimized simulation...\n");

        sampling_kernel(in, GSD);
        std::vector<int> host_mats(in.lookups);
        std::vector<double> host_energies(in.lookups);
        copy_samples_to_host(GSD, host_mats, host_energies);

        sort_samples_by_material(host_mats, host_energies);

        int n_lookups_per_material[12] = {0};
        for (int mat : host_mats)
                n_lookups_per_material[mat]++;

        copy_samples_to_device(GSD, host_mats, host_energies);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if (count == 0)
                        continue;
                xs_lookup_kernel_optimization_4(in, GSD, m, count, offset);
                offset += count;
        }

        copy_from_device(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long));
        unsigned long verification_scalar = reduce_verification(SD, in.lookups);

        free_sample_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        size_t sz = in.lookups * sizeof(double) + in.lookups * sizeof(int);
        allocate_sample_buffers(in, GSD);
        if( mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", sz/1024.0/1024.0);
        if( mype == 0) printf("Beginning optimized simulation...\n");

        sampling_kernel(in, GSD);
        std::vector<int> host_mats(in.lookups);
        std::vector<double> host_energies(in.lookups);
        copy_samples_to_host(GSD, host_mats, host_energies);

        std::vector<std::pair<int, double>> zipped(in.lookups);
        for (int i = 0; i < in.lookups; ++i)
                zipped[i] = {host_mats[i], host_energies[i]};

        auto mid = std::stable_partition(zipped.begin(), zipped.end(), [](const auto &entry) {
                return entry.first == 0;
        });

        int n_fuel_lookups = std::distance(zipped.begin(), mid);
        for (int i = 0; i < in.lookups; ++i) {
                host_mats[i] = zipped[i].first;
                host_energies[i] = zipped[i].second;
        }

        copy_samples_to_device(GSD, host_mats, host_energies);

        xs_lookup_kernel_optimization_5(in, GSD, n_fuel_lookups, 0);
        xs_lookup_kernel_optimization_5(in, GSD, in.lookups - n_fuel_lookups, n_fuel_lookups);

        copy_from_device(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long));
        unsigned long verification_scalar = reduce_verification(SD, in.lookups);

        free_sample_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        size_t sz = in.lookups * sizeof(double) + in.lookups * sizeof(int);
        allocate_sample_buffers(in, GSD);
        if( mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", sz/1024.0/1024.0);
        if( mype == 0) printf("Beginning optimized simulation...\n");

        sampling_kernel(in, GSD);
        std::vector<int> host_mats(in.lookups);
        std::vector<double> host_energies(in.lookups);
        copy_samples_to_host(GSD, host_mats, host_energies);

        sort_samples_by_material_then_energy(host_mats, host_energies);

        int n_lookups_per_material[12] = {0};
        for (int mat : host_mats)
                n_lookups_per_material[mat]++;

        copy_samples_to_device(GSD, host_mats, host_energies);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if (count == 0)
                        continue;
                xs_lookup_kernel_optimization_4(in, GSD, m, count, offset);
                offset += count;
        }

        copy_from_device(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long));
        unsigned long verification_scalar = reduce_verification(SD, in.lookups);

        free_sample_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}
