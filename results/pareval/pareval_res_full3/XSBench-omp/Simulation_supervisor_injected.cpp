#include "XSbench_header.cuh"
#include "gate.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#pragma omp declare target

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
        const int n_nucs = num_nucs[mat];
        const int * const mat_vec = mats + mat * max_num_nucs;
        const double * const conc_vec = concs + mat * max_num_nucs;

        for( int k = 0; k < 5; k++ )
                macro_xs_vector[k] = 0;

        if( grid_type == UNIONIZED )
                idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);
        else if( grid_type == HASH )
        {
                double du = 1.0 / hash_bins;
                idx = p_energy / du;
        }

        for( int j = 0; j < n_nucs; j++ )
        {
                double xs_vector[5];
                p_nuc = mat_vec[j];
                conc = conc_vec[j];
                calculate_micro_xs( p_energy, p_nuc, n_isotopes,
                                   n_gridpoints, egrid, index_data,
                                   nuclide_grids, idx, xs_vector, grid_type, hash_bins );
                for( int k = 0; k < 5; k++ )
                        macro_xs_vector[k] += xs_vector[k] * conc;
        }
}

#pragma omp end declare target

static void copy_samples_to_host(int n, int *device_mats, double *device_energies,
                                 std::vector<int> &host_mats, std::vector<double> &host_energies)
{
        if( n == 0 )
                return;
        int device = omp_get_default_device();
        int host_device = omp_get_initial_device();
        omp_target_memcpy(host_mats.data(), device_mats, n * sizeof(int), 0, 0, host_device, device);
        omp_target_memcpy(host_energies.data(), device_energies, n * sizeof(double), 0, 0, host_device, device);
}

static void copy_samples_to_device(int n, int *device_mats, double *device_energies,
                                   const std::vector<int> &host_mats, const std::vector<double> &host_energies)
{
        if( n == 0 )
                return;
        int device = omp_get_default_device();
        int host_device = omp_get_initial_device();
        omp_target_memcpy(device_mats, host_mats.data(), n * sizeof(int), 0, 0, device, host_device);
        omp_target_memcpy(device_energies, host_energies.data(), n * sizeof(double), 0, 0, device, host_device);
}

static void sort_samples_by_material(int n, std::vector<int> &mats, std::vector<double> &energies)
{
        if( n <= 1 )
                return;
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                if( mats[a] != mats[b] )
                        return mats[a] < mats[b];
                return energies[a] < energies[b];
        });
        std::vector<int> mats_copy(n);
        std::vector<double> energies_copy(n);
        for( int i = 0; i < n; i++ )
        {
                mats_copy[i] = mats[indices[i]];
                energies_copy[i] = energies[indices[i]];
        }
        std::memcpy(mats.data(), mats_copy.data(), n * sizeof(int));
        std::memcpy(energies.data(), energies_copy.data(), n * sizeof(double));
}

static void sort_chunk_by_energy(int offset, int count, std::vector<int> &mats, std::vector<double> &energies)
{
        if( count <= 1 )
                return;
        std::vector<int> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                return energies[offset + a] < energies[offset + b];
        });
        std::vector<int> mats_copy(count);
        std::vector<double> energies_copy(count);
        for( int i = 0; i < count; i++ )
        {
                mats_copy[i] = mats[offset + indices[i]];
                energies_copy[i] = energies[offset + indices[i]];
        }
        std::memcpy(mats.data() + offset, mats_copy.data(), count * sizeof(int));
        std::memcpy(energies.data() + offset, energies_copy.data(), count * sizeof(double));
}

static void partition_by_fuel(int n, int n_fuel, std::vector<int> &mats, std::vector<double> &energies)
{
        if( n == 0 )
                return;
        std::vector<int> mats_copy(n);
        std::vector<double> energies_copy(n);
        int fuel_pos = 0;
        int other_pos = n_fuel;
        for( int i = 0; i < n; i++ )
        {
                if( mats[i] == 0 )
                {
                        mats_copy[fuel_pos] = mats[i];
                        energies_copy[fuel_pos] = energies[i];
                        fuel_pos++;
                }
                else
                {
                        mats_copy[other_pos] = mats[i];
                        energies_copy[other_pos] = energies[i];
                        other_pos++;
                }
        }
        std::memcpy(mats.data(), mats_copy.data(), n * sizeof(int));
        std::memcpy(energies.data(), energies_copy.data(), n * sizeof(double));
}

void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD )
{
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        int * mats = GSD.mats;
        double * unionized = GSD.unionized_energy_array;
        int * index_grid = GSD.index_grid;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;
        int device = omp_get_default_device();

        #pragma omp target teams loop is_device_ptr(num_nucs, concs, mats, unionized, index_grid, nuclide_grid, verification) device(device)
        for( int i = 0; i < in.lookups; i++ )
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*i);

                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy,
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
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

void sampling_kernel(Inputs in, SimulationData GSD )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int device = omp_get_default_device();

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples) device(device)
        for( int i = 0; i < in.lookups; i++ )
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*i);

                double p_energy = LCG_random_double(&seed);
                int mat         = pick_mat(&seed);

                p_energy_samples[i] = p_energy;
                mat_samples[i] = mat;
        }
}

void xs_lookup_kernel_optimization_1(Inputs in, SimulationData GSD )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        int * index_grid = GSD.index_grid;
        double * unionized = GSD.unionized_energy_array;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;
        int device = omp_get_default_device();

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, index_grid, unionized, nuclide_grid, mats, verification) device(device)
        for( int i = 0; i < in.lookups; i++ )
        {
                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat_samples[i],
                        in.n_isotopes,
                        in.n_gridpoints,
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

void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        int * index_grid = GSD.index_grid;
        double * unionized = GSD.unionized_energy_array;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;
        int device = omp_get_default_device();

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, index_grid, unionized, nuclide_grid, mats, verification) device(device)
        for( int i = 0; i < in.lookups; i++ )
        {
                if( mat_samples[i] != m )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        m,
                        in.n_isotopes,
                        in.n_gridpoints,
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

void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        int * index_grid = GSD.index_grid;
        double * unionized = GSD.unionized_energy_array;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;
        int device = omp_get_default_device();

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, index_grid, unionized, nuclide_grid, mats, verification) device(device)
        for( int i = 0; i < in.lookups; i++ )
        {
                int mat = mat_samples[i];

                bool should_run = ((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0));
                if( !should_run )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
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

void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        int * index_grid = GSD.index_grid;
        double * unionized = GSD.unionized_energy_array;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;
        int device = omp_get_default_device();

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, index_grid, unionized, nuclide_grid, mats, verification) device(device)
        for( int local_idx = 0; local_idx < n_lookups; local_idx++ )
        {
                int i = local_idx + offset;

                if( mat_samples[i] != m )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat_samples[i],
                        in.n_isotopes,
                        in.n_gridpoints,
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

void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset )
{
        double * p_energy_samples = GSD.p_energy_samples;
        int * mat_samples = GSD.mat_samples;
        int * num_nucs = GSD.num_nucs;
        double * concs = GSD.concs;
        int * index_grid = GSD.index_grid;
        double * unionized = GSD.unionized_energy_array;
        NuclideGridPoint * nuclide_grid = GSD.nuclide_grid;
        int * mats = GSD.mats;
        unsigned long * verification = GSD.verification;
        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = GSD.max_num_nucs;
        int device = omp_get_default_device();

        #pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples, num_nucs, concs, index_grid, unionized, nuclide_grid, mats, verification) device(device)
        for( int local_idx = 0; local_idx < n_lookups; local_idx++ )
        {
                int i = local_idx + offset;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        p_energy_samples[i],
                        mat_samples[i],
                        in.n_isotopes,
                        in.n_gridpoints,
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

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        double start = get_time();
        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        profile->host_to_device_time = get_time() - start;

        if( mype == 0 )
                printf("Running baseline event-based simulation...\n");

        int nwarmups = in.num_warmups;
        start = 0.0;
        for (int i = 0; i < in.num_iterations + nwarmups; i++) {
                if (i == nwarmups)
                        start = get_time();
                xs_lookup_kernel_baseline(in, GSD);
        }
        profile->kernel_time = get_time() - start;

        if( mype == 0)
                printf("Reducing verification results...\n");
        start = get_time();
        int device = omp_get_default_device();
        int host_device = omp_get_initial_device();
        omp_target_memcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), 0, 0, host_device, device);
        GATE_CHECKSUM_BYTES("simulation_verification", SD.verification, in.lookups * sizeof(unsigned long));
        profile->device_to_host_time = get_time() - start;

        unsigned long verification_scalar = 0;
        for( int i =0; i < in.lookups; i++ )
                verification_scalar += SD.verification[i];

        release_device_memory(GSD);

        return verification_scalar;
}

static void allocate_sampling_buffers(Inputs in, SimulationData &GSD)
{
        int device = omp_get_default_device();
        size_t sz_double = in.lookups * sizeof(double);
        size_t sz_int = in.lookups * sizeof(int);
        GSD.p_energy_samples = (double *) omp_target_alloc(sz_double, device);
        GSD.length_p_energy_samples = in.lookups;
        GSD.mat_samples = (int *) omp_target_alloc(sz_int, device);
        GSD.length_mat_samples = in.lookups;
}

static void free_sampling_buffers(SimulationData &GSD)
{
        int device = omp_get_default_device();
        if (GSD.p_energy_samples)
                omp_target_free(GSD.p_energy_samples, device);
        if (GSD.mat_samples)
                omp_target_free(GSD.mat_samples, device);
}

static size_t verification_host_size(Inputs in)
{
        return in.lookups * sizeof(unsigned long);
}

static unsigned long reduce_verification(Inputs in, SimulationData SD, SimulationData GSD)
{
        int device = omp_get_default_device();
        int host_device = omp_get_initial_device();
        omp_target_memcpy(SD.verification, GSD.verification, verification_host_size(in), 0, 0, host_device, device);
        unsigned long verification_scalar = 0;
        for( int i =0; i < in.lookups; i++ )
                verification_scalar += SD.verification[i];
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";
        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        allocate_sampling_buffers(in, GSD);

        sampling_kernel(in, GSD);
        xs_lookup_kernel_optimization_1(in, GSD);

        if( mype == 0) printf("Reducing verification results...\n");
        unsigned long verification_scalar = reduce_verification(in, SD, GSD);

        free_sampling_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 2 - Material Lookup Kernels";
        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        allocate_sampling_buffers(in, GSD);

        sampling_kernel(in, GSD);
        for( int m = 0; m < 12; m++ )
                xs_lookup_kernel_optimization_2(in, GSD, m);

        if( mype == 0) printf("Reducing verification results...\n");
        unsigned long verification_scalar = reduce_verification(in, SD, GSD);

        free_sampling_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";
        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        allocate_sampling_buffers(in, GSD);

        sampling_kernel(in, GSD);
        xs_lookup_kernel_optimization_3(in, GSD, 0);
        xs_lookup_kernel_optimization_3(in, GSD, 1);

        if( mype == 0) printf("Reducing verification results...\n");
        unsigned long verification_scalar = reduce_verification(in, SD, GSD);

        free_sampling_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";
        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        allocate_sampling_buffers(in, GSD);

        sampling_kernel(in, GSD);

        int n = in.lookups;
        std::vector<int> mat_host(n);
        std::vector<double> energy_host(n);
        copy_samples_to_host(n, GSD.mat_samples, GSD.p_energy_samples, mat_host, energy_host);

        int n_lookups_per_material[12] = {0};
        for( int i = 0; i < n; i++ )
                n_lookups_per_material[mat_host[i]]++;

        sort_samples_by_material(n, mat_host, energy_host);
        copy_samples_to_device(n, GSD.mat_samples, GSD.p_energy_samples, mat_host, energy_host);

        if( mype == 0) printf("Beginning optimized simulation...\n");
        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if( count == 0 )
                        continue;
                xs_lookup_kernel_optimization_4(in, GSD, m, count, offset);
                offset += count;
        }

        if( mype == 0) printf("Reducing verification results...\n");
        unsigned long verification_scalar = reduce_verification(in, SD, GSD);

        free_sampling_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";
        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        allocate_sampling_buffers(in, GSD);

        sampling_kernel(in, GSD);

        int n = in.lookups;
        std::vector<int> mat_host(n);
        std::vector<double> energy_host(n);
        copy_samples_to_host(n, GSD.mat_samples, GSD.p_energy_samples, mat_host, energy_host);

        int n_fuel_lookups = 0;
        for( int i = 0; i < n; i++ )
                if( mat_host[i] == 0 )
                        n_fuel_lookups++;

        partition_by_fuel(n, n_fuel_lookups, mat_host, energy_host);
        copy_samples_to_device(n, GSD.mat_samples, GSD.p_energy_samples, mat_host, energy_host);

        xs_lookup_kernel_optimization_5(in, GSD, n_fuel_lookups, 0);
        xs_lookup_kernel_optimization_5(in, GSD, n - n_fuel_lookups, n_fuel_lookups);

        if( mype == 0) printf("Reducing verification results...\n");
        unsigned long verification_scalar = reduce_verification(in, SD, GSD);

        free_sampling_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";
        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        allocate_sampling_buffers(in, GSD);

        sampling_kernel(in, GSD);

        int n = in.lookups;
        std::vector<int> mat_host(n);
        std::vector<double> energy_host(n);
        copy_samples_to_host(n, GSD.mat_samples, GSD.p_energy_samples, mat_host, energy_host);

        int n_lookups_per_material[12] = {0};
        for( int i = 0; i < n; i++ )
                n_lookups_per_material[mat_host[i]]++;

        sort_samples_by_material(n, mat_host, energy_host);
        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if( count > 0 )
                        sort_chunk_by_energy(offset, count, mat_host, energy_host);
                offset += count;
        }
        copy_samples_to_device(n, GSD.mat_samples, GSD.p_energy_samples, mat_host, energy_host);

        offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int count = n_lookups_per_material[m];
                if( count == 0 )
                        continue;
                xs_lookup_kernel_optimization_4(in, GSD, m, count, offset);
                offset += count;
        }

        if( mype == 0) printf("Reducing verification results...\n");
        unsigned long verification_scalar = reduce_verification(in, SD, GSD);

        free_sampling_buffers(GSD);
        release_device_memory(GSD);

        return verification_scalar;
}
