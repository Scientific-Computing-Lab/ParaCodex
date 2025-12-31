#include "XSbench_header.cuh"
#include <algorithm>
#include <numeric>
#include <vector>

static void sort_pairs_by_key(int *keys, double *values, int length)
{
        if (length <= 1)
                return;

        std::vector<int> indices(length);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                return keys[a] < keys[b];
        });

        std::vector<int> sorted_keys(length);
        std::vector<double> sorted_values(length);
        for (int i = 0; i < length; ++i) {
                sorted_keys[i] = keys[indices[i]];
                sorted_values[i] = values[indices[i]];
        }
        std::copy(sorted_keys.begin(), sorted_keys.end(), keys);
        std::copy(sorted_values.begin(), sorted_values.end(), values);
}

static void sort_pairs_by_double_key(double *keys, int *values, int length)
{
        if (length <= 1)
                return;

        std::vector<int> indices(length);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                return keys[a] < keys[b];
        });

        std::vector<double> sorted_keys(length);
        std::vector<int> sorted_values(length);
        for (int i = 0; i < length; ++i) {
                sorted_keys[i] = keys[indices[i]];
                sorted_values[i] = values[indices[i]];
        }
        std::copy(sorted_keys.begin(), sorted_keys.end(), keys);
        std::copy(sorted_values.begin(), sorted_values.end(), values);
}

static void partition_fuel_samples(int *mat_samples, double *p_energy_samples, int lookups, int n_fuel)
{
        if (lookups == 0)
                return;

        std::vector<int> temp_mat(lookups);
        std::vector<double> temp_energy(lookups);
        int fuel_index = 0;
        int other_index = n_fuel;

        for (int i = 0; i < lookups; ++i) {
                if (mat_samples[i] == 0) {
                        temp_mat[fuel_index] = mat_samples[i];
                        temp_energy[fuel_index] = p_energy_samples[i];
                        ++fuel_index;
                } else {
                        temp_mat[other_index] = mat_samples[i];
                        temp_energy[other_index] = p_energy_samples[i];
                        ++other_index;
                }
        }

        std::copy(temp_mat.begin(), temp_mat.end(), mat_samples);
        std::copy(temp_energy.begin(), temp_energy.end(), p_energy_samples);
}

#pragma omp declare target
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                                   long n_gridpoints,
                                   double * __restrict__ egrid, int * __restrict__ index_data,
                                   NuclideGridPoint * __restrict__ nuclide_grids,
                                   long idx, double * __restrict__ xs_vector, int grid_type, int hash_bins ){

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
                                   double * __restrict__ macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
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
#pragma omp end declare target

void sampling_kernel(Inputs in, SimulationData GSD )
{
        int lookups = in.lookups;
        #pragma omp target is_device_ptr(GSD.p_energy_samples, GSD.mat_samples)
        #pragma omp teams
        #pragma omp loop
        for (int i = 0; i < lookups; ++i) {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                GSD.p_energy_samples[i] = p_energy;
                GSD.mat_samples[i] = mat;
        }
}

void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD )
{
        int lookups = in.lookups;
        #pragma omp target is_device_ptr(GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, GSD.verification)
        #pragma omp teams
        #pragma omp loop
        for (int i = 0; i < lookups; ++i) {
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
                        GSD.num_nucs,
                        GSD.concs,
                        GSD.unionized_energy_array,
                        GSD.index_grid,
                        GSD.nuclide_grid,
                        GSD.mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for(int j = 0; j < 5; j++ ) {
                        if( macro_xs_vector[j] > max ) {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                GSD.verification[i] = max_idx+1;
        }
}

void xs_lookup_kernel_optimization_1(Inputs in, SimulationData GSD )
{
        int lookups = in.lookups;
        #pragma omp target is_device_ptr(GSD.p_energy_samples, GSD.mat_samples, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, GSD.verification)
        #pragma omp teams
        #pragma omp loop
        for (int i = 0; i < lookups; ++i) {
                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        GSD.p_energy_samples[i],
                        GSD.mat_samples[i],
                        in.n_isotopes,
                        in.n_gridpoints,
                        GSD.num_nucs,
                        GSD.concs,
                        GSD.unionized_energy_array,
                        GSD.index_grid,
                        GSD.nuclide_grid,
                        GSD.mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for(int j = 0; j < 5; j++ ) {
                        if( macro_xs_vector[j] > max ) {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                GSD.verification[i] = max_idx+1;
        }
}

void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m )
{
        int lookups = in.lookups;
        #pragma omp target is_device_ptr(GSD.p_energy_samples, GSD.mat_samples, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, GSD.verification)
        #pragma omp teams
        #pragma omp loop
        for (int i = 0; i < lookups; ++i) {
                if( GSD.mat_samples[i] != m )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        GSD.p_energy_samples[i],
                        m,
                        in.n_isotopes,
                        in.n_gridpoints,
                        GSD.num_nucs,
                        GSD.concs,
                        GSD.unionized_energy_array,
                        GSD.index_grid,
                        GSD.nuclide_grid,
                        GSD.mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for(int j = 0; j < 5; j++ ) {
                        if( macro_xs_vector[j] > max ) {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                GSD.verification[i] = max_idx+1;
        }
}

void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel )
{
        int lookups = in.lookups;
        #pragma omp target is_device_ptr(GSD.p_energy_samples, GSD.mat_samples, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, GSD.verification)
        #pragma omp teams
        #pragma omp loop
        for (int i = 0; i < lookups; ++i) {
                int mat = GSD.mat_samples[i];

                if( ((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0 ) ))
                {
                        double macro_xs_vector[5] = {0};

                        calculate_macro_xs(
                                GSD.p_energy_samples[i],
                                mat,
                                in.n_isotopes,
                                in.n_gridpoints,
                                GSD.num_nucs,
                                GSD.concs,
                                GSD.unionized_energy_array,
                                GSD.index_grid,
                                GSD.nuclide_grid,
                                GSD.mats,
                                macro_xs_vector,
                                in.grid_type,
                                in.hash_bins,
                                GSD.max_num_nucs
                        );

                        double max = -1.0;
                        int max_idx = 0;
                        for(int j = 0; j < 5; j++ ) {
                                if( macro_xs_vector[j] > max ) {
                                        max = macro_xs_vector[j];
                                        max_idx = j;
                                }
                        }
                        GSD.verification[i] = max_idx+1;
                }
        }
}

void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset )
{
        #pragma omp target is_device_ptr(GSD.p_energy_samples, GSD.mat_samples, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, GSD.verification)
        #pragma omp teams
        #pragma omp loop
        for (int idx = 0; idx < n_lookups; ++idx) {
                int i = idx + offset;
                int mat = GSD.mat_samples[i];
                if( mat != m )
                        continue;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        GSD.p_energy_samples[i],
                        mat,
                        in.n_isotopes,
                        in.n_gridpoints,
                        GSD.num_nucs,
                        GSD.concs,
                        GSD.unionized_energy_array,
                        GSD.index_grid,
                        GSD.nuclide_grid,
                        GSD.mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for(int j = 0; j < 5; j++ ) {
                        if( macro_xs_vector[j] > max ) {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                GSD.verification[i] = max_idx+1;
        }
}

void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset )
{
        #pragma omp target teams loop
        for (int idx = 0; idx < n_lookups; ++idx) {
                int i = idx + offset;

                double macro_xs_vector[5] = {0};

                calculate_macro_xs(
                        GSD.p_energy_samples[i],
                        GSD.mat_samples[i],
                        in.n_isotopes,
                        in.n_gridpoints,
                        GSD.num_nucs,
                        GSD.concs,
                        GSD.unionized_energy_array,
                        GSD.index_grid,
                        GSD.nuclide_grid,
                        GSD.mats,
                        macro_xs_vector,
                        in.grid_type,
                        in.hash_bins,
                        GSD.max_num_nucs
                );

                double max = -1.0;
                int max_idx = 0;
                for(int j = 0; j < 5; j++ ) {
                        if( macro_xs_vector[j] > max ) {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                GSD.verification[i] = max_idx+1;
        }
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        double start = get_time();
        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        profile->host_to_device_time = get_time() - start;

        if( mype == 0)      printf("Running baseline event-based simulation...\n");

        int nwarmups = in.num_warmups;
        start = 0.0;
        int host = omp_get_initial_device();
        int device = omp_get_default_device();

        for (int i = 0; i < in.num_iterations + nwarmups; i++) {
                if (i == nwarmups) {
                        start = get_time();
                }
                xs_lookup_kernel_baseline(in, GSD);
        }

        profile->kernel_time = get_time() - start;

        if( mype == 0)      printf("Reducing verification results...\n");
        start = get_time();
        omp_target_memcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), 0, 0, host, device);
        profile->device_to_host_time = get_time() - start;

        unsigned long long verification_scalar = std::accumulate(SD.verification, SD.verification + in.lookups, 0ULL);

        release_device_memory(GSD);

        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        if( mype == 0) printf("Beginning optimized simulation...\n");
        int host = omp_get_initial_device();
        int device = omp_get_default_device();

        sampling_kernel(in, GSD);
        xs_lookup_kernel_optimization_1(in, GSD);

        omp_target_memcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), 0, 0, host, device);
        unsigned long long verification_scalar = std::accumulate(SD.verification, SD.verification + in.lookups, 0ULL);
        release_device_memory(GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 2 - Material Lookup Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        if( mype == 0) printf("Beginning optimized simulation...\n");
        int host = omp_get_initial_device();
        int device = omp_get_default_device();

        sampling_kernel(in, GSD);
        for( int m = 0; m < 12; m++ )
                xs_lookup_kernel_optimization_2(in, GSD, m);

        omp_target_memcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), 0, 0, host, device);
        unsigned long long verification_scalar = std::accumulate(SD.verification, SD.verification + in.lookups, 0ULL);
        release_device_memory(GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        if( mype == 0) printf("Beginning optimized simulation...\n");
        int host = omp_get_initial_device();
        int device = omp_get_default_device();

        sampling_kernel(in, GSD);
        xs_lookup_kernel_optimization_3(in, GSD, 0);
        xs_lookup_kernel_optimization_3(in, GSD, 1);

        omp_target_memcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), 0, 0, host, device);
        unsigned long long verification_scalar = std::accumulate(SD.verification, SD.verification + in.lookups, 0ULL);
        release_device_memory(GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        if( mype == 0) printf("Beginning optimized simulation...\n");
        int host = omp_get_initial_device();
        int device = omp_get_default_device();

        std::vector<double> energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);
        int n_lookups_per_material[12];

        sampling_kernel(in, GSD);
        omp_target_memcpy(energy_samples.data(), GSD.p_energy_samples, in.lookups * sizeof(double), 0, 0, host, device);
        omp_target_memcpy(mat_samples.data(), GSD.mat_samples, in.lookups * sizeof(int), 0, 0, host, device);

        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = std::count(mat_samples.begin(), mat_samples.end(), m);

        sort_pairs_by_key(mat_samples.data(), energy_samples.data(), in.lookups);

        omp_target_memcpy(GSD.p_energy_samples, energy_samples.data(), in.lookups * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(GSD.mat_samples, mat_samples.data(), in.lookups * sizeof(int), 0, 0, device, host);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                if( n_lookups_per_material[m] == 0 )
                        continue;
                xs_lookup_kernel_optimization_4(in, GSD, m, n_lookups_per_material[m], offset);
                offset += n_lookups_per_material[m];
        }

        omp_target_memcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), 0, 0, host, device);
        unsigned long long verification_scalar = std::accumulate(SD.verification, SD.verification + in.lookups, 0ULL);
        release_device_memory(GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        if( mype == 0) printf("Beginning optimized simulation...\n");
        int host = omp_get_initial_device();
        int device = omp_get_default_device();

        std::vector<double> energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);

        sampling_kernel(in, GSD);
        omp_target_memcpy(energy_samples.data(), GSD.p_energy_samples, in.lookups * sizeof(double), 0, 0, host, device);
        omp_target_memcpy(mat_samples.data(), GSD.mat_samples, in.lookups * sizeof(int), 0, 0, host, device);

        int n_fuel_lookups = std::count(mat_samples.begin(), mat_samples.end(), 0);
        partition_fuel_samples(mat_samples.data(), energy_samples.data(), in.lookups, n_fuel_lookups);

        omp_target_memcpy(GSD.p_energy_samples, energy_samples.data(), in.lookups * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(GSD.mat_samples, mat_samples.data(), in.lookups * sizeof(int), 0, 0, device, host);

        xs_lookup_kernel_optimization_5(in, GSD, n_fuel_lookups, 0);
        xs_lookup_kernel_optimization_5(in, GSD, in.lookups - n_fuel_lookups, n_fuel_lookups);

        omp_target_memcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), 0, 0, host, device);
        unsigned long long verification_scalar = std::accumulate(SD.verification, SD.verification + in.lookups, 0ULL);
        release_device_memory(GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData SD, int mype)
{
        const char * optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

        if( mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        if( mype == 0) printf("Allocating additional device data required by kernel...\n");
        if( mype == 0) printf("Beginning optimized simulation...\n");
        int host = omp_get_initial_device();
        int device = omp_get_default_device();

        std::vector<double> energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);
        int n_lookups_per_material[12];

        sampling_kernel(in, GSD);
        omp_target_memcpy(energy_samples.data(), GSD.p_energy_samples, in.lookups * sizeof(double), 0, 0, host, device);
        omp_target_memcpy(mat_samples.data(), GSD.mat_samples, in.lookups * sizeof(int), 0, 0, host, device);

        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = std::count(mat_samples.begin(), mat_samples.end(), m);

        sort_pairs_by_key(mat_samples.data(), energy_samples.data(), in.lookups);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                int length = n_lookups_per_material[m];
                if( length == 0 )
                        continue;
                sort_pairs_by_double_key(energy_samples.data() + offset, mat_samples.data() + offset, length);
                offset += length;
        }

        omp_target_memcpy(GSD.p_energy_samples, energy_samples.data(), in.lookups * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(GSD.mat_samples, mat_samples.data(), in.lookups * sizeof(int), 0, 0, device, host);

        offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                if( n_lookups_per_material[m] == 0 )
                        continue;
                xs_lookup_kernel_optimization_4(in, GSD, m, n_lookups_per_material[m], offset);
                offset += n_lookups_per_material[m];
        }

        omp_target_memcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), 0, 0, host, device);
        unsigned long long verification_scalar = std::accumulate(SD.verification, SD.verification + in.lookups, 0ULL);
        release_device_memory(GSD);
        return verification_scalar;
}
