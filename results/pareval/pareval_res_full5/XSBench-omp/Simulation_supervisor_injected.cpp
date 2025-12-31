#include "XSbench_header.cuh"
#include "gate.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>
#include <vector>

#pragma omp declare target

static constexpr double k_pick_mat_cdf[12] = {
        0.0,
        0.052,
        0.327,
        0.461,
        0.615,
        0.679,
        0.745,
        0.800,
        0.808,
        0.823,
        0.848,
        0.861,
};

long grid_search(long n, double quarry, const double * __restrict__ A)
{
        long lowerLimit = 0;
        long upperLimit = n - 1;
        long examinationPoint;
        long length = upperLimit - lowerLimit;

        while (length > 1)
        {
                examinationPoint = lowerLimit + (length / 2);

                if (A[examinationPoint] > quarry)
                        upperLimit = examinationPoint;
                else
                        lowerLimit = examinationPoint;

                length = upperLimit - lowerLimit;
        }

        return lowerLimit;
}

long grid_search_nuclide(long n, double quarry, const NuclideGridPoint * __restrict__ A, long low, long high)
{
        long lowerLimit = low;
        long upperLimit = high;
        long examinationPoint;
        long length = upperLimit - lowerLimit;

        while (length > 1)
        {
                examinationPoint = lowerLimit + (length / 2);

                if (A[examinationPoint].energy > quarry)
                        upperLimit = examinationPoint;
                else
                        lowerLimit = examinationPoint;

                length = upperLimit - lowerLimit;
        }

        return lowerLimit;
}

int pick_mat(uint64_t *seed)
{
        double roll = LCG_random_double(seed);
        for (int i = 1; i < 12; ++i)
        {
                if (roll < k_pick_mat_cdf[i])
                        return i;
        }
        return 0;
}

double LCG_random_double(uint64_t *seed)
{
        const uint64_t m = 9223372036854775808ULL;
        const uint64_t a = 2806196910506780709ULL;
        const uint64_t c = 1ULL;
        *seed = (a * (*seed) + c) % m;
        return (double)(*seed) / (double)m;
}

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
        const uint64_t m = 9223372036854775808ULL;
        uint64_t a = 2806196910506780709ULL;
        uint64_t c = 1ULL;

        n = n % m;

        uint64_t a_new = 1;
        uint64_t c_new = 0;

        while (n > 0)
        {
                if (n & 1)
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

void calculate_micro_xs(double p_energy, int nuc, long n_isotopes, long n_gridpoints, const double * __restrict__ egrid,
                        const int * __restrict__ index_data, const NuclideGridPoint * __restrict__ nuclide_grids, long idx,
                        double * __restrict__ xs_vector, int grid_type, int hash_bins)
{
        double f;
        const NuclideGridPoint *low;
        const NuclideGridPoint *high;

        if (grid_type == NUCLIDE)
        {
                idx = grid_search_nuclide(n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints], 0, n_gridpoints - 1);

                if (idx == n_gridpoints - 1)
                        low = &nuclide_grids[nuc * n_gridpoints + idx - 1];
                else
                        low = &nuclide_grids[nuc * n_gridpoints + idx];
        }
        else if (grid_type == UNIONIZED)
        {
                if (index_data[idx * n_isotopes + nuc] == n_gridpoints - 1)
                        low = &nuclide_grids[nuc * n_gridpoints + index_data[idx * n_isotopes + nuc] - 1];
                else
                        low = &nuclide_grids[nuc * n_gridpoints + index_data[idx * n_isotopes + nuc]];
        }
        else
        {
                int u_low = index_data[idx * n_isotopes + nuc];

                int u_high;
                if (idx == hash_bins - 1)
                        u_high = n_gridpoints - 1;
                else
                        u_high = index_data[(idx + 1) * n_isotopes + nuc] + 1;

                double e_low  = nuclide_grids[nuc * n_gridpoints + u_low].energy;
                double e_high = nuclide_grids[nuc * n_gridpoints + u_high].energy;
                int lower;
                if (p_energy <= e_low)
                        lower = 0;
                else if (p_energy >= e_high)
                        lower = n_gridpoints - 1;
                else
                        lower = grid_search_nuclide(n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints], u_low, u_high);

                if (lower == n_gridpoints - 1)
                        low = &nuclide_grids[nuc * n_gridpoints + lower - 1];
                else
                        low = &nuclide_grids[nuc * n_gridpoints + lower];
        }

        high = low + 1;

        f = (high->energy - p_energy) / (high->energy - low->energy);

        xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
        xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
        xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
        xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
        xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, const int * __restrict__ num_nucs,
                        const double * __restrict__ concs, const double * __restrict__ egrid, const int * __restrict__ index_data,
                        const NuclideGridPoint * __restrict__ nuclide_grids, const int * __restrict__ mats,
                        double * __restrict__ macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs)
{
        int p_nuc;
        long idx = -1;
        double conc;

        for (int k = 0; k < 5; k++)
                macro_xs_vector[k] = 0;

        if (grid_type == UNIONIZED)
                idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid);
        else if (grid_type == HASH)
        {
                double du = 1.0 / hash_bins;
                idx = p_energy / du;
        }

        for (int j = 0; j < num_nucs[mat]; j++)
        {
                double xs_vector[5];
                p_nuc = mats[mat * max_num_nucs + j];
                conc = concs[mat * max_num_nucs + j];
                calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints, egrid, index_data, nuclide_grids,
                                  idx, xs_vector, grid_type, hash_bins);
                for (int k = 0; k < 5; k++)
                        macro_xs_vector[k] += xs_vector[k] * conc;
        }
}

inline unsigned long evaluate_lookup(double p_energy, int mat, long n_isotopes, long n_gridpoints, const int *num_nucs,
                                     const double *concs, const double *unionized_energy_array, const int *index_grid,
                                     const NuclideGridPoint *nuclide_grid, const int *mats, int grid_type, int hash_bins,
                                     int max_num_nucs)
{
        double macro_xs_vector[5] = {0};
        calculate_macro_xs(p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs, unionized_energy_array,
                           index_grid, nuclide_grid, mats, macro_xs_vector, grid_type, hash_bins, max_num_nucs);
        double max_val = -1.0;
        int max_idx = 0;
        for (int j = 0; j < 5; j++)
        {
                if (macro_xs_vector[j] > max_val)
                {
                        max_val = macro_xs_vector[j];
                        max_idx = j;
                }
        }
        return (unsigned long) (max_idx + 1);
}

#pragma omp end declare target

static void sort_samples_by_material(int *mat_samples, double *p_energy_samples, int lookups)
{
        std::vector<std::pair<int, double>> zipped(lookups);
        for (int i = 0; i < lookups; ++i)
                zipped[i] = std::make_pair(mat_samples[i], p_energy_samples[i]);

        std::stable_sort(zipped.begin(), zipped.end(), [](const auto &a, const auto &b) {
                return a.first < b.first;
        });

        for (int i = 0; i < lookups; ++i)
        {
                mat_samples[i] = zipped[i].first;
                p_energy_samples[i] = zipped[i].second;
        }
}

static int partition_samples_by_fuel(int *mat_samples, double *p_energy_samples, int lookups)
{
        std::vector<int> ordered_mat(lookups);
        std::vector<double> ordered_energy(lookups);
        int write = 0;

        for (int i = 0; i < lookups; ++i)
        {
                if (mat_samples[i] == 0)
                {
                        ordered_mat[write] = mat_samples[i];
                        ordered_energy[write] = p_energy_samples[i];
                        ++write;
                }
        }
        int fuel_count = write;
        for (int i = 0; i < lookups; ++i)
        {
                if (mat_samples[i] != 0)
                {
                        ordered_mat[write] = mat_samples[i];
                        ordered_energy[write] = p_energy_samples[i];
                        ++write;
                }
        }
        std::copy(ordered_mat.begin(), ordered_mat.end(), mat_samples);
        std::copy(ordered_energy.begin(), ordered_energy.end(), p_energy_samples);
        return fuel_count;
}

static void sort_energy_within_range(int *mat_samples, double *p_energy_samples, int start, int count)
{
        if (count <= 1)
                return;
        std::vector<std::pair<double, int>> chunk(count);
        for (int i = 0; i < count; ++i)
                chunk[i] = std::make_pair(p_energy_samples[start + i], mat_samples[start + i]);

        std::sort(chunk.begin(), chunk.end(), [](const auto &a, const auto &b) {
                return a.first < b.first;
        });

        for (int i = 0; i < count; ++i)
        {
                p_energy_samples[start + i] = chunk[i].first;
                mat_samples[start + i] = chunk[i].second;
        }
}

static void xs_lookup_kernel_baseline(const Inputs &in, SimulationData &SD)
{
        int lookups = in.lookups;
        const int *num_nucs = SD.num_nucs;
        const double *concs = SD.concs;
        const double *unionized_energy_array = SD.unionized_energy_array;
        const int *index_grid = SD.index_grid;
        const NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        const int *mats = SD.mats;
        unsigned long *verification = SD.verification;

        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = SD.max_num_nucs;
        long n_isotopes = in.n_isotopes;
        long n_gridpoints = in.n_gridpoints;

        #pragma omp target teams distribute parallel for thread_limit(256) \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid]) \
            map(tofrom: verification[0:lookups])
        for (int i = 0; i < lookups; ++i)
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ull * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                verification[i] = evaluate_lookup(p_energy, mat, n_isotopes, n_gridpoints, num_nucs,
                                                   concs, unionized_energy_array, index_grid, nuclide_grid,
                                                   mats, grid_type, hash_bins, max_num_nucs);
        }
}

static void sampling_kernel(const Inputs &in, double *p_energy_samples, int *mat_samples)
{
        int lookups = in.lookups;
        #pragma omp target teams distribute parallel for thread_limit(32) \
            map(tofrom: p_energy_samples[0:in.lookups], mat_samples[0:in.lookups])
        for (int i = 0; i < lookups; ++i)
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ull * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                p_energy_samples[i] = p_energy;
                mat_samples[i] = mat;
        }
}

static void xs_lookup_kernel_samples(const Inputs &in, SimulationData &SD, const double *p_energy_samples,
                                     const int *mat_samples, int start, int n_lookups)
{
        const int *num_nucs = SD.num_nucs;
        const double *concs = SD.concs;
        const double *unionized_energy_array = SD.unionized_energy_array;
        const int *index_grid = SD.index_grid;
        const NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        const int *mats = SD.mats;
        unsigned long *verification = SD.verification;

        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = SD.max_num_nucs;
        long n_isotopes = in.n_isotopes;
        long n_gridpoints = in.n_gridpoints;

        #pragma omp target teams distribute parallel for thread_limit(256) \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid], p_energy_samples[0:in.lookups], mat_samples[0:in.lookups]) \
            map(tofrom: verification[0:in.lookups])
        for (int idx = 0; idx < n_lookups; ++idx)
        {
                int global_idx = start + idx;
                int mat = mat_samples[global_idx];
                double p_energy = p_energy_samples[global_idx];
                verification[global_idx] = evaluate_lookup(p_energy, mat, n_isotopes, n_gridpoints, num_nucs,
                                                           concs, unionized_energy_array, index_grid, nuclide_grid,
                                                           mats, grid_type, hash_bins, max_num_nucs);
        }
}

static void xs_lookup_kernel_filter_material(const Inputs &in, SimulationData &SD, const double *p_energy_samples,
                                             const int *mat_samples, int material)
{
        const int *num_nucs = SD.num_nucs;
        const double *concs = SD.concs;
        const double *unionized_energy_array = SD.unionized_energy_array;
        const int *index_grid = SD.index_grid;
        const NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        const int *mats = SD.mats;
        unsigned long *verification = SD.verification;

        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = SD.max_num_nucs;
        long n_isotopes = in.n_isotopes;
        long n_gridpoints = in.n_gridpoints;

        #pragma omp target teams distribute parallel for thread_limit(256) \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid], p_energy_samples[0:in.lookups], mat_samples[0:in.lookups]) \
            map(tofrom: verification[0:in.lookups])
        for (int idx = 0; idx < in.lookups; ++idx)
        {
                int mat = mat_samples[idx];
                if (mat != material)
                        continue;
                double p_energy = p_energy_samples[idx];
                verification[idx] = evaluate_lookup(p_energy, mat, n_isotopes, n_gridpoints, num_nucs,
                                                   concs, unionized_energy_array, index_grid, nuclide_grid,
                                                   mats, grid_type, hash_bins, max_num_nucs);
        }
}

static void xs_lookup_kernel_filter_fuel(const Inputs &in, SimulationData &SD, const double *p_energy_samples,
                                         const int *mat_samples, bool is_fuel)
{
        const int *num_nucs = SD.num_nucs;
        const double *concs = SD.concs;
        const double *unionized_energy_array = SD.unionized_energy_array;
        const int *index_grid = SD.index_grid;
        const NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        const int *mats = SD.mats;
        unsigned long *verification = SD.verification;

        int grid_type = in.grid_type;
        int hash_bins = in.hash_bins;
        int max_num_nucs = SD.max_num_nucs;
        long n_isotopes = in.n_isotopes;
        long n_gridpoints = in.n_gridpoints;

        #pragma omp target teams distribute parallel for thread_limit(256) \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid], p_energy_samples[0:in.lookups], mat_samples[0:in.lookups]) \
            map(tofrom: verification[0:in.lookups])
        for (int idx = 0; idx < in.lookups; ++idx)
        {
                int mat = mat_samples[idx];
                if ((is_fuel && mat != 0) || (!is_fuel && mat == 0))
                        continue;
                double p_energy = p_energy_samples[idx];
                verification[idx] = evaluate_lookup(p_energy, mat, n_isotopes, n_gridpoints, num_nucs,
                                                   concs, unionized_energy_array, index_grid, nuclide_grid,
                                                   mats, grid_type, hash_bins, max_num_nucs);
        }
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile *profile)
{
        double start = get_time();
        profile->host_to_device_time = 0.0;

        if (mype == 0)
                printf("Running baseline event-based simulation...\n");

        int total_runs = in.num_iterations + in.num_warmups;
        double kernel_start = 0.0;
        double kernel_end = 0.0;
        const int *num_nucs = SD.num_nucs;
        const double *concs = SD.concs;
        const int *mats = SD.mats;
        const double *unionized_energy_array = SD.unionized_energy_array;
        const int *index_grid = SD.index_grid;
        const NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        unsigned long *verification = SD.verification;

        #pragma omp target data \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid]) \
            map(tofrom: verification[0:in.lookups])
        {
                for (int iter = 0; iter < total_runs; ++iter)
                {
                        if (iter == in.num_warmups)
                                kernel_start = get_time();
                        xs_lookup_kernel_baseline(in, SD);
                        if (iter == total_runs - 1)
                                kernel_end = get_time();
                }
                double copy_start = get_time();
                #pragma omp target update from(SD.verification[0:in.lookups])
                profile->device_to_host_time = get_time() - copy_start;
        }

        profile->kernel_time = kernel_end - kernel_start;
        unsigned long verification_scalar = std::accumulate(SD.verification, SD.verification + in.lookups, 0UL);
        GATE_CHECKSUM_BYTES("verification_buffer", SD.verification, SD.length_verification * sizeof(unsigned long));
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if (mype == 0)
                printf("Simulation Kernel:\"%s\"\n", optimization_name);

        size_t total_sz = in.lookups * (sizeof(double) + sizeof(int));
        if (mype == 0)
                printf("Allocating an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

        std::vector<double> p_energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);
        double *p_energy_ptr = p_energy_samples.data();
        int *mat_ptr = mat_samples.data();

                if (mype == 0)
                        printf("Beginning optimized simulation...\n");

                sampling_kernel(in, p_energy_ptr, mat_ptr);
                xs_lookup_kernel_samples(in, SD, p_energy_ptr, mat_ptr, 0, in.lookups);
        return std::accumulate(SD.verification, SD.verification + in.lookups, 0UL);
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 2 - Material Lookup Kernels";

        if (mype == 0)
                printf("Simulation Kernel:\"%s\"\n", optimization_name);

        size_t total_sz = in.lookups * (sizeof(double) + sizeof(int));
        if (mype == 0)
                printf("Allocating an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

        std::vector<double> p_energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);
        double *p_energy_ptr = p_energy_samples.data();
        int *mat_ptr = mat_samples.data();
        int *num_nucs = SD.num_nucs;
        double *concs = SD.concs;
        int *mats = SD.mats;
        double *unionized_energy_array = SD.unionized_energy_array;
        int *index_grid = SD.index_grid;
        NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        unsigned long *verification = SD.verification;

        #pragma omp target data \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid], p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups]) \
            map(tofrom: verification[0:in.lookups])
        {
                if (mype == 0)
                        printf("Beginning optimized simulation...\n");

                sampling_kernel(in, p_energy_ptr, mat_ptr);
                for (int m = 0; m < 12; m++)
                        xs_lookup_kernel_filter_material(in, SD, p_energy_ptr, mat_ptr, m);
                #pragma omp target update from(SD.verification[0:in.lookups])
        }

        return std::accumulate(SD.verification, SD.verification + in.lookups, 0UL);
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

        if (mype == 0)
                printf("Simulation Kernel:\"%s\"\n", optimization_name);

        size_t total_sz = in.lookups * (sizeof(double) + sizeof(int));
        if (mype == 0)
                printf("Allocating an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

        std::vector<double> p_energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);
        double *p_energy_ptr = p_energy_samples.data();
        int *mat_ptr = mat_samples.data();
        int *num_nucs = SD.num_nucs;
        double *concs = SD.concs;
        int *mats = SD.mats;
        double *unionized_energy_array = SD.unionized_energy_array;
        int *index_grid = SD.index_grid;
        NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        unsigned long *verification = SD.verification;

        #pragma omp target data \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid], p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups]) \
            map(tofrom: verification[0:in.lookups])
        {
                if (mype == 0)
                        printf("Beginning optimized simulation...\n");

                sampling_kernel(in, p_energy_ptr, mat_ptr);
                xs_lookup_kernel_filter_fuel(in, SD, p_energy_ptr, mat_ptr, true);
                xs_lookup_kernel_filter_fuel(in, SD, p_energy_ptr, mat_ptr, false);
                #pragma omp target update from(SD.verification[0:in.lookups])
        }

        return std::accumulate(SD.verification, SD.verification + in.lookups, 0UL);
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";

        if (mype == 0)
                printf("Simulation Kernel:\"%s\"\n", optimization_name);

        size_t total_sz = in.lookups * (sizeof(double) + sizeof(int));
        if (mype == 0)
                printf("Allocating an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

        std::vector<double> p_energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);
        double *p_energy_ptr = p_energy_samples.data();
        int *mat_ptr = mat_samples.data();
        int *num_nucs = SD.num_nucs;
        double *concs = SD.concs;
        int *mats = SD.mats;
        double *unionized_energy_array = SD.unionized_energy_array;
        int *index_grid = SD.index_grid;
        NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        unsigned long *verification = SD.verification;

        #pragma omp target data \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid], p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups]) \
            map(tofrom: verification[0:in.lookups])
        {
                if (mype == 0)
                        printf("Beginning optimized simulation...\n");

                sampling_kernel(in, p_energy_ptr, mat_ptr);
                #pragma omp target update from(p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups])
                sort_samples_by_material(mat_ptr, p_energy_ptr, in.lookups);
                std::array<int, 12> n_lookups_per_material{};
                for (int i = 0; i < in.lookups; ++i)
                        ++n_lookups_per_material[mat_ptr[i]];

                std::array<int, 12> offsets{};
                int offset = 0;
                for (int m = 0; m < 12; ++m)
                {
                        offsets[m] = offset;
                        offset += n_lookups_per_material[m];
                }

                #pragma omp target update to(p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups])
                for (int m = 0; m < 12; ++m)
                {
                        int count = n_lookups_per_material[m];
                        if (count == 0)
                                continue;
                        xs_lookup_kernel_samples(in, SD, p_energy_ptr, mat_ptr, offsets[m], count);
                }
                #pragma omp target update from(SD.verification[0:in.lookups])
        }

        return std::accumulate(SD.verification, SD.verification + in.lookups, 0UL);
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

        if (mype == 0)
                printf("Simulation Kernel:\"%s\"\n", optimization_name);

        size_t total_sz = in.lookups * (sizeof(double) + sizeof(int));
        if (mype == 0)
                printf("Allocating an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

        std::vector<double> p_energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);
        double *p_energy_ptr = p_energy_samples.data();
        int *mat_ptr = mat_samples.data();
        int *num_nucs = SD.num_nucs;
        double *concs = SD.concs;
        int *mats = SD.mats;
        double *unionized_energy_array = SD.unionized_energy_array;
        int *index_grid = SD.index_grid;
        NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        unsigned long *verification = SD.verification;

        #pragma omp target data \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid], p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups]) \
            map(tofrom: verification[0:in.lookups])
        {
                if (mype == 0)
                        printf("Beginning optimized simulation...\n");

                sampling_kernel(in, p_energy_ptr, mat_ptr);
                #pragma omp target update from(p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups])
                int n_fuel_lookups = partition_samples_by_fuel(mat_ptr, p_energy_ptr, in.lookups);
                #pragma omp target update to(p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups])

                if (n_fuel_lookups > 0)
                        xs_lookup_kernel_samples(in, SD, p_energy_ptr, mat_ptr, 0, n_fuel_lookups);
                if (in.lookups - n_fuel_lookups > 0)
                        xs_lookup_kernel_samples(in, SD, p_energy_ptr, mat_ptr, n_fuel_lookups, in.lookups - n_fuel_lookups);
                #pragma omp target update from(SD.verification[0:in.lookups])
        }

        return std::accumulate(SD.verification, SD.verification + in.lookups, 0UL);
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

        if (mype == 0)
                printf("Simulation Kernel:\"%s\"\n", optimization_name);

        size_t total_sz = in.lookups * (sizeof(double) + sizeof(int));
        if (mype == 0)
                printf("Allocating an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

        std::vector<double> p_energy_samples(in.lookups);
        std::vector<int> mat_samples(in.lookups);
        double *p_energy_ptr = p_energy_samples.data();
        int *mat_ptr = mat_samples.data();
        int *num_nucs = SD.num_nucs;
        double *concs = SD.concs;
        int *mats = SD.mats;
        double *unionized_energy_array = SD.unionized_energy_array;
        int *index_grid = SD.index_grid;
        NuclideGridPoint *nuclide_grid = SD.nuclide_grid;
        unsigned long *verification = SD.verification;

        #pragma omp target data \
            map(to: num_nucs[0:SD.length_num_nucs], concs[0:SD.length_concs], mats[0:SD.length_mats], unionized_energy_array[0:SD.length_unionized_energy_array], index_grid[0:SD.length_index_grid], nuclide_grid[0:SD.length_nuclide_grid], p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups]) \
            map(tofrom: verification[0:in.lookups])
        {
                if (mype == 0)
                        printf("Beginning optimized simulation...\n");

                sampling_kernel(in, p_energy_ptr, mat_ptr);
                #pragma omp target update from(p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups])
                sort_samples_by_material(mat_ptr, p_energy_ptr, in.lookups);

                std::array<int, 12> n_lookups_per_material{};
                for (int i = 0; i < in.lookups; ++i)
                        ++n_lookups_per_material[mat_ptr[i]];

                std::array<int, 12> offsets{};
                int offset = 0;
                for (int m = 0; m < 12; ++m)
                {
                        offsets[m] = offset;
                        offset += n_lookups_per_material[m];
                }

                for (int m = 0; m < 12; ++m)
                        if (n_lookups_per_material[m] > 0)
                                sort_energy_within_range(mat_ptr, p_energy_ptr, offsets[m], n_lookups_per_material[m]);

                #pragma omp target update to(p_energy_ptr[0:in.lookups], mat_ptr[0:in.lookups])
                for (int m = 0; m < 12; ++m)
                {
                        int count = n_lookups_per_material[m];
                        if (count == 0)
                                continue;
                        xs_lookup_kernel_samples(in, SD, p_energy_ptr, mat_ptr, offsets[m], count);
                }
                #pragma omp target update from(SD.verification[0:in.lookups])
        }

        return std::accumulate(SD.verification, SD.verification + in.lookups, 0UL);
}
