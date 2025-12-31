#include "XSbench_header.cuh"

#include <algorithm>
#include <cassert>
#include <omp.h>
#include <utility>
#include <vector>

constexpr int NUM_MATERIALS = 12;

#define BASE_DATA_MAP \
        map(to: GSD.num_nucs[0:GSD.length_num_nucs], \
             GSD.concs[0:GSD.length_concs], \
             GSD.unionized_energy_array[0:GSD.length_unionized_energy_array], \
             GSD.index_grid[0:GSD.length_index_grid], \
             GSD.nuclide_grid[0:GSD.length_nuclide_grid], \
             GSD.mats[0:GSD.length_mats])

#pragma omp declare target

long grid_search(long n, double quarry, double * __restrict__ A)
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

long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high)
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
        double dist[NUM_MATERIALS] = {0.140, 0.052, 0.275, 0.134, 0.154, 0.064, 0.066, 0.055, 0.008, 0.015, 0.025, 0.013};

        double roll = LCG_random_double(seed);

        for (int i = 0; i < NUM_MATERIALS; i++)
        {
                double running = 0;
                for (int j = i; j > 0; j--)
                        running += dist[j];
                if (roll < running)
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

void calculate_micro_xs(double p_energy, int nuc, long n_isotopes,
                        long n_gridpoints,
                        double * __restrict__ egrid, int * __restrict__ index_data,
                        NuclideGridPoint * __restrict__ nuclide_grids,
                        long idx, double * __restrict__ xs_vector, int grid_type, int hash_bins)
{
        double f;
        NuclideGridPoint *low, *high;

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

                double e_low = nuclide_grids[nuc * n_gridpoints + u_low].energy;
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

void calculate_macro_xs(double p_energy, int mat, long n_isotopes,
                        long n_gridpoints, int * __restrict__ num_nucs,
                        double * __restrict__ concs,
                        double * __restrict__ egrid, int * __restrict__ index_data,
                        NuclideGridPoint * __restrict__ nuclide_grids,
                        int * __restrict__ mats,
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
                calculate_micro_xs(p_energy, p_nuc, n_isotopes,
                                   n_gridpoints, egrid, index_data,
                                   nuclide_grids, idx, xs_vector, grid_type, hash_bins);
                for (int k = 0; k < 5; k++)
                        macro_xs_vector[k] += xs_vector[k] * conc;
        }
}

static inline void accumulate_lookup(Inputs in,
                                     int idx,
                                     double p_energy,
                                     int mat,
                                     int *num_nucs,
                                     double *concs,
                                     double *unionized_array,
                                     int *index_grid,
                                     NuclideGridPoint *nuclide_grid,
                                     int *mats,
                                     unsigned long *verification,
                                     int max_num_nucs)
{
        double macro_xs_vector[5];
        calculate_macro_xs(p_energy,
                            mat,
                            in.n_isotopes,
                            in.n_gridpoints,
                            num_nucs,
                            concs,
                            unionized_array,
                            index_grid,
                            nuclide_grid,
                            mats,
                            macro_xs_vector,
                            in.grid_type,
                            in.hash_bins,
                            max_num_nucs);

        double max = -1.0;
        int max_idx = 0;
        for (int j = 0; j < 5; j++)
        {
                if (macro_xs_vector[j] > max)
                {
                        max = macro_xs_vector[j];
                        max_idx = j;
                }
        }
        verification[idx] = max_idx + 1;
}

#pragma omp end declare target

static void compute_material_counts(const int *mat_samples, int lookups, int counts[NUM_MATERIALS])
{
        for (int i = 0; i < NUM_MATERIALS; ++i)
                counts[i] = 0;

        for (int i = 0; i < lookups; ++i)
        {
                int mat = mat_samples[i];
                if (mat >= 0 && mat < NUM_MATERIALS)
                        ++counts[mat];
        }
}

static void sort_samples_by_material(int *mat_samples, double *p_energy_samples, int lookups)
{
        if (lookups <= 1)
                return;

        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(lookups);
        for (int i = 0; i < lookups; ++i)
                pairs.emplace_back(mat_samples[i], p_energy_samples[i]);

        std::stable_sort(pairs.begin(), pairs.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                return a.first < b.first;
        });

        for (int i = 0; i < lookups; ++i)
        {
                mat_samples[i] = pairs[i].first;
                p_energy_samples[i] = pairs[i].second;
        }
}

static void sort_samples_by_energy(int *mat_samples, double *p_energy_samples, int offset, int count)
{
        if (count <= 1)
                return;

        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(count);
        for (int i = 0; i < count; ++i)
                pairs.emplace_back(mat_samples[offset + i], p_energy_samples[offset + i]);

        std::sort(pairs.begin(), pairs.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                return a.second < b.second;
        });

        for (int i = 0; i < count; ++i)
        {
                mat_samples[offset + i] = pairs[i].first;
                p_energy_samples[offset + i] = pairs[i].second;
        }
}

static void partition_samples_by_fuel(int *mat_samples, double *p_energy_samples, int lookups, int fuel_lookups)
{
        if (lookups <= 1)
                return;

        std::vector<int> mat_temp(lookups);
        std::vector<double> energy_temp(lookups);
        int fuel_idx = 0;
        int other_idx = fuel_lookups;

        for (int i = 0; i < lookups; ++i)
        {
                if (mat_samples[i] == 0)
                {
                        mat_temp[fuel_idx] = mat_samples[i];
                        energy_temp[fuel_idx++] = p_energy_samples[i];
                }
                else
                {
                        mat_temp[other_idx] = mat_samples[i];
                        energy_temp[other_idx++] = p_energy_samples[i];
                }
        }

        for (int i = 0; i < lookups; ++i)
        {
                mat_samples[i] = mat_temp[i];
                p_energy_samples[i] = energy_temp[i];
        }
}

static void launch_baseline_lookup(Inputs in, SimulationData &GSD)
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(num_nucs, concs, unionized_array, index_grid, nuclide_grid, mats, verification)
        for (int i = 0; i < in.lookups; ++i)
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2 * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                accumulate_lookup(in,
                                  i,
                                  p_energy,
                                  mat,
                                  num_nucs,
                                  concs,
                                  unionized_array,
                                  index_grid,
                                  nuclide_grid,
                                  mats,
                                  verification,
                                  max_num_nucs);
        }
}

static void launch_sampling_kernel(Inputs in, SimulationData &GSD)
{
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;

#pragma omp target teams loop is_device_ptr(p_energy_samples, mat_samples)
        for (int i = 0; i < in.lookups; ++i)
        {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2 * i);
                p_energy_samples[i] = LCG_random_double(&seed);
                mat_samples[i] = pick_mat(&seed);
        }
}

static void launch_lookup_with_samples(Inputs in, SimulationData &GSD)
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(num_nucs, concs, unionized_array, index_grid, nuclide_grid, mats, verification, p_energy_samples, mat_samples)
        for (int i = 0; i < in.lookups; ++i)
        {
                accumulate_lookup(in,
                                  i,
                                  p_energy_samples[i],
                                  mat_samples[i],
                                  num_nucs,
                                  concs,
                                  unionized_array,
                                  index_grid,
                                  nuclide_grid,
                                  mats,
                                  verification,
                                  max_num_nucs);
        }
}

static void launch_lookup_filtered_by_material(Inputs in, SimulationData &GSD, int material)
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(num_nucs, concs, unionized_array, index_grid, nuclide_grid, mats, verification, p_energy_samples, mat_samples)
        for (int i = 0; i < in.lookups; ++i)
        {
                int mat = mat_samples[i];
                if (mat != material)
                        continue;
                accumulate_lookup(in,
                                  i,
                                  p_energy_samples[i],
                                  mat,
                                  num_nucs,
                                  concs,
                                  unionized_array,
                                  index_grid,
                                  nuclide_grid,
                                  mats,
                                  verification,
                                  max_num_nucs);
        }
}

static void launch_lookup_partitioned(Inputs in, SimulationData &GSD, bool include_fuel)
{
        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        double *p_energy_samples = GSD.p_energy_samples;
        int *mat_samples = GSD.mat_samples;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(num_nucs, concs, unionized_array, index_grid, nuclide_grid, mats, verification, p_energy_samples, mat_samples)
        for (int i = 0; i < in.lookups; ++i)
        {
                int mat = mat_samples[i];
                bool is_fuel = (mat == 0);
                if (include_fuel != is_fuel)
                        continue;
                accumulate_lookup(in,
                                  i,
                                  p_energy_samples[i],
                                  mat,
                                  num_nucs,
                                  concs,
                                  unionized_array,
                                  index_grid,
                                  nuclide_grid,
                                  mats,
                                  verification,
                                  max_num_nucs);
        }
}

static void launch_lookup_segment(Inputs in, SimulationData &GSD, int base_index, int length)
{
        if (length <= 0)
                return;

        int *num_nucs = GSD.num_nucs;
        double *concs = GSD.concs;
        double *unionized_array = GSD.unionized_energy_array;
        int *index_grid = GSD.index_grid;
        NuclideGridPoint *nuclide_grid = GSD.nuclide_grid;
        int *mats = GSD.mats;
        unsigned long *verification = GSD.verification;
        double *p_energy_samples = GSD.p_energy_samples + base_index;
        int *mat_samples = GSD.mat_samples + base_index;
        int max_num_nucs = GSD.max_num_nucs;

#pragma omp target teams loop is_device_ptr(num_nucs, concs, unionized_array, index_grid, nuclide_grid, mats, verification, p_energy_samples, mat_samples)
        for (int local_idx = 0; local_idx < length; ++local_idx)
        {
                int global_idx = base_index + local_idx;
                int mat = mat_samples[local_idx];
                double p_energy = p_energy_samples[local_idx];
                accumulate_lookup(in,
                                  global_idx,
                                  p_energy,
                                  mat,
                                  num_nucs,
                                  concs,
                                  unionized_array,
                                  index_grid,
                                  nuclide_grid,
                                  mats,
                                  verification,
                                  max_num_nucs);
        }
}

static unsigned long long reduce_verification(unsigned long *verification, int lookups)
{
        unsigned long long verification_scalar = 0;
        for (int i = 0; i < lookups; ++i)
                verification_scalar += verification[i];
        return verification_scalar;
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile *profile)
{
        SimulationData GSD = SD;
        if (mype == 0)
                printf("Running baseline event-based simulation...\n");

        int nwarmups = in.num_warmups;
        double kernel_start = 0.0;

        profile->host_to_device_time = 0.0;

#pragma omp target data BASE_DATA_MAP \
        map(alloc: GSD.verification[0:in.lookups])
        {
                for (int i = 0; i < in.num_iterations + nwarmups; i++)
                {
                        if (i == nwarmups)
                                kernel_start = get_time();
                        launch_baseline_lookup(in, GSD);
                }
                profile->kernel_time = (kernel_start == 0.0) ? 0.0 : get_time() - kernel_start;
                double copy_start = get_time();
                #pragma omp target update from(GSD.verification[0:in.lookups])
                profile->device_to_host_time = get_time() - copy_start;
        }

        if (mype == 0)
                printf("Reducing verification results...\n");

        unsigned long long verification_scalar = reduce_verification(GSD.verification, in.lookups);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if (mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = SD;
        size_t total_sz = 0;
        size_t sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *)malloc(sz);
        assert(GSD.p_energy_samples != NULL);
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *)malloc(sz);
        assert(GSD.mat_samples != NULL);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
        if (mype == 0) printf("Beginning optimized simulation...\n");

#pragma omp target data BASE_DATA_MAP \
        map(tofrom: GSD.p_energy_samples[0:in.lookups], GSD.mat_samples[0:in.lookups]) \
        map(alloc: GSD.verification[0:in.lookups])
        {
                launch_sampling_kernel(in, GSD);
                launch_lookup_with_samples(in, GSD);
                #pragma omp target update from(GSD.verification[0:in.lookups])
        }

        if (mype == 0) printf("Reducing verification results...\n");
        unsigned long long verification_scalar = reduce_verification(GSD.verification, in.lookups);
        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 2 - Material Lookup Kernels";

        if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if (mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = SD;
        size_t total_sz = 0;
        size_t sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *)malloc(sz);
        assert(GSD.p_energy_samples != NULL);
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *)malloc(sz);
        assert(GSD.mat_samples != NULL);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
        if (mype == 0) printf("Beginning optimized simulation...\n");

#pragma omp target data BASE_DATA_MAP \
        map(tofrom: GSD.p_energy_samples[0:in.lookups], GSD.mat_samples[0:in.lookups]) \
        map(alloc: GSD.verification[0:in.lookups])
        {
                launch_sampling_kernel(in, GSD);
                for (int m = 0; m < NUM_MATERIALS; ++m)
                        launch_lookup_filtered_by_material(in, GSD, m);
                #pragma omp target update from(GSD.verification[0:in.lookups])
        }

        if (mype == 0) printf("Reducing verification results...\n");
        unsigned long long verification_scalar = reduce_verification(GSD.verification, in.lookups);
        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

        if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if (mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = SD;
        size_t total_sz = 0;
        size_t sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *)malloc(sz);
        assert(GSD.p_energy_samples != NULL);
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *)malloc(sz);
        assert(GSD.mat_samples != NULL);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
        if (mype == 0) printf("Beginning optimized simulation...\n");

#pragma omp target data BASE_DATA_MAP \
        map(tofrom: GSD.p_energy_samples[0:in.lookups], GSD.mat_samples[0:in.lookups]) \
        map(alloc: GSD.verification[0:in.lookups])
        {
                launch_sampling_kernel(in, GSD);
                launch_lookup_partitioned(in, GSD, false);
                launch_lookup_partitioned(in, GSD, true);
                #pragma omp target update from(GSD.verification[0:in.lookups])
        }

        if (mype == 0) printf("Reducing verification results...\n");
        unsigned long long verification_scalar = reduce_verification(GSD.verification, in.lookups);
        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";

        if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if (mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = SD;
        size_t total_sz = 0;
        size_t sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *)malloc(sz);
        assert(GSD.p_energy_samples != NULL);
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *)malloc(sz);
        assert(GSD.mat_samples != NULL);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
        if (mype == 0) printf("Beginning optimized simulation...\n");

#pragma omp target data BASE_DATA_MAP \
        map(tofrom: GSD.p_energy_samples[0:in.lookups], GSD.mat_samples[0:in.lookups]) \
        map(alloc: GSD.verification[0:in.lookups])
        {
                launch_sampling_kernel(in, GSD);
                #pragma omp target update from(GSD.mat_samples[0:in.lookups], GSD.p_energy_samples[0:in.lookups])
                int n_lookups_per_material[NUM_MATERIALS];
                compute_material_counts(GSD.mat_samples, in.lookups, n_lookups_per_material);
                sort_samples_by_material(GSD.mat_samples, GSD.p_energy_samples, in.lookups);
                #pragma omp target update to(GSD.mat_samples[0:in.lookups], GSD.p_energy_samples[0:in.lookups])
                int offset = 0;
                for (int m = 0; m < NUM_MATERIALS; ++m)
                {
                        int n = n_lookups_per_material[m];
                        launch_lookup_segment(in, GSD, offset, n);
                        offset += n;
                }
                #pragma omp target update from(GSD.verification[0:in.lookups])
        }

        if (mype == 0) printf("Reducing verification results...\n");
        unsigned long long verification_scalar = reduce_verification(GSD.verification, in.lookups);
        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

        if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if (mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = SD;
        size_t total_sz = 0;
        size_t sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *)malloc(sz);
        assert(GSD.p_energy_samples != NULL);
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *)malloc(sz);
        assert(GSD.mat_samples != NULL);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
        if (mype == 0) printf("Beginning optimized simulation...\n");

#pragma omp target data BASE_DATA_MAP \
        map(tofrom: GSD.p_energy_samples[0:in.lookups], GSD.mat_samples[0:in.lookups]) \
        map(alloc: GSD.verification[0:in.lookups])
        {
                launch_sampling_kernel(in, GSD);
                #pragma omp target update from(GSD.mat_samples[0:in.lookups], GSD.p_energy_samples[0:in.lookups])
                int n_lookups_per_material[NUM_MATERIALS];
                compute_material_counts(GSD.mat_samples, in.lookups, n_lookups_per_material);
                int n_fuel_lookups = n_lookups_per_material[0];
                partition_samples_by_fuel(GSD.mat_samples, GSD.p_energy_samples, in.lookups, n_fuel_lookups);
                #pragma omp target update to(GSD.mat_samples[0:in.lookups], GSD.p_energy_samples[0:in.lookups])
                launch_lookup_segment(in, GSD, 0, n_fuel_lookups);
                launch_lookup_segment(in, GSD, n_fuel_lookups, in.lookups - n_fuel_lookups);
                #pragma omp target update from(GSD.verification[0:in.lookups])
        }

        if (mype == 0) printf("Reducing verification results...\n");
        unsigned long long verification_scalar = reduce_verification(GSD.verification, in.lookups);
        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData SD, int mype)
{
        const char *optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

        if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);
        if (mype == 0) printf("Allocating additional device data required by kernel...\n");

        SimulationData GSD = SD;
        size_t total_sz = 0;
        size_t sz = in.lookups * sizeof(double);
        GSD.p_energy_samples = (double *)malloc(sz);
        assert(GSD.p_energy_samples != NULL);
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        GSD.mat_samples = (int *)malloc(sz);
        assert(GSD.mat_samples != NULL);
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
        if (mype == 0) printf("Beginning optimized simulation...\n");

#pragma omp target data BASE_DATA_MAP \
        map(tofrom: GSD.p_energy_samples[0:in.lookups], GSD.mat_samples[0:in.lookups]) \
        map(alloc: GSD.verification[0:in.lookups])
        {
                launch_sampling_kernel(in, GSD);
                #pragma omp target update from(GSD.mat_samples[0:in.lookups], GSD.p_energy_samples[0:in.lookups])
                int n_lookups_per_material[NUM_MATERIALS];
                compute_material_counts(GSD.mat_samples, in.lookups, n_lookups_per_material);
                sort_samples_by_material(GSD.mat_samples, GSD.p_energy_samples, in.lookups);
                int offset = 0;
                for (int m = 0; m < NUM_MATERIALS; ++m)
                {
                        int count = n_lookups_per_material[m];
                        sort_samples_by_energy(GSD.mat_samples, GSD.p_energy_samples, offset, count);
                        offset += count;
                }
                #pragma omp target update to(GSD.mat_samples[0:in.lookups], GSD.p_energy_samples[0:in.lookups])
                offset = 0;
                for (int m = 0; m < NUM_MATERIALS; ++m)
                {
                        int count = n_lookups_per_material[m];
                        launch_lookup_segment(in, GSD, offset, count);
                        offset += count;
                }
                #pragma omp target update from(GSD.verification[0:in.lookups])
        }

        if (mype == 0) printf("Reducing verification results...\n");
        unsigned long long verification_scalar = reduce_verification(GSD.verification, in.lookups);
        free(GSD.p_energy_samples);
        free(GSD.mat_samples);
        return verification_scalar;
}
