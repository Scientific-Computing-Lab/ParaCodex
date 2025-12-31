#include "XSbench_header.h"

static SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD);
static void release_device_memory(SimulationData GSD);
static void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD);
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile);

unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        if (in.kernel_id != 0 && mype == 0)
                printf("Warning: kernel_id %d is not supported in the OpenMP offload build; using kernel 0.\n", in.kernel_id);
        return run_event_based_simulation_baseline(in, SD, mype, profile);
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        double start = get_time();
        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
        profile->host_to_device_time = get_time() - start;

        if (mype == 0)
                printf("Running baseline event-based simulation...\n");

        int nwarmups = in.num_warmups;
        start = 0.0;
        for (int i = 0; i < in.num_iterations + nwarmups; ++i) {
                if (i == nwarmups)
                        start = get_time();
                xs_lookup_kernel_baseline(in, GSD);
        }
        profile->kernel_time = get_time() - start;

        if (mype == 0)
                printf("Reducing verification results...\n");
        start = get_time();
        size_t verification_sz = (size_t) in.lookups * sizeof(unsigned long);
        if (verification_sz > 0)
                omp_target_memcpy(SD.verification, GSD.verification, verification_sz, 0, 0, omp_get_initial_device(), omp_get_default_device());
        profile->device_to_host_time = get_time() - start;

        unsigned long verification_scalar = 0;
        for (int i = 0; i < in.lookups; ++i)
                verification_scalar += SD.verification[i];

        release_device_memory(GSD);

        return verification_scalar;
}

static SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD)
{
        int device = omp_get_default_device();
        int host = omp_get_initial_device();
        SimulationData GSD = SD;
        size_t sz;
        size_t total_sz = 0;

        // num_nucs
        if (GSD.length_num_nucs > 0) {
                sz = GSD.length_num_nucs * sizeof(int);
                GSD.num_nucs = (int *) omp_target_alloc(sz, device);
                if (GSD.num_nucs == NULL) {
                        fprintf(stderr, "Failed to allocate num_nucs on device.\n");
                        exit(EXIT_FAILURE);
                }
                omp_target_memcpy(GSD.num_nucs, SD.num_nucs, sz, 0, 0, device, host);
                total_sz += sz;
        } else {
                GSD.num_nucs = NULL;
        }

        // concs
        if (GSD.length_concs > 0) {
                sz = GSD.length_concs * sizeof(double);
                GSD.concs = (double *) omp_target_alloc(sz, device);
                if (GSD.concs == NULL) {
                        fprintf(stderr, "Failed to allocate concs on device.\n");
                        exit(EXIT_FAILURE);
                }
                omp_target_memcpy(GSD.concs, SD.concs, sz, 0, 0, device, host);
                total_sz += sz;
        } else {
                GSD.concs = NULL;
        }

        // mats
        if (GSD.length_mats > 0) {
                sz = GSD.length_mats * sizeof(int);
                GSD.mats = (int *) omp_target_alloc(sz, device);
                if (GSD.mats == NULL) {
                        fprintf(stderr, "Failed to allocate mats on device.\n");
                        exit(EXIT_FAILURE);
                }
                omp_target_memcpy(GSD.mats, SD.mats, sz, 0, 0, device, host);
                total_sz += sz;
        } else {
                GSD.mats = NULL;
        }

        // unionized_energy_array (may be zero length)
        if (SD.length_unionized_energy_array > 0) {
                sz = SD.length_unionized_energy_array * sizeof(double);
                GSD.unionized_energy_array = (double *) omp_target_alloc(sz, device);
                if (GSD.unionized_energy_array == NULL) {
                        fprintf(stderr, "Failed to allocate unionized_energy_array on device.\n");
                        exit(EXIT_FAILURE);
                }
                omp_target_memcpy(GSD.unionized_energy_array, SD.unionized_energy_array, sz, 0, 0, device, host);
                total_sz += sz;
        } else {
                GSD.unionized_energy_array = NULL;
        }

        // index_grid (may be zero length)
        if (SD.length_index_grid > 0) {
                sz = SD.length_index_grid * sizeof(int);
                GSD.index_grid = (int *) omp_target_alloc(sz, device);
                if (GSD.index_grid == NULL) {
                        fprintf(stderr, "Failed to allocate index_grid on device.\n");
                        exit(EXIT_FAILURE);
                }
                omp_target_memcpy(GSD.index_grid, SD.index_grid, sz, 0, 0, device, host);
                total_sz += sz;
        } else {
                GSD.index_grid = NULL;
        }

        // nuclide_grid
        if (GSD.length_nuclide_grid > 0) {
                sz = GSD.length_nuclide_grid * sizeof(NuclideGridPoint);
                GSD.nuclide_grid = (NuclideGridPoint *) omp_target_alloc(sz, device);
                if (GSD.nuclide_grid == NULL) {
                        fprintf(stderr, "Failed to allocate nuclide_grid on device.\n");
                        exit(EXIT_FAILURE);
                }
                omp_target_memcpy(GSD.nuclide_grid, SD.nuclide_grid, sz, 0, 0, device, host);
                total_sz += sz;
        } else {
                GSD.nuclide_grid = NULL;
        }

        // verification buffer
        size_t verification_sz = (size_t) in.lookups * sizeof(unsigned long);
        if (verification_sz > 0) {
                GSD.verification = (unsigned long *) omp_target_alloc(verification_sz, device);
                if (GSD.verification == NULL) {
                        fprintf(stderr, "Failed to allocate verification buffer on device.\n");
                        exit(EXIT_FAILURE);
                }
                total_sz += verification_sz;
        } else {
                GSD.verification = NULL;
        }
        GSD.length_verification = in.lookups;

        if (mype == 0)
                printf("GPU Intialization complete. Allocated %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

        return GSD;
}

static void release_device_memory(SimulationData GSD)
{
        int device = omp_get_default_device();
        if (GSD.num_nucs)
                omp_target_free(GSD.num_nucs, device);
        if (GSD.concs)
                omp_target_free(GSD.concs, device);
        if (GSD.mats)
                omp_target_free(GSD.mats, device);
        if (GSD.unionized_energy_array)
                omp_target_free(GSD.unionized_energy_array, device);
        if (GSD.index_grid)
                omp_target_free(GSD.index_grid, device);
        if (GSD.nuclide_grid)
                omp_target_free(GSD.nuclide_grid, device);
        if (GSD.verification)
                omp_target_free(GSD.verification, device);
}

static void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD)
{
        int *d_num_nucs = GSD.num_nucs;
        double *d_concs = GSD.concs;
        int *d_mats = GSD.mats;
        double *d_unionized_energy_array = GSD.unionized_energy_array;
        int *d_index_grid = GSD.index_grid;
        NuclideGridPoint *d_nuclide_grid = GSD.nuclide_grid;
        unsigned long *d_verification = GSD.verification;

        #pragma omp target teams loop device(omp_get_default_device()) \
                is_device_ptr(d_num_nucs, d_concs, d_mats, d_unionized_energy_array, d_index_grid, d_nuclide_grid, d_verification)
        for (int i = 0; i < in.lookups; ++i) {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2 * i);

                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);

                double macro_xs_vector[5] = {0.0};

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
                for (int j = 0; j < 5; ++j) {
                        if (macro_xs_vector[j] > max) {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                d_verification[i] = max_idx + 1;
        }
}

#pragma omp declare target

void calculate_micro_xs( double p_energy, int nuc, long n_isotopes,
                          long n_gridpoints,
                          const double * __restrict__ egrid, const int * __restrict__ index_data,
                          const NuclideGridPoint * __restrict__ nuclide_grids,
                          long idx, double * __restrict__ xs_vector, int grid_type, int hash_bins )
{
        double f;
        const NuclideGridPoint * low;
        const NuclideGridPoint * high;

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

        for (int k = 0; k < 5; ++k)
                macro_xs_vector[k] = 0.0;

        if (grid_type == UNIONIZED)
                idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid);
        else if (grid_type == HASH)
        {
                double du = 1.0 / hash_bins;
                idx = p_energy / du;
        }

        for (int j = 0; j < num_nucs[mat]; ++j)
        {
                double xs_vector[5];
                p_nuc = mats[mat * max_num_nucs + j];
                conc = concs[mat * max_num_nucs + j];
                calculate_micro_xs(p_energy, p_nuc, n_isotopes,
                                   n_gridpoints, egrid, index_data,
                                   nuclide_grids, idx, xs_vector, grid_type, hash_bins);
                for (int k = 0; k < 5; ++k)
                        macro_xs_vector[k] += xs_vector[k] * conc;
        }
}

long grid_search(long n, double quarry, const double * A)
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

long grid_search_nuclide(long n, double quarry, const NuclideGridPoint * A, long low, long high)
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

int pick_mat(uint64_t * seed)
{
        const double dist[12] = {0.140, 0.052, 0.275, 0.134, 0.154, 0.064,
                                 0.066, 0.055, 0.008, 0.015, 0.025, 0.013};

        double roll = LCG_random_double(seed);

        for (int i = 0; i < 12; ++i)
        {
                double running = 0.0;
                for (int j = i; j > 0; --j)
                        running += dist[j];
                if (roll < running)
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
        return (double)(*seed) / (double) m;
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

#pragma omp end declare target
