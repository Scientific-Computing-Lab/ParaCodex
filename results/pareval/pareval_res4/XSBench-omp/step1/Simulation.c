#include "XSbench_header.h"

typedef struct {
        int mat;
        double energy;
} MatEnergy;

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile);

static int compare_by_material(const void *lhs, const void *rhs)
{
        const MatEnergy *a = lhs;
        const MatEnergy *b = rhs;
        return a->mat - b->mat;
}

static int compare_by_energy(const void *lhs, const void *rhs)
{
        const MatEnergy *a = lhs;
        const MatEnergy *b = rhs;
        if (a->energy < b->energy)
                return -1;
        if (a->energy > b->energy)
                return 1;
        return 0;
}

static void count_materials(const int *mat_samples, int lookups, int counts[12])
{
        for (int i = 0; i < 12; ++i)
                counts[i] = 0;
        for (int i = 0; i < lookups; ++i)
                ++counts[mat_samples[i]];
}

static void pack_samples(const SimulationData *SD, MatEnergy *buffer, int lookups)
{
        for (int i = 0; i < lookups; ++i) {
                buffer[i].mat = SD->mat_samples[i];
                buffer[i].energy = SD->p_energy_samples[i];
        }
}

static void unpack_samples(SimulationData *SD, const MatEnergy *buffer, int lookups)
{
        for (int i = 0; i < lookups; ++i) {
                SD->mat_samples[i] = buffer[i].mat;
                SD->p_energy_samples[i] = buffer[i].energy;
        }
}

static void partition_by_fuel(SimulationData *SD, int n_fuel, MatEnergy *buffer, int lookups)
{
        int fuel_idx = 0;
        int other_idx = n_fuel;
        for (int i = 0; i < lookups; ++i) {
                MatEnergy entry;
                entry.mat = SD->mat_samples[i];
                entry.energy = SD->p_energy_samples[i];
                if (entry.mat == 0)
                        buffer[fuel_idx++] = entry;
                else
                        buffer[other_idx++] = entry;
        }
        unpack_samples(SD, buffer, lookups);
}

static void allocate_sample_buffers(Inputs in, SimulationData *SD)
{
        SD->p_energy_samples = (double *) malloc(in.lookups * sizeof(double));
        assert(SD->p_energy_samples != NULL);
        SD->mat_samples = (int *) malloc(in.lookups * sizeof(int));
        assert(SD->mat_samples != NULL);
        SD->length_p_energy_samples = in.lookups;
        SD->length_mat_samples = in.lookups;
}

static void free_sample_buffers(SimulationData *SD)
{
        free(SD->p_energy_samples);
        SD->p_energy_samples = NULL;
        free(SD->mat_samples);
        SD->mat_samples = NULL;
        SD->length_p_energy_samples = 0;
        SD->length_mat_samples = 0;
}

static unsigned long long reduce_verification(unsigned long *verification, int lookups)
{
        unsigned long long sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < lookups; ++i)
                sum += verification[i];
        return sum;
}

#pragma omp declare target
static inline void evaluate_lookup(int idx, double p_energy, int mat, Inputs in,
                                   const int *num_nucs, const double *concs,
                                   const double *unionized_energy_array, const int *index_grid,
                                   const NuclideGridPoint *nuclide_grid, const int *mats,
                                   unsigned long *verification, int grid_type, int hash_bins, int max_num_nucs)
{
        double macro_xs_vector[5];
        calculate_macro_xs(p_energy,
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
                           grid_type,
                           hash_bins,
                           max_num_nucs);
        double max_val = -1.0;
        int max_idx = 0;
        for (int j = 0; j < 5; ++j) {
                if (macro_xs_vector[j] > max_val) {
                        max_val = macro_xs_vector[j];
                        max_idx = j;
                }
        }
        verification[idx] = max_idx + 1;
}

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

void xs_lookup_kernel_baseline(Inputs in, SimulationData *GSD)
{
        const int *num_nucs = GSD->num_nucs;
        const double *concs = GSD->concs;
        const double *unionized_energy_array = GSD->unionized_energy_array;
        const int *index_grid = GSD->index_grid;
        const NuclideGridPoint *nuclide_grid = GSD->nuclide_grid;
        const int *mats = GSD->mats;
        unsigned long *verification = GSD->verification;
        const int max_num_nucs = GSD->max_num_nucs;

        #pragma omp target teams loop
        for (int i = 0; i < in.lookups; ++i) {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                evaluate_lookup(i, p_energy, mat, in,
                                num_nucs, concs, unionized_energy_array, index_grid,
                                nuclide_grid, mats, verification, in.grid_type,
                                in.hash_bins, max_num_nucs);
        }
}

void sampling_kernel(Inputs in, SimulationData *GSD)
{
        double *p_energy_samples = GSD->p_energy_samples;
        int *mat_samples = GSD->mat_samples;

        #pragma omp target teams loop
        for (int i = 0; i < in.lookups; ++i) {
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2*i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                p_energy_samples[i] = p_energy;
                mat_samples[i] = mat;
        }
}

void xs_lookup_kernel_optimization_1(Inputs in, SimulationData *GSD)
{
        double *p_energy_samples = GSD->p_energy_samples;
        int *mat_samples = GSD->mat_samples;
        const int *num_nucs = GSD->num_nucs;
        const double *concs = GSD->concs;
        const double *unionized_energy_array = GSD->unionized_energy_array;
        const int *index_grid = GSD->index_grid;
        const NuclideGridPoint *nuclide_grid = GSD->nuclide_grid;
        const int *mats = GSD->mats;
        unsigned long *verification = GSD->verification;
        const int max_num_nucs = GSD->max_num_nucs;

        #pragma omp target teams loop
        for (int i = 0; i < in.lookups; ++i)
        {
                evaluate_lookup(i, p_energy_samples[i], mat_samples[i], in,
                                num_nucs, concs, unionized_energy_array, index_grid,
                                nuclide_grid, mats, verification, in.grid_type,
                                in.hash_bins, max_num_nucs);
        }
}

void xs_lookup_kernel_optimization_2(Inputs in, SimulationData *GSD, int m )
{
        double *p_energy_samples = GSD->p_energy_samples;
        int *mat_samples = GSD->mat_samples;
        const int *num_nucs = GSD->num_nucs;
        const double *concs = GSD->concs;
        const double *unionized_energy_array = GSD->unionized_energy_array;
        const int *index_grid = GSD->index_grid;
        const NuclideGridPoint *nuclide_grid = GSD->nuclide_grid;
        const int *mats = GSD->mats;
        unsigned long *verification = GSD->verification;
        const int max_num_nucs = GSD->max_num_nucs;

        #pragma omp target teams loop
        for (int i = 0; i < in.lookups; ++i)
        {
                if (mat_samples[i] != m)
                        continue;
                evaluate_lookup(i, p_energy_samples[i], m, in,
                                num_nucs, concs, unionized_energy_array, index_grid,
                                nuclide_grid, mats, verification, in.grid_type,
                                in.hash_bins, max_num_nucs);
        }
}

void xs_lookup_kernel_optimization_3(Inputs in, SimulationData *GSD, int is_fuel )
{
        double *p_energy_samples = GSD->p_energy_samples;
        int *mat_samples = GSD->mat_samples;
        const int *num_nucs = GSD->num_nucs;
        const double *concs = GSD->concs;
        const double *unionized_energy_array = GSD->unionized_energy_array;
        const int *index_grid = GSD->index_grid;
        const NuclideGridPoint *nuclide_grid = GSD->nuclide_grid;
        const int *mats = GSD->mats;
        unsigned long *verification = GSD->verification;
        const int max_num_nucs = GSD->max_num_nucs;

        #pragma omp target teams loop
        for (int i = 0; i < in.lookups; ++i)
        {
                int mat = mat_samples[i];
                if ((is_fuel == 1 && mat == 0) || (is_fuel == 0 && mat != 0)) {
                        evaluate_lookup(i, p_energy_samples[i], mat, in,
                                        num_nucs, concs, unionized_energy_array, index_grid,
                                        nuclide_grid, mats, verification, in.grid_type,
                                        in.hash_bins, max_num_nucs);
                }
        }
}

void xs_lookup_kernel_optimization_4(Inputs in, SimulationData *GSD, int m, int n_lookups, int offset )
{
        double *p_energy_samples = GSD->p_energy_samples;
        int *mat_samples = GSD->mat_samples;
        const int *num_nucs = GSD->num_nucs;
        const double *concs = GSD->concs;
        const double *unionized_energy_array = GSD->unionized_energy_array;
        const int *index_grid = GSD->index_grid;
        const NuclideGridPoint *nuclide_grid = GSD->nuclide_grid;
        const int *mats = GSD->mats;
        unsigned long *verification = GSD->verification;
        const int max_num_nucs = GSD->max_num_nucs;

        #pragma omp target teams loop
        for (int i = 0; i < n_lookups; ++i)
        {
                int idx = offset + i;
                if (mat_samples[idx] != m)
                        continue;
                evaluate_lookup(idx, p_energy_samples[idx], m, in,
                                num_nucs, concs, unionized_energy_array, index_grid,
                                nuclide_grid, mats, verification, in.grid_type,
                                in.hash_bins, max_num_nucs);
        }
}

void xs_lookup_kernel_optimization_5(Inputs in, SimulationData *GSD, int n_lookups, int offset )
{
        double *p_energy_samples = GSD->p_energy_samples;
        int *mat_samples = GSD->mat_samples;
        const int *num_nucs = GSD->num_nucs;
        const double *concs = GSD->concs;
        const double *unionized_energy_array = GSD->unionized_energy_array;
        const int *index_grid = GSD->index_grid;
        const NuclideGridPoint *nuclide_grid = GSD->nuclide_grid;
        const int *mats = GSD->mats;
        unsigned long *verification = GSD->verification;
        const int max_num_nucs = GSD->max_num_nucs;

        #pragma omp target teams loop
        for (int i = 0; i < n_lookups; ++i)
        {
                int idx = offset + i;
                evaluate_lookup(idx, p_energy_samples[idx], mat_samples[idx], in,
                                num_nucs, concs, unionized_energy_array, index_grid,
                                nuclide_grid, mats, verification, in.grid_type,
                                in.hash_bins, max_num_nucs);
        }
}

unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        return run_event_based_simulation_baseline(in, SD, mype, profile);
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile)
{
        double start = get_time();
        if (mype == 0)
                printf("Running baseline event-based simulation...\n");

        int lookups = in.lookups;
        int len_num_nucs = SD.length_num_nucs;
        int len_concs = SD.length_concs;
        int len_mats = SD.length_mats;
        int len_unionized = SD.length_unionized_energy_array;
        long len_index = SD.length_index_grid;
        int len_nuclide = SD.length_nuclide_grid;

        #pragma omp target data \
            map(to: SD.num_nucs[0:len_num_nucs], SD.concs[0:len_concs], SD.mats[0:len_mats], \
                SD.unionized_energy_array[0:len_unionized], SD.index_grid[0:len_index], \
                SD.nuclide_grid[0:len_nuclide]) \
            map(tofrom: SD.verification[0:lookups])
        {
                profile->host_to_device_time = get_time() - start;

                int total_iterations = in.num_iterations + in.num_warmups;
                double kernel_start = 0.0;
                for (int iter = 0; iter < total_iterations; ++iter) {
                        if (iter == in.num_warmups)
                                kernel_start = get_time();
                        xs_lookup_kernel_baseline(in, &SD);
                }
                profile->kernel_time = get_time() - kernel_start;

                double device_to_host_start = get_time();
                #pragma omp target update from(SD.verification[0:lookups])
                profile->device_to_host_time = get_time() - device_to_host_start;
        }

        unsigned long long verification_scalar = reduce_verification(SD.verification, lookups);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype)
{
        const int lookups = in.lookups;
        if (mype == 0)
                printf("Simulation Kernel:\"Optimization 1 - basic sample/lookup kernel splitting\"\n");

        allocate_sample_buffers(in, &GSD);

        int len_num_nucs = GSD.length_num_nucs;
        int len_concs = GSD.length_concs;
        int len_mats = GSD.length_mats;
        int len_unionized = GSD.length_unionized_energy_array;
        long len_index = GSD.length_index_grid;
        int len_nuclide = GSD.length_nuclide_grid;

        #pragma omp target data \
            map(to: GSD.num_nucs[0:len_num_nucs], GSD.concs[0:len_concs], GSD.mats[0:len_mats], \
                GSD.unionized_energy_array[0:len_unionized], GSD.index_grid[0:len_index], \
                GSD.nuclide_grid[0:len_nuclide]) \
            map(tofrom: GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups], GSD.verification[0:lookups])
        {
                sampling_kernel(in, &GSD);
                xs_lookup_kernel_optimization_1(in, &GSD);
                #pragma omp target update from(GSD.verification[0:lookups])
        }

        unsigned long long verification_scalar = reduce_verification(GSD.verification, lookups);
        free_sample_buffers(&GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype)
{
        const int lookups = in.lookups;
        if (mype == 0)
                printf("Simulation Kernel:\"Optimization 2 - Material Lookup Kernels\"\n");

        allocate_sample_buffers(in, &GSD);

        int len_num_nucs = GSD.length_num_nucs;
        int len_concs = GSD.length_concs;
        int len_mats = GSD.length_mats;
        int len_unionized = GSD.length_unionized_energy_array;
        long len_index = GSD.length_index_grid;
        int len_nuclide = GSD.length_nuclide_grid;

        #pragma omp target data \
            map(to: GSD.num_nucs[0:len_num_nucs], GSD.concs[0:len_concs], GSD.mats[0:len_mats], \
                GSD.unionized_energy_array[0:len_unionized], GSD.index_grid[0:len_index], \
                GSD.nuclide_grid[0:len_nuclide]) \
            map(tofrom: GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups], GSD.verification[0:lookups])
        {
                sampling_kernel(in, &GSD);
                for (int m = 0; m < 12; ++m)
                        xs_lookup_kernel_optimization_2(in, &GSD, m);
                #pragma omp target update from(GSD.verification[0:lookups])
        }

        unsigned long long verification_scalar = reduce_verification(GSD.verification, lookups);
        free_sample_buffers(&GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype)
{
        const int lookups = in.lookups;
        if (mype == 0)
                printf("Simulation Kernel:\"Optimization 3 - Fuel or Other Lookup Kernels\"\n");

        allocate_sample_buffers(in, &GSD);

        int len_num_nucs = GSD.length_num_nucs;
        int len_concs = GSD.length_concs;
        int len_mats = GSD.length_mats;
        int len_unionized = GSD.length_unionized_energy_array;
        long len_index = GSD.length_index_grid;
        int len_nuclide = GSD.length_nuclide_grid;

        #pragma omp target data \
            map(to: GSD.num_nucs[0:len_num_nucs], GSD.concs[0:len_concs], GSD.mats[0:len_mats], \
                GSD.unionized_energy_array[0:len_unionized], GSD.index_grid[0:len_index], \
                GSD.nuclide_grid[0:len_nuclide]) \
            map(tofrom: GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups], GSD.verification[0:lookups])
        {
                sampling_kernel(in, &GSD);
                xs_lookup_kernel_optimization_3(in, &GSD, 0);
                xs_lookup_kernel_optimization_3(in, &GSD, 1);
                #pragma omp target update from(GSD.verification[0:lookups])
        }

        unsigned long long verification_scalar = reduce_verification(GSD.verification, lookups);
        free_sample_buffers(&GSD);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype)
{
        const int lookups = in.lookups;
        if (mype == 0)
                printf("Simulation Kernel:\"Optimization 4 - All Material Lookup Kernels + Material Sort\"\n");

        allocate_sample_buffers(in, &GSD);
        MatEnergy *buffer = (MatEnergy *) malloc(lookups * sizeof(MatEnergy));
        assert(buffer != NULL);

        int len_num_nucs = GSD.length_num_nucs;
        int len_concs = GSD.length_concs;
        int len_mats = GSD.length_mats;
        int len_unionized = GSD.length_unionized_energy_array;
        long len_index = GSD.length_index_grid;
        int len_nuclide = GSD.length_nuclide_grid;
        int n_lookups_per_material[12];

        #pragma omp target data \
            map(to: GSD.num_nucs[0:len_num_nucs], GSD.concs[0:len_concs], GSD.mats[0:len_mats], \
                GSD.unionized_energy_array[0:len_unionized], GSD.index_grid[0:len_index], \
                GSD.nuclide_grid[0:len_nuclide]) \
            map(tofrom: GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups], GSD.verification[0:lookups])
        {
                sampling_kernel(in, &GSD);
                #pragma omp target update from(GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups])
                count_materials(GSD.mat_samples, lookups, n_lookups_per_material);
                pack_samples(&GSD, buffer, lookups);
                qsort(buffer, lookups, sizeof(MatEnergy), compare_by_material);
                unpack_samples(&GSD, buffer, lookups);
                #pragma omp target update to(GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups])

                int offset = 0;
                for (int m = 0; m < 12; ++m) {
                        int n_slice = n_lookups_per_material[m];
                        if (n_slice == 0)
                                continue;
                        xs_lookup_kernel_optimization_4(in, &GSD, m, n_slice, offset);
                        offset += n_slice;
                }
                #pragma omp target update from(GSD.verification[0:lookups])
        }

        unsigned long long verification_scalar = reduce_verification(GSD.verification, lookups);
        free_sample_buffers(&GSD);
        free(buffer);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype)
{
        const int lookups = in.lookups;
        if (mype == 0)
                printf("Simulation Kernel:\"Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort\"\n");

        allocate_sample_buffers(in, &GSD);
        MatEnergy *buffer = (MatEnergy *) malloc(lookups * sizeof(MatEnergy));
        assert(buffer != NULL);

        int len_num_nucs = GSD.length_num_nucs;
        int len_concs = GSD.length_concs;
        int len_mats = GSD.length_mats;
        int len_unionized = GSD.length_unionized_energy_array;
        long len_index = GSD.length_index_grid;
        int len_nuclide = GSD.length_nuclide_grid;
        int n_fuel_lookups = 0;

        #pragma omp target data \
            map(to: GSD.num_nucs[0:len_num_nucs], GSD.concs[0:len_concs], GSD.mats[0:len_mats], \
                GSD.unionized_energy_array[0:len_unionized], GSD.index_grid[0:len_index], \
                GSD.nuclide_grid[0:len_nuclide]) \
            map(tofrom: GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups], GSD.verification[0:lookups])
        {
                sampling_kernel(in, &GSD);
                #pragma omp target update from(GSD.mat_samples[0:lookups], GSD.p_energy_samples[0:lookups])
                for (int i = 0; i < lookups; ++i)
                        if (GSD.mat_samples[i] == 0)
                                ++n_fuel_lookups;
                partition_by_fuel(&GSD, n_fuel_lookups, buffer, lookups);
                #pragma omp target update to(GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups])

                if (n_fuel_lookups > 0)
                        xs_lookup_kernel_optimization_5(in, &GSD, n_fuel_lookups, 0);
                if (n_fuel_lookups < lookups)
                        xs_lookup_kernel_optimization_5(in, &GSD, lookups - n_fuel_lookups, n_fuel_lookups);
                #pragma omp target update from(GSD.verification[0:lookups])
        }

        unsigned long long verification_scalar = reduce_verification(GSD.verification, lookups);
        free_sample_buffers(&GSD);
        free(buffer);
        return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype)
{
        const int lookups = in.lookups;
        if (mype == 0)
                printf("Simulation Kernel:\"Optimization 6 - Material & Energy Sorts + Material-specific Kernels\"\n");

        allocate_sample_buffers(in, &GSD);
        MatEnergy *buffer = (MatEnergy *) malloc(lookups * sizeof(MatEnergy));
        assert(buffer != NULL);

        int len_num_nucs = GSD.length_num_nucs;
        int len_concs = GSD.length_concs;
        int len_mats = GSD.length_mats;
        int len_unionized = GSD.length_unionized_energy_array;
        long len_index = GSD.length_index_grid;
        int len_nuclide = GSD.length_nuclide_grid;
        int n_lookups_per_material[12];

        #pragma omp target data \
            map(to: GSD.num_nucs[0:len_num_nucs], GSD.concs[0:len_concs], GSD.mats[0:len_mats], \
                GSD.unionized_energy_array[0:len_unionized], GSD.index_grid[0:len_index], \
                GSD.nuclide_grid[0:len_nuclide]) \
            map(tofrom: GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups], GSD.verification[0:lookups])
        {
                sampling_kernel(in, &GSD);
                #pragma omp target update from(GSD.mat_samples[0:lookups], GSD.p_energy_samples[0:lookups])
                count_materials(GSD.mat_samples, lookups, n_lookups_per_material);
                pack_samples(&GSD, buffer, lookups);
                qsort(buffer, lookups, sizeof(MatEnergy), compare_by_material);

                int offset = 0;
                for (int m = 0; m < 12; ++m) {
                        int count = n_lookups_per_material[m];
                        if (count == 0)
                                continue;
                        qsort(buffer + offset, count, sizeof(MatEnergy), compare_by_energy);
                        offset += count;
                }
                unpack_samples(&GSD, buffer, lookups);
                #pragma omp target update to(GSD.p_energy_samples[0:lookups], GSD.mat_samples[0:lookups])

                int offset_kernel = 0;
                for (int m = 0; m < 12; ++m) {
                        int count = n_lookups_per_material[m];
                        if (count == 0)
                                continue;
                        xs_lookup_kernel_optimization_4(in, &GSD, m, count, offset_kernel);
                        offset_kernel += count;
                }
                #pragma omp target update from(GSD.verification[0:lookups])
        }

        unsigned long long verification_scalar = reduce_verification(GSD.verification, lookups);
        free_sample_buffers(&GSD);
        free(buffer);
        return verification_scalar;
}
