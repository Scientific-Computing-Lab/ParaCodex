#include "XSbench_header.h"

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype, Profile* profile)
{
	double start = get_time();
        SimulationData GSD = move_simulation_data_to_device(in, mype, SD);
	profile->host_to_device_time = get_time() - start;

        if( mype == 0)	printf("Running baseline event-based simulation...\n");

        int nthreads = 256;
        int nblocks = ceil( (double) in.lookups / (double) nthreads);

	int nwarmups = in.num_warmups;
	start = 0.0;
	for (int i = 0; i < in.num_iterations + nwarmups; i++) {
		if (i == nwarmups) {
			gpuErrchk( cudaDeviceSynchronize() );
			start = get_time();
		}
		xs_lookup_kernel_baseline<<<nblocks, nthreads>>>( in, GSD );
	}
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	profile->kernel_time = get_time() - start;

        if( mype == 0)	printf("Reducing verification results...\n");
	start = get_time();
        gpuErrchk(cudaMemcpy(SD.verification, GSD.verification, in.lookups * sizeof(unsigned long), cudaMemcpyDeviceToHost) );
	profile->device_to_host_time = get_time() - start;

        unsigned long verification_scalar = 0;
        for( int i =0; i < in.lookups; i++ )
                verification_scalar += SD.verification[i];

        release_device_memory(GSD);

        return verification_scalar;
}

__global__ void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD )
{
        const int i = blockIdx.x *blockDim.x + threadIdx.x;

        if( i >= in.lookups )
                return;

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
        for(int j = 0; j < 5; j++ )
        {
                if( macro_xs_vector[j] > max )
                {
                        max = macro_xs_vector[j];
                        max_idx = j;
                }
        }
        GSD.verification[i] = max_idx+1;
}

__device__ void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
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

__device__ void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
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

__device__ long grid_search( long n, double quarry, double * __restrict__ A)
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

__host__ __device__ long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high)
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

__device__ int pick_mat( uint64_t * seed )
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

__host__ __device__ double LCG_random_double(uint64_t * seed)
{

        const uint64_t m = 9223372036854775808ULL;
        const uint64_t a = 2806196910506780709ULL;
        const uint64_t c = 1ULL;
        *seed = (a * (*seed) + c) % m;
        return (double) (*seed) / (double) m;
}

__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
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

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        gpuErrchk( cudaMalloc((void **) &GSD.p_energy_samples, sz) );
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        gpuErrchk( cudaMalloc((void **) &GSD.mat_samples, sz) );
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        int nthreads = 32;
        int nblocks = ceil( (double) in.lookups / 32.0);

        sampling_kernel<<<nblocks, nthreads>>>( in, GSD );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        xs_lookup_kernel_optimization_1<<<nblocks, nthreads>>>( in, GSD );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        return verification_scalar;
}

__global__ void sampling_kernel(Inputs in, SimulationData GSD )
{

        const int i = blockIdx.x *blockDim.x + threadIdx.x;

        if( i >= in.lookups )
                return;

        uint64_t seed = STARTING_SEED;

        seed = fast_forward_LCG(seed, 2*i);

        double p_energy = LCG_random_double(&seed);
        int mat         = pick_mat(&seed);

        GSD.p_energy_samples[i] = p_energy;
        GSD.mat_samples[i] = mat;
}

__global__ void xs_lookup_kernel_optimization_1(Inputs in, SimulationData GSD )
{

        const int i = blockIdx.x *blockDim.x + threadIdx.x;

        if( i >= in.lookups )
                return;

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
        for(int j = 0; j < 5; j++ )
        {
                if( macro_xs_vector[j] > max )
                {
                        max = macro_xs_vector[j];
                        max_idx = j;
                }
        }
        GSD.verification[i] = max_idx+1;
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 2 - Material Lookup Kernels";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        gpuErrchk( cudaMalloc((void **) &GSD.p_energy_samples, sz) );
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        gpuErrchk( cudaMalloc((void **) &GSD.mat_samples, sz) );
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        int nthreads = 32;
        int nblocks = ceil( (double) in.lookups / 32.0);

        sampling_kernel<<<nblocks, nthreads>>>( in, GSD );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        for( int m = 0; m < 12; m++ )
                xs_lookup_kernel_optimization_2<<<nblocks, nthreads>>>( in, GSD, m );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        return verification_scalar;
}

__global__ void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m )
{

        const int i = blockIdx.x *blockDim.x + threadIdx.x;

        if( i >= in.lookups )
                return;

        int mat = GSD.mat_samples[i];
        if( mat != m )
                return;

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
        for(int j = 0; j < 5; j++ )
        {
                if( macro_xs_vector[j] > max )
                {
                        max = macro_xs_vector[j];
                        max_idx = j;
                }
        }
        GSD.verification[i] = max_idx+1;
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        gpuErrchk( cudaMalloc((void **) &GSD.p_energy_samples, sz) );
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        gpuErrchk( cudaMalloc((void **) &GSD.mat_samples, sz) );
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        int nthreads = 32;
        int nblocks = ceil( (double) in.lookups / 32.0);

        sampling_kernel<<<nblocks, nthreads>>>( in, GSD );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        xs_lookup_kernel_optimization_3<<<nblocks, nthreads>>>( in, GSD, 0 );
        xs_lookup_kernel_optimization_3<<<nblocks, nthreads>>>( in, GSD, 1 );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        return verification_scalar;
}

__global__ void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel )
{

        const int i = blockIdx.x *blockDim.x + threadIdx.x;

        if( i >= in.lookups )
                return;

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
                for(int j = 0; j < 5; j++ )
                {
                        if( macro_xs_vector[j] > max )
                        {
                                max = macro_xs_vector[j];
                                max_idx = j;
                        }
                }
                GSD.verification[i] = max_idx+1;
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
        gpuErrchk( cudaMalloc((void **) &GSD.p_energy_samples, sz) );
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        gpuErrchk( cudaMalloc((void **) &GSD.mat_samples, sz) );
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        int nthreads = 32;
        int nblocks = ceil( (double) in.lookups / 32.0);

        sampling_kernel<<<nblocks, nthreads>>>( in, GSD );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        int n_lookups_per_material[12];
        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = thrust::count(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, m);

        thrust::sort_by_key(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, GSD.p_energy_samples);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                nthreads = 32;
                nblocks = ceil((double) n_lookups_per_material[m] / (double) nthreads);
                xs_lookup_kernel_optimization_4<<<nblocks, nthreads>>>( in, GSD, m, n_lookups_per_material[m], offset );
                offset += n_lookups_per_material[m];
        }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        return verification_scalar;
}

__global__ void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset )
{

        int i = blockIdx.x *blockDim.x + threadIdx.x;

        if( i >= n_lookups )
                return;

        i += offset;

        int mat = GSD.mat_samples[i];
        if( mat != m )
                return;

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
        for(int j = 0; j < 5; j++ )
        {
                if( macro_xs_vector[j] > max )
                {
                        max = macro_xs_vector[j];
                        max_idx = j;
                }
        }
        GSD.verification[i] = max_idx+1;
}

struct is_mat_fuel{
        __host__ __device__
        bool operator()(const int & a)
        {
                return a == 0;
        }
};

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        gpuErrchk( cudaMalloc((void **) &GSD.p_energy_samples, sz) );
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        gpuErrchk( cudaMalloc((void **) &GSD.mat_samples, sz) );
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        int nthreads = 32;
        int nblocks = ceil( (double) in.lookups / 32.0);

        sampling_kernel<<<nblocks, nthreads>>>( in, GSD );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        int n_fuel_lookups = thrust::count(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, 0);

        thrust::partition(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, GSD.p_energy_samples, is_mat_fuel());

        nblocks = ceil( (double) n_fuel_lookups / (double) nthreads);
        xs_lookup_kernel_optimization_5<<<nblocks, nthreads>>>( in, GSD, n_fuel_lookups, 0 );

        nblocks = ceil( (double) (in.lookups - n_fuel_lookups) / (double) nthreads);
        xs_lookup_kernel_optimization_5<<<nblocks, nthreads>>>( in, GSD, in.lookups-n_fuel_lookups, n_fuel_lookups );

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        return verification_scalar;
}

__global__ void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset )
{

        int i = blockIdx.x *blockDim.x + threadIdx.x;

        if( i >= n_lookups )
                return;

        i += offset;

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
        for(int j = 0; j < 5; j++ )
        {
                if( macro_xs_vector[j] > max )
                {
                        max = macro_xs_vector[j];
                        max_idx = j;
                }
        }
        GSD.verification[i] = max_idx+1;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype)
{
        const char * optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

        if( mype == 0)	printf("Simulation Kernel:\"%s\"\n", optimization_name);

        if( mype == 0)	printf("Allocating additional device data required by kernel...\n");
        size_t sz;
        size_t total_sz = 0;

        sz = in.lookups * sizeof(double);
        gpuErrchk( cudaMalloc((void **) &GSD.p_energy_samples, sz) );
        total_sz += sz;
        GSD.length_p_energy_samples = in.lookups;

        sz = in.lookups * sizeof(int);
        gpuErrchk( cudaMalloc((void **) &GSD.mat_samples, sz) );
        total_sz += sz;
        GSD.length_mat_samples = in.lookups;

        if( mype == 0)	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

        if( mype == 0)	printf("Beginning optimized simulation...\n");

        int nthreads = 32;
        int nblocks = ceil( (double) in.lookups / 32.0);

        sampling_kernel<<<nblocks, nthreads>>>( in, GSD );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        int n_lookups_per_material[12];
        for( int m = 0; m < 12; m++ )
                n_lookups_per_material[m] = thrust::count(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, m);

        thrust::sort_by_key(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, GSD.p_energy_samples);

        int offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                thrust::sort_by_key(thrust::device, GSD.p_energy_samples + offset, GSD.p_energy_samples + offset + n_lookups_per_material[m], GSD.mat_samples + offset);
                offset += n_lookups_per_material[m];
        }

        offset = 0;
        for( int m = 0; m < 12; m++ )
        {
                nthreads = 32;
                nblocks = ceil((double) n_lookups_per_material[m] / (double) nthreads);
                xs_lookup_kernel_optimization_4<<<nblocks, nthreads>>>( in, GSD, m, n_lookups_per_material[m], offset );
                offset += n_lookups_per_material[m];
        }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        if( mype == 0)	printf("Reducing verification results...\n");

        unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        return verification_scalar;
}
