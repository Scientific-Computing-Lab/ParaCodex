#include "XSbench_header.cuh"

static void * allocate_and_copy(void **dst_device, const void *src, size_t count, size_t elem_size, int device, int host)
{
        *dst_device = omp_target_alloc(count * elem_size, device);
        omp_target_memcpy(*dst_device, src, count * elem_size, 0, 0, device, host);
        return *dst_device;
}

SimulationData move_simulation_data_to_device( Inputs in, int mype, SimulationData SD )
{
        if(mype == 0) printf("Mapping simulation data onto the OpenMP target device...\n");

        SimulationData GSD = SD;
        int device = omp_get_default_device();
        int host = omp_get_initial_device();

        if (GSD.length_num_nucs > 0)
                allocate_and_copy((void **) &GSD.num_nucs, SD.num_nucs, GSD.length_num_nucs, sizeof(int), device, host);

        if (GSD.length_concs > 0)
                allocate_and_copy((void **) &GSD.concs, SD.concs, GSD.length_concs, sizeof(double), device, host);

        if (GSD.length_mats > 0)
                allocate_and_copy((void **) &GSD.mats, SD.mats, GSD.length_mats, sizeof(int), device, host);

        if (GSD.length_unionized_energy_array > 0)
                allocate_and_copy((void **) &GSD.unionized_energy_array, SD.unionized_energy_array, GSD.length_unionized_energy_array, sizeof(double), device, host);

        if (GSD.length_index_grid > 0)
                allocate_and_copy((void **) &GSD.index_grid, SD.index_grid, GSD.length_index_grid, sizeof(int), device, host);

        if (GSD.length_nuclide_grid > 0)
                allocate_and_copy((void **) &GSD.nuclide_grid, SD.nuclide_grid, GSD.length_nuclide_grid, sizeof(NuclideGridPoint), device, host);

        GSD.verification = (unsigned long *) omp_target_alloc(in.lookups * sizeof(unsigned long), device);
        GSD.length_verification = in.lookups;

        GSD.p_energy_samples = (double *) omp_target_alloc(in.lookups * sizeof(double), device);
        GSD.length_p_energy_samples = in.lookups;

        GSD.mat_samples = (int *) omp_target_alloc(in.lookups * sizeof(int), device);
        GSD.length_mat_samples = in.lookups;

        if(mype == 0 ) printf("OpenMP target initialization complete.\n");
        return GSD;
}

void release_device_memory(SimulationData GSD) {
        int device = omp_get_default_device();
        if (GSD.num_nucs) omp_target_free(GSD.num_nucs, device);
        if (GSD.concs) omp_target_free(GSD.concs, device);
        if (GSD.mats) omp_target_free(GSD.mats, device);
        if (GSD.unionized_energy_array) omp_target_free(GSD.unionized_energy_array, device);
        if (GSD.index_grid) omp_target_free(GSD.index_grid, device);
        if (GSD.nuclide_grid) omp_target_free(GSD.nuclide_grid, device);
        if (GSD.verification) omp_target_free(GSD.verification, device);
        if (GSD.p_energy_samples) omp_target_free(GSD.p_energy_samples, device);
        if (GSD.mat_samples) omp_target_free(GSD.mat_samples, device);
}

void release_memory(SimulationData SD) {
        free(SD.num_nucs);
        free(SD.concs);
        free(SD.mats);
        if (SD.length_unionized_energy_array > 0) free(SD.unionized_energy_array);
        free(SD.nuclide_grid);
        free(SD.verification);
}

SimulationData grid_init_do_not_profile( Inputs in, int mype )
{
        // Structure to hold all allocated simuluation data arrays
        SimulationData SD;



        // Keep track of how much data we're allocating
        size_t nbytes = 0;

        // Set the initial seed value
        uint64_t seed = 42;

        ////////////////////////////////////////////////////////////////////
        // Initialize Nuclide Grids
        ////////////////////////////////////////////////////////////////////

        if(mype == 0) printf("Intializing nuclide grids...\n");

        // First, we need to initialize our nuclide grid. This comes in the form
        // of a flattened 2D array that hold all the information we need to define
        // the cross sections for all isotopes in the simulation.
        // The grid is composed of "NuclideGridPoint" structures, which hold the
        // energy level of the grid point and all associated XS data at that level.
        // An array of structures (AOS) is used instead of
        // a structure of arrays, as the grid points themselves are accessed in
        // a random order, but all cross section interaction channels and the
        // energy level are read whenever the gridpoint is accessed, meaning the
        // AOS is more cache efficient.

        // Initialize Nuclide Grid
        SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
        SD.nuclide_grid     = (NuclideGridPoint *) malloc( SD.length_nuclide_grid * sizeof(NuclideGridPoint));
        assert(SD.nuclide_grid != NULL);
        nbytes += SD.length_nuclide_grid * sizeof(NuclideGridPoint);
        for( int i = 0; i < SD.length_nuclide_grid; i++ )
        {
                SD.nuclide_grid[i].energy        = LCG_random_double(&seed);
                SD.nuclide_grid[i].total_xs      = LCG_random_double(&seed);
                SD.nuclide_grid[i].elastic_xs    = LCG_random_double(&seed);
                SD.nuclide_grid[i].absorbtion_xs = LCG_random_double(&seed);
                SD.nuclide_grid[i].fission_xs    = LCG_random_double(&seed);
                SD.nuclide_grid[i].nu_fission_xs = LCG_random_double(&seed);
        }

        // Sort so that each nuclide has data stored in ascending energy order.
        for( int i = 0; i < in.n_isotopes; i++ )
                qsort( &SD.nuclide_grid[i*in.n_gridpoints], in.n_gridpoints, sizeof(NuclideGridPoint), NGP_compare);

        // error debug check
        /*
        for( int i = 0; i < in.n_isotopes; i++ )
        {
                printf("NUCLIDE %d ==============================\n", i);
                for( int j = 0; j < in.n_gridpoints; j++ )
                printf("E%d = %lf\n", j, SD.nuclide_grid[i * in.n_gridpoints + j].energy);
        }
        */

        // Allocate Verification Array
        size_t sz = in.lookups * sizeof(unsigned long);
        SD.verification = (unsigned long *) malloc(sz);
        nbytes += sz;
        SD.length_verification = in.lookups;


        ////////////////////////////////////////////////////////////////////
        // Initialize Acceleration Structure
        ////////////////////////////////////////////////////////////////////

        if( in.grid_type == NUCLIDE )
        {
                SD.length_unionized_energy_array = 0;
                SD.length_index_grid = 0;
        }

        if( in.grid_type == UNIONIZED )
        {
                if(mype == 0) printf("Intializing unionized grid...\n");

                // Allocate space to hold the union of all nuclide energy data
                SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
                SD.unionized_energy_array = (double *) malloc( SD.length_unionized_energy_array * sizeof(double));
                assert(SD.unionized_energy_array != NULL );
                nbytes += SD.length_unionized_energy_array * sizeof(double);

                // Copy energy data over from the nuclide energy grid
                for( int i = 0; i < SD.length_unionized_energy_array; i++ )
                        SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;

                // Sort unionized energy array
                qsort( SD.unionized_energy_array, SD.length_unionized_energy_array, sizeof(double), double_compare);

                // Allocate space to hold the acceleration grid indices
                SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
                SD.index_grid = (int *) malloc( SD.length_index_grid * sizeof(int));
                assert(SD.index_grid != NULL);
                nbytes += SD.length_index_grid * sizeof(int);

                // Generates the double indexing grid
                int * idx_low = (int *) calloc( in.n_isotopes, sizeof(int));
                assert(idx_low != NULL );
                double * energy_high = (double *) malloc( in.n_isotopes * sizeof(double));
                assert(energy_high != NULL );

                for( int i = 0; i < in.n_isotopes; i++ )
                        energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + 1].energy;

                for( long e = 0; e < SD.length_unionized_energy_array; e++ )
                {
                        double unionized_energy = SD.unionized_energy_array[e];
                        for( long i = 0; i < in.n_isotopes; i++ )
                        {
                                if( unionized_energy < energy_high[i]  )
                                        SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                                        else if( idx_low[i] == in.n_gridpoints - 2 )
                                        SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                                        else
                                        {
                                        idx_low[i]++;
                                        SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                                        energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + idx_low[i] + 1].energy;
                                }
                        }
                }

                free(idx_low);
                free(energy_high);
        }

        if( in.grid_type == HASH )
        {
                if(mype == 0) printf("Intializing hash grid...\n");
                SD.length_unionized_energy_array = 0;
                SD.length_index_grid  = in.hash_bins * in.n_isotopes;
                SD.index_grid = (int *) malloc( SD.length_index_grid * sizeof(int));
                assert(SD.index_grid != NULL);
                nbytes += SD.length_index_grid * sizeof(int);

                double du = 1.0 / in.hash_bins;

                // For each energy level in the hash table
                for( long e = 0; e < in.hash_bins; e++ )
                {
                        double energy = e * du;

                        // We need to determine the bounding energy levels for all isotopes
                        for( long i = 0; i < in.n_isotopes; i++ )
                        {
                                SD.index_grid[e * in.n_isotopes + i] = grid_search_nuclide( in.n_gridpoints, energy, SD.nuclide_grid + i * in.n_gridpoints, 0, in.n_gridpoints-1);
                        }
                }
        }

        ////////////////////////////////////////////////////////////////////
        // Initialize Materials and Concentrations
        ////////////////////////////////////////////////////////////////////
        if(mype == 0) printf("Intializing material data...\n");

        // Set the number of nuclides in each material
        SD.num_nucs  = load_num_nucs(in.n_isotopes);
        SD.length_num_nucs = 12; // There are always 12 materials in XSBench

        // Intialize the flattened 2D grid of material data. The grid holds
        // a list of nuclide indices for each of the 12 material types. The
        // grid is allocated as a full square grid, even though not all
        // materials have the same number of nuclides.
        SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
        SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;

        // Intialize the flattened 2D grid of nuclide concentration data. The grid holds
        // a list of nuclide concentrations for each of the 12 material types. The
        // grid is allocated as a full square grid, even though not all
        // materials have the same number of nuclides.
        SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
        SD.length_concs = SD.length_mats;

        if(mype == 0) printf("Intialization complete. Allocated %.0lf MB of data on CPU.\n", nbytes/1024.0/1024.0 );

        return SD;
}
