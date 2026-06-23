#define LIMIT -999
//#define TRACE

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#ifdef GATE_VERIFY
#include "gate.h"
#endif

#define BLOCK_SIZE 16

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// Returns the current system time in microseconds
long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;

}

#pragma omp declare target
static inline int maximum( int a,
		 int b,
		 int c){

	int k;
	if( a <= b )
		k = b;
	else
	k = a;

	if( k <=c )
	return(c);
	else
	return(k);
}

static inline void compute_tile(int *input_itemsets, int *referrence,
        int max_cols, int penalty, int b_index_x, int b_index_y)
{
    int input_itemsets_l[(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)];
    int reference_l[BLOCK_SIZE * BLOCK_SIZE];

    for ( int i = 0; i < BLOCK_SIZE; ++i )
    {
        for ( int j = 0; j < BLOCK_SIZE; ++j)
        {
            reference_l[i*BLOCK_SIZE + j] = referrence[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1];
        }
    }

    for ( int i = 0; i < BLOCK_SIZE + 1; ++i )
    {
        for ( int j = 0; j < BLOCK_SIZE + 1; ++j)
        {
            input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i) + b_index_x*BLOCK_SIZE +  j];
        }
    }

    for ( int i = 1; i < BLOCK_SIZE + 1; ++i )
    {
        for ( int j = 1; j < BLOCK_SIZE + 1; ++j)
        {
            input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = maximum( input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_l[(i - 1)*BLOCK_SIZE + j - 1],
                    input_itemsets_l[i*(BLOCK_SIZE + 1) + j - 1] - penalty,
                    input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j] - penalty);
        }
    }

    for ( int i = 0; i < BLOCK_SIZE; ++i )
    {
        for ( int j = 0; j < BLOCK_SIZE; ++j)
        {
            input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1] = input_itemsets_l[(i + 1)*(BLOCK_SIZE+1) + j + 1];
        }
    }
}
#pragma omp end declare target

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
    runTest( argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads>\n", argv[0]);
	fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
	fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
	fprintf(stderr, "\t<num_threads>    - no. of threads\n");
	exit(1);
}

void nw_optimized(int *input_itemsets, int *output_itemsets, int *referrence,
        int max_rows, int max_cols, int penalty)
{
    const int num_blocks = (max_cols-1)/BLOCK_SIZE;
    const int total_items = max_rows * max_cols;
    const size_t total_bytes = (size_t)total_items * sizeof(int);
    const int device = omp_get_default_device();
    const int host = omp_get_initial_device();

    int *device_input = (int *)omp_target_alloc(total_bytes, device);
    int *device_reference = (int *)omp_target_alloc(total_bytes, device);

    if (!device_input || !device_reference)
    {
        fprintf(stderr, "error: can not allocate device memory\n");
        if (device_input)
            omp_target_free(device_input, device);
        if (device_reference)
            omp_target_free(device_reference, device);
        exit(EXIT_FAILURE);
    }

    omp_target_memcpy(device_input, input_itemsets, total_bytes, 0, 0, device, host);
    omp_target_memcpy(device_reference, referrence, total_bytes, 0, 0, device, host);

    for ( int blk = 1; blk <= num_blocks; blk++ )
    {
        const int tile_count = blk;
#pragma omp target teams distribute is_device_ptr(device_input, device_reference) num_teams(tile_count)
        for( int b_index_x = 0; b_index_x < tile_count; ++b_index_x)
        {
            int b_index_y = blk - 1 - b_index_x;
            compute_tile(device_input, device_reference, max_cols, penalty, b_index_x, b_index_y);
        }
    }

    printf("Processing bottom-right matrix\n");

    for ( int blk = 2; blk <= num_blocks; blk++ )
    {
        const int tile_count = num_blocks - blk + 1;
#pragma omp target teams distribute is_device_ptr(device_input, device_reference) num_teams(tile_count)
        for( int b_index_x = blk - 1; b_index_x < num_blocks; ++b_index_x)
        {
            int b_index_y = num_blocks + blk - 2 - b_index_x;
            compute_tile(device_input, device_reference, max_cols, penalty, b_index_x, b_index_y);
        }
    }

    omp_target_memcpy(input_itemsets, device_input, total_bytes, 0, 0, host, device);

    omp_target_free(device_input, device);
    omp_target_free(device_reference, device);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv)
{
    int max_rows, max_cols, penalty;
    int *input_itemsets, *output_itemsets, *referrence;
    int omp_num_threads;


    // the lengths of the two sequences should be able to divided by 16.
    // And at current stage  max_rows needs to equal max_cols
    if (argc == 4)
    {
        max_rows = atoi(argv[1]);
        max_cols = atoi(argv[1]);
        penalty = atoi(argv[2]);
        omp_num_threads = atoi(argv[3]);
    }
    else{
        usage(argc, argv);
    }

    omp_set_num_threads(omp_num_threads);

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
    output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );


    if (!input_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );

    for (int i = 0 ; i < max_cols; i++){
        for (int j = 0 ; j < max_rows; j++){
            input_itemsets[i*max_cols+j] = 0;
        }
    }

    printf("Start Needleman-Wunsch\n");

    for( int i=1; i< max_rows ; i++){    //please define your own sequence.
        input_itemsets[i*max_cols] = rand() % 10 + 1;
    }
    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
        input_itemsets[j] = rand() % 10 + 1;
    }


    for (int i = 1 ; i < max_cols; i++){
        for (int j = 1 ; j < max_rows; j++){
            referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
        }
    }

    for( int i = 1; i< max_rows ; i++)
        input_itemsets[i*max_cols] = -i * penalty;
    for( int j = 1; j< max_cols ; j++)
        input_itemsets[j] = -j * penalty;


    printf("Num of threads: %d\n", omp_num_threads);
    printf("Processing top-left matrix\n");

    long long start_time = get_time();

    nw_optimized( input_itemsets, output_itemsets, referrence,
        max_rows, max_cols, penalty );

#ifdef GATE_VERIFY
    GATE_CHECKSUM_BYTES("input_itemsets", input_itemsets,
            (size_t)max_rows * (size_t)max_cols * sizeof(int));
#endif

    long long end_time = get_time();

    printf("Total time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));

#define TRACEBACK
#ifdef TRACEBACK

    FILE *fpo = fopen("result.txt","w");
    fprintf(fpo, "print traceback value GPU:\n");

    for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
        int nw, n, w, traceback;
        if ( i == max_rows - 2 && j == max_rows - 2 )
            fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]); //print the first element
        if ( i == 0 && j == 0 )
            break;
        if ( i > 0 && j > 0 ){
            nw = input_itemsets[(i - 1) * max_cols + j - 1];
            w  = input_itemsets[ i * max_cols + j - 1 ];
            n  = input_itemsets[(i - 1) * max_cols + j];
        }
        else if ( i == 0 ){
            nw = n = LIMIT;
            w  = input_itemsets[ i * max_cols + j - 1 ];
        }
        else if ( j == 0 ){
            nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max_cols + j];
        }
        else{
        }

        int new_nw, new_w, new_n;
        new_nw = nw + referrence[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if(traceback == new_nw)
            traceback = nw;
        if(traceback == new_w)
            traceback = w;
        if(traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);

        if(traceback == nw )
        {i--; j--; continue;}

        else if(traceback == w )
        {j--; continue;}

        else if(traceback == n )
        {i--; continue;}

        else
            ;
    }

    fclose(fpo);

#endif

    free(referrence);
    free(input_itemsets);
    free(output_itemsets);

}
