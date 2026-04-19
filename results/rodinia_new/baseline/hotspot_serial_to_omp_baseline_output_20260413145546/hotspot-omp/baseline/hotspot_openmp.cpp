#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Returns the current system time in microseconds
long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

using namespace std;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD (3.0e6)
/* required precision in degrees */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor */
#define FACTOR_CHIP 0.5

typedef float FLOAT;

/* chip parameters */
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

/* ambient temperature, assuming no package at all */
const FLOAT amb_temp = 80.0;

int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations
 * by one time step
 */
void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int row, int col,
                      FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1,
                      FLOAT step)
{
    (void)step;
    int total = row * col;

#pragma omp target teams distribute parallel for collapse(2) default(none) \
    map(to : temp[0:total], power[0:total]) map(from : result[0:total]) \
    shared(result, temp, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1)
    for (int r = 0; r < row; ++r) {
        for (int c = 0; c < col; ++c) {
            int idx = r * col + c;
            FLOAT cur = temp[idx];
            FLOAT delta = power[idx];

            if (r > 0) {
                delta += (temp[(r - 1) * col + c] - cur) * Ry_1;
            }
            if (r + 1 < row) {
                delta += (temp[(r + 1) * col + c] - cur) * Ry_1;
            }
            if (c > 0) {
                delta += (temp[r * col + c - 1] - cur) * Rx_1;
            }
            if (c + 1 < col) {
                delta += (temp[r * col + c + 1] - cur) * Rx_1;
            }

            result[idx] = cur + Cap_1 * (delta + (amb_temp - cur) * Rz_1);
        }
    }
}

/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int row, int col)
{
#ifdef VERBOSE
    int i = 0;
#endif

    FLOAT grid_height = chip_height / row;
    FLOAT grid_width = chip_width / col;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1 = 1.f / Rx;
    FLOAT Ry_1 = 1.f / Ry;
    FLOAT Rz_1 = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;
#ifdef VERBOSE
    fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
    fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
#endif

    {
        FLOAT *r = result;
        FLOAT *t = temp;
        for (int i = 0; i < num_iterations; i++) {
#ifdef VERBOSE
            fprintf(stdout, "iteration %d\n", i++);
#endif
            single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
            FLOAT *tmp = t;
            t = r;
            r = tmp;
        }
    }
#ifdef VERBOSE
    fprintf(stdout, "iteration %d\n", i++);
#endif
}

void fatal(const char *s)
{
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, char *file)
{
    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++) {
            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }

    fclose(fp);
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file)
{
    int i;
    FILE *fp;
    char str[STR_SIZE];
    FLOAT val;

    fp = fopen(file, "r");
    if (!fp)
        fatal("file could not be opened for reading");

    for (i = 0; i < grid_rows * grid_cols; i++) {
        fgets(str, STR_SIZE, fp);
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1))
            fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
    fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<no. of threads>   - number of threads\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

int main(int argc, char **argv)
{
    int grid_rows, grid_cols, sim_time;
#ifdef OUTPUT
    int i;
#endif
    FLOAT *temp, *power, *result;
    char *tfile, *pfile, *ofile;

    /* check validity of inputs */
    if (argc != 8)
        usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 ||
        (grid_cols = atoi(argv[2])) <= 0 ||
        (sim_time = atoi(argv[3])) <= 0 ||
        (num_omp_threads = atoi(argv[4])) <= 0)
        usage(argc, argv);

    /* allocate memory for the temperature and power arrays */
    temp = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    power = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    result = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    if (!temp || !power || !result)
        fatal("unable to allocate memory");

    /* read initial temperatures and input power */
    tfile = argv[5];
    pfile = argv[6];
    ofile = argv[7];

    read_input(temp, grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);

    printf("Start computing the transient temperature\n");

    long long start_time = get_time();

    compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);

    long long end_time = get_time();

    printf("Ending simulation\n");
    printf("Total time: %.3f seconds\n", ((float)(end_time - start_time)) / (1000 * 1000));

    writeoutput((1 & sim_time) ? result : temp, grid_rows, grid_cols, ofile);

    /* output results */
#ifdef VERBOSE
    fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
    for (i = 0; i < grid_rows * grid_cols; i++)
        fprintf(stdout, "%d\t%g\n", i, temp[i]);
#endif
    /* cleanup */
    free(temp);
    free(power);
    free(result);

    return 0;
}
/* vim: set ts=4 sw=4  sts=4 et si ai: */
