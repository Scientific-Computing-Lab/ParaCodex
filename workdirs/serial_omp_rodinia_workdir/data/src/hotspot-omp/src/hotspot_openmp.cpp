#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include "../../common/rodiniaUtilFunctions.h"

#define STR_SIZE	256

/* mptogpu */
#define GPU_DEVICE 1
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
#define OPEN
//#define NUM_THREAD 4

/* chip parameters	*/
double t_chip = 0.0005;
double chip_height = 0.016;
double chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/


int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
void single_iteration_gpu(double *result, double *temp, double *power, int row, int col,
					  double Cap, double Rx, double Ry, double Rz, 
					  double step)
{
	double delta;
	int r, c;
	double amb_temp = 80.0;

	#pragma omp target device(GPU_DEVICE) map(to: power[0:row*col], temp[0:row*col]) map(tofrom: result[0:row*col])
	{
		#pragma omp parallel for collapse(2)
		for (r = 0; r < row; r++) {
			for (c = 0; c < col; c++) {
				/*	Corner 1	*/
				if ((r == 0) && (c == 0)) {
					delta = (step / Cap) * (power[0] +
							(temp[1] - temp[0]) / Rx +
							(temp[col] - temp[0]) / Ry +
							(amb_temp - temp[0]) / Rz);
				}	/*	Corner 2	*/
				else if ((r == 0) && (c == col-1)) {
					delta = (step / Cap) * (power[c] +
							(temp[c-1] - temp[c]) / Rx +
							(temp[c+col] - temp[c]) / Ry +
							(amb_temp - temp[c]) / Rz);
				}	/*	Corner 3	*/
				else if ((r == row-1) && (c == col-1)) {
					delta = (step / Cap) * (power[r*col+c] + 
							(temp[r*col+c-1] - temp[r*col+c]) / Rx + 
							(temp[(r-1)*col+c] - temp[r*col+c]) / Ry + 
							(amb_temp - temp[r*col+c]) / Rz);					
				}	/*	Corner 4	*/
				else if ((r == row-1) && (c == 0)) {
					delta = (step / Cap) * (power[r*col] + 
							(temp[r*col+1] - temp[r*col]) / Rx + 
							(temp[(r-1)*col] - temp[r*col]) / Ry + 
							(amb_temp - temp[r*col]) / Rz);
				}	/*	Edge 1	*/
				else if (r == 0) {
					delta = (step / Cap) * (power[c] + 
							(temp[c+1] + temp[c-1] - 2.0*temp[c]) / Rx + 
							(temp[col+c] - temp[c]) / Ry + 
							(amb_temp - temp[c]) / Rz);
				}	/*	Edge 2	*/
				else if (c == col-1) {
					delta = (step / Cap) * (power[r*col+c] + 
							(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) / Ry + 
							(temp[r*col+c-1] - temp[r*col+c]) / Rx + 
							(amb_temp - temp[r*col+c]) / Rz);
				}	/*	Edge 3	*/
				else if (r == row-1) {
					delta = (step / Cap) * (power[r*col+c] + 
							(temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) / Rx + 
							(temp[(r-1)*col+c] - temp[r*col+c]) / Ry + 
							(amb_temp - temp[r*col+c]) / Rz);
				}	/*	Edge 4	*/
				else if (c == 0) {
					delta = (step / Cap) * (power[r*col] + 
							(temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) / Ry + 
							(temp[r*col+1] - temp[r*col]) / Rx + 
							(amb_temp - temp[r*col]) / Rz);
				}	/*	Inside the chip	*/
				else {
					delta = (step / Cap) * (power[r*col+c] + 
							(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) / Ry + 
							(temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) / Rx + 
							(amb_temp - temp[r*col+c]) / Rz);
				}
	  			
				/*	Update Temperatures	*/
				result[r*col+c] =temp[r*col+c]+ delta;
			}
		}
	}


#ifdef OPEN
	omp_set_num_threads(num_omp_threads);
	#pragma omp parallel for shared(result, temp) private(r, c) schedule(static)
#endif
	for (r = 0; r < row; r++) {
		for (c = 0; c < col; c++) {
			temp[r*col+c]=result[r*col+c];
		}
	}
}


void single_iteration_cpu(double *result, double *temp, double *power, int row, int col,
					  double Cap, double Rx, double Ry, double Rz, 
					  double step)
{
	double delta;
	int r, c;
	double amb_temp = 80.0;

	for (r = 0; r < row; r++) {
		for (c = 0; c < col; c++) {
			/*	Corner 1	*/
			if ((r == 0) && (c == 0)) {
				delta = (step / Cap) * (power[0] +
						(temp[1] - temp[0]) / Rx +
						(temp[col] - temp[0]) / Ry +
						(amb_temp - temp[0]) / Rz);
			}	/*	Corner 2	*/
			else if ((r == 0) && (c == col-1)) {
				delta = (step / Cap) * (power[c] +
						(temp[c-1] - temp[c]) / Rx +
						(temp[c+col] - temp[c]) / Ry +
						(amb_temp - temp[c]) / Rz);
			}	/*	Corner 3	*/
			else if ((r == row-1) && (c == col-1)) {
				delta = (step / Cap) * (power[r*col+c] + 
						(temp[r*col+c-1] - temp[r*col+c]) / Rx + 
						(temp[(r-1)*col+c] - temp[r*col+c]) / Ry + 
						(amb_temp - temp[r*col+c]) / Rz);					
			}	/*	Corner 4	*/
			else if ((r == row-1) && (c == 0)) {
				delta = (step / Cap) * (power[r*col] + 
						(temp[r*col+1] - temp[r*col]) / Rx + 
						(temp[(r-1)*col] - temp[r*col]) / Ry + 
						(amb_temp - temp[r*col]) / Rz);
			}	/*	Edge 1	*/
			else if (r == 0) {
				delta = (step / Cap) * (power[c] + 
						(temp[c+1] + temp[c-1] - 2.0*temp[c]) / Rx + 
						(temp[col+c] - temp[c]) / Ry + 
						(amb_temp - temp[c]) / Rz);
			}	/*	Edge 2	*/
			else if (c == col-1) {
				delta = (step / Cap) * (power[r*col+c] + 
						(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) / Ry + 
						(temp[r*col+c-1] - temp[r*col+c]) / Rx + 
						(amb_temp - temp[r*col+c]) / Rz);
			}	/*	Edge 3	*/
			else if (r == row-1) {
				delta = (step / Cap) * (power[r*col+c] + 
						(temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) / Rx + 
						(temp[(r-1)*col+c] - temp[r*col+c]) / Ry + 
						(amb_temp - temp[r*col+c]) / Rz);
			}	/*	Edge 4	*/
			else if (c == 0) {
				delta = (step / Cap) * (power[r*col] + 
						(temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) / Ry + 
						(temp[r*col+1] - temp[r*col]) / Rx + 
						(amb_temp - temp[r*col]) / Rz);
			}	/*	Inside the chip	*/
			else {
				delta = (step / Cap) * (power[r*col+c] + 
						(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) / Ry + 
						(temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) / Rx + 
						(amb_temp - temp[r*col+c]) / Rz);
			}
  			
			/*	Update Temperatures	*/
			result[r*col+c] =temp[r*col+c]+ delta;
		}
	}

	for (r = 0; r < row; r++) {
		for (c = 0; c < col; c++) {
			temp[r*col+c]=result[r*col+c];
		}
	}
}


/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(double *result, int num_iterations, double *temp, double *power, int row, int col, int dev) 
{
	#ifdef VERBOSE
	int i = 0;
	#endif

	double grid_height = chip_height / row;
	double grid_width = chip_width / col;

	double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	double Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	double Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	double Rz = t_chip / (K_SI * grid_height * grid_width);

	double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	double step = PRECISION / max_slope;
	double t;

	#ifdef VERBOSE
	fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
	fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
	#endif

     for (int i = 0; i < num_iterations ; i++)
	{
		#ifdef VERBOSE
		fprintf(stdout, "iteration %d\n", i++);
		#endif
		if(dev == 0)
			single_iteration_cpu(result, temp, power, row, col, Cap, Rx, Ry, Rz, step);
		else
			single_iteration_gpu(result, temp, power, row, col, Cap, Rx, Ry, Rz, step);	
	}	

	#ifdef VERBOSE
	fprintf(stdout, "iteration %d\n", i++);
	#endif
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
	exit(1);
}

void read_input(double *vect, int grid_rows, int grid_cols, char *file)
{
  	int i, index;
	FILE *fp;
	char str[STR_SIZE];
	double val;

	fp = fopen (file, "r");
	if (!fp)
		fatal ("file could not be opened for reading");

	for (i=0; i < grid_rows * grid_cols; i++) {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		if ((sscanf(str, "%lf", &val) != 1) )
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
	exit(1);
}

void compareResults(double *result_cpu, double *result_gpu, int grid_rows, int grid_cols) {
	int i,fail;
	fail = 0;

	for (i = 0; i < grid_rows * grid_cols; i++) {
		if (percentDiff(result_cpu[i], result_gpu[i])> PERCENT_DIFF_ERROR_THRESHOLD) {
			fail++;
		}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char **argv)
{
	int grid_rows, grid_cols, sim_time, i;
	double *temp_cpu, *temp_gpu, *power, *result_cpu, *result_gpu;
	char *tfile, *pfile;
	double t_start, t_end;
	
	/* check validity of inputs	*/
	if (argc != 7)
		usage(argc, argv);
	if ((grid_rows = atoi(argv[1])) <= 0 ||
		(grid_cols = atoi(argv[2])) <= 0 ||
		(sim_time = atoi(argv[3])) <= 0 || 
		(num_omp_threads = atoi(argv[4])) <= 0
		)
		usage(argc, argv);

	/* allocate memory for the temperature and power arrays	*/
	temp_cpu = (double *) calloc (grid_rows * grid_cols, sizeof(double));
	temp_gpu = (double *) calloc (grid_rows * grid_cols, sizeof(double));
	power = (double *) calloc (grid_rows * grid_cols, sizeof(double));
	result_cpu = (double *) calloc (grid_rows * grid_cols, sizeof(double));
	result_gpu = (double *) calloc (grid_rows * grid_cols, sizeof(double));

	if(!temp_cpu || !temp_gpu || !power)
		fatal("unable to allocate memory");

	/* read initial temperatures and input power	*/
	tfile = argv[5];
	pfile = argv[6];
	read_input(temp_cpu, grid_rows, grid_cols, tfile);
	read_input(temp_gpu, grid_rows, grid_cols, tfile);
	read_input(power, grid_rows, grid_cols, pfile);

	printf("<< Start computing the transient temperature >>\n");

	t_start = rtclock();
	compute_tran_temp(result_cpu, sim_time, temp_cpu, power, grid_rows, grid_cols, 0);
	t_end = rtclock();
    	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	t_start = rtclock();
	compute_tran_temp(result_gpu, sim_time, temp_gpu, power, grid_rows, grid_cols, 1);
	t_end = rtclock();
    	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(result_cpu, result_gpu, grid_rows, grid_cols);

	printf("Ending simulation\n");


	/* output results	*/
#ifdef VERBOSE
	fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
	for(i=0; i < grid_rows * grid_cols; i++)
	fprintf(stdout, "%d\t%g\n", i, temp[i]);
#endif
	/* cleanup	*/
	free(temp_gpu);
	free(temp_cpu);
	free(power);

	return 0;
}

