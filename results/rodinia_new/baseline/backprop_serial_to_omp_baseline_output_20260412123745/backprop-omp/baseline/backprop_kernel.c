#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  int input_cols, hidden_cols;
  int j, k;
  float out_err, hid_err;
  float *input_units;
  float *hidden_units;
  float *output_units;
  float *hidden_delta;
  float *output_delta;
  float *target_units;
  float *input_weights;
  float *hidden_weights;
  float *input_prev_weights;
  float *hidden_prev_weights;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  input_cols = hid + 1;
  hidden_cols = out + 1;

  input_units = net->input_units;
  hidden_units = net->hidden_units;
  output_units = net->output_units;
  hidden_delta = net->hidden_delta;
  output_delta = net->output_delta;
  target_units = net->target;
  input_weights = net->input_weights[0];
  hidden_weights = net->hidden_weights[0];
  input_prev_weights = net->input_prev_weights[0];
  hidden_prev_weights = net->hidden_prev_weights[0];

  printf("Performing GPU computation\n");

  input_units[0] = 1.0f;
  hidden_units[0] = 1.0f;

  #pragma omp target data map(to: input_units[0:(in + 1)], target_units[0:(out + 1)], input_weights[0:((in + 1) * (hid + 1))], hidden_weights[0:((hid + 1) * (out + 1))], input_prev_weights[0:((in + 1) * (hid + 1))], hidden_prev_weights[0:((hid + 1) * (out + 1))]) map(tofrom: hidden_units[0:(hid + 1)], output_units[0:(out + 1)], hidden_delta[0:(hid + 1)], output_delta[0:(out + 1)])
  {
    #pragma omp target teams distribute parallel for private(k)
    for (j = 1; j <= hid; ++j) {
      float sum = 0.0f;
      for (k = 0; k <= in; ++k) {
        sum += input_weights[(k * input_cols) + j] * input_units[k];
      }
      hidden_units[j] = squash(sum);
    }

    hidden_units[0] = 1.0f;

    #pragma omp target teams distribute parallel for private(k)
    for (j = 1; j <= out; ++j) {
      float sum = 0.0f;
      for (k = 0; k <= hid; ++k) {
        sum += hidden_weights[(k * hidden_cols) + j] * hidden_units[k];
      }
      output_units[j] = squash(sum);
    }

    out_err = 0.0f;
    #pragma omp target teams distribute parallel for reduction(+:out_err)
    for (j = 1; j <= out; ++j) {
      float o = output_units[j];
      float t = target_units[j];
      output_delta[j] = o * (1.0f - o) * (t - o);
      out_err += fabsf(output_delta[j]);
    }

    hid_err = 0.0f;
    #pragma omp target teams distribute parallel for reduction(+:hid_err)
    for (j = 1; j <= hid; ++j) {
      float h = hidden_units[j];
      float sum = 0.0f;
      for (k = 1; k <= out; ++k) {
        sum += output_delta[k] * hidden_weights[(j * hidden_cols) + k];
      }
      hidden_delta[j] = h * (1.0f - h) * sum;
      hid_err += fabsf(hidden_delta[j]);
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (j = 1; j <= out; ++j) {
      for (k = 0; k <= hid; ++k) {
        float new_dw = ((ETA * output_delta[j] * hidden_units[k]) + (MOMENTUM * hidden_prev_weights[(k * hidden_cols) + j]));
        hidden_weights[(k * hidden_cols) + j] += new_dw;
        hidden_prev_weights[(k * hidden_cols) + j] = new_dw;
      }
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (j = 1; j <= hid; ++j) {
      for (k = 0; k <= in; ++k) {
        float new_dw = ((ETA * hidden_delta[j] * input_units[k]) + (MOMENTUM * input_prev_weights[(k * input_cols) + j]));
        input_weights[(k * input_cols) + j] += new_dw;
        input_prev_weights[(k * input_cols) + j] = new_dw;
      }
    }
  }

  *eo = out_err;
  *eh = hid_err;

}
