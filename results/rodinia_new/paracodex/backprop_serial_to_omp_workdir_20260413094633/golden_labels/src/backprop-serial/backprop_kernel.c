#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"
#include "gate.h"

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

#ifdef GATE_VERIFY
static void pack_matrix(float **src, float *dst, int rows, int cols)
{
  int i;

  for (i = 0; i < rows; ++i) {
    memcpy(dst + ((size_t)i * (size_t)cols), src[i], (size_t)cols * sizeof(float));
  }
}
#endif

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
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#ifdef GATE_VERIFY
  {
    size_t input_count = (size_t)(in + 1) * (size_t)(hid + 1);
    size_t hidden_count = (size_t)(hid + 1) * (size_t)(out + 1);
    float *input_weights_flat = (float *) malloc(input_count * sizeof(float));
    float *hidden_weights_flat = (float *) malloc(hidden_count * sizeof(float));
    float *input_prev_weights_flat = (float *) malloc(input_count * sizeof(float));
    float *hidden_prev_weights_flat = (float *) malloc(hidden_count * sizeof(float));

    if (input_weights_flat != NULL && hidden_weights_flat != NULL &&
        input_prev_weights_flat != NULL && hidden_prev_weights_flat != NULL) {
      pack_matrix(net->input_weights, input_weights_flat, in + 1, hid + 1);
      pack_matrix(net->hidden_weights, hidden_weights_flat, hid + 1, out + 1);
      pack_matrix(net->input_prev_weights, input_prev_weights_flat, in + 1, hid + 1);
      pack_matrix(net->hidden_prev_weights, hidden_prev_weights_flat, hid + 1, out + 1);

      GATE_CHECKSUM_BYTES("input_weights", input_weights_flat, input_count * sizeof(float));
      GATE_STATS_F32("hidden_weights", hidden_weights_flat, hidden_count);
      GATE_CHECKSUM_BYTES("input_prev_weights", input_prev_weights_flat, input_count * sizeof(float));
      GATE_STATS_F32("hidden_prev_weights", hidden_prev_weights_flat, hidden_count);
    }

    free(input_weights_flat);
    free(hidden_weights_flat);
    free(input_prev_weights_flat);
    free(hidden_prev_weights_flat);
  }
#endif

}
