#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "backprop.h"
#include "gate.h"

////////////////////////////////////////////////////////////////////////////////

extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

static void pack_matrix(float **src, float *dst, int rows, int cols)
{
  int i;

  for (i = 0; i < rows; ++i) {
    memcpy(dst + ((size_t)i * (size_t)cols), src[i], (size_t)cols * sizeof(float));
  }
}

static void unpack_matrix(const float *src, float **dst, int rows, int cols)
{
  int i;

  for (i = 0; i < rows; ++i) {
    memcpy(dst[i], src + ((size_t)i * (size_t)cols), (size_t)cols * sizeof(float));
  }
}

static void bpnn_layerforward_device(float *restrict l1, float *restrict l2, const float *restrict conn, int n1, int n2, int conn_cols)
{
  int j, k;

#pragma omp target teams distribute parallel for is_device_ptr(l1, l2, conn) private(k)
  for (j = 1; j <= n2; ++j) {
    float sum = 0.0f;
    for (k = 0; k <= n1; ++k) {
      sum += conn[((size_t)k * (size_t)conn_cols) + (size_t)j] * l1[k];
    }
    l2[j] = 1.0f / (1.0f + expf(-sum));
  }
}

static void bpnn_output_error_device(float *restrict delta, const float *restrict target, const float *restrict output, int nj, float *err)
{
  int j;
  float errsum = 0.0f;

#pragma omp target teams distribute parallel for is_device_ptr(delta, target, output) reduction(+:errsum)
  for (j = 1; j <= nj; ++j) {
    float o = output[j];
    float t = target[j];
    delta[j] = o * (1.0f - o) * (t - o);
    errsum += fabsf(delta[j]);
  }

  *err = errsum;
}

static void bpnn_hidden_error_device(float *restrict delta_h, int nh, const float *restrict delta_o, int no, const float *restrict who, const float *restrict hidden, int who_cols, float *err)
{
  int j, k;
  float errsum = 0.0f;

#pragma omp target teams distribute parallel for is_device_ptr(delta_h, delta_o, who, hidden) reduction(+:errsum) private(k)
  for (j = 1; j <= nh; ++j) {
    float h = hidden[j];
    float sum = 0.0f;
    for (k = 1; k <= no; ++k) {
      sum += delta_o[k] * who[((size_t)j * (size_t)who_cols) + (size_t)k];
    }
    delta_h[j] = h * (1.0f - h) * sum;
    errsum += fabsf(delta_h[j]);
  }

  *err = errsum;
}

static void bpnn_adjust_weights_device(float *restrict delta, int ndelta, const float *restrict ly, int nly, float *restrict w, float *restrict oldw, int w_cols)
{
  int j, k;
  float new_dw;

#pragma omp target teams distribute parallel for collapse(2) is_device_ptr(delta, ly, w, oldw) private(new_dw)
  for (j = 1; j <= ndelta; ++j) {
    for (k = 0; k <= nly; ++k) {
      size_t idx = ((size_t)k * (size_t)w_cols) + (size_t)j;
      new_dw = (ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[idx]);
      w[idx] += new_dw;
      oldw[idx] = new_dw;
    }
  }
}

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
  int input_rows, input_cols, hidden_rows, hidden_cols;
  size_t input_weight_count, hidden_weight_count;
  float out_err, hid_err;
  float *input_weights_flat;
  float *hidden_weights_flat;
  float *input_prev_weights_flat;
  float *hidden_prev_weights_flat;
  float *d_input_units = NULL;
  float *d_hidden_units = NULL;
  float *d_output_units = NULL;
  float *d_hidden_delta = NULL;
  float *d_output_delta = NULL;
  float *d_target = NULL;
  float *d_input_weights_flat = NULL;
  float *d_hidden_weights_flat = NULL;
  float *d_input_prev_weights_flat = NULL;
  float *d_hidden_prev_weights_flat = NULL;
  int device;
  int host_device;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  input_rows = in + 1;
  input_cols = hid + 1;
  hidden_rows = hid + 1;
  hidden_cols = out + 1;
  input_weight_count = (size_t)input_rows * (size_t)input_cols;
  hidden_weight_count = (size_t)hidden_rows * (size_t)hidden_cols;

  input_weights_flat = (float *) malloc(input_weight_count * sizeof(float));
  hidden_weights_flat = (float *) malloc(hidden_weight_count * sizeof(float));
  input_prev_weights_flat = (float *) malloc(input_weight_count * sizeof(float));
  hidden_prev_weights_flat = (float *) malloc(hidden_weight_count * sizeof(float));

  if (input_weights_flat == NULL || hidden_weights_flat == NULL ||
      input_prev_weights_flat == NULL || hidden_prev_weights_flat == NULL) {
    fprintf(stderr, "Unable to allocate packed weight buffers\n");
    free(input_weights_flat);
    free(hidden_weights_flat);
    free(input_prev_weights_flat);
    free(hidden_prev_weights_flat);
    return;
  }

  device = omp_get_default_device();
  host_device = omp_get_initial_device();

  d_input_units = (float *) omp_target_alloc((size_t)input_rows * sizeof(float), device);
  d_hidden_units = (float *) omp_target_alloc((size_t)hidden_rows * sizeof(float), device);
  d_output_units = (float *) omp_target_alloc((size_t)(out + 1) * sizeof(float), device);
  d_hidden_delta = (float *) omp_target_alloc((size_t)hidden_rows * sizeof(float), device);
  d_output_delta = (float *) omp_target_alloc((size_t)(out + 1) * sizeof(float), device);
  d_target = (float *) omp_target_alloc((size_t)(out + 1) * sizeof(float), device);
  d_input_weights_flat = (float *) omp_target_alloc(input_weight_count * sizeof(float), device);
  d_hidden_weights_flat = (float *) omp_target_alloc(hidden_weight_count * sizeof(float), device);
  d_input_prev_weights_flat = (float *) omp_target_alloc(input_weight_count * sizeof(float), device);
  d_hidden_prev_weights_flat = (float *) omp_target_alloc(hidden_weight_count * sizeof(float), device);

  if (d_input_units == NULL || d_hidden_units == NULL || d_output_units == NULL ||
      d_hidden_delta == NULL || d_output_delta == NULL || d_target == NULL ||
      d_input_weights_flat == NULL || d_hidden_weights_flat == NULL ||
      d_input_prev_weights_flat == NULL || d_hidden_prev_weights_flat == NULL) {
    fprintf(stderr, "Unable to allocate device buffers\n");
    goto cleanup;
  }

  net->input_units[0] = 1.0f;
  net->hidden_units[0] = 1.0f;

  pack_matrix(net->input_weights, input_weights_flat, input_rows, input_cols);
  pack_matrix(net->hidden_weights, hidden_weights_flat, hidden_rows, hidden_cols);
  pack_matrix(net->input_prev_weights, input_prev_weights_flat, input_rows, input_cols);
  pack_matrix(net->hidden_prev_weights, hidden_prev_weights_flat, hidden_rows, hidden_cols);

  printf("Performing CPU computation\n");

  (void) omp_target_memcpy(d_input_units, net->input_units, (size_t)input_rows * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_hidden_units, net->hidden_units, (size_t)hidden_rows * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_output_units, net->output_units, (size_t)(out + 1) * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_hidden_delta, net->hidden_delta, (size_t)hidden_rows * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_output_delta, net->output_delta, (size_t)(out + 1) * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_target, net->target, (size_t)(out + 1) * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_input_weights_flat, input_weights_flat, input_weight_count * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_hidden_weights_flat, hidden_weights_flat, hidden_weight_count * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_input_prev_weights_flat, input_prev_weights_flat, input_weight_count * sizeof(float), 0, 0, device, host_device);
  (void) omp_target_memcpy(d_hidden_prev_weights_flat, hidden_prev_weights_flat, hidden_weight_count * sizeof(float), 0, 0, device, host_device);

  bpnn_layerforward_device(d_input_units, d_hidden_units, d_input_weights_flat, in, hid, input_cols);
  bpnn_layerforward_device(d_hidden_units, d_output_units, d_hidden_weights_flat, hid, out, hidden_cols);
  bpnn_output_error_device(d_output_delta, d_target, d_output_units, out, &out_err);
  bpnn_hidden_error_device(d_hidden_delta, hid, d_output_delta, out, d_hidden_weights_flat, d_hidden_units, hidden_cols, &hid_err);
  bpnn_adjust_weights_device(d_output_delta, out, d_hidden_units, hid, d_hidden_weights_flat, d_hidden_prev_weights_flat, hidden_cols);
  bpnn_adjust_weights_device(d_hidden_delta, hid, d_input_units, in, d_input_weights_flat, d_input_prev_weights_flat, input_cols);

  (void) omp_target_memcpy(net->hidden_units, d_hidden_units, (size_t)hidden_rows * sizeof(float), 0, 0, host_device, device);
  (void) omp_target_memcpy(net->output_units, d_output_units, (size_t)(out + 1) * sizeof(float), 0, 0, host_device, device);
  (void) omp_target_memcpy(net->hidden_delta, d_hidden_delta, (size_t)hidden_rows * sizeof(float), 0, 0, host_device, device);
  (void) omp_target_memcpy(net->output_delta, d_output_delta, (size_t)(out + 1) * sizeof(float), 0, 0, host_device, device);
  (void) omp_target_memcpy(input_weights_flat, d_input_weights_flat, input_weight_count * sizeof(float), 0, 0, host_device, device);
  (void) omp_target_memcpy(hidden_weights_flat, d_hidden_weights_flat, hidden_weight_count * sizeof(float), 0, 0, host_device, device);
  (void) omp_target_memcpy(input_prev_weights_flat, d_input_prev_weights_flat, input_weight_count * sizeof(float), 0, 0, host_device, device);
  (void) omp_target_memcpy(hidden_prev_weights_flat, d_hidden_prev_weights_flat, hidden_weight_count * sizeof(float), 0, 0, host_device, device);

  unpack_matrix(input_weights_flat, net->input_weights, input_rows, input_cols);
  unpack_matrix(hidden_weights_flat, net->hidden_weights, hidden_rows, hidden_cols);
  unpack_matrix(input_prev_weights_flat, net->input_prev_weights, input_rows, input_cols);
  unpack_matrix(hidden_prev_weights_flat, net->hidden_prev_weights, hidden_rows, hidden_cols);

#ifdef GATE_VERIFY
  GATE_CHECKSUM_BYTES("input_weights", input_weights_flat, input_weight_count * sizeof(float));
  GATE_STATS_F32("hidden_weights", hidden_weights_flat, hidden_weight_count);
  GATE_CHECKSUM_BYTES("input_prev_weights", input_prev_weights_flat, input_weight_count * sizeof(float));
  GATE_STATS_F32("hidden_prev_weights", hidden_prev_weights_flat, hidden_weight_count);
#endif

cleanup:
  if (d_input_units != NULL) omp_target_free(d_input_units, device);
  if (d_hidden_units != NULL) omp_target_free(d_hidden_units, device);
  if (d_output_units != NULL) omp_target_free(d_output_units, device);
  if (d_hidden_delta != NULL) omp_target_free(d_hidden_delta, device);
  if (d_output_delta != NULL) omp_target_free(d_output_delta, device);
  if (d_target != NULL) omp_target_free(d_target, device);
  if (d_input_weights_flat != NULL) omp_target_free(d_input_weights_flat, device);
  if (d_hidden_weights_flat != NULL) omp_target_free(d_hidden_weights_flat, device);
  if (d_input_prev_weights_flat != NULL) omp_target_free(d_input_prev_weights_flat, device);
  if (d_hidden_prev_weights_flat != NULL) omp_target_free(d_hidden_prev_weights_flat, device);

  free(input_weights_flat);
  free(hidden_weights_flat);
  free(input_prev_weights_flat);
  free(hidden_prev_weights_flat);

  *eo = out_err;
  *eh = hid_err;
}
