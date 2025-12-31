#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <new>
#include <string>
#include <omp.h>

#include "gate.h"

#define BSIZE 256

#pragma omp declare target

template <class T>
class AvgPoolGrad {
  public:
    void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
      *dx += (scale * dy);
    }
};

template <class T>
class MaxPoolGrad {
  public:
    void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
      *dx += dy * static_cast<T>(x == y);
    }
};

#pragma omp end declare target

#include "reference.h"

template <typename PoolProcess, typename T>
void KernelPool2DGrad(
    const int nthreads,
    const T* __restrict input_data,
    const T* __restrict output_data,
    const T* __restrict output_grad,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int ksize_height,
    const int ksize_width,
    const int stride_height,
    const int stride_width,
    const int padding_height,
    const int padding_width,
    PoolProcess pool_process,
    bool exclusive,
    T* __restrict input_grad,
    bool channel_last = false,
    bool use_gpu = true) {
  if (nthreads == 0) {
    return;
  }

  const int spatial_in = input_height * input_width;
  const int spatial_out = output_height * output_width;
  const int batch = nthreads / (channels * spatial_in);
  const int output_numel = batch * channels * spatial_out;
  (void)output_numel;

  // Ada Lovelace (SM89) likes to see at least 4-8 warps per block; 256 threads
  // per team balances occupancy with register pressure for this compute-heavy kernel.
  const int threads_per_team = 256;
  const int num_team_hint =
      std::max(1, (nthreads + threads_per_team - 1) / threads_per_team);

  // Offload the main gradient accumulation loop to the GPU.
  // Data is expected to be present due to the surrounding target data region.
  // Cache common extents so the device kernels avoid redundant integer math.
  const int output_hw = output_height * output_width;
  const int kernel_area = ksize_height * ksize_width;
  const T inv_kernel_area =
      kernel_area > 0 ? static_cast<T>(1.0) / static_cast<T>(kernel_area)
                      : static_cast<T>(0.0);

  if (channel_last) {
#pragma omp target teams distribute parallel for collapse(4)                 \
    num_teams(num_team_hint) thread_limit(threads_per_team)                  \
        map(present, to : input_data[0:nthreads], output_data[0:output_numel], \
                              output_grad[0:output_numel])                   \
        map(present, tofrom : input_grad[0:nthreads])                        \
            firstprivate(pool_process) if (use_gpu)
    for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
      for (int h = 0; h < input_height; ++h) {
        for (int w = 0; w < input_width; ++w) {
          for (int offsetC = 0; offsetC < channels; ++offsetC) {
            const int index =
                (((batch_idx * input_height) + h) * input_width + w) * channels +
                offsetC;
            const int w_offset = w + padding_width;
            const int h_offset = h + padding_height;

            int phstart = (h_offset < ksize_height)
                              ? 0
                              : (h_offset - ksize_height) / stride_height + 1;
            int pwstart = (w_offset < ksize_width)
                              ? 0
                              : (w_offset - ksize_width) / stride_width + 1;
            const int phend =
                std::min(h_offset / stride_height + 1, output_height);
            const int pwend =
                std::min(w_offset / stride_width + 1, output_width);

            T gradient = static_cast<T>(0.0);
            const T input = input_data[index];

            const int per_batch_stride = output_hw * channels;
            const int output_stride = batch_idx * per_batch_stride;
            const T* __restrict output_data_t = output_data + output_stride;
            const T* __restrict output_grad_t = output_grad + output_stride;
            const int row_stride = output_width * channels;
            const int channel_stride = channels;

            if (exclusive) {
              for (int ph = phstart; ph < phend; ++ph) {
                const int hstart_raw = ph * stride_height - padding_height;
                int hend = hstart_raw + ksize_height;
                if (hend > input_height) {
                  hend = input_height;
                }
                int hstart = hstart_raw;
                if (hstart < 0) {
                  hstart = 0;
                }
                const int effective_h = hend - hstart;

                int output_sub_idx =
                    ph * row_stride + pwstart * channel_stride + offsetC;
                for (int pw = pwstart; pw < pwend; ++pw) {
                  const int wstart_raw = pw * stride_width - padding_width;
                  int wend = wstart_raw + ksize_width;
                  if (wend > input_width) {
                    wend = input_width;
                  }
                  int wstart = wstart_raw;
                  if (wstart < 0) {
                    wstart = 0;
                  }
                  const int effective_w = wend - wstart;
                  const int pool_size = effective_h * effective_w;
                  const T scale = pool_size > 0
                                      ? static_cast<T>(1.0) /
                                            static_cast<T>(pool_size)
                                      : static_cast<T>(0.0);
                  pool_process.compute(input, output_data_t[output_sub_idx],
                                       output_grad_t[output_sub_idx], scale,
                                       &gradient);
                  output_sub_idx += channel_stride;
                }
              }
            } else {
              for (int ph = phstart; ph < phend; ++ph) {
                int output_sub_idx =
                    ph * row_stride + pwstart * channel_stride + offsetC;
                for (int pw = pwstart; pw < pwend; ++pw) {
                  pool_process.compute(input, output_data_t[output_sub_idx],
                                       output_grad_t[output_sub_idx],
                                       inv_kernel_area, &gradient);
                  output_sub_idx += channel_stride;
                }
              }
            }
            input_grad[index] = gradient;
          }
        }
      }
    }
  } else {
#pragma omp target teams distribute parallel for collapse(3)                 \
    num_teams(num_team_hint) thread_limit(threads_per_team)                  \
        map(present, to : input_data[0:nthreads], output_data[0:output_numel], \
                              output_grad[0:output_numel])                   \
        map(present, tofrom : input_grad[0:nthreads])                        \
            firstprivate(pool_process) if (use_gpu)
    for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
      for (int offsetC = 0; offsetC < channels; ++offsetC) {
        for (int spatial_idx = 0; spatial_idx < spatial_in; ++spatial_idx) {
          const int h = spatial_idx / input_width;
          const int w = spatial_idx % input_width;
          const int index =
              (((batch_idx * channels) + offsetC) * input_height + h) *
                  input_width +
              w;
          const int w_offset = w + padding_width;
          const int h_offset = h + padding_height;

          int phstart = (h_offset < ksize_height)
                            ? 0
                            : (h_offset - ksize_height) / stride_height + 1;
          int pwstart = (w_offset < ksize_width)
                            ? 0
                            : (w_offset - ksize_width) / stride_width + 1;
          const int phend =
              std::min(h_offset / stride_height + 1, output_height);
          const int pwend =
              std::min(w_offset / stride_width + 1, output_width);

          T gradient = static_cast<T>(0.0);
          const T input = input_data[index];

          const int per_batch_stride = output_hw * channels;
          const int per_channel_stride = output_hw;
          const int output_stride =
              batch_idx * per_batch_stride + offsetC * per_channel_stride;

          const T* __restrict output_data_t = output_data + output_stride;
          const T* __restrict output_grad_t = output_grad + output_stride;
          const int row_stride = output_width;

          if (exclusive) {
            for (int ph = phstart; ph < phend; ++ph) {
              const int hstart_raw = ph * stride_height - padding_height;
              int hend = hstart_raw + ksize_height;
              if (hend > input_height) {
                hend = input_height;
              }
              int hstart = hstart_raw;
              if (hstart < 0) {
                hstart = 0;
              }
              const int effective_h = hend - hstart;

              int output_sub_idx = ph * row_stride + pwstart;
              for (int pw = pwstart; pw < pwend; ++pw) {
                const int wstart_raw = pw * stride_width - padding_width;
                int wend = wstart_raw + ksize_width;
                if (wend > input_width) {
                  wend = input_width;
                }
                int wstart = wstart_raw;
                if (wstart < 0) {
                  wstart = 0;
                }
                const int effective_w = wend - wstart;
                const int pool_size = effective_h * effective_w;
                const T scale = pool_size > 0
                                    ? static_cast<T>(1.0) /
                                          static_cast<T>(pool_size)
                                    : static_cast<T>(0.0);
                pool_process.compute(input, output_data_t[output_sub_idx],
                                     output_grad_t[output_sub_idx], scale,
                                     &gradient);
                ++output_sub_idx;
              }
            }
          } else {
            for (int ph = phstart; ph < phend; ++ph) {
              int output_sub_idx = ph * row_stride + pwstart;
              for (int pw = pwstart; pw < pwend; ++pw) {
                pool_process.compute(input, output_data_t[output_sub_idx],
                                     output_grad_t[output_sub_idx],
                                     inv_kernel_area, &gradient);
                ++output_sub_idx;
              }
            }
          }
          input_grad[index] = gradient;
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 8) {
    printf("Usage: %s <batch> <input channels> <input height> ", argv[0]);
    printf("<input width> <output height> <output width> <repeat>\n");
    return 1;
  }

  const int batch_size = atoi(argv[1]);
  const int input_channels = atoi(argv[2]);
  const int input_height = atoi(argv[3]);
  const int input_width = atoi(argv[4]);

  const int output_height = atoi(argv[5]);
  const int output_width = atoi(argv[6]);

  const int repeat = atoi(argv[7]);

  const int input_numel =
      batch_size * input_channels * input_height * input_width;
  const int output_numel =
      batch_size * input_channels * output_height * output_width;

  const int ksize_height = 11;
  const int ksize_width = 11;
  const int stride_height = 4;
  const int stride_width = 4;
  const int padding_height = 1;
  const int padding_width = 1;
  const bool exclusive = true;
  const std::string data_format = "NCHW";
  const bool channel_last = (data_format == "NHWC");

  int nthreads = batch_size * input_channels * input_height * input_width;

  AvgPoolGrad<float> pool_process;

  float* input = new float[input_numel];
  float* output = new float[output_numel];
  float* output_grad = new float[output_numel];
  float* input_grad = new float[input_numel];
  float* input_grad_ref = new float[input_numel];

  const int num_devices = omp_get_num_devices();
  const bool use_gpu = num_devices > 0;

  srand(123);
  for (int i = 0; i < input_numel; ++i) {
    input[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    input_grad[i] = 0.f;
  }

  for (int i = 0; i < output_numel; ++i) {
    output[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    output_grad[i] = input_width * input_height;
  }

  {
    auto start = std::chrono::steady_clock::now();

    // Keep tensors resident on the device to amortize host-device transfers.
    #pragma omp target data                                               \
        map(to : input[0:input_numel], output[0:output_numel],            \
                 output_grad[0:output_numel])                             \
        map(from : input_grad[0:input_numel]) if (use_gpu)
    {
      for (int i = 0; i < repeat; i++) {
        KernelPool2DGrad<AvgPoolGrad<float>, float>(
            nthreads, input, output, output_grad, input_channels,
            input_height, input_width, output_height, output_width,
            ksize_height, ksize_width, stride_height, stride_width,
            padding_height, padding_width, pool_process, exclusive, input_grad,
            channel_last, use_gpu);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("Average kernel execution time: %f (s)\n",
           (time * 1e-9f) / repeat);
  }

  reference<AvgPoolGrad<float>, float>(
      nthreads, input, output, output_grad, input_channels, input_height,
      input_width, output_height, output_width, ksize_height, ksize_width,
      stride_height, stride_width, padding_height, padding_width, pool_process,
      exclusive, input_grad_ref, channel_last);

  bool ok = true;
  for (int i = 0; i < input_numel; ++i) {
    if (fabsf(input_grad[i] - input_grad_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  GATE_CHECKSUM_BYTES("input_grad", input_grad, static_cast<size_t>(input_numel) * sizeof(float));
  GATE_CHECKSUM_BYTES("input_grad_ref", input_grad_ref, static_cast<size_t>(input_numel) * sizeof(float));

  delete[] input;
  delete[] output;
  delete[] input_grad;
  delete[] input_grad_ref;
  delete[] output_grad;
  return 0;
}
