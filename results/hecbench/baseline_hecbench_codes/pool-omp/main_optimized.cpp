#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#include <omp.h>

#include "gate.h"

constexpr int THREADS_PER_TEAM = 256;

#pragma omp declare target
template <typename T>
constexpr T device_min(const T a, const T b) noexcept {
  return (a < b) ? a : b;
}

template <typename T>
constexpr T device_max(const T a, const T b) noexcept {
  return (a > b) ? a : b;
}

template <class T>
class AvgPoolGrad {
 public:
  void compute(const T&, const T&, const T& dy, T scale, T* dx) const noexcept {
    *dx += scale * dy;
  }
};

template <class T>
class MaxPoolGrad {
 public:
  void compute(const T& x, const T& y, const T& dy, T scale, T* dx) const noexcept {
    (void)scale;
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
    const bool exclusive,
    T* __restrict input_grad,
    const bool channel_last = false) {
  if (nthreads <= 0) {
    return;
  }

  const std::int64_t spatial_hw =
      static_cast<std::int64_t>(input_height) * static_cast<std::int64_t>(input_width);
  const std::int64_t denom =
      static_cast<std::int64_t>(channels) * (spatial_hw == 0 ? 1 : spatial_hw);
  const int batch_size =
      denom == 0 ? 0 : static_cast<int>(static_cast<std::int64_t>(nthreads) / denom);

  const int input_hw = input_height * input_width;
  const int output_hw = output_height * output_width;
  const int channel_stride = input_hw;
  const int batch_stride = channels * channel_stride;
  const int batch_stride_nhwc = input_height * input_width * channels;
  const T full_pool_inv = static_cast<T>(1.f / (ksize_height * ksize_width));
  const std::size_t input_count = static_cast<std::size_t>(nthreads);
  const std::size_t output_count =
      static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(channels) *
      static_cast<std::size_t>(output_height) * static_cast<std::size_t>(output_width);
  (void)input_count;
  (void)output_count;

  const int team_size = THREADS_PER_TEAM;
  const int computed_teams = (nthreads + team_size - 1) / team_size;
  const int num_teams = computed_teams > 0 ? computed_teams : 1;
  (void)num_teams;

#pragma omp target teams distribute parallel for \
    num_teams(num_teams) thread_limit(team_size) schedule(static, 1) \
    map(present: input_data[0:input_count], \
                 output_data[0:output_count], \
                 output_grad[0:output_count], \
                 input_grad[0:input_count]) \
    firstprivate(pool_process)
  for (int index = 0; index < nthreads; ++index) {
    int batch_idx = 0;
    int offsetC = 0;
    int h_idx = 0;
    int w_idx = 0;

    if (!channel_last) {
      int remaining = index;
      batch_idx = remaining / batch_stride;
      remaining -= batch_idx * batch_stride;
      offsetC = remaining / channel_stride;
      remaining -= offsetC * channel_stride;
      h_idx = remaining / input_width;
      w_idx = remaining - h_idx * input_width;
    } else {
      int remaining = index;
      batch_idx = remaining / batch_stride_nhwc;
      remaining -= batch_idx * batch_stride_nhwc;
      h_idx = remaining / (input_width * channels);
      remaining -= h_idx * input_width * channels;
      w_idx = remaining / channels;
      offsetC = remaining - w_idx * channels;
    }

    const int w_offset = w_idx + padding_width;
    const int h_offset = h_idx + padding_height;

    const int phstart = (h_offset < ksize_height)
                            ? 0
                            : (h_offset - ksize_height) / stride_height + 1;
    const int pwstart = (w_offset < ksize_width)
                            ? 0
                            : (w_offset - ksize_width) / stride_width + 1;
    const int phend = device_min(h_offset / stride_height + 1, output_height);
    const int pwend = device_min(w_offset / stride_width + 1, output_width);

    if (phstart >= phend || pwstart >= pwend) {
      input_grad[index] = static_cast<T>(0);
      continue;
    }

    T gradient = static_cast<T>(0);
    const T input_value = input_data[index];

    int output_stride = 0;
    if (!channel_last) {
      output_stride = (batch_idx * channels + offsetC) * output_hw;
    } else {
      output_stride = batch_idx * output_hw * channels;
    }

    const T* __restrict output_data_t = output_data + output_stride;
    const T* __restrict output_grad_t = output_grad + output_stride;

    for (int ph = phstart; ph < phend; ++ph) {
      const int hstart_unclamped = ph * stride_height - padding_height;
      const int hstart = device_max(hstart_unclamped, 0);
      const int hend = device_min(hstart_unclamped + ksize_height, input_height);
      const int effective_h = hend - hstart;

      for (int pw = pwstart; pw < pwend; ++pw) {
        const int wstart_unclamped = pw * stride_width - padding_width;
        const int wstart = device_max(wstart_unclamped, 0);
        const int wend = device_min(wstart_unclamped + ksize_width, input_width);
        const int effective_w = wend - wstart;

        const T scale = exclusive
                            ? static_cast<T>(1.f / (effective_h * effective_w))
                            : full_pool_inv;

        const int output_sub_idx =
            channel_last ? (ph * output_width + pw) * channels + offsetC
                         : ph * output_width + pw;

        pool_process.compute(input_value,
                             output_data_t[output_sub_idx],
                             output_grad_t[output_sub_idx],
                             scale,
                             &gradient);
      }
    }

    input_grad[index] = gradient;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 8) {
    std::printf("Usage: %s <batch> <input channels> <input height> ", argv[0]);
    std::printf("<input width> <output height> <output width> <repeat>\n");
    return 1;
  }

  const int batch_size = std::atoi(argv[1]);
  const int input_channels = std::atoi(argv[2]);
  const int input_height = std::atoi(argv[3]);
  const int input_width = std::atoi(argv[4]);
  const int output_height = std::atoi(argv[5]);
  const int output_width = std::atoi(argv[6]);
  const int repeat = std::atoi(argv[7]);

  if (batch_size <= 0 || input_channels <= 0 || input_height <= 0 ||
      input_width <= 0 || output_height <= 0 || output_width <= 0) {
    std::fprintf(stderr, "All tensor dimensions must be positive.\n");
    return 1;
  }

  if (repeat <= 0) {
    std::fprintf(stderr, "Repeat count must be positive.\n");
    return 1;
  }

  const int input_numel = batch_size * input_channels * input_height * input_width;
  const int output_numel = batch_size * input_channels * output_height * output_width;

  const int ksize_height = 11;
  const int ksize_width = 11;
  const int stride_height = 4;
  const int stride_width = 4;
  const int padding_height = 1;
  const int padding_width = 1;
  const bool exclusive = true;
  const bool channel_last = false;  // Data format fixed to NCHW.

  const int nthreads = input_numel;

  AvgPoolGrad<float> pool_process;

  float* input = new float[input_numel];
  float* output = new float[output_numel];
  float* output_grad = new float[output_numel];
  float* input_grad = new float[input_numel];
  float* input_grad_ref = new float[input_numel];

  std::srand(123);
  for (int i = 0; i < input_numel; ++i) {
    input[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }

  for (int i = 0; i < output_numel; ++i) {
    output[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    output_grad[i] = static_cast<float>(input_width * input_height);
  }

  long long compute_time_ns = 0;

#pragma omp target data map(to : input[0:input_numel], \
                                output[0:output_numel], \
                                output_grad[0:output_numel]) \
    map(from : input_grad[0:input_numel])
  {
    const auto compute_start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < repeat; ++iter) {
      KernelPool2DGrad<AvgPoolGrad<float>, float>(
          nthreads,
          input,
          output,
          output_grad,
          input_channels,
          input_height,
          input_width,
          output_height,
          output_width,
          ksize_height,
          ksize_width,
          stride_height,
          stride_width,
          padding_height,
          padding_width,
          pool_process,
          exclusive,
          input_grad,
          channel_last);
    }
    const auto compute_end = std::chrono::steady_clock::now();
    compute_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(compute_end - compute_start)
            .count();
  }

  std::printf("Average kernel execution time: %f (s)\n",
              (compute_time_ns * 1e-9f) / repeat);

  reference<AvgPoolGrad<float>, float>(
      nthreads,
      input,
      output,
      output_grad,
      input_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      ksize_height,
      ksize_width,
      stride_height,
      stride_width,
      padding_height,
      padding_width,
      pool_process,
      exclusive,
      input_grad_ref,
      channel_last);

  bool ok = true;
  for (int i = 0; i < input_numel; ++i) {
    if (std::fabs(input_grad[i] - input_grad_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  GATE_CHECKSUM_BYTES("input_grad", input_grad, static_cast<size_t>(input_numel) * sizeof(float));

  delete[] input;
  delete[] output;
  delete[] input_grad;
  delete[] input_grad_ref;
  delete[] output_grad;

  return ok ? 0 : 1;
}
