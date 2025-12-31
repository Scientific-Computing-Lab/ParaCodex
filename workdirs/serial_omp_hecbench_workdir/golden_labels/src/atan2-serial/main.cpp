





#include <cstdio>
#include <cmath>
#include <limits>
#include <chrono>




template <int DEGREE>
constexpr float approx_atan2f_P(float x);



template <>
constexpr float approx_atan2f_P<3>(float x) {
  return x * (float(-0xf.8eed2p-4) + x * x * float(0x3.1238p-4));
}



template <>
constexpr float approx_atan2f_P<5>(float x) {
  auto z = x * x;
  return x * (float(-0xf.ecfc8p-4) + z * (float(0x4.9e79dp-4) + z * float(-0x1.44f924p-4)));
}



template <>
constexpr float approx_atan2f_P<7>(float x) {
  auto z = x * x;
  return x * (float(-0xf.fcc7ap-4) + z * (float(0x5.23886p-4) + z * (float(-0x2.571968p-4) + z * float(0x9.fb05p-8))));
}



template <>
constexpr float approx_atan2f_P<9>(float x) {
  auto z = x * x;
  return x * (float(-0xf.ff73ep-4) +
              z * (float(0x5.48ee1p-4) +
                   z * (float(-0x2.e1efe8p-4) + z * (float(0x1.5cce54p-4) + z * float(-0x5.56245p-8)))));
}



template <>
constexpr float approx_atan2f_P<11>(float x) {
  auto z = x * x;
  return x * (float(-0xf.ffe82p-4) +
              z * (float(0x5.526c8p-4) +
                   z * (float(-0x3.18bea8p-4) +
                        z * (float(0x1.dce3bcp-4) + z * (float(-0xd.7a64ap-8) + z * float(0x3.000eap-8))))));
}



template <>
constexpr float approx_atan2f_P<13>(float x) {
  auto z = x * x;
  return x * (float(-0xf.fffbep-4) +
              z * (float(0x5.54adp-4) +
                   z * (float(-0x3.2b4df8p-4) +
                        z * (float(0x2.1df79p-4) +
                             z * (float(-0x1.46081p-4) + z * (float(0x8.99028p-8) + z * float(-0x1.be0bc4p-8)))))));
}



template <>
constexpr float approx_atan2f_P<15>(float x) {
  auto z = x * x;
  return x * (float(-0xf.ffff4p-4) +
              z * (float(0x5.552f9p-4 + z * (float(-0x3.30f728p-4) +
                                             z * (float(0x2.39826p-4) +
                                                  z * (float(-0x1.8a880cp-4) +
                                                       z * (float(0xe.484d6p-8) +
                                                            z * (float(-0x5.93d5p-8) + z * float(0x1.0875dcp-8)))))))));
}

template <int DEGREE>
constexpr float unsafe_atan2f_impl(float y, float x) {
  constexpr float pi4f = 3.1415926535897932384626434 / 4;
  constexpr float pi34f = 3.1415926535897932384626434 * 3 / 4;

  auto r = (fabsf(x) - fabsf(y)) / (fabsf(x) + fabsf(y));
  if (x < 0)
    r = -r;

  auto angle = (x >= 0) ? pi4f : pi34f;
  angle += approx_atan2f_P<DEGREE>(r);

  return ((y < 0)) ? -angle : angle;
}

template <int DEGREE>
constexpr float unsafe_atan2f(float y, float x) {
  return unsafe_atan2f_impl<DEGREE>(y, x);
}

template <int DEGREE>
constexpr float safe_atan2f(float y, float x) {
  return unsafe_atan2f_impl<DEGREE>(y, ((y == 0.f) & (x == 0.f)) ? 0.2f : x);
}






template <int DEGREE>
constexpr float approx_atan2i_P(float x);



template <>
constexpr float approx_atan2i_P<3>(float x) {
  auto z = x * x;
  return x * (-664694912.f + z * 131209024.f);
}



template <>
constexpr float approx_atan2i_P<5>(float x) {
  auto z = x * x;
  return x * (-680392064.f + z * (197338400.f + z * (-54233256.f)));
}



template <>
constexpr float approx_atan2i_P<7>(float x) {
  auto z = x * x;
  return x * (-683027840.f + z * (219543904.f + z * (-99981040.f + z * 26649684.f)));
}



template <>
constexpr float approx_atan2i_P<9>(float x) {
  auto z = x * x;
  return x * (-683473920.f + z * (225785056.f + z * (-123151184.f + z * (58210592.f + z * (-14249276.f)))));
}



template <>
constexpr float approx_atan2i_P<11>(float x) {
  auto z = x * x;
  return x *
         (-683549696.f + z * (227369312.f + z * (-132297008.f + z * (79584144.f + z * (-35987016.f + z * 8010488.f)))));
}



template <>
constexpr float approx_atan2i_P<13>(float x) {
  auto z = x * x;
  return x * (-683562624.f +
              z * (227746080.f +
                   z * (-135400128.f + z * (90460848.f + z * (-54431464.f + z * (22973256.f + z * (-4657049.f)))))));
}

template <>
constexpr float approx_atan2i_P<15>(float x) {
  auto z = x * x;
  return x * (-683562624.f +
              z * (227746080.f +
                   z * (-135400128.f + z * (90460848.f + z * (-54431464.f + z * (22973256.f + z * (-4657049.f)))))));
}

template <int DEGREE>
constexpr int unsafe_atan2i_impl(float y, float x) {
  constexpr long long maxint = (long long)(std::numeric_limits<int>::max()) + 1LL;
  constexpr int pi4 = int(maxint / 4LL);
  constexpr int pi34 = int(3LL * maxint / 4LL);

  auto r = (fabsf(x) - fabsf(y)) / (fabsf(x) + fabsf(y));
  if (x < 0)
    r = -r;

  auto angle = (x >= 0) ? pi4 : pi34;
  angle += int(approx_atan2i_P<DEGREE>(r));

  return (y < 0) ? -angle : angle;
}

template <int DEGREE>
constexpr int unsafe_atan2i(float y, float x) {
  return unsafe_atan2i_impl<DEGREE>(y, x);
}



template <int DEGREE>
constexpr float approx_atan2s_P(float x);



template <>
constexpr float approx_atan2s_P<3>(float x) {
  auto z = x * x;
  return x * ((-10142.439453125f) + z * 2002.0908203125f);
}



template <>
constexpr float approx_atan2s_P<5>(float x) {
  auto z = x * x;
  return x * ((-10381.9609375f) + z * ((3011.1513671875f) + z * (-827.538330078125f)));
}



template <>
constexpr float approx_atan2s_P<7>(float x) {
  auto z = x * x;
  return x * ((-10422.177734375f) + z * (3349.97412109375f + z * ((-1525.589599609375f) + z * 406.64190673828125f)));
}



template <>
constexpr float approx_atan2s_P<9>(float x) {
  auto z = x * x;
  return x * ((-10428.984375f) + z * (3445.20654296875f + z * ((-1879.137939453125f) +
                                                               z * (888.22314453125f + z * (-217.42669677734375f)))));
}

template <int DEGREE>
constexpr short unsafe_atan2s_impl(float y, float x) {
  constexpr int maxshort = (int)(std::numeric_limits<short>::max()) + 1;
  constexpr short pi4 = short(maxshort / 4);
  constexpr short pi34 = short(3 * maxshort / 4);

  auto r = (fabsf(x) - fabsf(y)) / (fabsf(x) + fabsf(y));
  if (x < 0)
    r = -r;

  auto angle = (x >= 0) ? pi4 : pi34;
  angle += short(approx_atan2s_P<DEGREE>(r));

  return (y < 0) ? -angle : angle;
}

template <int DEGREE>
constexpr short unsafe_atan2s(float y, float x) {
  return unsafe_atan2s_impl<DEGREE>(y, x);
}


void compute_f (const int n,
                const float *x,
                const float *y,
                      float *r)
{
    for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = safe_atan2f< 3>(vy, vx) +
           safe_atan2f< 5>(vy, vx) +
           safe_atan2f< 7>(vy, vx) +
           safe_atan2f< 9>(vy, vx) +
           safe_atan2f<11>(vy, vx) +
           safe_atan2f<13>(vy, vx) +
           safe_atan2f<15>(vy, vx);
  }
}


void compute_s (const int n,
                const float *x,
                const float *y,
                      short *r)
{
    for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = unsafe_atan2s< 3>(vy, vx) +
           unsafe_atan2s< 5>(vy, vx) +
           unsafe_atan2s< 7>(vy, vx) +
           unsafe_atan2s< 9>(vy, vx);
  }
}


void compute_i (const int n,
                const float *x,
                const float *y,
                      int *r)
{
    for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = unsafe_atan2i< 3>(vy, vx) +
           unsafe_atan2i< 5>(vy, vx) +
           unsafe_atan2i< 7>(vy, vx) +
           unsafe_atan2i< 9>(vy, vx) +
           unsafe_atan2i<11>(vy, vx) +
           unsafe_atan2i<13>(vy, vx) +
           unsafe_atan2i<15>(vy, vx);
  }
}

void reference_f (const int n,
                  const float *x,
                  const float *y,
                        float *r)
{
  for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = safe_atan2f< 3>(vy, vx) +
           safe_atan2f< 5>(vy, vx) +
           safe_atan2f< 7>(vy, vx) +
           safe_atan2f< 9>(vy, vx) +
           safe_atan2f<11>(vy, vx) +
           safe_atan2f<13>(vy, vx) +
           safe_atan2f<15>(vy, vx);
  }
}

void reference_s (const int n,
                  const float *x,
                  const float *y,
                        short *r)
{
  for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = unsafe_atan2s< 3>(vy, vx) +
           unsafe_atan2s< 5>(vy, vx) +
           unsafe_atan2s< 7>(vy, vx) +
           unsafe_atan2s< 9>(vy, vx);
  }
}

void reference_i (const int n,
                  const float *x,
                  const float *y,
                        int *r)
{
  for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = unsafe_atan2i< 3>(vy, vx) +
           unsafe_atan2i< 5>(vy, vx) +
           unsafe_atan2i< 7>(vy, vx) +
           unsafe_atan2i< 9>(vy, vx) +
           unsafe_atan2i<11>(vy, vx) +
           unsafe_atan2i<13>(vy, vx) +
           unsafe_atan2i<15>(vy, vx);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of coordinates> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const size_t input_bytes = sizeof(float) * n;
  const size_t output_float_bytes = sizeof(float) * n;
  const size_t output_int_bytes = sizeof(int) * n;
  const size_t output_short_bytes = sizeof(short) * n;

  float *x = (float*) malloc (input_bytes);
  float *y = (float*) malloc (input_bytes);

  float *hf = (float*) malloc (output_float_bytes);
    int *hi = (int*) malloc (output_int_bytes);
  short *hs = (short*) malloc (output_short_bytes);

  

  float *rf = (float*) malloc (output_float_bytes);
    int *ri = (int*) malloc (output_int_bytes);
  short *rs = (short*) malloc (output_short_bytes);

  srand(123);
  for (int i = 0; i < n; i++) {
    x[i] = rand() / (float)RAND_MAX + 1.57f;
    y[i] = rand() / (float)RAND_MAX + 1.57f;
  }
  
    {
    printf("\n======== output type is f32 ========\n");
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      compute_f(n, y, x, hf);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

    
    reference_f (n, y, x, rf);
    float error = 0;
    for (int i = 0; i < n; i++) {
      if (fabsf(rf[i] - hf[i]) > 1e-3f) {
        error += (ri[i] - hi[i]) * (ri[i] - hi[i]);
      }
    }
    printf("RMSE: %f\n", sqrtf(error / n));

    printf("\n======== output type is i32 ========\n");

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      compute_i(n, y, x, hi);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

    
    reference_i (n, y, x, ri);
    error = 0;
    for (int i = 0; i < n; i++) {
      if (abs(ri[i] - hi[i]) > 0) {
        error += (ri[i] - hi[i]) * (ri[i] - hi[i]);
      }
    }
    printf("RMSE: %f\n", sqrtf(error / n));

    printf("\n======== output type is i16 ========\n");

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      compute_s(n, y, x, hs);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

    
    error = 0;
    reference_s (n, y, x, rs);
    for (int i = 0; i < n; i++) {
      if (abs(rs[i] - hs[i]) > 0) {
        error += (rs[i] - hs[i]) * (rs[i] - hs[i]);
      }
    }
    printf("RMSE: %f\n", sqrtf(error / n));
  }

  free(x);
  free(y);
  free(hf);
  free(hi);
  free(hs);
  free(rf);
  free(ri);
  free(rs);
  return 0;
}