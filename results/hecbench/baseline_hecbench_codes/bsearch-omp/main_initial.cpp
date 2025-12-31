#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>

#include "gate.h"

#ifndef Real_t
#define Real_t float
#endif

namespace {
constexpr int kTeamSize = 256;

inline bool has_target_device() {
  return omp_get_num_devices() > 0;
}

inline int compute_num_teams(size_t workload) {
  if (workload == 0) {
    return 1;
  }

  long long raw = static_cast<long long>((workload + kTeamSize - 1) / kTeamSize);
  int max_teams = omp_get_max_teams();
  if (max_teams > 0 && raw > max_teams) {
    raw = max_teams;
  }
  return raw > 0 ? static_cast<int>(raw) : 1;
}

inline unsigned bit_length(size_t value) {
  unsigned bits = 0;
  while (value >> bits) {
    ++bits;
  }
  return bits;
}
}  // namespace

template <typename T>
void bs(const size_t aSize,
        const size_t zSize,
        const T *__restrict__ acc_a,
        const T *__restrict__ acc_z,
        size_t *__restrict__ acc_r,
        const size_t n,
        const int repeat) {
  long long exec_ns = 0;

  if (has_target_device()) {
    const int numTeams = compute_num_teams(zSize);
    #pragma omp target data map(to : acc_a[0:aSize], acc_z[0:zSize]) map(from : acc_r[0:zSize])
    {
      const auto start = std::chrono::steady_clock::now();
      for (int iter = 0; iter < repeat; ++iter) {
        #pragma omp target teams distribute parallel for num_teams(numTeams) thread_limit(kTeamSize) schedule(static, 1)
        for (size_t idx = 0; idx < zSize; ++idx) {
          const T z = acc_z[idx];
          size_t low = 0;
          size_t high = n;
          while (high - low > 1) {
            const size_t mid = low + ((high - low) >> 1);
            if (z < acc_a[mid]) {
              high = mid;
            } else {
              low = mid;
            }
          }
          acc_r[idx] = low;
        }
      }
      const auto end = std::chrono::steady_clock::now();
      exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
  } else {
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < repeat; ++iter) {
      for (size_t idx = 0; idx < zSize; ++idx) {
        const T z = acc_z[idx];
        size_t low = 0;
        size_t high = n;
        while (high - low > 1) {
          const size_t mid = low + ((high - low) >> 1);
          if (z < acc_a[mid]) {
            high = mid;
          } else {
            low = mid;
          }
        }
        acc_r[idx] = low;
      }
    }
    const auto end = std::chrono::steady_clock::now();
    exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  std::cout << "Average device execution time (bs1) "
            << (exec_ns * 1e-9f) / static_cast<float>(repeat) << " (s)\n";
}

template <typename T>
void bs2(const size_t aSize,
         const size_t zSize,
         const T *__restrict__ acc_a,
         const T *__restrict__ acc_z,
         size_t *__restrict__ acc_r,
         const size_t n,
         const int repeat) {
  const unsigned nbits = bit_length(n);
  size_t initial_pivot = 0;
  if (nbits > 0) {
    initial_pivot = size_t{1} << (nbits - 1);
    if (initial_pivot > n) {
      initial_pivot = n;
    }
  }

  long long exec_ns = 0;

  if (has_target_device()) {
    const int numTeams = compute_num_teams(zSize);
    #pragma omp target data map(to : acc_a[0:aSize], acc_z[0:zSize]) map(from : acc_r[0:zSize])
    {
      const auto start = std::chrono::steady_clock::now();
      for (int iter = 0; iter < repeat; ++iter) {
        #pragma omp target teams distribute parallel for num_teams(numTeams) thread_limit(kTeamSize) schedule(static, 1) firstprivate(initial_pivot)
        for (size_t idx = 0; idx < zSize; ++idx) {
          const T z = acc_z[idx];
          size_t k = initial_pivot;
          size_t local_idx = (k < aSize && acc_a[k] <= z) ? k : 0;
          size_t bit = k;
          while (bit >>= 1) {
            const size_t candidate = local_idx | bit;
            if (candidate < n && z >= acc_a[candidate]) {
              local_idx = candidate;
            }
          }
          acc_r[idx] = local_idx;
        }
      }
      const auto end = std::chrono::steady_clock::now();
      exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
  } else {
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < repeat; ++iter) {
      for (size_t idx = 0; idx < zSize; ++idx) {
        const T z = acc_z[idx];
        size_t k = initial_pivot;
        size_t local_idx = (k < aSize && acc_a[k] <= z) ? k : 0;
        size_t bit = k;
        while (bit >>= 1) {
          const size_t candidate = local_idx | bit;
          if (candidate < n && z >= acc_a[candidate]) {
            local_idx = candidate;
          }
        }
        acc_r[idx] = local_idx;
      }
    }
    const auto end = std::chrono::steady_clock::now();
    exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  std::cout << "Average device execution time (bs2) "
            << (exec_ns * 1e-9f) / static_cast<float>(repeat) << " (s)\n";
}

template <typename T>
void bs3(const size_t aSize,
         const size_t zSize,
         const T *__restrict__ acc_a,
         const T *__restrict__ acc_z,
         size_t *__restrict__ acc_r,
         const size_t n,
         const int repeat) {
  const unsigned nbits = bit_length(n);
  size_t initial_pivot = 0;
  if (nbits > 0) {
    initial_pivot = size_t{1} << (nbits - 1);
    if (initial_pivot > n) {
      initial_pivot = n;
    }
  }

  long long exec_ns = 0;

  if (has_target_device()) {
    const int numTeams = compute_num_teams(zSize);
    #pragma omp target data map(to : acc_a[0:aSize], acc_z[0:zSize]) map(from : acc_r[0:zSize])
    {
      const auto start = std::chrono::steady_clock::now();
      for (int iter = 0; iter < repeat; ++iter) {
        #pragma omp target teams distribute parallel for num_teams(numTeams) thread_limit(kTeamSize) schedule(static, 1) firstprivate(initial_pivot)
        for (size_t idx = 0; idx < zSize; ++idx) {
          const T z = acc_z[idx];
          size_t bit = initial_pivot;
          size_t local_idx = (bit < aSize && acc_a[bit] <= z) ? bit : 0;
          while (bit >>= 1) {
            const size_t candidate = local_idx | bit;
            const size_t clamp = candidate < n ? candidate : n;
            if (z >= acc_a[clamp]) {
              local_idx = candidate;
            }
          }
          acc_r[idx] = local_idx;
        }
      }
      const auto end = std::chrono::steady_clock::now();
      exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
  } else {
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < repeat; ++iter) {
      for (size_t idx = 0; idx < zSize; ++idx) {
        const T z = acc_z[idx];
        size_t bit = initial_pivot;
        size_t local_idx = (bit < aSize && acc_a[bit] <= z) ? bit : 0;
        while (bit >>= 1) {
          const size_t candidate = local_idx | bit;
          const size_t clamp = candidate < n ? candidate : n;
          if (z >= acc_a[clamp]) {
            local_idx = candidate;
          }
        }
        acc_r[idx] = local_idx;
      }
    }
    const auto end = std::chrono::steady_clock::now();
    exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  std::cout << "Average device execution time (bs3) "
            << (exec_ns * 1e-9f) / static_cast<float>(repeat) << " (s)\n";
}

template <typename T>
void bs4(const size_t aSize,
         const size_t zSize,
         const T *__restrict__ acc_a,
         const T *__restrict__ acc_z,
         size_t *__restrict__ acc_r,
         const size_t n,
         const int repeat) {
  const unsigned nbits = bit_length(n);
  size_t initial_pivot = 0;
  if (nbits > 0) {
    initial_pivot = size_t{1} << (nbits - 1);
    if (initial_pivot > n) {
      initial_pivot = n;
    }
  }

  long long exec_ns = 0;

  if (has_target_device()) {
    const int numTeams = compute_num_teams(zSize);
    #pragma omp target data map(to : acc_a[0:aSize], acc_z[0:zSize]) map(from : acc_r[0:zSize])
    {
      const auto start = std::chrono::steady_clock::now();
      for (int iter = 0; iter < repeat; ++iter) {
        #pragma omp target teams distribute parallel for num_teams(numTeams) thread_limit(kTeamSize) schedule(static, 1) firstprivate(initial_pivot)
        for (size_t gid = 0; gid < zSize; ++gid) {
          const T z = acc_z[gid];
          size_t bit = initial_pivot;
          size_t local_idx = (bit < aSize && acc_a[bit] <= z) ? bit : 0;
          while (bit >>= 1) {
            const size_t candidate = local_idx | bit;
            const size_t clamp = candidate < n ? candidate : n;
            if (z >= acc_a[clamp]) {
              local_idx = candidate;
            }
          }
          acc_r[gid] = local_idx;
        }
      }
      const auto end = std::chrono::steady_clock::now();
      exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
  } else {
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < repeat; ++iter) {
      for (size_t gid = 0; gid < zSize; ++gid) {
        const T z = acc_z[gid];
        size_t bit = initial_pivot;
        size_t local_idx = (bit < aSize && acc_a[bit] <= z) ? bit : 0;
        while (bit >>= 1) {
          const size_t candidate = local_idx | bit;
          const size_t clamp = candidate < n ? candidate : n;
          if (z >= acc_a[clamp]) {
            local_idx = candidate;
          }
        }
        acc_r[gid] = local_idx;
      }
    }
    const auto end = std::chrono::steady_clock::now();
    exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  std::cout << "Average device execution time (bs4) "
            << (exec_ns * 1e-9f) / static_cast<float>(repeat) << " (s)\n";
}

#ifdef DEBUG
void verify(Real_t *a, Real_t *z, size_t *r, size_t aSize, size_t zSize, std::string msg) {
  for (size_t i = 0; i < zSize; ++i) {
    if (!(r[i] + 1 < aSize && a[r[i]] <= z[i] && z[i] < a[r[i] + 1])) {
      std::cout << msg << ": incorrect result:" << std::endl;
      std::cout << "index = " << i << " r[index] = " << r[i] << std::endl;
      std::cout << a[r[i]] << " <= " << z[i] << " < " << a[r[i] + 1] << std::endl;
      break;
    }
    r[i] = 0xFFFFFFFF;
  }
}
#endif

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage ./main <number of elements> <repeat>\n";
    return 1;
  }

  const size_t numElem = std::strtoull(argv[1], nullptr, 10);
  const int repeat = std::atoi(argv[2]);

  srand(2);
  const size_t aSize = numElem;
  const size_t zSize = 2 * aSize;

  Real_t *a = nullptr;
  Real_t *z = nullptr;
  size_t *r = nullptr;

  posix_memalign(reinterpret_cast<void **>(&a), 1024, aSize * sizeof(Real_t));
  posix_memalign(reinterpret_cast<void **>(&z), 1024, zSize * sizeof(Real_t));
  posix_memalign(reinterpret_cast<void **>(&r), 1024, zSize * sizeof(size_t));

  const size_t N = aSize - 1;

  for (size_t i = 0; i < aSize; i++) {
    a[i] = static_cast<Real_t>(i);
  }

  for (size_t i = 0; i < zSize; i++) {
    z[i] = static_cast<Real_t>(rand() % N);
  }

  bs(aSize, zSize, a, z, r, N, repeat);
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs1");
#endif

  bs2(aSize, zSize, a, z, r, N, repeat);
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs2");
#endif

  bs3(aSize, zSize, a, z, r, N, repeat);
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs3");
#endif

  bs4(aSize, zSize, a, z, r, N, repeat);
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs4");
#endif

  GATE_CHECKSUM_BYTES("r", r, zSize * sizeof(size_t));

  free(a);
  free(z);
  free(r);
  return 0;
}
