#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <omp.h>
#include "gate.h"

#ifndef Real_t
#define Real_t float
#endif

template <typename T>
void bs(const size_t aSize,
        const size_t zSize,
        const T *acc_a,
        const T *acc_z,
        size_t *acc_r,
        const size_t n,
        const int repeat,
        const bool use_gpu)
{
  auto start = std::chrono::steady_clock::now();
  if (use_gpu) {
    // Use launch geometry sized for the RTX 4060 Laptop (Ada, 24 SM) to boost team concurrency.
    const int threads_per_team = 256;
    const size_t raw_team_count =
      (zSize + static_cast<size_t>(threads_per_team) - 1) / static_cast<size_t>(threads_per_team);
    const size_t capped_team_count =
      raw_team_count == 0 ? 1u
                          : (raw_team_count > static_cast<size_t>(65535) ? static_cast<size_t>(65535) : raw_team_count);
    const int launch_teams = static_cast<int>(capped_team_count);
    // Keep the search data resident across repeats to minimize host-device transfers.
#pragma omp target data if (use_gpu) map(to : acc_a[0:aSize], acc_z[0:zSize]) map(from : acc_r[0:zSize])
    {
      for (int rep = 0; rep < repeat; rep++) {
        // Offload independent searches across z-elements to the GPU.
#pragma omp target teams distribute parallel for if (use_gpu) map(present : acc_a[0:aSize], acc_z[0:zSize], acc_r[0:zSize]) \
    num_teams(launch_teams) thread_limit(threads_per_team)
        for (size_t zi = 0; zi < zSize; zi++) {
          T z = acc_z[zi];
          size_t low = 0;
          size_t high = n;
          while (high - low > 1) {
            size_t mid = low + (high - low) / 2;
            if (z < acc_a[mid])
              high = mid;
            else
              low = mid;
          }
          acc_r[zi] = low;
        }
      }
    }
  } else {
    for (int rep = 0; rep < repeat; rep++) {
      for (size_t zi = 0; zi < zSize; zi++) {
        T z = acc_z[zi];
        size_t low = 0;
        size_t high = n;
        while (high - low > 1) {
          size_t mid = low + (high - low) / 2;
          if (z < acc_a[mid])
            high = mid;
          else
            low = mid;
        }
        acc_r[zi] = low;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device execution time (bs1) " << (time * 1e-9f) / repeat << " (s)\n";
}

template <typename T>
void bs2(const size_t aSize,
         const size_t zSize,
         const T *acc_a,
         const T *acc_z,
         size_t *acc_r,
         const size_t n,
         const int repeat,
         const bool use_gpu)
{
  auto start = std::chrono::steady_clock::now();
  if (use_gpu) {
    // Use launch geometry sized for the RTX 4060 Laptop (Ada, 24 SM) to boost team concurrency.
    const int threads_per_team = 256;
    const size_t raw_team_count =
      (zSize + static_cast<size_t>(threads_per_team) - 1) / static_cast<size_t>(threads_per_team);
    const size_t capped_team_count =
      raw_team_count == 0 ? 1u
                          : (raw_team_count > static_cast<size_t>(65535) ? static_cast<size_t>(65535) : raw_team_count);
    const int launch_teams = static_cast<int>(capped_team_count);
    // Keep the search data resident across repeats to minimize host-device transfers.
#pragma omp target data if (use_gpu) map(to : acc_a[0:aSize], acc_z[0:zSize]) map(from : acc_r[0:zSize])
    {
      for (int rep = 0; rep < repeat; rep++) {
        // Offload independent searches across z-elements to the GPU.
#pragma omp target teams distribute parallel for if (use_gpu) map(present : acc_a[0:aSize], acc_z[0:zSize], acc_r[0:zSize]) \
    num_teams(launch_teams) thread_limit(threads_per_team)
        for (size_t zi = 0; zi < zSize; zi++) {
          unsigned nbits = 0;
          while (n >> nbits)
            nbits++;
          size_t k = 1ULL << (nbits - 1);
          T z = acc_z[zi];
          size_t idx = (acc_a[k] <= z) ? k : 0;
          while (k >>= 1) {
            size_t r = idx | k;
            if (r < n && z >= acc_a[r]) {
              idx = r;
            }
          }
          acc_r[zi] = idx;
        }
      }
    }
  } else {
    for (int rep = 0; rep < repeat; rep++) {
      for (size_t zi = 0; zi < zSize; zi++) {
        unsigned nbits = 0;
        while (n >> nbits)
          nbits++;
        size_t k = 1ULL << (nbits - 1);
        T z = acc_z[zi];
        size_t idx = (acc_a[k] <= z) ? k : 0;
        while (k >>= 1) {
          size_t r = idx | k;
          if (r < n && z >= acc_a[r]) {
            idx = r;
          }
        }
        acc_r[zi] = idx;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device execution time (bs2) " << (time * 1e-9f) / repeat << " (s)\n";
}

template <typename T>
void bs3(const size_t aSize,
         const size_t zSize,
         const T *acc_a,
         const T *acc_z,
         size_t *acc_r,
         const size_t n,
         const int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    for (int i = 0; i < zSize; i++) {
      unsigned nbits = 0;
      while (n >> nbits)
        nbits++;
      size_t k = 1ULL << (nbits - 1);
      T z = acc_z[i];
      size_t idx = (acc_a[k] <= z) ? k : 0;
      while (k >>= 1) {
        size_t r = idx | k;
        size_t w = r < n ? r : n;
        if (z >= acc_a[w]) {
          idx = r;
        }
      }
      acc_r[i] = idx;
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device execution time (bs3) " << (time * 1e-9f) / repeat << " (s)\n";
}

template <typename T>
void bs4(const size_t aSize,
         const size_t zSize,
         const T *acc_a,
         const T *acc_z,
         size_t *acc_r,
         const size_t n,
         const int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    {
      size_t k;
      {
        size_t lid = omp_get_thread_num();
        size_t gid = omp_get_team_num() * omp_get_num_threads() + lid;
        if (lid == 0) {
          unsigned nbits = 0;
          while (n >> nbits)
            nbits++;
          k = 1ULL << (nbits - 1);
        }

        size_t p = k;
        T z = acc_z[gid];
        size_t idx = (acc_a[p] <= z) ? p : 0;
        while (p >>= 1) {
          size_t r = idx | p;
          size_t w = r < n ? r : n;
          if (z >= acc_a[w]) {
            idx = r;
          }
        }
        acc_r[gid] = idx;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device execution time (bs4) " << (time * 1e-9f) / repeat << " (s)\n";
}

#ifdef DEBUG
void verify(Real_t *a, Real_t *z, size_t *r, size_t aSize, size_t zSize, std::string msg)
{
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

int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cout << "Usage ./main <number of elements> <repeat>\n";
    return 1;
  }

  size_t numElem = atol(argv[1]);
  uint repeat = atoi(argv[2]);

  srand(2);
  size_t aSize = numElem;
  size_t zSize = 2 * aSize;
  Real_t *a = NULL;
  Real_t *z = NULL;
  size_t *r = NULL;
  posix_memalign((void **)&a, 1024, aSize * sizeof(Real_t));
  posix_memalign((void **)&z, 1024, zSize * sizeof(Real_t));
  posix_memalign((void **)&r, 1024, zSize * sizeof(size_t));

  size_t N = aSize - 1;
  const char *force_gpu_env = std::getenv("FORCE_OMP_GPU");
  const bool use_gpu = (force_gpu_env && force_gpu_env[0] != '0') && omp_get_num_devices() > 0;

  for (size_t i = 0; i < aSize; i++)
    a[i] = i;

  for (size_t i = 0; i < zSize; i++) {
    z[i] = rand() % N;
  }

  {
    bs(aSize, zSize, a, z, r, N, repeat, use_gpu);

#ifdef DEBUG
    verify(a, z, r, aSize, zSize, "bs1");
#endif

    bs2(aSize, zSize, a, z, r, N, repeat, use_gpu);

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
  }

  GATE_CHECKSUM_BYTES("result_r", r, zSize * sizeof(size_t));

  free(a);
  free(z);
  free(r);
  return 0;
}
