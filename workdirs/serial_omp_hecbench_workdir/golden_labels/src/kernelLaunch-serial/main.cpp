


#include <cstdio>
#include <cstdlib>
#include <chrono>


#define DO_NOT_OPTIMIZE_AWAY                                                       \
  unsigned i = omp_get_num_teams() * omp_get_num_threads() + omp_get_thread_num(); \
  if (out) *out = args.args[i];

struct SmallKernelArgs {
  char args[16];
};

struct MediumKernelArgs {
  char args[256];
};

struct LargeKernelArgs {
  char args[4096];
};

void KernelWithSmallArgs(SmallKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }

void KernelWithMediumArgs(MediumKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }

void KernelWithLargeArgs(LargeKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  SmallKernelArgs small_kernel_args;
  MediumKernelArgs medium_kernel_args;
  LargeKernelArgs large_kernel_args;

  

  for (int i = 0; i < repeat; i++) {
        KernelWithSmallArgs(small_kernel_args, nullptr);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
        KernelWithSmallArgs(small_kernel_args, nullptr);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithSmallArgs: %f (us)\n", (time * 1e-3f) / repeat);

  

  for (int i = 0; i < repeat; i++) {
        KernelWithMediumArgs(medium_kernel_args, nullptr);
  }

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
        KernelWithMediumArgs(medium_kernel_args, nullptr);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithMediumArgs: %f (us)\n", (time * 1e-3f) / repeat);

  

  for (int i = 0; i < repeat; i++) {
        KernelWithLargeArgs(large_kernel_args, nullptr);
  }

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
        KernelWithLargeArgs(large_kernel_args, nullptr);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithLargeArgs: %f (us)\n", (time * 1e-3f) / repeat);

  return 0;
}