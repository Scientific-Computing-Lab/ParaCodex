




























































































































































































































#include <assert.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include <stdlib.h>
using namespace std;



constexpr int kMaxN = 31;





constexpr int64_t kLimit = 100000000;



constexpr bool kPrint = false;

static_assert(sizeof(int64_t) == 8, "int64_t is not 8 bytes");
static_assert(sizeof(int32_t) == 4, "int32_t is not 4 bytes");
static_assert(sizeof(int8_t) == 1, "int64_t is not 1 byte");

constexpr int64_t lsb = 1;
constexpr int32_t lsb32 = 1;

constexpr int div_up(int p, int q) {
  return (p + (q - 1)) / q;
};



int ffsll(int64_t x) {
  for (int i = 0; i < 64; i++)
    if ((x >> i) & 1) return (i+1);
  return 0;
};





template <int n>
using Positions = array<int8_t, n>;



template <int n>
using Results = vector<Positions<n>>;



template <int n>
using PositionsGPUAligned = int64_t[(n + 7) / 8];

template <int n>
using PositionsGPU = int8_t[div_up(n, 8) * 8];





template <int n>
using Availability = int32_t[2 * n + 1];







template <int n>
using Open = int64_t[4 * n + 2];

template <int n>
using Stack = int8_t[24 * n];

template <int n>
void print(const Positions<n>& pos);



constexpr int kThreadsPerBlock = 4;





constexpr int kNumLogicalThreads = 16383;



template <int n>
void dfs(int64_t* p_count,
         int64_t* p_result,
         Availability<n> &availability,
         Open<n> &open,
         Stack<n> &stack,
         PositionsGPUAligned<n>& pgpualigned,
         const int32_t logical_thread_index)
{
  constexpr int two_n = 2 * n;
  constexpr int64_t msb = lsb << (int64_t)(n - 1);
  constexpr int64_t nn1 = lsb << (2 * n - 1);
  PositionsGPU<n> &pos = *((PositionsGPU<n>*)(&pgpualigned[0]));
  

  

  availability[0] = msb | (msb - 1);
  open[0] = 0;
  open[1] = 0;
  int top = 0;
  int8_t k, m, d, num_open;
  

  

  

#define push(k, m, d, num_open) do { \
  stack[top++] = k; \
  stack[top++] = m; \
  stack[top++] = d; \
  stack[top++] = num_open; \
} while (0)
#define pop(k, m, d, num_open) do { \
  num_open = stack[--top]; \
  d = stack[--top]; \
  m = stack[--top]; \
  k = stack[--top]; \
} while (0)
  

  push(0, -1, 0, 0);
  while (top) {
    pop(k, m, d, num_open);
    int64_t* openings = open + 2 * k + 2;
    openings[0] = openings[-2];
    openings[1] = openings[-1];
    int32_t avail = availability[k];
    

    

    

#define place_macro(d) do { \
  if (m>=0) { \
    pos[m] = k; \
    avail ^= (lsb32 << m); \
    openings[d] &= (openings[d] - 1); \
  } else { \
    openings[d] |= (nn1 >> k); \
    ++num_open; \
  } \
} while (0)
    if (d) {
      place_macro(1);
    } else {
      place_macro(0);
    }
++k;
availability[k] = avail;
if (k == two_n) {
  

  

  

  int64_t  cnt;
  cnt = p_count[0]++;
  if (cnt < kLimit) {
    constexpr int kAlignedCnt = (n + 7) / 8;
    int64_t* dst = p_result + kAlignedCnt * cnt;
    for (int i=0; i<kAlignedCnt; ++i) {
      dst[i] = pgpualigned[i];
    }
  }
  

} else {
  

  

  

  

  constexpr int8_t k_limit = (n > 19 ? (8 + (n / 3)) : (n - 5));
  if (kNumLogicalThreads > 1 &&
      k == k_limit &&
      

      uint64_t(131071 * (openings[1] - openings[0]) + avail) % kNumLogicalThreads != logical_thread_index) {
    

    continue;
  }
  

  int8_t offset = k - two_n - 2;
  for (d=0; d<2; ++d) {
    if (openings[d]) { 

      

      m = offset + ffsll(openings[d]);
      

      

      if (((unsigned)m < n) && ((avail >> m) & 1)) {
        if (m || k <= n) { 

          push(k, m, d, num_open);
        }
      }
    }
  }
  if (num_open < n) {
    push(k, -1, 1, num_open);
    push(k, -1, 0, num_open);
  }
}
}
}





template <int n>
int64_t unique_count(Results<n> &results) {
  int64_t total = results.size();
  int64_t unique = total;
  sort(results.begin(), results.end());
  if (kPrint && total) {
    print<n>(results[0]);
  }
  for (int i=1; i<total; ++i) {
    if (results[i] == results[i-1]) {
      --unique;
    } else if (kPrint) {
      print<n>(results[i]);
    }
  }
  return unique;
}



long unixtime() {
  using namespace chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}



template <int n>
void run_gpu_d(int64_t* count, Results<n>& final_results) {
  assert(sizeof(int64_t) == 8);

  constexpr int64_t kAlignedCnt = (n + 7) / 8;
  int64_t *results = (int64_t*) malloc (sizeof(int64_t) * kLimit * kAlignedCnt); 

  int blocks_x = div_up(kNumLogicalThreads, kThreadsPerBlock);

{
  auto start = std::chrono::steady_clock::now();

    {
    Availability<n> availability[kThreadsPerBlock]; 
    Open<n> open[kThreadsPerBlock]; 
    Stack<n> stack[kThreadsPerBlock];
    PositionsGPUAligned<n> pgpualigned[kThreadsPerBlock];

        {
      int lid = omp_get_thread_num();
      int tid = omp_get_team_num();

      const int32_t result_index = tid * kThreadsPerBlock + lid;
      dfs<n>(
      count,
      results,
      availability[lid],
      open[lid],
      stack[lid],
      pgpualigned[lid],
      result_index);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cout << "Kernel execution time:  " << time * 1e-9f << " (s)\n";


  if (*count >= kLimit) {
    cout << "Result for n = " << n << " will be bogus because GPU "
      << " exceeded " << kLimit << " solutions.\n";
  }

  int64_t r_count = *count;
  int64_t data_size = r_count * kAlignedCnt;


  

  for (int i=0; i<r_count; ++i) {
    Positions<n> pos;
    PositionsGPU<n>& gpos = *((PositionsGPU<n>*)(results + (kAlignedCnt * i)));
    for (int j=0; j<n; ++j) {
      pos[j] = gpos[j];
    }
    final_results.push_back(pos);
  }

  free(results);
} 
}



template <int n>
void run_gpu(const int64_t* known_results) {
  cout << "\n";
  cout << "------\n";
  cout << unixtime() << " Computing PL(2, " << n << ")\n";
  if (n > kMaxN) {
    cout << unixtime() << " Sorry, n = " << n << " exceeds the max allowed " << kMaxN << "\n";
    return;
  }

  int64_t count = 0;
  int64_t total;
  Results<n> final_results;

  run_gpu_d<n>(&count, final_results);

  

  total = unique_count<n>(final_results);
  cout << unixtime() << " Result " << total << " for n = " << n;
  if (n < 0 || n >= 64 || known_results[n] == -1) {
    cout << " is NEW";
  } else if (known_results[n] == total) {
    cout << " MATCHES previously published result";
  } else {
    cout << " MISMATCHES previously published result " << known_results[n];
  }
  cout << "\n------\n\n";
}

void init_known_results(int64_t (&known_results)[64]) {
  for (int i=0;  i<64; ++i) {
    known_results[i] = 0;
  }
  

  for (int i = 29;  i<64;  ++i) {
    if (i % 4 == 3 || i % 4 == 0) {
      known_results[i] = -1;
    }
  }
  known_results[3]  = 1;
  known_results[4]  = 0;
  known_results[7]  = 0;
  known_results[8]  = 4;
  known_results[11] = 16;
  known_results[12] = 40;
  known_results[15] = 194;
  known_results[16] = 274;
  known_results[19] = 2384;
  known_results[20] = 4719;
  known_results[23] = 31856;
  known_results[24] = 62124;
  known_results[27] = 426502;
  known_results[28] = 817717;
}

template <int n>
void print(const Positions<n>& pos) {
  cout << unixtime() << " Sequence ";
  int s[2 * n];
  for (int i=0; i<2*n; ++i) {
    s[i] = -1;
  }
  for (int m=1;  m<=n;  ++m) {
    int k2 = pos[m-1];
    int k1 = k2 - m - 1;
    assert(0 <= k1);
    assert(k2 < 2*n);
    assert(s[k1] == -1);
    assert(s[k2] == -1);
    s[k1] = s[k2] = m;
  }
  for (int i=0;  i<2*n;  ++i) {
    const int64_t m = s[i];
    assert(0 <= m);
    assert(m <= n);
    cout << std::setw(3) << m;
  }
  cout << "\n";
}

int main(int argc, char **argv) {
  int64_t known_results[64];

  init_known_results(known_results);
  

  run_gpu<7>(known_results);
  run_gpu<8>(known_results);
  run_gpu<11>(known_results);
  run_gpu<12>(known_results);
  run_gpu<15>(known_results);

  

  

  

  

  

  

  

  

  return 0;
}