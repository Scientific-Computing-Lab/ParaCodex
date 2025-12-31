#include <algorithm>
#include <chrono>
#include <cstdio>

typedef unsigned char uchar;

template <typename Td, typename Ts>
void convert(int nelems, int niters)
{
  Ts *src = (Ts*) malloc (nelems * sizeof(Ts));
  Td *dst = (Td*) malloc (nelems * sizeof(Td));

  const int ls = std::min(nelems, 256);
  const int gs = (nelems + ls - 1) / ls;

    {
    

        for (int i = 0; i < nelems; i++) {
      dst[i] = static_cast<Td>(src[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < niters; i++) {
            for (int i = 0; i < nelems; i++)
        dst[i] = static_cast<Td>(src[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>
                  (end - start).count() / niters / 1.0e6;
    double size = (sizeof(Td) + sizeof(Ts)) * nelems / 1e9;
    printf("size(GB):%.2f, average time(sec):%f, BW:%f\n", size, time, size / time);
  }
  free(src);
  free(dst);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int niters = atoi(argv[2]);




  printf("float -> float\n");
  convert<float, float>(nelems, niters); 
  

  

  printf("float -> int\n");
  convert<int, float>(nelems, niters); 
  printf("float -> char\n");
  convert<char, float>(nelems, niters); 
  printf("float -> uchar\n");
  convert<uchar, float>(nelems, niters); 

  printf("int -> int\n");
  convert<int, int>(nelems, niters); 
  printf("int -> float\n");
  convert<float, int>(nelems, niters); 
  

  

  printf("int -> char\n");
  convert<char, int>(nelems, niters); 
  printf("int -> uchar\n");
  convert<uchar, int>(nelems, niters); 

  printf("char -> int\n");
  convert<int, char>(nelems, niters); 
  printf("char -> float\n");
  convert<float, char>(nelems, niters); 
  

  

  printf("char -> char\n");
  convert<char, char>(nelems, niters); 
  printf("char -> uchar\n");
  convert<uchar, char>(nelems, niters); 

  printf("uchar -> int\n");
  convert<int, uchar>(nelems, niters); 
  printf("uchar -> float\n");
  convert<float, uchar>(nelems, niters); 
  

  

  printf("uchar -> char\n");
  convert<char, uchar>(nelems, niters); 
  printf("uchar -> uchar\n");
  convert<uchar, uchar>(nelems, niters); 

  return 0;
}