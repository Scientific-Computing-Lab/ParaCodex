


#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>




#define TBSIZE 256



#define DOT_NUM_BLOCKS 256



#define SCALAR (0.4)



int ARRAY_SIZE = 33554432;
unsigned int num_times = 100;


template <class T>
void init_arrays(
  T *__restrict a,
  T *__restrict b,
  T *__restrict c,
  T initA, T initB, T initC)
{
  const int array_size = ARRAY_SIZE; 
    for (int i = 0; i < array_size; i++) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
}


template <class T>
void copy(const T *__restrict a, T *__restrict c)
{
  const int array_size = ARRAY_SIZE;
    for (int i = 0; i < array_size; i++)
    c[i] = a[i];
}

template <class T>
void mul(T *__restrict b, const T *__restrict c)
{
  const int array_size = ARRAY_SIZE;
    for (int i = 0; i < array_size; i++) {
    const T scalar = SCALAR;
    b[i] = scalar * c[i];
  }
}

template <class T>
void add(const T *__restrict a, const T *__restrict b, T *__restrict c)
{
  const int array_size = ARRAY_SIZE;
    for (int i = 0; i < array_size; i++) {
    c[i] = a[i] + b[i];
  }
}


template <class T>
void triad(T *__restrict a, const T *__restrict b, const T *__restrict c)
{
  const int array_size = ARRAY_SIZE;
    for (int i = 0; i < array_size; i++) {
    const T scalar = SCALAR;
    a[i] = b[i] + scalar * c[i];
  }
}


template <class T>
void nstream(T *__restrict a, const T *__restrict b, const T *__restrict c)
{
  const int array_size = ARRAY_SIZE;
    for (int i = 0; i < array_size; i++) {
    const T scalar = SCALAR;
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
T dot(const T *__restrict a, const T *__restrict b)
{
  const int array_size = ARRAY_SIZE;
  T sum = 0.0;
    for (int i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }
  return sum;
}




template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  std::cout << "Running kernels " << num_times << " times" << std::endl;

  

  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  const int array_size = ARRAY_SIZE; 
  T *a = (T*)aligned_alloc(1024, sizeof(T)*array_size);
  T *b = (T*)aligned_alloc(1024, sizeof(T)*array_size);
  T *c = (T*)aligned_alloc(1024, sizeof(T)*array_size);

  if (sizeof(T) == sizeof(float))
    std::cout << "Precision: float" << std::endl;
  else
    std::cout << "Precision: double" << std::endl;

  

  std::cout << std::setprecision(1) << std::fixed
    << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout.precision(ss);

    {
    

    init_arrays(a, b, c, (T)0.1, (T)0.2, T(0.0));

    

    std::vector<std::vector<double>> timings(6);

    

    std::chrono::high_resolution_clock::time_point t1, t2;

    

    for (unsigned int k = 0; k < num_times; k++)
    {
      

      t1 = std::chrono::high_resolution_clock::now();
      copy(a, c);
      t2 = std::chrono::high_resolution_clock::now();
      timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


      

      t1 = std::chrono::high_resolution_clock::now();
      mul(b, c);
      t2 = std::chrono::high_resolution_clock::now();
      timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

      

      t1 = std::chrono::high_resolution_clock::now();
      add(a, b, c);
      t2 = std::chrono::high_resolution_clock::now();
      timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

      

      t1 = std::chrono::high_resolution_clock::now();
      triad(a, b, c);
      t2 = std::chrono::high_resolution_clock::now();
      timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

      

      t1 = std::chrono::high_resolution_clock::now();
      dot(a, b);
      t2 = std::chrono::high_resolution_clock::now();
      timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

      

      t1 = std::chrono::high_resolution_clock::now();
      nstream(a, b, c);
      t2 = std::chrono::high_resolution_clock::now();
      timings[5].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }

    

    std::cout
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << "MBytes/sec"
      << std::left << std::setw(12) << "Min (sec)"
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average"
      << std::endl
      << std::fixed;

    std::vector<std::string> labels;
    std::vector<size_t> sizes;

    labels = {"Copy", "Mul", "Add", "Triad", "Dot", "Nstream"};
    sizes = {
      2 * sizeof(T) * ARRAY_SIZE,
      2 * sizeof(T) * ARRAY_SIZE,
      3 * sizeof(T) * ARRAY_SIZE,
      3 * sizeof(T) * ARRAY_SIZE,
      2 * sizeof(T) * ARRAY_SIZE,
      4 * sizeof(T) * ARRAY_SIZE};

    for (size_t i = 0; i < timings.size(); ++i)
    {
      

      auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

      

      double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

      double bandwidth = 1.0E-6 * sizes[i] / (*minmax.first);

      std::cout
        << std::left << std::setw(12) << labels[i]
        << std::left << std::setw(12) << std::setprecision(3) << bandwidth
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
        << std::left << std::setw(12) << std::setprecision(5) << average
        << std::endl;
    }
    

    std::cout << std::endl;

  }

  free(a);
  free(b);
  free(c);
}


int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

int parseInt(const char *str, int *output)
{
  char *next;
  *output = strtol(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--arraysize").compare(argv[i]) ||
        !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseInt(argv[i], &ARRAY_SIZE) || ARRAY_SIZE <= 0)
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
        !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--help").compare(argv[i]) ||
        !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
        << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char *argv[])
{
  parseArguments(argc, argv);
  run<float>();
  run<double>();
}

