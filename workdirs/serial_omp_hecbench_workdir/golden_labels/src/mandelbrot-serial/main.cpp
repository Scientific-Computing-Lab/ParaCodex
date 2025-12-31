












#include "util.hpp"
#include "mandel.hpp"

void Execute() {
  

  MandelParallel m_par(row_size, col_size, max_iterations);
  MandelSerial m_ser(row_size, col_size, max_iterations);

  

  m_par.Evaluate();

  double kernel_time = 0;

  

  common::MyTimer t_par;

  for (int i = 0; i < repetitions; ++i) 
    kernel_time += m_par.Evaluate();

  common::Duration parallel_time = t_par.elapsed();

  

  m_par.Print();

  

  common::MyTimer t_ser;
  m_ser.Evaluate();
  common::Duration serial_time = t_ser.elapsed();

  

  std::cout << std::setw(20) << "serial time: " << serial_time.count() << "s\n";
  std::cout << std::setw(20) << "Average parallel time: "
                        << (parallel_time / repetitions).count() * 1e3 << " ms\n";
  std::cout << std::setw(20) << "Average kernel execution time: "
                        << kernel_time / repetitions * 1e3 << " ms\n";

  

  m_par.Verify(m_ser);
}

void Usage(std::string program_name) {
  

  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: ";
  std::cout << program_name << " <repeat>\n\n";
  exit(-1);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    Usage(argv[0]);
  }

  try {
    repetitions = atoi(argv[1]);
    Execute();
  } catch (...) {
    std::cout << "Failure\n";
    std::terminate();
  }
  std::cout << "Success\n";
  return 0;
}