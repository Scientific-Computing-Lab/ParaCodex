


#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>






void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " <Number of iterations within the kernel> ";
  std::cout << "<Kernel execution count>\n\n";
}



template <typename T>
void print_matrix(T** matrix, size_t size_X, size_t size_Y) {
  std::cout << std::endl;
  for (size_t i = 0; i < size_X; ++i) {
    for (size_t j = 0; j < size_Y; ++j) {
      std::cout << std::setw(3) << matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
}



template <typename T>
void print_vector(T* vector, size_t n) {
  std::cout << std::endl;
  for (size_t i = 0; i < n; ++i) {
    std::cout << vector[i] << " ";
  }
  std::cout << std::endl;
}



void motion_device(float* particleX, float* particleY,
                   float* randomX, float* randomY, int** grid, size_t grid_size,
                   size_t n_particles, int nIterations, float radius,
                   size_t* map, int nRepeat) {
  srand(17);

  

  const size_t scale = 100;

  

  for (size_t i = 0; i < n_particles * nIterations; i++) {
    randomX[i] = rand() % scale;
    randomY[i] = rand() % scale;
  }

  const size_t MAP_SIZE = n_particles * grid_size * grid_size;

    {
    std::cout << " The number of kernel execution is " << nRepeat << std::endl;
    std::cout << " The number of particles is " << n_particles << std::endl;

    double time_total = 0.0;

    for (int i = 0; i < nRepeat; i++) {

                  
      auto start = std::chrono::steady_clock::now();

            for (int ii = 0; ii < n_particles; ii++) {

        

        

        

        

        

        size_t iter = 0;
        float pX = particleX[ii];
        float pY = particleY[ii];
        size_t map_base = ii * grid_size * grid_size;

        while (iter < nIterations) {
          

          

          

          


          float randnumX = randomX[iter * n_particles + ii];
          float randnumY = randomY[iter * n_particles + ii];

          

          float displacementX = randnumX / 1000.0f - 0.0495f;
          float displacementY = randnumY / 1000.0f - 0.0495f;

          

          pX += displacementX;
          pY += displacementY;

          

          float dX = pX - truncf(pX);
          float dY = pY - truncf(pY);

          

          int iX = floorf(pX);
          int iY = floorf(pY);

          

          if ((pX < grid_size) && (pY < grid_size) && (pX >= 0) && (pY >= 0)) {
            

            

            if ((dX * dX + dY * dY <= radius * radius))
              

              map[map_base + iY * grid_size + iX]++;
          }

          iter++;

        }  


        particleX[ii] = pX;
        particleY[ii] = pY;
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      time_total += time;
    }

    std::cout << std::endl;
    std::cout << "Average kernel execution time: " << (time_total * 1e-9) / nRepeat << " (s)";
    std::cout << std::endl;
  }

  

  

  

  for (size_t i = 0; i < n_particles; ++i) {
    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        if (map[i * grid_size * grid_size + y * grid_size + x] > 0) {
          grid[y][x] += map[i * grid_size * grid_size + y * grid_size + x];
        }
      }
    }
  }  

}  


int main(int argc, char* argv[]) {
  if (argc != 3) {
    usage(argv[0]);
    return 1;
  }

  

  int nIterations = std::stoi(argv[1]);
  int nRepeat = std::stoi(argv[2]);

  

  const size_t grid_size = 21;    

  const size_t n_particles = 147456;  

  const float radius = 0.5;       


  

  int** grid = new int*[grid_size];
  for (size_t i = 0; i < grid_size; i++) grid[i] = new int[grid_size];

  

  float* randomX = new float[n_particles * nIterations];
  float* randomY = new float[n_particles * nIterations];

  

  float* particleX = new float[n_particles];
  float* particleY = new float[n_particles];

  

  const size_t MAP_SIZE = n_particles * grid_size * grid_size;
  size_t* map = new size_t[MAP_SIZE];

  

  for (size_t i = 0; i < n_particles; i++) {
    

    particleX[i] = 10.0;
    particleY[i] = 10.0;

    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        map[i * grid_size * grid_size + y * grid_size + x] = 0;
      }
    }
  }

  for (size_t y = 0; y < grid_size; y++) {
    for (size_t x = 0; x < grid_size; x++) {
      grid[y][x] = 0;
    }
  }

  

  auto start = std::chrono::steady_clock::now();

  

  motion_device(particleX, particleY, randomX, randomY, grid, grid_size,
                n_particles, nIterations, radius, map, nRepeat);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << std::endl;
  std::cout << "Simulation time: " << time * 1e-9 << " (s) ";
  std::cout << std::endl;

  

  if (grid_size <= 64) {
    std::cout << "\n ********************** OUTPUT GRID: " << std::endl;
    print_matrix<int>(grid, grid_size, grid_size);
  }

  

  for (size_t i = 0; i < grid_size; i++) delete grid[i];

  delete[] grid;
  delete[] particleX;
  delete[] particleY;
  delete[] randomX;
  delete[] randomY;
  delete[] map;

  return 0;
}