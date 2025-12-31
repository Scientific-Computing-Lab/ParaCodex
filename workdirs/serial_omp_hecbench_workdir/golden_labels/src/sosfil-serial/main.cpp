
























#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>







#define MAX_THREADS 256
#define THREADS 32
#define sos_width  6   


template <typename T>
void filtering (const int repeat, const int n_signals, const int n_samples,
                const int n_sections, const int zi_width)
{
  

  assert(MAX_THREADS >= n_sections);

  

  assert(n_samples >= n_sections);

  

  srand(2);

  const int blocks = n_signals;

  

  const int sos_size = n_sections * sos_width ;

  T* sos = (T*) malloc (sizeof(T) * sos_size);
  for (int i = 0; i < n_sections; i++)
    for (int j = 0; j < sos_width; j++)
      sos[i*sos_width+j] = (T)1 ; 


  

  const int z_size = (n_sections + 1) * blocks * zi_width;
  T* zi = (T*) malloc (sizeof(T) * z_size);
  for (int i = 0; i < z_size; i++) zi[i] = (T)1; 


  

  const int x_size = n_signals * n_samples;
  T* x_in = (T*) malloc (sizeof(T) * x_size);
  for (int i = 0; i < n_signals; i++) 
    for (int j = 0; j < n_samples; j++) 
      x_in[i*n_samples+j] = (T)sin(2*3.14*(i+1+j));


  

  

#ifdef DEBUG
  

  

  const int shared_mem_size = 32 + (32+1)*2*2 + 32*6;
#else
  

  

  const int shared_mem_size = 32 + (32+1)*8*2 + 32*6;
#endif

{
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
        {
      T smem[shared_mem_size];  

            {

        T *s_out = smem ;
        T *s_zi =  &s_out[n_sections] ;
        T *s_sos = &s_zi[n_sections * zi_width] ;

        const int tx = static_cast<int>( omp_get_thread_num() );
        const int ty = static_cast<int>( omp_get_team_num() );

        

        s_out[tx] = 0;

        

        for ( int i = 0; i < zi_width; i++ ) {
          s_zi[tx * zi_width + i] = zi[ty * n_sections * zi_width + tx * zi_width + i];
        }

        

                for ( int i = 0; i < sos_width; i++ ) {
          s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
        }

        
        const int load_size = n_sections - 1 ;
        const int unload_size = n_samples - load_size ;

        T temp;
        T x_n;

        if ( ty < n_signals ) {
          

          for ( int n = 0; n < load_size; n++ ) {
            if ( tx == 0 ) {
              x_n = x_in[ty * n_samples + n];
            } else {
              x_n = s_out[tx - 1];
            }

            

            temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

            s_zi[tx * zi_width + 0] =
              s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

            s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            s_out[tx] = temp;

                      }

          

          for ( int n = load_size; n < n_samples; n++ ) {
            if ( tx == 0 ) {
              x_n = x_in[ty * n_samples + n];
            } else {
              x_n = s_out[tx - 1];
            }

            

            temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

            s_zi[tx * zi_width + 0] =
              s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

            s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            if ( tx < load_size ) {
              s_out[tx] = temp;
            } else {
              x_in[ty * n_samples + ( n - load_size )] = temp;
            }

                      }

          

          for ( int n = 0; n < n_sections; n++ ) {
            

            if ( tx > n ) {
              x_n = s_out[tx - 1];

              

              temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

              s_zi[tx * zi_width + 0] =
                s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

              s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

              if ( tx < load_size ) {
                s_out[tx] = temp;
              } else {
                x_in[ty * n_samples + ( n + unload_size )] = temp;
              }
            }
                      }
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %lf (s)\n", time * 1e-9 / repeat);
}

#ifdef DEBUG
  for (int i = 0; i < n_signals; i++) { 
    for (int j = 0; j < n_samples; j++) 
      printf("%.2f ", x_in[i*n_samples+j]);
    printf("\n");
  }
#endif

  free(x_in);
  free(sos);
  free(zi);
}

int main(int argc, char** argv) {
  if (argc != 2) 
  {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int numSections = THREADS; 

#ifdef DEBUG
  const int numSignals = 2; 
  const int numSamples = THREADS+1;
#else
  

  const int numSignals = 8;  
  const int numSamples = 100000;
#endif

  const int zi_width = 2;
  filtering<float> (repeat, numSignals, numSamples, numSections, zi_width);
  filtering<double> (repeat, numSignals, numSamples, numSections, zi_width);
  return 0;
}