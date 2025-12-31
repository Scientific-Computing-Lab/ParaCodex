




#include <algorithm>
#include <cstdio>
#include <sys/time.h>
#include <time.h>
#include <math.h>



#define BLOCK_DIM 16








float compute_distance(const float *ref, int ref_nb, const float *query,
                       int query_nb, int dim, int ref_index, int query_index) {
  float sum = 0.f;
  for (int d = 0; d < dim; ++d) {
    const float diff =
        ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

void modified_insertion_sort(float *dist, int *index, int length, int k) {

  

  index[0] = 0;

  

  for (int i = 1; i < length; ++i) {

    

    float curr_dist = dist[i];
    int curr_index = i;

    

    

    if (i >= k && curr_dist >= dist[k - 1]) {
      continue;
    }

    

    int j = std::min(i, k - 1);
    while (j > 0 && dist[j - 1] > curr_dist) {
      dist[j] = dist[j - 1];
      index[j] = index[j - 1];
      --j;
    }

    

    dist[j] = curr_dist;
    index[j] = curr_index;
  }
}

bool knn_serial(const float *ref, int ref_nb, const float *query, int query_nb,
           int dim, int k, float *knn_dist, int *knn_index) {
  

  

  float *dist = (float *)malloc(ref_nb * sizeof(float));
  int *index = (int *)malloc(ref_nb * sizeof(int));

  

  if (!dist || !index) {
    printf("Memory allocation error\n");
    free(dist);
    free(index);
    return false;
  }

  

  for (int i = 0; i < query_nb; ++i) {

    

    for (int j = 0; j < ref_nb; ++j) {
      dist[j] = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
      index[j] = j;
    }

    

    modified_insertion_sort(dist, index, ref_nb, k);

    

    for (int j = 0; j < k; ++j) {
      knn_dist[j * query_nb + i] = dist[j];
      knn_index[j * query_nb + i] = index[j];
    }
  }

  

  free(dist);
  free(index);
  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int iterations = atoi(argv[1]);

  float *ref;          

  float *query;        

  float *dist;         

  int *ind;            

  int ref_nb = 4096;   

  int query_nb = 4096; 

  int dim = 68;        

  int k = 20;          

  int c_iterations = 1;
  int i;
  const float precision = 0.001f; 

  int nb_correct_precisions = 0;
  int nb_correct_indexes = 0;
  

  ref = (float *)malloc(ref_nb * dim * sizeof(float));
  query = (float *)malloc(query_nb * dim * sizeof(float));
  

  dist = (float *)malloc(query_nb * ref_nb * sizeof(float));
  ind = (int *)malloc(query_nb * k * sizeof(float));

  

  srand(2);
  for (i = 0; i < ref_nb * dim; i++)
    ref[i] = (float)rand() / (float)RAND_MAX;
  for (i = 0; i < query_nb * dim; i++)
    query[i] = (float)rand() / (float)RAND_MAX;

  

  printf("Number of reference points      : %6d\n", ref_nb);
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim);
  printf("Number of neighbors to consider : %4d\n", k);
  printf("Processing kNN search           :\n");

  float *knn_dist = (float *)malloc(query_nb * k * sizeof(float));
  int *knn_index = (int *)malloc(query_nb * k * sizeof(int));
  printf("Ground truth computation in progress...\n\n");
  if (!knn_serial(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);
    return EXIT_FAILURE;
  }

  struct timeval tic;
  struct timeval toc;
  float elapsed_time;

  printf("On CPU: \n");
  gettimeofday(&tic, NULL);
  for (i = 0; i < c_iterations; i++) {
    knn_serial(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  }
  gettimeofday(&toc, NULL);
  elapsed_time = toc.tv_sec - tic.tv_sec;
  elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
  printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time,
         c_iterations, elapsed_time / (c_iterations));

  printf("on GPU: \n");
  gettimeofday(&tic, NULL);

  for (i = 0; i < iterations; i++) {
        {
      

            {
        float shared_A[BLOCK_DIM*BLOCK_DIM];
        float shared_B[BLOCK_DIM*BLOCK_DIM];
        int begin_A;
        int begin_B;
        int step_A;
        int step_B;
        int end_A;
        
                {
          

          int tx = omp_get_thread_num() % 16;
          int ty = omp_get_thread_num() / 16;
      
          

          float tmp;
          float ssd = 0;
      
          

          begin_A = BLOCK_DIM * (omp_get_team_num() / ((query_nb+15)/16));
          begin_B = BLOCK_DIM * (omp_get_team_num() % ((query_nb+15)/16));
          step_A  = BLOCK_DIM * ref_nb;
          step_B  = BLOCK_DIM * query_nb;
          end_A   = begin_A + (dim - 1) * ref_nb;
      
          

          int cond0 = (begin_A + tx < ref_nb); 

          int cond1 = (begin_B + tx < query_nb); 

                                           

          int cond2 =
              (begin_A + ty < ref_nb); 

      
          

          

          for (int a = begin_A, b = begin_B; 
                   a <= end_A; a += step_A, b += step_B) {
            

            

            if (a / ref_nb + ty < dim) {
              shared_A[ty*BLOCK_DIM+tx] = (cond0) ? ref[a + ref_nb * ty + tx] : 0;
              shared_B[ty*BLOCK_DIM+tx] = (cond1) ? query[b + query_nb * ty + tx] : 0;
            } else {
              shared_A[ty*BLOCK_DIM+tx] = 0;
              shared_B[ty*BLOCK_DIM+tx] = 0;
            }
      
            

                  
            

            

            if (cond2 && cond1) {
              for (int k = 0; k < BLOCK_DIM; ++k) {
                tmp = shared_A[k*BLOCK_DIM+ty] - shared_B[k*BLOCK_DIM+tx];
                ssd += tmp * tmp;
              }
            }
      
            

            

                      }
      
          

          if (cond2 && cond1) dist[(begin_A + ty) * query_nb + begin_B + tx] = ssd;
        }
      }
      
      

            for (unsigned int xIndex = 0; xIndex < query_nb; xIndex++) {
        

        float* p_dist = &dist[xIndex];
        int* p_ind = &ind[xIndex];
        float max_dist = p_dist[0];
        p_ind[0] = 0;
      
        

        for (int l = 1; l < k; l++) {
          int curr_row = l * query_nb;
          float curr_dist = p_dist[curr_row];
          if (curr_dist < max_dist) {
            int i = l - 1;
            for (int a = 0; a < l - 1; a++) {
              if (p_dist[a * query_nb] > curr_dist) {
                i = a;
                break;
              }
            }
            for (int j = l; j > i; j--) {
              p_dist[j * query_nb] = p_dist[(j - 1) * query_nb];
              p_ind[j * query_nb] = p_ind[(j - 1) * query_nb];
            }
            p_dist[i * query_nb] = curr_dist;
            p_ind[i * query_nb] = l;
          } else {
            p_ind[l * query_nb] = l;
          }
          max_dist = p_dist[curr_row];
        }
      
        

        int max_row = (k - 1) * query_nb;
        for (int l = k; l < ref_nb; l++) {
          float curr_dist = p_dist[l * query_nb];
          if (curr_dist < max_dist) {
            int i = k - 1;
            for (int a = 0; a < k - 1; a++) {
              if (p_dist[a * query_nb] > curr_dist) {
                i = a;
                break;
              }
            }
            for (int j = k - 1; j > i; j--) {
              p_dist[j * query_nb] = p_dist[(j - 1) * query_nb];
              p_ind[j * query_nb] = p_ind[(j - 1) * query_nb];
            }
            p_dist[i * query_nb] = curr_dist;
            p_ind[i * query_nb] = l;
            max_dist = p_dist[max_row];
          }
        }
      }
      
      

            for (unsigned int i = 0; i < query_nb * k; i++)
        dist[i] = sqrtf(dist[i]);

                }
  }

  gettimeofday(&toc, NULL);
  elapsed_time = toc.tv_sec - tic.tv_sec;
  elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
  printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time,
         iterations, elapsed_time / (iterations));

  for (int i = 0; i < query_nb * k; ++i) {
    if (fabs(dist[i] - knn_dist[i]) <= precision) {
      nb_correct_precisions++;
    }
    if (ind[i] == knn_index[i]) {
      nb_correct_indexes++;
    } else {
      printf("Mismatch @index %d: %d %d\n", i, ind[i], knn_index[i]);
    }
  }

  float precision_accuracy = nb_correct_precisions / ((float)query_nb * k);
  float index_accuracy = nb_correct_indexes / ((float)query_nb * k);
  printf("Precision accuracy %f\nIndex accuracy %f\n", precision_accuracy, index_accuracy);

  free(ind);
  free(dist);
  free(query);
  free(ref);
}