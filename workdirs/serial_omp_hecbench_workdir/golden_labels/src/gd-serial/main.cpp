#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "utils.h"

int main(int argc, const char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <path to file> <lambda> <alpha> <repeat>\n", argv[0]);
    return 1;
  }
  const std::string file_path = argv[1]; 
  const float lambda = atof(argv[2]);
  const float alpha = atof(argv[3]);
  const int iters = atof(argv[4]);

  

  Classification_Data_CRS A;
  get_CRSM_from_svm(A, file_path);

  const int m = A.m; 

  const int n = A.n; 


  std::vector<float> x(n, 0.f);
  std::vector<float> grad (n);

  float *d_grad        = grad.data();
  float *d_x           = x.data();
  int   *d_A_row_ptr   = A.row_ptr.data(); 
  int   *d_A_col_index = A.col_index.data();
  float *d_A_value     = A.values.data();
  int   *d_A_y_label   = A.y_label.data();

  float total_obj_val[1];
  float l2_norm[1];
  int   correct[1];

  {
  long long train_start = get_time();

  float obj_val = 0.f;
  float train_error = 0.f;

  for (int k = 0; k < iters; k++) {

    

    total_obj_val[0] = 0.f;
    l2_norm[0] = 0.f;
    correct[0] = 0;
    
            
    

    std::fill(grad.begin(), grad.end(), 0.f);

        
    

        for (int i = 0; i < m; ++i) {

      

      float xp = 0.f;
           
      for( int j = d_A_row_ptr[i]; j < d_A_row_ptr[i+1]; ++j) {
        xp += d_A_value[j] * d_x[d_A_col_index[j]];
      }

      

      float v = logf(1.f + expf(-xp * d_A_y_label[i]));
            total_obj_val[0] += v;

      

      float prediction = 1.f / (1.f + expf(-xp));
      int t = (prediction >= 0.5f) ? 1 : -1;
      if (d_A_y_label[i] == t) {
                correct[0]++;
      }

      

      float accum = expf(-d_A_y_label[i] * xp);
      accum = accum / (1.f + accum);
      for(int j = d_A_row_ptr[i]; j < d_A_row_ptr[i+1]; ++j) {
        float temp = -accum * d_A_value[j] * d_A_y_label[i];
                d_grad[d_A_col_index[j]] += temp;
      }
    }

    

        for (int i = 0; i < n; ++i) {
              l2_norm[0] += d_x[i] * d_x[i];
    }

            
    obj_val = total_obj_val[0] / (float)m + 0.5f * lambda * l2_norm[0];
    train_error = 1.f - (correct[0]/(float)m); 

    

        for (int i = 0; i < n; ++i) {
      float g = d_grad[i] / (float)m + lambda * d_x[i]; 
      d_x[i] = d_x[i] - alpha * g;
    }
  }

  long long train_end = get_time();
  printf("Training time takes %lld(us) for %d iterations\n\n", 
         train_end - train_start, iters);

  

  printf("object value = %f train_error = %f\n", obj_val, train_error);
}

  return 0; 
}