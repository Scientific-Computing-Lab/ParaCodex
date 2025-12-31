#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "./main.h"
#include "./util/graphics/graphics.h"
#include "./util/graphics/resize.h"
#include "./util/timer/timer.h"

int main(int argc, char* argv []) {

  

  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;
  long long time7;
  long long time8;
  long long time9;
  long long time10;
  long long time11;
  long long time12;

  time0 = get_time();

  

  fp* image_ori;                      

  int image_ori_rows;
  int image_ori_cols;
  long image_ori_elem;

  

  fp* image;                          

  int Nr,Nc;                          

  long Ne;

  

  int niter;                          

  fp lambda;                          


  

  int r1,r2,c1,c2;                    

  long NeROI;                         


  

  int* iN, *iS, *jE, *jW;

  

  int iter;   

  long i,j;     


  

  int mem_size_i;
  int mem_size_j;

  int blocks_x;
  int blocks_work_size, blocks_work_size2;
  size_t local_work_size;
  int no;
  int mul;
  fp meanROI;
  fp meanROI2;
  fp varROI;
  fp q0sqr;

  time1 = get_time();

  if(argc != 5){
    printf("Usage: %s <repeat> <lambda> <number of rows> <number of columns>\n", argv[0]);
    return 1;
  }
  else{
    niter = atoi(argv[1]);
    lambda = atof(argv[2]);
    Nr = atoi(argv[3]);
    Nc = atoi(argv[4]);
  }

  time2 = get_time();

  

  

  


  image_ori_rows = 502;
  image_ori_cols = 458;
  image_ori_elem = image_ori_rows * image_ori_cols;
  image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

  const char* input_image_path = "../data/srad/image.pgm";
  if ( !read_graphics( input_image_path, image_ori, image_ori_rows, image_ori_cols, 1) ) {
    printf("ERROR: failed to read input image at %s\n", input_image_path);
    if (image_ori != NULL) free(image_ori);
    return -1; 

  }

  time3 = get_time();

  

  

  


  Ne = Nr*Nc;

  image = (fp*)malloc(sizeof(fp) * Ne);

  resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

  time4 = get_time();

  


  

  r1     = 0;      

  r2     = Nr - 1; 

  c1     = 0;      

  c2     = Nc - 1; 


  

  NeROI = (r2-r1+1)*(c2-c1+1);                      


  

  mem_size_i = sizeof(int) * Nr;                      

  iN = (int *)malloc(mem_size_i) ;                    

  iS = (int *)malloc(mem_size_i) ;                    

  mem_size_j = sizeof(int) * Nc;                      

  jW = (int *)malloc(mem_size_j) ;                    

  jE = (int *)malloc(mem_size_j) ;                    


  

  for (i=0; i<Nr; i++) {
    iN[i] = i-1;                            

    iS[i] = i+1;                            

  }
  for (j=0; j<Nc; j++) {
    jW[j] = j-1;                            

    jE[j] = j+1;                            

  }

  

  iN[0]    = 0;                             

  iS[Nr-1] = Nr-1;                          

  jW[0]    = 0;                             

  jE[Nc-1] = Nc-1;                          


  fp *dN = (fp*) malloc (sizeof(fp)*Ne);
  fp *dS = (fp*) malloc (sizeof(fp)*Ne);
  fp *dW = (fp*) malloc (sizeof(fp)*Ne);
  fp *dE = (fp*) malloc (sizeof(fp)*Ne);
  fp *c = (fp*) malloc (sizeof(fp)*Ne);
  fp *sums = (fp*) malloc (sizeof(fp)*Ne);
  fp *sums2 = (fp*) malloc (sizeof(fp)*Ne);

  

  local_work_size = NUMBER_THREADS;

  

  blocks_x = Ne/(int)local_work_size;
  if (Ne % (int)local_work_size != 0){ 

    blocks_x = blocks_x + 1;                                  
  }
  blocks_work_size = blocks_x;

  time5 = get_time();

  

  

  

    {
  time6 = get_time();

    for (int ei = 0; ei < Ne; ei++)
    image[ei] = expf(image[ei]/(fp)255); 


  time7 = get_time();

  for (iter=0; iter<niter; iter++){ 

        for (int ei = 0; ei < Ne; ei++) {
      sums[ei] = image[ei];
      sums2[ei] = image[ei]*image[ei];
    }

    blocks_work_size2 = blocks_work_size;  

    no = Ne;  

    mul = 1;  


    while(blocks_work_size2 != 0){

      

            {
        fp psum[NUMBER_THREADS];
        fp psum2[NUMBER_THREADS];
                {
          int bx = omp_get_team_num();
          int tx = omp_get_thread_num();
          int ei = (bx*NUMBER_THREADS)+tx;

          int nf = NUMBER_THREADS-(blocks_work_size2*NUMBER_THREADS-no);

          int df = 0;


          

          int i;

          

          if(ei<no){

            psum[tx] = sums[ei*mul];
            psum2[tx] = sums2[ei*mul];
          }

          
          

          if(nf == NUMBER_THREADS){
            

            for(i=2; i<=NUMBER_THREADS; i=2*i){
              

              if((tx+1) % i == 0){                      

                psum[tx] = psum[tx] + psum[tx-i/2];
                psum2[tx] = psum2[tx] + psum2[tx-i/2];
              }
              

                          }
            

            if(tx==(NUMBER_THREADS-1)){                      

              sums[bx*mul*NUMBER_THREADS] = psum[tx];
              sums2[bx*mul*NUMBER_THREADS] = psum2[tx];
            }
          }
          

          else{ 
            

            if(bx != (blocks_work_size2 - 1)){                      

              

              for(i=2; i<=NUMBER_THREADS; i=2*i){                

                

                if((tx+1) % i == 0){                    

                  psum[tx] = psum[tx] + psum[tx-i/2];
                  psum2[tx] = psum2[tx] + psum2[tx-i/2];
                }
                

                              }
              

              if(tx==(NUMBER_THREADS-1)){                    

                sums[bx*mul*NUMBER_THREADS] = psum[tx];
                sums2[bx*mul*NUMBER_THREADS] = psum2[tx];
              }
            }
            

            else{                                

              

              for(i=2; i<=NUMBER_THREADS; i=2*i){                

                if(nf >= i){
                  df = i;
                }
              }
              

              for(i=2; i<=df; i=2*i){                      

                

                if((tx+1) % i == 0 && tx<df){                

                  psum[tx] = psum[tx] + psum[tx-i/2];
                  psum2[tx] = psum2[tx] + psum2[tx-i/2];
                }
                

                              }
              

              if(tx==(df-1)){                    

                

                for(i=(bx*NUMBER_THREADS)+df; i<(bx*NUMBER_THREADS)+nf; i++){            

                  psum[tx] = psum[tx] + sums[i];
                  psum2[tx] = psum2[tx] + sums2[i];
                }
                

                sums[bx*mul*NUMBER_THREADS] = psum[tx];
                sums2[bx*mul*NUMBER_THREADS] = psum2[tx];
              }
            }
          }
        }
      }

      

      no = blocks_work_size2;  
      if(blocks_work_size2 == 1){
          blocks_work_size2 = 0;
      }
      else{
        mul = mul * NUMBER_THREADS; 

        blocks_x = blocks_work_size2/(int)local_work_size; 

        if (blocks_work_size2 % (int)local_work_size != 0){ 

            blocks_x = blocks_x + 1;
        }
        blocks_work_size2 = blocks_x;
      }
    } 


        
    


    meanROI  = sums[0] / (fp)(NeROI); 

    meanROI2 = meanROI * meanROI;
    varROI = (sums2[0] / (fp)(NeROI)) - meanROI2; 

    q0sqr = varROI / meanROI2; 


    

        for (int ei = 0; ei < Ne; ei++) {
      

      int row = (ei+1) % Nr - 1; 

      int col = (ei+1) / Nr + 1 - 1; 

      if((ei+1) % Nr == 0){
        row = Nr - 1;
        col = col - 1;
      }

      

      fp d_Jc = image[ei];                            


      

      fp N_loc = image[iN[row] + Nr*col] - d_Jc;            

      fp S_loc = image[iS[row] + Nr*col] - d_Jc;            

      fp W_loc = image[row + Nr*jW[col]] - d_Jc;            

      fp E_loc = image[row + Nr*jE[col]] - d_Jc;            


      

      fp d_G2 = (N_loc*N_loc + S_loc*S_loc + W_loc*W_loc + E_loc*E_loc) / (d_Jc*d_Jc);  


      

      fp d_L = (N_loc + S_loc + W_loc + E_loc) / d_Jc;      


      

      fp d_num  = ((fp)0.5*d_G2) - (((fp)1.0/(fp)16.0)*(d_L*d_L)) ;            

      fp d_den  = (fp)1 + ((fp)0.25*d_L);                        

      fp d_qsqr = d_num/(d_den*d_den);                    


      

      d_den = (d_qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;        

      fp d_c_loc = (fp)1.0 / ((fp)1.0+d_den) ;                    


      

      if (d_c_loc < 0){                          

        d_c_loc = 0;                          

      }
      else if (d_c_loc > 1){                        

        d_c_loc = 1;                          

      }

      

      dN[ei] = N_loc; 
      dS[ei] = S_loc; 
      dW[ei] = W_loc; 
      dE[ei] = E_loc;
      c[ei] = d_c_loc;
    }

        for (int ei = 0; ei < Ne; ei++){              

      

      int row = (ei+1) % Nr - 1;  

      int col = (ei+1) / Nr ;     

      if((ei+1) % Nr == 0){
        row = Nr - 1;
        col = col - 1;
      }

      

      fp d_cN = c[ei];  

      fp d_cS = c[iS[row] + Nr*col];  

      fp d_cW = c[ei];  

      fp d_cE = c[row + Nr * jE[col]];  


      

      fp d_D = d_cN*dN[ei] + d_cS*dS[ei] + d_cW*dW[ei] + d_cE*dE[ei];

      

      image[ei] += (fp)0.25*lambda*d_D; 

    }
  }

  time8 = get_time();

  


    for (int ei = 0; ei < Ne; ei++)
    image[ei] = logf(image[ei])*(fp)255; 


  time9 = get_time();

  
  time10 = get_time();

  


  write_graphics(
      "./image_out.pgm",
      image,
      Nr,
      Nc,
      1,
      255);

  time11 = get_time();

  

  
  } 


  free(image_ori);
  free(image);
  free(iN); 
  free(iS); 
  free(jW); 
  free(jE);

  time12 = get_time();

  


  printf("Time spent in different stages of the application:\n");
  printf("%15.12f s, %15.12f %% : SETUP VARIABLES\n",
      (float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : READ COMMAND LINE PARAMETERS\n",
      (float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : READ IMAGE FROM FILE\n",
      (float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : RESIZE IMAGE\n", 
      (float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n",
      (float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : COPY DATA TO CPU->GPU\n",
      (float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : EXTRACT IMAGE\n", 
      (float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : COMPUTE (%d iterations)\n", 
      (float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time12-time0) * 100, niter);
  printf("%15.12f s, %15.12f %% : COMPRESS IMAGE\n", 
      (float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : COPY DATA TO GPU->CPU\n", 
      (float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : SAVE IMAGE INTO FILE\n", 
      (float) (time11-time10) / 1000000, (float) (time11-time10) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : FREE MEMORY\n", 
      (float) (time12-time11) / 1000000, (float) (time12-time11) / (float) (time12-time0) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", (float) (time12-time0) / 1000000);

  return 0;
}