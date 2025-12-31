#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "./util/timer/timer.h"
#include "./util/num/num.h"
#include "main.h"

int main(int argc, char *argv [])
{
  

  int i, j, k, l, m, n;

  

  par_str par_cpu;
  dim_str dim_cpu;
  box_str* box_cpu;
  FOUR_VECTOR* rv_cpu;
  fp* qv_cpu;
  FOUR_VECTOR* fv_cpu;
  int nh;

  printf("WG size of kernel = %d \n", NUMBER_THREADS);

  

  dim_cpu.arch_arg = 0;
  dim_cpu.cores_arg = 1;
  dim_cpu.boxes1d_arg = 1;

  

  if(argc==3){
    for(dim_cpu.cur_arg=1; dim_cpu.cur_arg<argc; dim_cpu.cur_arg++){
      

      if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0){
        

        if(argc>=dim_cpu.cur_arg+1){
          

          if(isInteger(argv[dim_cpu.cur_arg+1])==1){
            dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
            if(dim_cpu.boxes1d_arg<0){
              printf("ERROR: Wrong value to -boxes1d argument, cannot be <=0\n");
              return 0;
            }
            dim_cpu.cur_arg = dim_cpu.cur_arg+1;
          }
          

          else{
            printf("ERROR: Value to -boxes1d argument in not a number\n");
            return 0;
          }
        }
        

        else{
          printf("ERROR: Missing value to -boxes1d argument\n");
          return 0;
        }
      }
      

      else{
        printf("ERROR: Unknown argument\n");
        return 0;
      }
    }
    

    printf("Configuration used: arch = %d, cores = %d, boxes1d = %d\n", dim_cpu.arch_arg, dim_cpu.cores_arg, dim_cpu.boxes1d_arg);
  }
  else{
    printf("Provide boxes1d argument, example: -boxes1d 16");
    return 0;
  }

  par_cpu.alpha = 0.5;

  

  dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg; 


  

  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;              

  dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  

  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

  

  box_cpu = (box_str*)malloc(dim_cpu.box_mem);

  

  nh = 0;

  

  for(i=0; i<dim_cpu.boxes1d_arg; i++){
    

    for(j=0; j<dim_cpu.boxes1d_arg; j++){
      

      for(k=0; k<dim_cpu.boxes1d_arg; k++){

        

        box_cpu[nh].x = k;
        box_cpu[nh].y = j;
        box_cpu[nh].z = i;
        box_cpu[nh].number = nh;
        box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

        

        box_cpu[nh].nn = 0;

        

        for(l=-1; l<2; l++){
          

          for(m=-1; m<2; m++){
            

            for(n=-1; n<2; n++){

              

              if(    (((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)  &&
                  (l==0 && m==0 && n==0)==false  ){

                

                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                box_cpu[nh].nei[box_cpu[nh].nn].number =  (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
                  (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
                  box_cpu[nh].nei[box_cpu[nh].nn].x;
                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                

                box_cpu[nh].nn = box_cpu[nh].nn + 1;

              }

            } 

          } 

        } 


        

        nh = nh + 1;

      } 

    } 

  } 


  

  

  


  

  srand(2);

  

  rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    rv_cpu[i].v = (rand()%10 + 1) / 10.0;      

    

    rv_cpu[i].x = (rand()%10 + 1) / 10.0;      

    

    rv_cpu[i].y = (rand()%10 + 1) / 10.0;      

    

    rv_cpu[i].z = (rand()%10 + 1) / 10.0;      

    

  }

  

  qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    qv_cpu[i] = (rand()%10 + 1) / 10.0;      

    

  }

  

  fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    fv_cpu[i].v = 0;                

    fv_cpu[i].x = 0;                

    fv_cpu[i].y = 0;                

    fv_cpu[i].z = 0;                

  }

  long long kstart, kend;
  long long start = get_time();

  

  int dim_cpu_number_boxes = dim_cpu.number_boxes;

{
  kstart = get_time();
    {
    FOUR_VECTOR rA_shared[100];
    FOUR_VECTOR rB_shared[100];
    fp qB_shared[100];

        {
      int bx = omp_get_team_num();
      int tx = omp_get_thread_num();
      int wtx = tx;

      


      if(bx<dim_cpu_number_boxes){

        


        

        fp a2 = 2*par_cpu.alpha*par_cpu.alpha;

        

        int first_i;
        


        

        int pointer;
        int k = 0;
        int first_j;
        int j = 0;
        


        

        fp r2;
        fp u2;
        fp vij;
        fp fs;
        fp fxij;
        fp fyij;
        fp fzij;
        THREE_VECTOR d;

        


        


        

        first_i = box_cpu[bx].offset;

        


        

        

        while(wtx<NUMBER_PAR_PER_BOX){
          rA_shared[wtx] = rv_cpu[first_i+wtx];
          wtx = wtx + NUMBER_THREADS;
        }
        wtx = tx;

        

        


        


        

        for (k=0; k<(1+box_cpu[bx].nn); k++){

          

          

          


          if(k==0){
            pointer = bx;                          

          }
          else{
            pointer = box_cpu[bx].nei[k-1].number;              

          }

          


          

          first_j = box_cpu[pointer].offset;

          

          

          while(wtx<NUMBER_PAR_PER_BOX){
            rB_shared[wtx] = rv_cpu[first_j+wtx];
            qB_shared[wtx] = qv_cpu[first_j+wtx];
            wtx = wtx + NUMBER_THREADS;
          }
          wtx = tx;

          

          


          


          

          while(wtx<NUMBER_PAR_PER_BOX){

            

            for (j=0; j<NUMBER_PAR_PER_BOX; j++){

              r2 = rA_shared[wtx].v + rB_shared[j].v - DOT(rA_shared[wtx],rB_shared[j]); 
              u2 = a2*r2;
              vij= exp(-u2);
              fs = 2*vij;
              d.x = rA_shared[wtx].x  - rB_shared[j].x;
              fxij=fs*d.x;
              d.y = rA_shared[wtx].y  - rB_shared[j].y;
              fyij=fs*d.y;
              d.z = rA_shared[wtx].z  - rB_shared[j].z;
              fzij=fs*d.z;
              fv_cpu[first_i+wtx].v +=  qB_shared[j]*vij;
              fv_cpu[first_i+wtx].x +=  qB_shared[j]*fxij;
              fv_cpu[first_i+wtx].y +=  qB_shared[j]*fyij;
              fv_cpu[first_i+wtx].z +=  qB_shared[j]*fzij;

            }

            

            wtx = wtx + NUMBER_THREADS;

          }

          

          wtx = tx;

          


        }
      }
    }
  }
  kend = get_time();
}

  long long end = get_time();
  printf("Device offloading time:\n"); 
  printf("%.12f s\n", (float) (end-start) / 1000000);

  printf("Kernel execution time:\n"); 
  printf("%.12f s\n", (float) (kend-kstart) / 1000000); 

#ifdef DEBUG
  int offset = 395;
  for(int g=0; g<10; g++){
    printf("g=%d %f, %f, %f, %f\n", \
        g, fv_cpu[offset+g].v, fv_cpu[offset+g].x, fv_cpu[offset+g].y, fv_cpu[offset+g].z);
  }
#endif

  

#ifdef OUTPUT
  FILE *fptr;
  fptr = fopen("result.txt", "w");  
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
  }
  fclose(fptr);
#endif         

  free(rv_cpu);
  free(qv_cpu);
  free(fv_cpu);
  free(box_cpu);

  return 0;
}
