


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>

typedef struct { float x; float y; float z; float w; } float4 ;

#define MAX_DETECTIONS  4096
#define N_PARTITIONS    32

void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt>\n\n");
  printf("               detections.txt -> Input file containing the coordinates, width, and scores of detected objects\n");
  printf("               output.txt     -> Output file after performing NMS\n");
  printf("               repeat         -> Kernel execution count\n\n");
}



int get_optimal_dim(int val)
{
  int div, neg, cntneg, cntpos;

  


  neg = 1;
  div = 16;
  cntneg = div;
  cntpos = div;

  


  for(int i=0; i<5; i++)
  {
    if(val % div == 0)
      return div;

    if(neg)
    {
      cntneg--;
      div = cntneg;
      neg = 0;
    }
    else
    {
      cntpos++;
      div = cntpos;
      neg = 1;
    }
  }

  return 16;
}




int get_upper_limit(int val, int mul)
{
  int cnt = mul;

  


  while(cnt < val)
    cnt += mul;

  if(cnt > MAX_DETECTIONS)
    cnt = MAX_DETECTIONS;

  return cnt;
}

int main(int argc, char *argv[])
{
  int x, y, w;
  float score;

  if(argc != 4)
  {
    print_help();
    return 0;
  }

  

  int ndetections = 0;

  FILE *fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("Error: Unable to open file %s for input detection coordinates.\n", argv[1]);
    return -1;
  }

  

  float4* points = (float4*) malloc(sizeof(float4) * MAX_DETECTIONS);
  if(!points)
  {
    printf("Error: Unable to allocate CPU memory.\n");
    return -1;
  }

  memset(points, 0, sizeof(float4) * MAX_DETECTIONS);

  while(!feof(fp))
  {
    int cnt = fscanf(fp, "%d,%d,%d,%f\n", &x, &y, &w, &score);

    if (cnt !=4)
    {
      printf("Error: Invalid file format in line %d when reading %s\n", ndetections, argv[1]);
      return -1;
    }

    points[ndetections].x = (float) x;       

    points[ndetections].y = (float) y;       

    points[ndetections].z = (float) w;       

    points[ndetections].w = score;           


    ndetections++;
  }

  printf("Number of detections read from input file (%s): %d\n", argv[1], ndetections);

  fclose(fp);

  

  unsigned char* pointsbitmap = (unsigned char*) malloc(sizeof(unsigned char) * MAX_DETECTIONS);
  memset(pointsbitmap, 0, sizeof(unsigned char) * MAX_DETECTIONS);

  unsigned char* nmsbitmap = (unsigned char*) malloc(sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS);
  memset(nmsbitmap, 1, sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS);

  

  const int repeat = atoi(argv[3]);
  const int limit = get_upper_limit(ndetections, 16);
  const int threads = get_optimal_dim(limit) * get_optimal_dim(limit);

    {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
            for (int i = 0; i < limit; i++) {
        for (int j = 0; j < limit; j++) {
          if(points[i].w < points[j].w)
          {
            float area = (points[j].z + 1.0f) * (points[j].z + 1.0f);
            float w = fmaxf(0.0f, fminf(points[i].x + points[i].z, points[j].x + points[j].z) - 
                fmaxf(points[i].x, points[j].x) + 1.0f);
            float h = fmaxf(0.0f, fminf(points[i].y + points[i].z, points[j].y + points[j].z) - 
                fmaxf(points[i].y, points[j].y) + 1.0f);
            nmsbitmap[i * MAX_DETECTIONS + j] = (((w * h) / area) < 0.3f) && (points[j].z != 0);
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (generate_nms_bitmap): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    

    for (int n = 0; n < repeat; n++) {
            {
        #ifdef BYTE_ATOMIC
        unsigned char s;
        #else
        unsigned s;
        #endif
                {
          int bid = omp_get_team_num();
          int lid = omp_get_thread_num();
          int idx = bid * MAX_DETECTIONS + lid;

          if (lid == 0) s = 1;
          
                    s &= 
            #ifndef BYTE_ATOMIC
            (unsigned int)
            #endif
            nmsbitmap[idx];
            
          for(int i=0; i<(N_PARTITIONS-1); i++)
          {
            idx += MAX_DETECTIONS / N_PARTITIONS;
                        s &= nmsbitmap[idx];
                      }
          pointsbitmap[bid] = s;
        }
      }
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (reduce_nms_bitmap): %f (s)\n", (time * 1e-9f) / repeat);
  }

  fp = fopen(argv[2], "w");
  if (!fp)
  {
    printf("Error: Unable to open file %s for detection outcome.\n", argv[2]);
    return -1;
  }

  int totaldets = 0;
  for(int i = 0; i < ndetections; i++)
  {
    if(pointsbitmap[i])
    {
      x = (int) points[i].x;          

      y = (int) points[i].y;          

      w = (int) points[i].z;          

      score = points[i].w;            

      fprintf(fp, "%d,%d,%d,%f\n", x, y, w, score);
      totaldets++; 
    }
  }
  fclose(fp);
  printf("Detections after NMS: %d\n", totaldets);

  free(points);
  free(pointsbitmap);
  free(nmsbitmap);

  return 0;
}