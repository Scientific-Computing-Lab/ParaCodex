


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <chrono>

#define MAXDISTANCE    (200)



unsigned int minimum(unsigned int a, unsigned int b) 
{
  return (b < a) ? b : a;
}



void floydWarshallCPUReference(unsigned int * pathDistanceMatrix,
    unsigned int * pathMatrix, unsigned int numNodes)
{
  unsigned int distanceYtoX, distanceYtoK, distanceKtoX, indirectDistance;

  

  unsigned int width = numNodes;
  unsigned int yXwidth;

  


  for(unsigned int k = 0; k < numNodes; ++k)
  {
    for(unsigned int y = 0; y < numNodes; ++y)
    {
      yXwidth =  y*numNodes;
      for(unsigned int x = 0; x < numNodes; ++x)
      {
        distanceYtoX = pathDistanceMatrix[yXwidth + x];
        distanceYtoK = pathDistanceMatrix[yXwidth + k];
        distanceKtoX = pathDistanceMatrix[k * width + x];

        indirectDistance = distanceYtoK + distanceKtoX;

        if(indirectDistance < distanceYtoX)
        {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x]         = k;
        }
      }
    }
  }
}






int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
    return 1;
  }
  

  unsigned int numNodes = atoi(argv[1]);
  unsigned int numIterations = atoi(argv[2]);
  unsigned int blockSize = atoi(argv[3]);

  

  if(numNodes % blockSize != 0) {
    numNodes = (numNodes / blockSize + 1) * blockSize;
  }

  

  unsigned int* pathMatrix = NULL;
  unsigned int* pathDistanceMatrix = NULL;
  unsigned int* verificationPathDistanceMatrix = NULL;
  unsigned int* verificationPathMatrix = NULL;
  unsigned int matrixSize;
  unsigned int matrixSizeBytes;

  matrixSize = numNodes * numNodes;
  matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);
  pathDistanceMatrix = (unsigned int *) malloc(matrixSizeBytes);
  assert (pathDistanceMatrix != NULL) ;

  pathMatrix = (unsigned int *) malloc(matrixSizeBytes);
  assert (pathMatrix != NULL) ;

  

  srand(2);
  for(unsigned int i = 0; i < numNodes; i++)
    for(unsigned int j = 0; j < numNodes; j++)
    {
      int index = i*numNodes + j;
      pathDistanceMatrix[index] = rand() % (MAXDISTANCE + 1);
    }
  for(unsigned int i = 0; i < numNodes; ++i)
  {
    unsigned int iXWidth = i * numNodes;
    pathDistanceMatrix[iXWidth + i] = 0;
  }

  

  for(unsigned int i = 0; i < numNodes; ++i)
  {
    for(unsigned int j = 0; j < i; ++j)
    {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }

  verificationPathDistanceMatrix = (unsigned int *) malloc(matrixSizeBytes);
  assert (verificationPathDistanceMatrix != NULL);

  verificationPathMatrix = (unsigned int *) malloc(matrixSizeBytes);
  assert(verificationPathMatrix != NULL);

  memcpy(verificationPathDistanceMatrix, pathDistanceMatrix, matrixSizeBytes);
  memcpy(verificationPathMatrix, pathMatrix, matrixSizeBytes);

  unsigned int numPasses = numNodes;

    {
    float total_time = 0.f;

    for (unsigned int n = 0; n < numIterations; n++) {
      


      
      auto start = std::chrono::steady_clock::now();

      for(unsigned int k = 0; k < numPasses; k++)
      {
                for(unsigned int y = 0; y < numNodes; ++y)
        {
          for(unsigned int x = 0; x < numNodes; ++x)
          {
            unsigned int distanceYtoX = pathDistanceMatrix[y*numNodes + x];
            unsigned int distanceYtoK = pathDistanceMatrix[y*numNodes + k];
            unsigned int distanceKtoX = pathDistanceMatrix[k*numNodes + x];
            unsigned int indirectDistance = distanceYtoK + distanceKtoX;

            if(indirectDistance < distanceYtoX)
            {
              pathDistanceMatrix[y*numNodes + x] = indirectDistance;
              pathMatrix[y*numNodes + x]         = k;
            }
          }
        }
      }
      
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      total_time += time;
    }

    printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / numIterations);

      }

  

  floydWarshallCPUReference(verificationPathDistanceMatrix,
      verificationPathMatrix, numNodes);
  if(memcmp(pathDistanceMatrix, verificationPathDistanceMatrix,
        numNodes*numNodes*sizeof(unsigned int)) == 0)
  {
    printf("PASS\n");
  }
  else
  {
    printf("FAIL\n");
    if (numNodes <= 8) 
    {
      for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++)
          printf("host: %u ", verificationPathDistanceMatrix[i*numNodes+j]);
        printf("\n");
      }
      for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++)
          printf("device: %u ", pathDistanceMatrix[i*numNodes+j]);
        printf("\n");
      }
    }
  }

  free(pathDistanceMatrix);
  free(pathMatrix);
  free(verificationPathDistanceMatrix);
  free(verificationPathMatrix);
  return 0;
}