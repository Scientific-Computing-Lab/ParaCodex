


#include <chrono>
#include "scan.h"

void bScan(const unsigned int blockSize,
           const unsigned int len,
           const float *input,
           float *output,
           float *sumBuffer)
{
    {
    float block[256];
        {
      int tid = omp_get_thread_num();
      int bid = omp_get_team_num();
      int gid = bid * blockSize/2 + tid;
      
      

      block[2*tid]     = input[2*gid];
      block[2*tid + 1] = input[2*gid + 1];
            
      float cache0 = block[0];
      float cache1 = cache0 + block[1];
      
      

      for(int stride = 1; stride < blockSize; stride *=2) {
        if(2*tid>=stride) {
          cache0 = block[2*tid-stride]+block[2*tid];
          cache1 = block[2*tid+1-stride]+block[2*tid+1];
        }
              
        block[2*tid] = cache0;
        block[2*tid+1] = cache1;
      
              }
      
      
   
      sumBuffer[bid] = block[blockSize-1];
      
      

      if(tid==0) {
        output[2*gid]     = 0;
        output[2*gid+1]   = block[2*tid];
      } else {
        output[2*gid]     = block[2*tid-1];
        output[2*gid + 1] = block[2*tid];
      }
    }
  }
}

void pScan(const unsigned int blockSize,
           const unsigned int len,
           const float *input,
           float *output)
{
    {
    

    float block[4];
        {
      int tid = omp_get_thread_num();
      int bid = omp_get_team_num();
      int gid = bid * len/2 + tid;

      

      block[2*tid]     = input[2*gid];
      block[2*tid + 1] = input[2*gid + 1];
      
      float cache0 = block[0];
      float cache1 = cache0 + block[1];

      

      for(int stride = 1; stride < blockSize; stride *=2) {

        if(2*tid>=stride) {
          cache0 = block[2*tid-stride]+block[2*tid];
          cache1 = block[2*tid+1-stride]+block[2*tid+1];
        }
        
        block[2*tid] = cache0;
        block[2*tid+1] = cache1;

              }

      

      if(tid==0) {
        output[2*gid]     = 0;
        output[2*gid+1]   = block[2*tid];
      } else {
        output[2*gid]     = block[2*tid-1];
        output[2*gid + 1] = block[2*tid];
      }
    }
  }
}

void bAddition(const unsigned int blockSize,
               const unsigned int len,
               const float *input,
               float *output)
{
    {
    float value;
        {
      int tid = omp_get_thread_num();
      int bid = omp_get_team_num();
      int gid = bid * blockSize + tid;
      

      if(tid == 0) value = input[bid];
      
      output[gid] += value;
    }
  }
}




void scanLargeArraysCPUReference(
    float * output,
    float * input,
    const unsigned int length)
{
  output[0] = 0;

  for(unsigned int i = 1; i < length; ++i)
  {
    output[i] = input[i-1] + output[i-1];
  }
}


int main(int argc, char * argv[])
{
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input length> <block size>\n";
    return 1;
  }
  int iterations = atoi(argv[1]);
  int length = atoi(argv[2]);
  int blockSize = atoi(argv[3]);

  if(iterations < 1)
  {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }
  if(!isPowerOf2(length))
  {
    length = roundToPowerOf2(length);
  }

  if((length/blockSize>GROUP_SIZE)&&(((length)&(length-1))!=0))
  {
    std::cout << "Invalid length: " << length << std::endl;
    return -1;
  }

  

  unsigned int sizeBytes = length * sizeof(float);

  float* inputBuffer = (float*) malloc (sizeBytes);

  

  fillRandom<float>(inputBuffer, length, 1, 0, 255);

  blockSize = (blockSize < length/2) ? blockSize : length/2;

  

  float t = std::log((float)length) / std::log((float)blockSize);
  unsigned int pass = (unsigned int)t;

  

  if(std::fabs(t - (float)pass) < 1e-7)
  {
    pass--;
  }
  
  

  int outputBufferSize = 0;
  int* outputBufferSizeOffset = (int*) malloc (sizeof(int) * pass);
  for(unsigned int i = 0; i < pass; i++)
  {
    outputBufferSizeOffset[i] = outputBufferSize;
    outputBufferSize += (int)(length / std::pow((float)blockSize,(float)i));
  }

  float* outputBuffer = (float*) malloc (sizeof(float) * outputBufferSize);

  

  int blockSumBufferSize = 0;
  int* blockSumBufferSizeOffset = (int*) malloc (sizeof(int) * pass);
  for(unsigned int i = 0; i < pass; i++)
  {
    blockSumBufferSizeOffset[i] = blockSumBufferSize;
    blockSumBufferSize += (int)(length / std::pow((float)blockSize,(float)(i + 1)));
  }
  float* blockSumBuffer = (float*) malloc (sizeof(float) * blockSumBufferSize);

  

  int tempLength = (int)(length / std::pow((float)blockSize, (float)pass));
  float* tempBuffer = (float*) malloc (sizeof(float) * tempLength);

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";

{
  auto start = std::chrono::steady_clock::now();

  for(int n = 0; n < iterations; n++)
  {
    

    bScan(blockSize, length, inputBuffer, 
          outputBuffer + outputBufferSizeOffset[0], 
          blockSumBuffer + blockSumBufferSizeOffset[0]);

    for(int i = 1; i < (int)pass; i++)
    {
      int size = (int)(length / std::pow((float)blockSize,(float)i));
      bScan(blockSize, size, blockSumBuffer + blockSumBufferSizeOffset[i - 1], 
            outputBuffer + outputBufferSizeOffset[i], 
            blockSumBuffer + blockSumBufferSizeOffset[i]);
    }

    

    pScan(blockSize, tempLength, 
          blockSumBuffer + blockSumBufferSizeOffset[pass - 1], tempBuffer);

    

    bAddition(blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(pass - 1))),
          tempBuffer, 
          outputBuffer + outputBufferSizeOffset[pass - 1]);

    for(int i = pass - 1; i > 0; i--)
    {
      bAddition(blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(i - 1))),
            outputBuffer + outputBufferSizeOffset[i], 
            outputBuffer + outputBufferSizeOffset[i - 1]);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of scan kernels: " << time * 1e-3f / iterations
            << " (us)\n";

  }

  

  float* verificationOutput = (float*)malloc(sizeBytes);
  memset(verificationOutput, 0, sizeBytes);

  

  scanLargeArraysCPUReference(verificationOutput, inputBuffer, length);

  

  if (compare<float>(outputBuffer, verificationOutput, length, (float)0.001))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  free(verificationOutput);
  free(inputBuffer);
  free(tempBuffer);
  free(blockSumBuffer);
  free(blockSumBufferSizeOffset);
  free(outputBuffer);
  free(outputBufferSizeOffset);
  return 0;
}