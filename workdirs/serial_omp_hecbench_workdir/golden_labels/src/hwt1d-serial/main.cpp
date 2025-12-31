


#include "hwt.h"



template<typename T>
T roundToPowerOf2(T val)
{
  int bytes = sizeof(T);
  val--;
  for(int i = 0; i < bytes; i++)
    val |= val >> (1<<i);
  val++;
  return val;
}

int main(int argc, char * argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <signal length> <repeat>\n";
    return 1;
  }
  unsigned int signalLength = atoi(argv[1]);
  const int iterations = atoi(argv[2]);

  

  signalLength = roundToPowerOf2<unsigned int>(signalLength);

  unsigned int levels = 0;
  if (getLevels(signalLength, &levels) == 1) {
    std::cerr << "signalLength > 2 ^ 23 not supported\n";
    return 1;
  }

  

  float *inData = (float*)malloc(signalLength * sizeof(float));

  srand(2);
  for(unsigned int i = 0; i < signalLength; i++)
  {
    inData[i] = (float)(rand() % 10);
  }

  float *dOutData = (float*) malloc(signalLength * sizeof(float));

  memset(dOutData, 0, signalLength * sizeof(float));

  float *dPartialOutData = (float*) malloc(signalLength * sizeof(float));

  memset(dPartialOutData, 0, signalLength * sizeof(float));

  float *hOutData = (float*)malloc(signalLength * sizeof(float));

  memset(hOutData, 0, signalLength * sizeof(float));

  std::cout << "Executing kernel for " 
            << iterations << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

{
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    unsigned int levels = 0;

    getLevels(signalLength, &levels);  


    unsigned int actualLevels = levels;

    

    

    

    

    const int maxLevelsOnDevice = 9;

    float* temp = (float*)malloc(signalLength * sizeof(float));
    memcpy(temp, inData, signalLength * sizeof(float));

    int levelsDone = 0;
    int one = 1;
    unsigned int curLevels = 0;
    unsigned int curSignalLength;
    while((unsigned int)levelsDone < actualLevels)
    {
      curLevels = (levels < maxLevelsOnDevice) ? levels : maxLevelsOnDevice;

      

      if(levelsDone == 0)
      {
        curSignalLength = signalLength;
      }
      else
      {
        curSignalLength = (one << levels);
      }

      

      unsigned int groupSize = (1 << curLevels) / 2;

      unsigned int totalLevels = levels;

      
      const int teams = (curSignalLength >> 1) / groupSize;

            {
        float lmem [512];
                {
          size_t localId = omp_get_thread_num();
          size_t groupId = omp_get_team_num();
          size_t localSize = omp_get_num_threads();
          
          

          float t0 = inData[groupId * localSize * 2 + localId];
          float t1 = inData[groupId * localSize * 2 + localSize + localId];
          

          if(0 == levelsDone)
          {
             float r = 1.f / sqrtf((float)curSignalLength);
             t0 *= r;
             t1 *= r;
          }
          lmem[localId] = t0;
          lmem[localSize + localId] = t1;
           
                    
          unsigned int levels = totalLevels > maxLevelsOnDevice ? maxLevelsOnDevice: totalLevels;
          unsigned int activeThreads = (1 << levels) / 2;
          unsigned int midOutPos = curSignalLength / 2;
          
          const float rsqrt_two = 0.7071f;
          for(unsigned int i = 0; i < levels; ++i)
          {

              float data0, data1;
              if(localId < activeThreads)
              {
                  data0 = lmem[2 * localId];
                  data1 = lmem[2 * localId + 1];
              }

              

              
              if(localId < activeThreads)
              {
                  lmem[localId] = (data0 + data1) * rsqrt_two;
                  unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
                  dOutData[globalPos] = (data0 - data1) * rsqrt_two;
             
                  midOutPos >>= 1;
              }
              activeThreads >>= 1;
                        }
      
          

          
           if(0 == localId)
              dPartialOutData[groupId] = lmem[0];
        }
      }

            
      if(levels <= maxLevelsOnDevice)
      {
        dOutData[0] = dPartialOutData[0];
        memcpy(hOutData, dOutData, (one << curLevels) * sizeof(float));
        memcpy(dOutData + (one << curLevels), hOutData + (one << curLevels),
            (signalLength  - (one << curLevels)) * sizeof(float));
        break;
      }
      else
      {
        levels -= maxLevelsOnDevice;
        memcpy(hOutData, dOutData, curSignalLength * sizeof(float));
        memcpy(inData, dPartialOutData, (one << levels) * sizeof(float));
        levelsDone += (int)maxLevelsOnDevice;
      }
    }

    memcpy(inData, temp, signalLength * sizeof(float));
    free(temp);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device offload time " << (time * 1e-9f) / iterations << " (s)\n";
}

  

  calApproxFinalOnHost(inData, hOutData, signalLength);

  bool ok = true;
  for(unsigned int i = 0; i < signalLength; ++i)
  {
    if(fabs(dOutData[i] - hOutData[i]) > 0.1f)
    {
      ok = false;
      break;
    }
  }

  free(inData);
  free(dOutData);
  free(dPartialOutData);
  free(hOutData);

  if(ok)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  return 0;
}