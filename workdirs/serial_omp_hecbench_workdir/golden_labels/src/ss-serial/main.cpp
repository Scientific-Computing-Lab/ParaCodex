


#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <climits>
#include "StringSearch.h"

int verify(uint* resultCount, uint workGroupCount, 
    uint* result, uint searchLenPerWG, 
    std::vector<uint> &cpuResults) 
{
  uint count = resultCount[0];
  for(uint i=1; i<workGroupCount; ++i)
  {
    uint found = resultCount[i];
    if(found > 0)
    {
      memcpy((result + count), (result + (i * searchLenPerWG)),
          found * sizeof(uint));
      count += found;
    }
  }
  std::sort(result, result+count);

  std::cout << "Device: found " << count << " times\n"; 

  

  bool pass = (count == cpuResults.size());
  pass = pass && std::equal (result, result+count, cpuResults.begin());
  if(pass)
  {
    std::cout << "Passed!\n" << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Failed\n" << std::endl;
    return -1;
  }
}



int compare(const uchar* text, const uchar* pattern, uint length)
{
    for(uint l=0; l<length; ++l)
    {
        if (TOLOWER(text[l]) != pattern[l]) return 0;
    }
    return 1;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <path to file> <substring> <repeat>\n", argv[0]);
    return -1;
  }
  std::string file = std::string(argv[1]); 

  std::string subStr = std::string(argv[2]);
  int iterations = atoi(argv[3]);

  if(iterations < 1)
  {
    std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
    exit(0);
  }

  

  if(file.length() == 0)
  {
    std::cout << "\n Error: Input File not specified..." << std::endl;
    return -1;
  }

  

  std::ifstream textFile(file.c_str(),
      std::ios::in|std::ios::binary|std::ios::ate);
  if(! textFile.is_open())
  {
    std::cout << "\n Unable to open file: " << file << std::endl;
    return -1;
  }

  uint textLength = (uint)(textFile.tellg());
  uchar* text = (uchar*)malloc(textLength+1);
  memset(text, 0, textLength+1);
  textFile.seekg(0, std::ios::beg);
  if (!textFile.read ((char*)text, textLength))
  {
    std::cout << "\n Reading file failed " << std::endl;
    textFile.close();
    return -1;
  }
  textFile.close();

  uint subStrLength = subStr.length();
  if(subStrLength == 0)
  {
    std::cout << "\nError: Sub-String not specified..." << std::endl;
    return -1;
  }

  if (textLength < subStrLength)
  {
    std::cout << "\nText size less than search pattern (" << textLength
      << " < " << subStrLength << ")" << std::endl;
    return -1;
  }

#ifdef ENABLE_2ND_LEVEL_FILTER
  if(subStrLength != 1 && subStrLength <= 16)
  {
    std::cout << "\nSearch pattern size should be longer than 16" << std::endl;
    return -1;
  }
#endif

  std::cout << "Search Pattern : " << subStr << std::endl;

  

  std::vector<uint> cpuResults;

  uint last = subStrLength - 1;
  uint badCharSkip[UCHAR_MAX + 1];

  

  uint scan = 0;
  for(scan = 0; scan <= UCHAR_MAX; ++scan)
  {
    badCharSkip[scan] = subStrLength;
  }

  

  for(scan = 0; scan < last; ++scan)
  {
    badCharSkip[toupper(subStr[scan])] = last - scan;
    badCharSkip[tolower(subStr[scan])] = last - scan;
  }

  

  uint curPos = 0;
  while((textLength - curPos) > last)
  {
    int p=last;
    for(scan=(last+curPos); COMPARE(text[scan], subStr[p--]); scan -= 1)
    {
      if (scan == curPos)
      {
        cpuResults.push_back(curPos);
        break;
      }
    }
    curPos += (scan == curPos) ? 1 : badCharSkip[text[last+curPos]];
  }

  std::cout << "CPU: found " << cpuResults.size() << " times\n"; 

  

  const uchar* pattern = (const uchar*) subStr.c_str();

  uint totalSearchPos = textLength - subStrLength + 1;
  uint searchLenPerWG = SEARCH_BYTES_PER_WORKITEM * LOCAL_SIZE;
  uint workGroupCount = (totalSearchPos + searchLenPerWG - 1) / searchLenPerWG;

  uint* resultCount = (uint*) malloc(workGroupCount * sizeof(uint));
  uint* result = (uint*) malloc((textLength - subStrLength + 1) * sizeof(uint));

  const uint patternLength = subStrLength;
  const uint maxSearchLength = searchLenPerWG;

  double time = 0.0;

{



  if(subStrLength == 1)
  {
    std::cout <<
      "\nRun only Naive-Kernel version of String Search for pattern size = 1" <<
      std::endl;
    std::cout << "\nExecuting String search naive for " <<
      iterations << " iterations" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; i++)
    {
            {
        uchar localPattern[1];
        uint groupSuccessCounter;
        	{
          int localIdx = omp_get_thread_num();
          int localSize = omp_get_num_threads();
          int groupIdx = omp_get_team_num(); 

          

          uint lastSearchIdx = textLength - patternLength + 1;

          

          uint beginSearchIdx = groupIdx * maxSearchLength;
          uint endSearchIdx = beginSearchIdx + maxSearchLength;
          if(beginSearchIdx <= lastSearchIdx) 
	  {
            if(endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;

            

            for(int idx = localIdx; idx < patternLength; idx+=localSize)
            {
              localPattern[idx] = TOLOWER(pattern[idx]);
            }

            if(localIdx == 0) groupSuccessCounter = 0;
            
            

            for(uint stringPos=beginSearchIdx+localIdx; stringPos<endSearchIdx; stringPos+=localSize)
            {
              if (compare(text+stringPos, localPattern, patternLength) == 1)
              {
                int count;
                                count = groupSuccessCounter++;
                result[beginSearchIdx+count] = stringPos;
              }
            }

                        if(localIdx == 0) resultCount[groupIdx] = groupSuccessCounter;
          }
        }
      }
    }
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    

        
    verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults); 
  }

  

  if(subStrLength > 1) {
    std::cout << "\nExecuting String search with load balance for " <<
      iterations << " iterations" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; i++) {
            {  

        #ifdef ENABLE_2ND_LEVEL_FILTER
        uchar localPattern[32];
        #else
        uchar localPattern[16];
        #endif
        uint stack1[LOCAL_SIZE*2];
        uint stack2[LOCAL_SIZE*2];
        uint stack1Counter;
        uint stack2Counter;
        uint groupSuccessCounter;
        	{
          int localIdx = omp_get_thread_num();
          int localSize = omp_get_num_threads();
          int groupIdx = omp_get_team_num(); 
            
            

            if(localIdx == 0)
            {
                groupSuccessCounter = 0;
                stack1Counter = 0;
                stack2Counter = 0;
            }
            
            

            uint lastSearchIdx = textLength - patternLength + 1;
            uint stackSize = 0;
        
            

            uint beginSearchIdx = groupIdx * maxSearchLength;
            uint endSearchIdx = beginSearchIdx + maxSearchLength;
            if(beginSearchIdx <= lastSearchIdx) {
              if(endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;
              uint searchLength = endSearchIdx - beginSearchIdx;
        
              

              for(uint idx = localIdx; idx < patternLength; idx+=localSize)
              {
                localPattern[idx] = TOLOWER(pattern[idx]);
              }
        
                      
              uchar first = localPattern[0];
              uchar second = localPattern[1];
              int stringPos = localIdx;
              int stackPos = 0;
              int revStackPos = 0;
        
              while (true)    

              {
        
                

                if(stringPos < searchLength)
                {
                  

                  if ((first == TOLOWER(text[beginSearchIdx+stringPos])) && (second == TOLOWER(text[beginSearchIdx+stringPos+1])))
                  {
                                        stackPos = stack1Counter++;
                    stack1[stackPos] = stringPos;
                  }
                }
        
                stringPos += localSize;     

        
                                stackSize = stack1Counter;
                                
                

                if((stackSize < localSize) && ((((stringPos)/localSize)*localSize) < searchLength)) continue;
        
               #ifdef ENABLE_2ND_LEVEL_FILTER
               

               

                if(localIdx < stackSize)
                {
                                        revStackPos = stack1Counter--;
                    int pos = stack1[--revStackPos];
                    bool status = (localPattern[2] == TOLOWER(text[beginSearchIdx+pos+2]));
                    status = status && (localPattern[3] == TOLOWER(text[beginSearchIdx+pos+3]));
                    status = status && (localPattern[4] == TOLOWER(text[beginSearchIdx+pos+4]));
                    status = status && (localPattern[5] == TOLOWER(text[beginSearchIdx+pos+5]));
                    status = status && (localPattern[6] == TOLOWER(text[beginSearchIdx+pos+6]));
                    status = status && (localPattern[7] == TOLOWER(text[beginSearchIdx+pos+7]));
                    status = status && (localPattern[8] == TOLOWER(text[beginSearchIdx+pos+8]));
                    status = status && (localPattern[9] == TOLOWER(text[beginSearchIdx+pos+9]));
        
                    if (status)
                    {
                                                stackPos = stack2Counter++;
                        stack2[stackPos] = pos;
                    }
                }
        
                                stackSize = stack2Counter;
                        
                

                if((stackSize < localSize) && ((((stringPos)/localSize)*localSize) < searchLength)) continue;
                #endif
        
        
              

                if(localIdx < stackSize)
                {
                    #ifdef ENABLE_2ND_LEVEL_FILTER
                                        revStackPos = stack2Counter--;
                    int pos = stack2[--revStackPos];
                    if (compare(text+beginSearchIdx+pos+10, localPattern+10, patternLength-10) == 1)
                    #else
                                        revStackPos = stack1Counter--;
                    int pos = stack1[--revStackPos];
                    if (compare(text+beginSearchIdx+pos+2, localPattern+2, patternLength-2) == 1)
                    #endif
                    {
                        

			int count;
                                                count = groupSuccessCounter++;
                        result[beginSearchIdx+count] = beginSearchIdx+pos;
                    }
                }
        
                                if((((stringPos/localSize)*localSize) >= searchLength) && 
                   (stack1Counter <= 0) && (stack2Counter <= 0)) break;
              }
        
              if(localIdx == 0) resultCount[groupIdx] = groupSuccessCounter;
            }
          }
      }
    }

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    

        
    verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults); 
  }

  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);
}

  free(text);
  free(result);
  free(resultCount);
  return 0;
}