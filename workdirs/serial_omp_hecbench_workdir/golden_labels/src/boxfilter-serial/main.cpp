


#include <chrono>
#include <memory>
#include <iostream>
#include "shrUtils.h"

typedef struct __attribute__((__aligned__(4)))
{
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
} uchar4;

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;

extern
void BoxFilterHost( unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                    unsigned int uiWidth, unsigned int uiHeight, int iRadius, float fScale );


const unsigned int RADIUS = 10;                    

const float SCALE = 1.0f/(2.0f * RADIUS + 1.0f);  


inline uint DivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}





float4 rgbaUintToFloat4(unsigned int c)
{
    float4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

uchar4 rgbaUintToUchar4(unsigned int c)
{
    uchar4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}



unsigned int rgbaFloat4ToUint(float4 rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
    return uiPackedPix;
}

inline float4 operator*(float4 a, float4 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w};
}

inline void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

void BoxFilterGPU ( unsigned int *uiInput,
                    unsigned int *uiTmp,
                    unsigned int *uiDevOutput,
                    const unsigned int uiWidth, 
                    const unsigned int uiHeight, 
                    const int iRadius,
                    const float fScale,
                    const float iCycles )
{
  const int szMaxWorkgroupSize = 256;
  const int iRadiusAligned = ((iRadius + 15)/16) * 16;  

  unsigned int uiNumOutputPix = 64;  


  if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + iRadius))
    uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - iRadius;

  

  const int uiBlockWidth = DivUp((size_t)uiWidth, (size_t)uiNumOutputPix);
  const int numTeams = uiHeight * uiBlockWidth;
  const int blockSize = iRadiusAligned + uiNumOutputPix + iRadius;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iCycles; i++) {
    

        {
      uchar4 uc4LocalData[90]; 

            {
        int lid = omp_get_thread_num(); 
        int gidx = omp_get_team_num() % uiBlockWidth;
        int gidy = omp_get_team_num() / uiBlockWidth;

        int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
        int globalPosY = gidy;
        int iGlobalOffset = globalPosY * uiWidth + globalPosX;

        

        if (globalPosX >= 0 && globalPosX < uiWidth)
            

            uc4LocalData[lid] = rgbaUintToUchar4(uiInput[iGlobalOffset]);
        else
            uc4LocalData[lid] = {0, 0, 0, 0}; 

        
        if((globalPosX >= 0) && (globalPosX < uiWidth) && (lid >= iRadiusAligned) && 
           (lid < (iRadiusAligned + (int)uiNumOutputPix)))
        {
            

            float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};

            

            int iOffsetX = lid - iRadius;
            int iLimit = iOffsetX + (2 * iRadius) + 1;
            for(; iOffsetX < iLimit; iOffsetX++)
            {
                f4Sum.x += uc4LocalData[iOffsetX].x;
                f4Sum.y += uc4LocalData[iOffsetX].y;
                f4Sum.z += uc4LocalData[iOffsetX].z;
                f4Sum.w += uc4LocalData[iOffsetX].w; 
            }

            

            

            uiTmp[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
        }
      }
    }

    

        for (size_t globalPosX = 0; globalPosX < uiWidth; globalPosX++) {
      unsigned int* uiInputImage = &uiTmp[globalPosX];
      unsigned int* uiOutputImage = &uiDevOutput[globalPosX];

      float4 f4Sum;
      float4 f4iRadius = {(float)iRadius, (float)iRadius, (float)iRadius, (float)iRadius};
      float4 top_color = rgbaUintToFloat4(uiInputImage[0]);
      float4 bot_color = rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);

      f4Sum = top_color * f4iRadius;
      for (int y = 0; y < iRadius + 1; y++) 
      {
          f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
      }
      uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);
      for(int y = 1; y < iRadius + 1; y++) 
      {
          f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
          f4Sum -= top_color;
          uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }
      
      for(int y = iRadius + 1; y < uiHeight - iRadius; y++) 
      {
          f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
          f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
          uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }

      for (int y = uiHeight - iRadius; y < uiHeight; y++) 
      {
          f4Sum += bot_color;
          f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
          uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / iCycles);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage %s <PPM image> <repeat>\n", argv[0]);
    return 1;
  }
  unsigned int uiImageWidth = 0;     

  unsigned int uiImageHeight = 0;    

  unsigned int* uiInput = NULL;      

  unsigned int* uiTmp = NULL;        

  unsigned int* uiDevOutput = NULL;      
  unsigned int* uiHostOutput = NULL;      

  shrLoadPPM4ub(argv[1], (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
  printf("Image Width = %u, Height = %u, bpp = %u, Mask Radius = %u\n", 
      uiImageWidth, uiImageHeight, unsigned(sizeof(unsigned int) * 8), RADIUS);
  printf("Using Local Memory for Row Processing\n\n");

  size_t szBuff= uiImageWidth * uiImageHeight;
  size_t szBuffBytes = szBuff * sizeof (unsigned int);

  

  uiTmp = (unsigned int*)malloc(szBuffBytes);
  uiDevOutput = (unsigned int*)malloc(szBuffBytes);
  uiHostOutput = (unsigned int*)malloc(szBuffBytes);

    {
    const int iCycles = atoi(argv[2]);

    printf("Warmup..\n");
    BoxFilterGPU (uiInput, uiTmp, uiDevOutput, 
                  uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);


    printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);
    BoxFilterGPU (uiInput, uiTmp, uiDevOutput,
                  uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);
  }

  

  BoxFilterHost(uiInput, uiTmp, uiHostOutput, uiImageWidth, uiImageHeight, RADIUS, SCALE);

  

  

  int error = 0;
  for (unsigned i = RADIUS * uiImageWidth; i < (uiImageHeight-RADIUS)*uiImageWidth; i++)
  {
    if (uiDevOutput[i] != uiHostOutput[i]) {
      printf("%d %08x %08x\n", i, uiDevOutput[i], uiHostOutput[i]);
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(uiInput);
  free(uiTmp);
  free(uiDevOutput);
  free(uiHostOutput);
  return 0;
}