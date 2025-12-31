


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>



#define BLOCK_SIZE 256



#define integrateXcoord 1
#define integrateYcoord 0

void pad_projections(
    double* d_img,
    const int nDetXMap,
    const int nDetYMap,
    const int nElem,
    const int np)
{
  for (int gid = 0; gid < nElem; gid++)
    d_img[(np*nDetYMap *nDetXMap) + (gid*nDetYMap)] = 0;
}

void map_boudaries_kernel(
    double* d_pBound,
    const int nElem,
    const double valueLeftBound,
    const double sizeElem,
    const double offset)
{
    for (int gid = 0; gid < nElem; gid++)
    d_pBound[gid] = (gid - valueLeftBound) * sizeElem + offset;
}

void rot_detector_kernel(
          double* __restrict d_pRdetY,
          double* __restrict d_pRdetZ,
    const double* __restrict d_pYcoord,
    const double* __restrict d_pZcoord,
    const double yOffset,
    const double zOffset,
    const double phi,
    const int nElem)
{
    for (int gid = 0; gid < nElem; gid++) {
    

    d_pRdetY[gid] = ((d_pYcoord[gid] - yOffset) * cos(phi) - 
                     (d_pZcoord[gid] - zOffset) * sin(phi)) + yOffset;
    d_pRdetZ[gid] = ((d_pYcoord[gid] - yOffset) * sin(phi) +
                     (d_pZcoord[gid] - zOffset) * cos(phi)) + zOffset;
  }
}

void mapDet2Slice_kernel(
           double* __restrict const pXmapp,
           double* __restrict const pYmapp,
    double tubeX,
    double tubeY,
    double tubeZ,
    const double* __restrict const pXcoord,
    const double* __restrict const pYcoord,
    const double* __restrict const pZcoord,
    const double* __restrict const pZSlicecoord,
    const int nDetXMap,
    const int nDetYMap,
    const int nz)
{
    for (int py = 0; py < nDetXMap; py++) {
    for (int px = 0; px < nDetYMap; px++) {

      const int pos = py * nDetYMap + px;

      pXmapp[pos] = ((pXcoord[py] - tubeX)*(pZSlicecoord[nz] - pZcoord[px]) - 
          (pXcoord[py] * tubeZ) + (pXcoord[py] * pZcoord[px])) / (-tubeZ + pZcoord[px]);

      if (py == 0)
        pYmapp[px] = ((pYcoord[px] - tubeY)*(pZSlicecoord[nz] - pZcoord[px]) -
            (pYcoord[px] * tubeZ) + (pYcoord[px] * pZcoord[px])) / (-tubeZ + pZcoord[px]);
    }
  }
}

void img_integration_kernel(
    double* d_img,
    const int nPixX,
    const int nPixY,
    const bool direction,
    const int offsetX,
    const int offsetY,
    const int nSlices,
    const int teams,
    const int teamX,
    const int teamY,
    const int threads,
    const int threadX,
    const int threadY,
    const int threadZ)
{
    {
    {
  const int lx = omp_get_thread_num() % threadX;
  const int ly = omp_get_thread_num() / threadX % threadY;
  const int lz = omp_get_thread_num() / (threadX * threadY);
  const int bx = omp_get_team_num() % teamX;
  const int by = omp_get_team_num() / teamX % teamY;
  const int bz = omp_get_team_num() / (teamX * teamY);
  const int tx = bx * threadX + lx;
  const int ty = by * threadY + ly;
  const int px = tx + offsetX;
  const int py = ty + offsetY;
  const int pz = bz * threadZ + lz;
  


  if (px < nPixY && py < nPixX && pz < nSlices) {
    if (direction == integrateXcoord) {

      for (int s = 1; s <= threadY; s *= 2) {

        int spot = ty - s;

        double val = 0;

        if (spot >= 0) {
          val = d_img[(pz*nPixY*nPixX) + (offsetY + spot) * nPixY + px];
        }

        if (spot >= 0) {
          d_img[(pz*nPixY*nPixX) + (py * nPixY) + px] += val;
        }
      }
    }
    else
    {
      for (int s = 1; s <= threadX; s *= 2) {

        int spot = tx - s;

        double val = 0;

        if (spot >= 0) {
          val = d_img[(pz*nPixY*nPixX) + py * nPixY + spot + offsetX];
        }

        if (spot >= 0) {
          d_img[(pz*nPixY*nPixX) + (py * nPixY) + px] += val;
        }
      }
    }
  }
  }
  }
}

void bilinear_interpolation_kernel(
          double* __restrict d_sliceI,
    const double* __restrict d_pProj,
    const double* __restrict d_pObjX,
    const double* __restrict d_pObjY,
    const double* __restrict d_pDetmX,
    const double* __restrict d_pDetmY,
    const int nPixXMap,
    const int nPixYMap,
    const int nDetXMap,
    const int nDetYMap,
    const int nDetX,
    const int nDetY,
    const int np) 
{
    for (int py = 0; py < nPixXMap; py++) {
    for (int px = 0; px < nPixYMap; px++) {

      


      

      

      const double xNormData = nDetX - d_pObjX[py] / d_pDetmX[0];
      const int    xData = floor(xNormData);
      const double alpha = xNormData - xData;

      

      

      const double yNormData = (d_pObjY[px] / d_pDetmX[0]) - (d_pDetmY[0] / d_pDetmX[0]);
      const int    yData = floor(yNormData);
      const double beta = yNormData - yData;

      double d00, d01, d10, d11;
      if (((xNormData) >= 0) && ((xNormData) <= nDetX) && ((yNormData) >= 0) && ((yNormData) <= nDetY)) 
        d00 = d_pProj[(np*nDetYMap*nDetXMap) + (xData*nDetYMap + yData)];
      else
        d00 = 0.0;

      if (((xData + 1) > 0) && ((xData + 1) <= nDetX) && ((yNormData) >= 0) && ((yNormData) <= nDetY))
        d10 = d_pProj[(np*nDetYMap*nDetXMap) + ((xData + 1)*nDetYMap + yData)];
      else
        d10 = 0.0;

      if (((xNormData) >= 0) && ((xNormData) <= nDetX) && ((yData + 1) > 0) && ((yData + 1) <= nDetY))
        d01 = d_pProj[(np*nDetYMap*nDetXMap) + (xData*nDetYMap + yData + 1)];
      else
        d01 = 0.0;

      if (((xData + 1) > 0) && ((xData + 1) <= nDetX) && ((yData + 1) > 0) && ((yData + 1) <= nDetY))
        d11 = d_pProj[(np*nDetYMap*nDetXMap) + ((xData + 1)*nDetYMap + yData + 1)];
      else
        d11 = 0.0;

      double result_temp1 = alpha * d10 + (-d00 * alpha + d00);
      double result_temp2 = alpha * d11 + (-d01 * alpha + d01);

      d_sliceI[py * nPixYMap + px] = beta * result_temp2 + (-result_temp1 * beta + result_temp1);
    }
  }
}

void differentiation_kernel(
          double* __restrict d_pVolume,
    const double* __restrict d_sliceI,
    double tubeX,
    double rtubeY,
    double rtubeZ,
    const double* __restrict const d_pObjX,
    const double* __restrict const d_pObjY,
    const double* __restrict const d_pObjZ,
    const int nPixX,
    const int nPixY,
    const int nPixXMap,
    const int nPixYMap,
    const double du,
    const double dv,
    const double dx,
    const double dy,
    const double dz,
    const int nz) 
{
  


    for (int py = 0; py < nPixX; py++) {
    for (int px = 0; px < nPixY; px++) {

      const int pos = (nPixX*nPixY*nz) + (py * nPixY) + px;

      int coordA = py * nPixYMap + px;
      int coordB = ((py + 1) * nPixYMap) + px;
      int coordC = coordA + 1;
      int coordD = coordB + 1;

      

      double gamma = atan((d_pObjX[py] + (dx / 2.0) - tubeX) / (rtubeZ - d_pObjZ[nz]));

      

      double alpha = atan((d_pObjY[px] + (dy / 2.0) - rtubeY) / (rtubeZ - d_pObjZ[nz]));

      double dA, dB, dC, dD;

      dA = d_sliceI[coordA];
      dB = d_sliceI[coordB];
      dC = d_sliceI[coordC];
      dD = d_sliceI[coordD];

      

      if (dC == 0 && dD == 0) {
        dC = dA;
        dD = dB;
      }

      

      d_pVolume[pos] += ((dD - dC - dB + dA)*(du*dv*dz / (cos(alpha)*cos(gamma)*dx*dy)));
    }
  }
}

void division_kernel(
    double* d_img,
    const int nPixX,
    const int nPixY,
    const int nSlices,
    const int nProj)
{
    for (int pz = 0; pz < nSlices; pz++) {
    for (int py = 0; py < nPixX; py++) {
      for (int px = 0; px < nPixY; px++) {
        const int pos = (nPixX*nPixY*pz) + (py * nPixY) + px;
        d_img[pos] /= (double) nProj;
      }
    }
  }
}




void backprojectionDDb(double* const h_pVolume,
    const double* const h_pProj,
    const double* const h_pTubeAngle,
    const double* const h_pDetAngle,
    const int idXProj,
    const int nProj,
    const int nPixX,
    const int nPixY,
    const int nSlices,
    const int nDetX,
    const int nDetY,
    const double dx,
    const double dy,
    const double dz,
    const double du,
    const double dv,
    const double DSD,
    const double DDR,
    const double DAG)
{
  

  const int nDetXMap = nDetX + 1;
  const int nDetYMap = nDetY + 1;

  

  const int nPixXMap = nPixX + 1;
  const int nPixYMap = nPixY + 1;

  double* d_pProj = (double*) malloc (nDetXMap*nDetYMap*nProj * sizeof(double));
  double* d_sliceI = (double*) malloc (nPixXMap*nPixYMap * sizeof(double));
  double* d_pVolume = h_pVolume;

  


  

  const double* h_pProj_tmp;
  double* d_pProj_tmp;

  for (int np = 0; np < nProj; np++) {

    

    pad_projections (d_pProj, nDetXMap, nDetYMap, nDetXMap, np);

    

    d_pProj_tmp = d_pProj + (nDetXMap*nDetYMap*np) + 1;
    memset(d_pProj_tmp, 0, nPixY * sizeof(double));
  }

  

  for (int np = 0; np < nProj; np++)
    for (int c = 0; c < nDetX; c++) {
      h_pProj_tmp = h_pProj + (c * nDetY) + (nDetX*nDetY*np);
      d_pProj_tmp = d_pProj + (((c + 1) * nDetYMap) + 1) + (nDetXMap*nDetYMap*np);
      memcpy(d_pProj_tmp, h_pProj_tmp, nDetY * sizeof(double));
    }

  

  double* d_pDetX = (double*) malloc (nDetXMap * sizeof(double));
  double* d_pDetY = (double*) malloc (nDetYMap * sizeof(double));
  double* d_pDetZ = (double*) malloc (nDetYMap * sizeof(double));
  double* d_pObjX = (double*) malloc (nPixXMap * sizeof(double));
  double* d_pObjY = (double*) malloc (nPixYMap * sizeof(double));
  double* d_pObjZ = (double*) malloc (nSlices * sizeof(double));

  

  double* d_pDetmY = (double*) malloc (nDetYMap * sizeof(double));
  double* d_pDetmX = (double*) malloc (nDetYMap * nDetXMap * sizeof(double));

  

  double* d_pRdetY = (double*) malloc (nDetYMap * sizeof(double));
  double* d_pRdetZ = (double*) malloc (nDetYMap * sizeof(double));

  

    {

  auto start = std::chrono::steady_clock::now();

  map_boudaries_kernel(d_pDetX, nDetXMap, (double)nDetX, -du, 0.0);

  map_boudaries_kernel(d_pDetY, nDetYMap, nDetY / 2.0, dv, 0.0);

  map_boudaries_kernel(d_pDetZ, nDetYMap, 0.0, 0.0, 0.0);

  map_boudaries_kernel(d_pObjX, nPixXMap, (double)nPixX, -dx, 0.0);

  map_boudaries_kernel(d_pObjY, nPixYMap, nPixY / 2.0, dy, 0.0);

  map_boudaries_kernel(d_pObjZ, nSlices, 0.0, dz, DAG + (dz / 2.0));

    for (int i = 0; i < nPixX * nPixY * nSlices; i++)
    d_pVolume[i] = 0.0;

  

  double tubeX = 0;
  double tubeY = 0;
  double tubeZ = DSD;

  

  double isoY = 0;
  double isoZ = DDR;

  int threadX = 8;
  int threadY = 4;
  int threadZ = 8;
  int threads = threadX * threadY * threadZ;

  int teamX = (int)ceilf((float)nDetYMap / (threadX - 1));
  int teamY = 1;
  int teamZ = (int)ceilf((float)nProj / threadZ);
  int teams = teamX * teamY * teamZ;
  

  


  

  int Xk = (int)ceilf((float)nDetXMap / (threadX - 1));
  for (int k = 0; k < Xk; k++) {
    img_integration_kernel(
        d_pProj, nDetXMap, nDetYMap, integrateXcoord, 0, k * 9, nProj,
        teams, teamX, teamY, threads, threadX, threadY, threadZ);
  }

  threadX = 4;
  threadY = 8;
  threadZ = 8;
  threads = threadX * threadY * threadZ;

  teamX = 1;
  teamY = (int)ceilf((float)nDetXMap / (threadY - 1));
  teamZ = (int)ceilf((float)nProj / threadZ);
  teams = teamX * teamY * teamZ;

  

  int Yk = (int)ceilf((float)nDetYMap / (threadY - 1));
  for (int k = 0; k < Yk; k++) {
    img_integration_kernel(
        d_pProj, nDetXMap, nDetYMap, integrateYcoord, k * 9, 0, nProj,
        teams, teamX, teamY, threads, threadX, threadY, threadZ);
  }

  double* d_pDetmX_tmp = d_pDetmX + (nDetYMap * (nDetXMap-2));

  

  int projIni, projEnd, nProj2Run;
  if (idXProj == -1) {
    projIni = 0;
    projEnd = nProj;
    nProj2Run = nProj;
  }
  else {
    projIni = idXProj;
    projEnd = idXProj + 1;
    nProj2Run = 1;
  }

  

  for (int p = projIni; p < projEnd; p++) {

    

    double theta = h_pTubeAngle[p] * M_PI / 180.0;

    

    double phi = h_pDetAngle[p] * M_PI / 180.0;

    


    

    double rtubeY = ((tubeY - isoY)*cos(theta) - (tubeZ - isoZ)*sin(theta)) + isoY;
    double rtubeZ = ((tubeY - isoY)*sin(theta) + (tubeZ - isoZ)*cos(theta)) + isoZ;

    


    

    rot_detector_kernel(
        d_pRdetY, d_pRdetZ, d_pDetY, d_pDetZ, isoY, isoZ, phi, nDetYMap);

    

    for (int nz = 0; nz < nSlices; nz++) {

      


      mapDet2Slice_kernel(
          d_pDetmX, d_pDetmY, tubeX, rtubeY, rtubeZ, d_pDetX,
          d_pRdetY, d_pRdetZ, d_pObjZ, nDetXMap, nDetYMap, nz);

      


      bilinear_interpolation_kernel(
          d_sliceI, d_pProj, d_pObjX, d_pObjY, d_pDetmX_tmp, d_pDetmY,
          nPixXMap, nPixYMap, nDetXMap, nDetYMap, nDetX, nDetY, p);

      


      differentiation_kernel(
          d_pVolume, d_sliceI, tubeX, rtubeY, rtubeZ, d_pObjX, d_pObjY, d_pObjZ,
          nPixX, nPixY, nPixXMap, nPixYMap, du, dv, dx, dy, dz, nz);

    } 


  } 


  

  division_kernel(d_pVolume, nPixX, nPixY, nSlices, nProj2Run);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution %f (s)\n", time * 1e-9f);

  }

  free(d_pProj);
  free(d_pDetX);
  free(d_pDetY);
  free(d_pDetZ);
  free(d_pObjX);
  free(d_pObjY);
  free(d_pObjZ);
  free(d_pDetmY);
  free(d_pDetmX);
  free(d_pRdetY);
  free(d_pRdetZ);
}

int main() 
{
                            

  const int nPixX = 1996;   

  const int nPixY = 2457;   

  const int nSlices = 78;  

                            

  const int nDetX = 1664;   

  const int nDetY = 2048;   


  const int nProj = 15;     

  const int idXProj = -1;   


  const double dx = 0.112;  

  const double dy = 0.112;
  const double dz = 1.0;

  const double du = 0.14;   

  const double dv = 0.14;

  const double DSD = 700;   

  const double DDR = 0.0;   

  const double DAG = 25.0;  


  const size_t pixVol = nPixX * nPixY * nSlices;
  const size_t detVol = nDetX * nDetY * nProj;
  double *h_pVolume = (double*) malloc (pixVol * sizeof(double));
  double *h_pProj = (double*) malloc (detVol * sizeof(double));

  double *h_pTubeAngle = (double*) malloc (nProj * sizeof(double));
  double *h_pDetAngle = (double*) malloc (nProj * sizeof(double));
  
  

  for (int i = 0; i < nProj; i++) 
    h_pTubeAngle[i] = -7.5 + i * 15.0/nProj;

  

  for (int i = 0; i < nProj; i++) 
    h_pDetAngle[i] = -2.1 + i * 4.2/nProj;

  

  srand(123);
  for (size_t i = 0; i < pixVol; i++) 
    h_pVolume[i] = (double)rand() / (double)RAND_MAX;

  for (size_t i = 0; i < detVol; i++) 
    h_pProj[i] = (double)rand() / (double)RAND_MAX;

  backprojectionDDb(
    h_pVolume,
    h_pProj,
    h_pTubeAngle,
    h_pDetAngle,
    idXProj,
    nProj,
    nPixX, nPixY,
    nSlices,
    nDetX, nDetY,
    dx, dy, dz,
    du, dv,
    DSD, DDR, DAG);

  double checkSum = 0;
  for (size_t i = 0; i < pixVol; i++)
    checkSum += h_pVolume[i];
  printf("checksum = %lf\n", checkSum);

  free(h_pVolume);
  free(h_pTubeAngle);
  free(h_pDetAngle);
  free(h_pProj);
  return 0;
}