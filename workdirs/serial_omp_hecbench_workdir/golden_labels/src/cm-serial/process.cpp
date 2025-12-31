#include "utils.h"
#include "kernels.h"










int processQuery(const std::vector<std::string> &refFiles, 
                 const std::vector<std::string> &sigGeneNameList,
                 const std::vector<int> &sigRegValue,
                 const int nRandomGenerations,
                 const int compoundChoice,
                 const int ENFPvalue, std::ofstream &outputStream) 
{
  

  const int nGenesTotal = U133AArrayLength;

  

  int sigNGenes = sigGeneNameList.size();
  double UCmax = computeUCMax(sigNGenes, nGenesTotal);

  

  std::vector<std::string> refGeneNameList;
  populateRefGeneNameList(refFiles.front(), refGeneNameList);
  

  int qIndex[nGenesTotal] = {0}; 

  

  int errorFlag = queryToIndex(qIndex, sigGeneNameList, sigRegValue, refGeneNameList);
  

  if (errorFlag != 0) {
    std::cout << "Error finding all required genes" << std::endl;
    return -1;
  }

  

  int signatureByRNGs = nRandomGenerations * sigNGenes;

  

  

  std::default_random_engine generator (123);
  std::uniform_real_distribution<float> distribution(0.f, 1.f);

  float *randomValues = (float*) malloc(sizeof(float) * signatureByRNGs);
  float *arraysAdded = (float*) malloc(sizeof(float) * nRandomGenerations);

  

  int refRegValues[nGenesTotal];

  

  int blocksPerGrid_Gene = (int)ceil((float)nGenesTotal / (float)threadsPerBlock);

  

  int blocksPerGrid_Gen = (int)ceil((float)nRandomGenerations / (float)threadsPerBlock);

  

  

  

  int *aboveThresholdAccumulator = (int*) malloc (sizeof(int) * blocksPerGrid_Gen);

  

  

  int *dotProductResult = (int*) malloc (sizeof(int) * blocksPerGrid_Gene);

    {

  

  int setSize = 0;

  

  outputStream << "The results against the reference profiles are listed below" << std::endl;
  outputStream << "Compound" << "\t" << "setSize" << "\t" << "averageSetScore" << "\t"
               << "P-value result"<<"\t " << "ENFP"<< "\t" << "Significant"<< std::endl;

  

  for (size_t refFileLoop = 0; refFileLoop < refFiles.size(); refFileLoop++) {

    

    if(refFileLoop % 1500 == 0) {
      std::cout << "Completed : " << (int)(((float)refFileLoop / refFiles.size()) * 100) << "%" << std::endl;
    }

    

    populateRefRegulationValues(refFiles[refFileLoop], refRegValues, setSize > 0);
    setSize++;

    

    std::string drug = parseDrugInfoFromFilename(refFiles[refFileLoop], compoundChoice);

    

    

    

    if ((refFileLoop < refFiles.size() - 1) &&
        (drug == parseDrugInfoFromFilename(refFiles[refFileLoop+1], compoundChoice)))
      continue;

    


    
    double cumulativeSetScore = computeDotProduct(refRegValues, qIndex, dotProductResult,
                                nGenesTotal, blocksPerGrid_Gene, threadsPerBlock) / UCmax;

    

    double averageSetScore = cumulativeSetScore / setSize;

    

    for (int i = 0; i < signatureByRNGs; ++i) randomValues[i] = distribution(generator);
    
    

    

    double pValue = computePValue(nRandomGenerations, blocksPerGrid_Gen, threadsPerBlock,
                                  averageSetScore, 
                                  setSize, signatureByRNGs, UCmax, 
                                  aboveThresholdAccumulator,
                                  randomValues, refRegValues, arraysAdded);

    

    

    int nDrugs = getNDrugs(compoundChoice);
    double ENFP = pValue * nDrugs;
    int significant = ENFP < ENFPvalue;

    

    outputStream << drug << "\t" << setSize << "\t" << averageSetScore << "\t"
                 << pValue << "\t" << ENFP << "\t" << significant << std::endl;

    

    setSize = 0;
  }

  } 


  free(randomValues);
  free(arraysAdded);
  free(aboveThresholdAccumulator);
  free(dotProductResult);

  return 0;
}











int queryToIndex(int *qIndex, const std::vector<std::string> &sigGeneNameList,
                 const std::vector<int> &sigRegValue, const std::vector<std::string> &refGeneNameList) {
  int nMatches = 0;
  int nSigNames = sigGeneNameList.size();
  for (size_t r = 0; r < refGeneNameList.size(); r++) {
    

    

    const std::string &refGeneName = refGeneNameList[r];
    

    for (size_t g = 0; g < sigGeneNameList.size(); g++) {
      

      

      if (refGeneName.compare(sigGeneNameList[g]) == 0) {
        qIndex[r] = sigRegValue[g];
        

        nMatches++;
        

        if (nMatches == nSigNames)
          return 0;
        break;
      }
    }
  }
  

  std::cout << "nSigNames: " << nSigNames << ", nMatches: " << nMatches << std::endl;
  return -1;
}




inline int getNDrugs(const int compoundChoice) {
  switch (compoundChoice) {
    case 1:
      return 1309;
    case 2:
      return 1409;
    case 3:
      return 3738;
    case 4:
      return 6100;
  }
  return -1; 

}




double computePValue(
  const int nRandomGenerations,
  const int blocksPerGrid,
  const int threadsPerBlock, 
  const double averageSetScore, const int setSize, const int signatureByRNGs, const double UCmax,
  int *device_aboveThresholdAccumulator,
  const float *device_randomIndexArray, const int *device_refRegNum, float *device_arraysAdded) {


  

  computeRandomConnectionScores (
    device_randomIndexArray, device_refRegNum, device_arraysAdded, signatureByRNGs, UCmax, setSize, nRandomGenerations);

  countAboveThresholdHelper (device_arraysAdded, averageSetScore, device_aboveThresholdAccumulator, 
                              blocksPerGrid, nRandomGenerations);

  
  int aboveThresholdSum = 0;
  for (int ii = 0; ii < blocksPerGrid; ii++)
    aboveThresholdSum += device_aboveThresholdAccumulator[ii];

  

  return computePValueHelper(aboveThresholdSum, nRandomGenerations);
}




double computePValueHelper(const double nAboveThreshold, const int nRandomGenerations) {

  double pValueR = 0.0;
  double pValueL = 0.0;
  double pValue = 0.0;

  if (nAboveThreshold < 1) {
    pValueL = (0.5 / nRandomGenerations);
    pValueR = ((nRandomGenerations - 0.5) / nRandomGenerations);
  }
  else if (nAboveThreshold > nRandomGenerations-1) {
    pValueR = (0.5 / nRandomGenerations);
    pValueL = (nRandomGenerations - 0.5) / nRandomGenerations;
  }
  else {
    pValueL = nAboveThreshold / nRandomGenerations;
    pValueR = (nRandomGenerations - nAboveThreshold) / nRandomGenerations;
  }

  if (pValueR < pValueL)
    pValue = pValueR * 2;

  if (pValueR > pValueL)
    pValue = pValueL * 2;

  return pValue;
}



inline double computeUCMax(const int sigNGenes, const int nGenesTotal) {
  return ((sigNGenes * nGenesTotal) - (sigNGenes * (sigNGenes + 1))/2 + sigNGenes);
}



double computeDotProduct(const int *device_v1, const int *device_v2, int *result,
                         const int vLength, const int blockSize, 
                         const int nThreads) {
  

  computeDotProductHelper (result, device_v1, device_v2, blockSize, vLength);

  
  double dot = 0.0;
  for (int z = 0; z < blockSize; z++) {
    dot += result[z];
  }
  return dot;
}


