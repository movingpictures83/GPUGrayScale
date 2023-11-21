
#include "GPUGrayScalePlugin.h"

void GPUGrayScalePlugin::input(std::string myInputfile) {
  inputfile = myInputfile;
}

void GPUGrayScalePlugin::run() {

}

void GPUGrayScalePlugin::output(std::string outputfile) {
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  //args = gpuTKArg_read(argc, argv); /* parse the input arguments */

  hostInputImageData = readPPM(inputfile.c_str(), &imageWidth, &imageHeight);
  hostOutputImageData = (float*) malloc(3*imageWidth*imageHeight*sizeof(float));

  cudaMalloc((void **)&deviceInputImageData,
             3*imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             3*imageWidth * imageHeight * sizeof(float));
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             3*imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice);

  ///////////////////////////////////////////////////////
  //@@ INSERT CODE HERE
  dim3 dimGrid(ceil((float)3*imageWidth / TILE_WIDTH),
               ceil((float)imageHeight / TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  rgb2gray<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                    deviceInputImageData, 3, imageWidth,
                                    imageHeight);

  ///////////////////////////////////////////////////////
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             3*imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  writePPM(outputfile.c_str(), imageWidth, imageHeight, hostOutputImageData);
  //gpuTKSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  free(hostInputImageData);
  free(hostOutputImageData);
}

PluginProxy<GPUGrayScalePlugin> GPUGrayScalePluginProxy = PluginProxy<GPUGrayScalePlugin>("GPUGrayScale", PluginManager::getInstance());

