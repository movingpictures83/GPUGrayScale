#ifndef GPUGRAYSCALEPLUGIN_H
#define GPUGRAYSCALEPLUGIN_H

#include "Plugin.h"
#include "Tool.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUGrayScalePlugin : public Plugin, public Tool {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
 //               std::map<std::string, std::string> parameters;
};

#define TILE_WIDTH 16
__global__ void rgb2gray(float *grayImage, float *rgbImage, int channels,
                         int width, int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    // get 1D coordinate for the grayscale image
    int grayOffset = y * width + x;
    // one can think of the RGB image having
    // CHANNEL times columns than the gray scale image
    int rgbOffset = grayOffset * channels;
    float r       = rgbImage[rgbOffset];     // red value for pixel
    float g       = rgbImage[rgbOffset + 1]; // green value for pixel
    float b       = rgbImage[rgbOffset + 2]; // blue value for pixel
    // perform the rescaling and store it
    // We multiply by floating point constants
    grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}
#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

static float *readPPM(const char *filename, int* width, int* height)
{
         char buff[16];
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    /*if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }*/

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    int myX, myY;
    //read image size information
    if (fscanf(fp, "%d %d", &myX, &myY) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }
    *height = myY;
    *width = myX;

    float* retval = (float*) malloc(3*myY*myX*sizeof(float));
    unsigned char* mychararr = (unsigned char*) malloc(3*myY*myX*sizeof(unsigned char));
    fread(mychararr, 1, 3*myX*myY, fp);
    for (int i = 0; i < 3*myX*myY; i++) {
       retval[i] = (float) mychararr[i];
    }
    fclose(fp);
    return retval;
}
void writePPM(const char *filename, int width, int height, float *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",width, height);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    unsigned char* result = (unsigned char*) malloc(3*width*height);
    for (int i = 0; i < 3*width*height; i++) {
       result[i] = img[i];
    }
    // pixel data
    fwrite(img, 1, 3 * width * height, fp);
    fclose(fp);
}

#endif

