#include <math.h>

int saveBMPcomplexImag(char *filename, long xRes, long yRes, int normalize, float *data);

//rowSize = 4 * ceil((n*width)/32);
long rowSize(int width, int bitsPixel)
{
	return 4 * ceil((bitsPixel*width) / 32.0);
}

#ifndef myBitmap_H
#define myBitmap_H


//int saveBMPcomplex(char *filename, long xRes, long yRes, int normalize, blenderOceanComplexT *data);
//int saveBMP(char *filename, int xRes, int yRes, int normalize, blenderOceanComplexT *data);
//HBITMAP CreateBitmapFromPixelData( HDC hDC, UINT uWidth, UINT uHeight, UINT uBitsPerPixel, LPVOID pBits );
//unsigned char *LoadBitmapFile(char *filename, dibHeader *bitmapInfoHeader)


#endif