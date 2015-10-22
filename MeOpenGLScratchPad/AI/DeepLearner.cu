#include <GL/glew.h>
#include <glm\glm.hpp>
#include <gl\GL.h>
#include <qt\qdebug.h>

#include "DeepLearner.h"
#include <random>
#include <time.h>
#include <algorithm>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "StateAction.h"
#include "CudaCode\ScreenManipulation.cu"

using std::vector;

#pragma region Initialization
void DeepLearner::Initialize(int* scorePoint, int* widthPoint, int* heightPoint, int Number_Of_Inputs, float learningRate, type algoType){
	score = scorePoint;
	width = widthPoint;
	rWidth = *width / 2;
	qDebug() << "Width: " << width << endl << "Reduce Width: " << rWidth << endl;
	height = heightPoint;
	rHeight = *height / 2;
	qDebug() << "height: " << height << endl << "Reduce height: " << rHeight << endl;
	numInput = Number_Of_Inputs;
	algo = algoType;
	lr = learningRate;
	srand(time(NULL));
	numCalls = 0;
	lastInput = 0;
	reduceScreen = new float[rWidth * rHeight];
}

DeepLearner::DeepLearner()
{

}

DeepLearner::~DeepLearner()
{
	//free(reduceScreen);
}
#pragma endregion

#pragma region Cuda Code
//This is where I get calculations, and pass the screengrab down to the neurons.
__global__ void CalcInput(float* screen, int* d_Input, int* d_numInput){
	//printf("Test\n");
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	int intensity;
	intensity = (screen[80314] * 100.0f);
	*d_Input = intensity % *d_numInput;// *d_numInput - 1;
}

__global__ void GreyScreen(float* d_pixelsR, float* d_pixelsG, float* d_pixelsB,
	float* d_reducePixels, int numPixels){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("Test ID: %u ", numPixels);
	if (id < numPixels){
		d_reducePixels[id] = (d_pixelsR[id] + d_pixelsG[id] + d_pixelsB[id]) / 3;
		//printf("Reduce Pixels ");
		//printf("%f ", d_reducePixels[id]);
	}
}
#pragma endregion

#pragma region input
//Find what input would be best.
int  DeepLearner::GetInput(vector<float*> screengrab){
	numCalls++;
	if (numCalls > 3){
		if (rand.randomInRange(0, 1) < f_RandomChance){
			lastInput = rand.randomInRange(0, numInput);
		}
		else{
			GetScreen();

			float* d_screen = reduceScreen;
			int* d_Input = &lastInput;
			//std::cout << *d_Input << std::endl;
			int* d_numInput = &numInput;
			//std::cout << *d_numInput << std::endl;
			int sizeInt = sizeof(int);
			int sizeArray = (rWidth * rHeight) * sizeof(float) / 4;
			qDebug() << "Array size: " << sizeArray /4 << endl;
			cudaMalloc((void**)&d_Input, sizeInt);
			cudaMalloc((void**)&d_numInput, sizeInt);
			cudaMalloc((void**)&d_screen, sizeArray);
			cudaMemcpy(d_numInput, &numInput, sizeInt, cudaMemcpyHostToDevice);
			cudaMemcpy(d_screen, reduceScreen, sizeArray, cudaMemcpyHostToDevice);
			CalcInput <<<1, 1>>>(d_screen, d_Input, d_numInput);
			cudaMemcpy(&lastInput, d_Input, sizeInt, cudaMemcpyDeviceToHost);
			//std::cout << *d_Input << std::endl;
			//std::cout << *d_numInput << std::endl;
			cudaFree(d_Input);
			cudaFree(d_numInput);
			cudaFree(d_screen);
			numCalls = 0;
		//free(d_Input); free(d_numInput);
		}
	}
	//Store state action pairs
	//Seenms like an array of 200-250 values is what I have with full screen.
	//Multiple arrays didn't work either. I'm seriously just limited in how many I get...

	return lastInput;
}

void DeepLearner::GetScreen(){
	glReadBuffer(GL_FRONT);
	int numPixels = *width * *height;
	GLfloat* pixelsR = new GLfloat[numPixels];
	GLfloat* pixelsG = new GLfloat[numPixels];
	GLfloat* pixelsB = new GLfloat[numPixels];
	glReadPixels(0, 0, *width, *height, GL_RED, GL_FLOAT, pixelsR);
	glReadPixels(0, 0, *width, *height, GL_GREEN, GL_FLOAT, pixelsG);
	glReadPixels(0, 0, *width, *height, GL_BLUE, GL_FLOAT, pixelsB);
	float* greyScreen = new float[numPixels];
#pragma region Serial Implementation

	//Greyscale the image
	for (int i = 0; i<numPixels; ++i){
		greyScreen[i] = 0.144*pixelsR[i] + 0.587*pixelsG[i] + 0.299*pixelsB[i];
	}

	//for (int i = 0; i < numPixels; i++){
	//	if (greyScreen[i] > 0.0f){
	//		qDebug() << "Pixel: " << i << " Intensity: " << greyScreen[i];
	//	}
	//}

	//Shrink the image
	int i = 0;
	for (int c = 0; c < *height; c += 2){
		for (int r = 0; r < *width; r += 2){
			float x1 = greyScreen[r + c * 800];
			float x2 = greyScreen[r + 1 + c * 800];
			float x3 = greyScreen[r + (c + 1) * 800];
			float x4 = greyScreen[r + 1 + (c + 1)  * 800];

			float avg = (x1 + x2 + x3 + x4) / 4;

			reduceScreen[i] = avg;
			++i;
		}
	}

	//for (int i = 0; i < *width/2 * *height/2; i++){
	//	if (reduceScreen[i] > 0.0f){
	//		qDebug() << "Pixel: " << i << " Intensity: " << reduceScreen[i];
	//	}
	//}
#pragma endregion

#pragma region CUDA implementation, currently broke
	//float* d_pixelsR;
	//float* d_pixelsG;
	//float* d_pixelsB;
	//float* d_reducePixels;
	//cudaMalloc((void**)&d_pixelsR, numPixels);
	//cudaMalloc((void**)&d_pixelsG, numPixels);
	//cudaMalloc((void**)&d_pixelsB, numPixels);
	//cudaMalloc((void**)&d_reducePixels, numPixels);

	//cudaMemcpy(d_pixelsR, pixelsR, numPixels, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_pixelsG, pixelsG, numPixels, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_pixelsB, pixelsB, numPixels, cudaMemcpyHostToDevice);
	////qDebug() << "It is running";
	////qDebug() << "Num Pixels / 1024" << numPixels / 1024 << endl;
	//GreyScreen <<<1, 1024>>> (d_pixelsR, d_pixelsG, d_pixelsB, d_reducePixels, numPixels);

	////cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	////GreyScreen <<<1, numPixels >>>(d_pixelsR, d_pixelsG, d_pixelsB, d_reducePixels, numPixels);
	//cudaMemcpy(greyScreen, d_reducePixels, numPixels, cudaMemcpyDeviceToHost);

	//for (int i = 0; i < numPixels; i++){
	//	if (greyScreen[i] > 0.0f){
	//		//qDebug() << "Pixel: " << i << " Intensity: " << greyScreen[i];
	//	}
	//}

	//cudaFree(d_reducePixels); 
	//cudaFree(d_pixelsR); 
	//cudaFree(d_pixelsG); 
	//cudaFree(d_pixelsB);
	
#pragma endregion

	free(pixelsR); free(pixelsG); free(pixelsB); free(greyScreen);
	//qDebug() << "Got a screen, mebe";
}

void DeepLearner::GameOver(bool isWin){
	//Modify weights to decrease the value of what happened.
}

void DeepLearner::SwitchAlgorithm(type algoType){
	//Changes the algorith, may remove.
	algo = algoType;
}

void DeepLearner::learn(){
	//Keep practicing Games over and over
}

void DeepLearner::play(){
	//Play a game with a lr of 0;
}

#pragma endregion