#include <GL/glew.h>
#include <glm\glm.hpp>
#include <gl\GL.h>
#include <qt\qdebug.h>

#include "Simple OpenGL Image Library\src\SOIL.h"

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

//Might want to reduce size of initial screen. Will allow for more screens stored. Longer learning time though :(
//Need to cuda-ify the learning algorithm.

#pragma region BMP functions

void writeBMP(float* data, int width, int height, char* dest = "D:/DefaultSoil.bmp", int channels = 1){
	std::vector< unsigned char > rgbdata(width * height * channels);
	
	for (int i = 0; i < width * height; ++i){
		rgbdata[i] = data[i] * 255;
	}

	SOIL_save_image(dest, SOIL_SAVE_TYPE_BMP, width, height, channels, rgbdata.data());

}

#pragma endregion

#pragma region Initialization
void DeepLearner::Initialize(int* scorePoint, int* widthPoint, int* heightPoint, int Number_Of_Inputs, float learningRate, type algoType){
	score = scorePoint;
	width = widthPoint;
	rWidth = *width / 2;
	//qDebug() << "Width: " << width << endl << "Reduce Width: " << rWidth << endl;
	height = heightPoint;
	rHeight = *height / 2;
	//qDebug() << "height: " << height << endl << "Reduce height: " << rHeight << endl;
	numInput = Number_Of_Inputs;
	algo = algoType;
	lr = learningRate;
	srand(time(NULL));
	numCalls = 0;
	lastInput = 0;
	lastScore = 0;
	numFirstHiddenNeurons = 1000;
	//postedge31-35 postgradient31-34
	screenSize = 73 * 73 * 8;// 9;
	firstHiddenWeightsSize = numFirstHiddenNeurons * screenSize;
	numScreenHistory = 15;

	//300 x 300
	reduceScreen = new float[90000];// rHeight * rHeight];

	//inputWeights = new float[50 * 50];
	//for (int i = 0; i < 50 * 50; ++i){
	//	inputWeights[i] = 1;//rand.randomFloat();
	//}

	//InputBias = new float[48 * 48];
	//for (int i = 0; i < 48 * 48; ++i){
	//	InputBias[i] = -1.0f *  rand.randomFloat();
	//}

	//Number of hidden nodes * number of input nodes
	firstHiddenWeights = new float[numFirstHiddenNeurons * (screenSize)];
	for (int i = 0; i < numFirstHiddenNeurons * screenSize; ++i){
		firstHiddenWeights[i] = rand.randomFloat();
	}

	//Number of hidden nodes * number of input nodes
	secondHiddenWeights = new float[20 * 15];
	for (int i = 0; i < 20 * 15; ++i){
		secondHiddenWeights[i] = rand.randomFloat();
	}

	//Number of hidden nodes * number of input nodes
	thirdHiddenWeights = new float[15 * 10];
	for (int i = 0; i < 15 * 10; ++i){
		thirdHiddenWeights[i] = 1;//rand.randomFloat();
	}

	firstBias = new float[numFirstHiddenNeurons];
	//Bias is currently generic numbers
	//Bias is set for specifically for breakout
	for (int i = 0; i < numFirstHiddenNeurons; ++i){
		firstBias[i] = -1 * rand.randomInRange(7500, 7700);// 11500, 12500);// 7750, 7900);
		//22000, 27000);
		//20000, 21500);
		//1500, 4000);
		//20, 2000);
		// 35, 50);
		//215, 230);
		// full 3x3, 395, 410);
		// 43, 52);
		// 175, 200);
	}

	secondBias = new float[15];
	//Bias is currently generic numbers
	//Bias is set for specifically for breakout
	for (int i = 0; i < 15; ++i){
		secondBias[i] =  -1 * rand.randomInRange(2, 10);
	}

	//1-2 layer
	outputWeights = new float[numFirstHiddenNeurons * numInput];
	for (int i = 0; i < numFirstHiddenNeurons * numInput; ++i){
		outputWeights[i] = rand.randomFloat();
	}

	//300x300 is screen size, 225 frames is 15 seconds
	screenStorageCount = 0;
	inputStorage = new int[screenSize * numScreenHistory];
	keypressStorage = new int[numScreenHistory];

	FirstLayerFire = new float[numFirstHiddenNeurons];
	OutputLayerTotals = new float[numInput];

	FirstLayerStorage = new float[numFirstHiddenNeurons * numScreenHistory];
	OutputLayerStorage = new float[numInput * numScreenHistory];
	
	//Input for storing what keys were pressed
	//inputStorage = new int[225];
	FullStorage = false;
}

DeepLearner::DeepLearner() : f_RandomChance(0.950)
{

}

DeepLearner::~DeepLearner()
{
	//delete[] screenStorage;
	//delete[] outputWeights;
	//delete[] bias;
	//delete[] firstHiddenWeights;
	//delete[] inputWeights;
	//delete[] reduceScreen;
	//free(reduceScreen);
}
#pragma endregion

#pragma region Cuda Code
//id = 0, positions I want to operate on are 0, 1, 50, 51
//id = 1, positions I want to operate on are 2, 3, 52, 53
//id = 2, positions I want to operate on are 4, 5, 54, 55
//id = 25, positions I want to operate on are 100, 101, 150, 151
__global__ void CalcInput(float* screen, float* weight, float* d_Votes, int stride){

	//Current implementation, idk if it works. Probably doesn't, but it is worth a try, I think.
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	d_Votes[id] = 0;

	d_Votes[id] += screen[id] * weight[id];
	d_Votes[id] += screen[id + 1] * weight[id + 1];
	d_Votes[id] += screen[stride] * weight[stride];
	d_Votes[id] += screen[stride + 1] * weight[stride + 1];

	d_Votes[id] /= 4;
}

__global__ void FirstHidden(float* input, float* weight, float* bias, int d_numVotes, float* d_votes){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	float total = 0.0f;

	//printf("Num Votes: %i", d_numVotes);

	for (int i = 0; i < d_numVotes; ++i){
		//if (weight[id*d_numVotes + i] > 0) printf("Weight higher than 0: %f", weight[id*d_numVotes + i]);
		//if (input[i] > 0) printf("Input: %f ", input[i]);
		//printf("Weight: %f\n", weight[id * d_numVotes + i]);
		//printf("Input: %f, Weight: %f\n", input[i], weight[id * d_numVotes + i]);
		float sig = input[i] * weight[id * d_numVotes + i];
		total += sig;// (1 / (1 + exp(-sig)));
	}

	//total /= d_numVotes;

	printf("total: %f, Bias: %f\n", total, bias[id]);
	total += bias[id];
	//printf("Total: %f\n", total);
	//printf("Bias: %f\n", bias[id]);
	total = (1 / (1 + exp(-total)));
	//total = ((int)(total)) % 3;
	//if (total < 0.1) printf("Total %i: %f\n", id, total);

	//printf("Total: %f\n", total);
	d_votes[id] = total;
}

__global__ void OutputLayer(float* hiddenVotes, float* weight, int d_numHiddenNodes, float* d_votes){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	float total = 0.0f;

	for (int i = 0; i < d_numHiddenNodes; ++i){
		//printf("Hidden Votes: %i\n", hiddenVotes[i]);
		//printf("Hidden Votes: %f, Weight: %f\n", hiddenVotes[i], weight[id * d_numHiddenNodes + i]);
		total += hiddenVotes[i] * weight[id * d_numHiddenNodes + i];
		//printf("Weight: %f", weight[id * d_numHiddenNodes + i]);
		//printf("\n");
	}

	d_votes[id] = total;
	//printf("Votes: %f\n", d_votes[id]);
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

__global__ void ApplyPool5(float* input, float* output){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (int i = 0; i < 148; ++i){
		//float total = input[i * 2 + id * 2 * 296] +
		//	input[i * 2 + 1 + id * 2 * 296] + input[i * 2 + id * 2 * 296 + 296] + input[i * 2 + 1 + id * 2 * 296 + 296];
		//total /= 4;

		float total = 0;
		total = max(	   input[i * 2 + id * 2 * 296], 
						   input[i * 2 + id * 2 * 296 + 1]);
		total = max(total, input[i * 2 + id * 2 * 296 + 296]);
		total = max(total, input[i * 2 + id * 2 * 296 + 296 + 1]);

		//float total = ((float)i) / 148.0f; // input[i * 2 + id * 2 * 296];

		//if (total < -0.1f){
		//	printf("ApplyFirstPool total: %f\n", total);
		//}
		output[i + id * 148] = total;
		output[i + id * 148] = 1 / (1 + exp(-(output[i + id * 148] * 2 - 1)));
	}
}

__global__ void ApplySecondPool(float* input, float* output){
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < 73; ++i){
		//float total = input[i * 2 + id * 2 * 296] +
		//	input[i * 2 + 1 + id * 2 * 296] + input[i * 2 + id * 2 * 296 + 296] + input[i * 2 + 1 + id * 2 * 296 + 296];
		//total /= 4;

		float total = 0;
		total = max(	   input[i * 2 + id * 2 * 146], 
						   input[i * 2 + id * 2 * 146 + 1]);
		total = max(total, input[i * 2 + id * 2 * 146 + 146]);
		total = max(total, input[i * 2 + id * 2 * 146 + 146 + 1]);

		output[i + id * 73] = total;//((float)i) / 73.0f;
		output[i + id * 73] = 1 / (1 + exp(-(output[i + id * 73] * 2 - 1)));
	}
}

//Each thread will be a row, ID is for that row, Input 300x300, output 296x296
__global__ void ApplyMat5(float* input, float* output, float* matrix){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	for (int i = 0; i < 296; ++i){
		float total = 0.0f;
		total += input[id * 300 + i] * matrix[0];
		total += input[id * 300 + i + 1] * matrix[1];
		total += input[id * 300 + i + 2] * matrix[2];
		total += input[id * 300 + i + 3] * matrix[3];
		total += input[id * 300 + i + 4] * matrix[4];

		total += input[id * 300 + i + 300 * 1] * matrix[5];
		total += input[id * 300 + i + 300 * 1 + 1] * matrix[6];
		total += input[id * 300 + i + 300 * 1 + 2] * matrix[7];
		total += input[id * 300 + i + 300 * 1 + 3] * matrix[8];
		total += input[id * 300 + i + 300 * 1 + 4] * matrix[9];

		total += input[id * 300 + i + 300 * 2] * matrix[10];
		total += input[id * 300 + i + 300 * 2 + 1] * matrix[11];
		total += input[id * 300 + i + 300 * 2 + 2] * matrix[12];
		total += input[id * 300 + i + 300 * 2 + 3] * matrix[13];
		total += input[id * 300 + i + 300 * 2 + 4] * matrix[14];

		total += input[id * 300 + i + 300 * 3] * matrix[15];
		total += input[id * 300 + i + 300 * 3 + 1] * matrix[16];
		total += input[id * 300 + i + 300 * 3 + 2] * matrix[17];
		total += input[id * 300 + i + 300 * 3 + 3] * matrix[18];
		total += input[id * 300 + i + 300 * 3 + 4] * matrix[19];

		total += input[id * 300 + i + 300 * 4] * matrix[20];
		total += input[id * 300 + i + 300 * 4 + 1] * matrix[21];
		total += input[id * 300 + i + 300 * 4 + 2] * matrix[22];
		total += input[id * 300 + i + 300 * 4 + 3] * matrix[23];
		total += input[id * 300 + i + 300 * 4 + 4] * matrix[24];

		total = fmax(0.0f, total);

		output[i + id * 296] = total;
	}
}

//Each thread will be a row, ID is for that row, Input 300x300, output 296x296
__global__ void ApplyMat3(float* input, float* output, float* matrix){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	//for (int i = 0; i < 148 * 148; ++i){
	//	if(input[i] > 0.1f) printf("Input above 0, %i", i);
	//}

	for (int i = 0; i < 146; ++i){
		float total = 0.0f;

		//if (input[id * 148 + i] > 0.1f) printf("Input above 0, %i", id * 148 + i);

		total += input[id * 148 + i] * matrix[0];
		total += input[id * 148 + i + 1] * matrix[1];
		total += input[id * 148 + i + 2] * matrix[2];

		total += input[id * 148 + i + 148 * 1] * matrix[3];
		total += input[id * 148 + i + 148 * 1 + 1] * matrix[4];
		total += input[id * 148 + i + 148 * 1 + 2] * matrix[5];

		total += input[id * 148 + i + 148 * 2] * matrix[6];
		total += input[id * 148 + i + 148 * 2 + 1] * matrix[7];
		total += input[id * 148 + i + 148 * 2 + 2] * matrix[8];

		//if (total < -0.1f || total > 0.1f) printf("Total: %f", total);

		total = fmax(0.0f, total);

		output[i + id * 146] = total;
	}
}

__global__ void CombineScreen(float* d_postEdge1, 
	float* d_postEdge2,
	float* d_postGradient1, 
	float* d_postGradient2, 
	float* d_postGradient3,
	float* d_postSobel3LR,
	float* d_postSobel3UD,
	float* d_postSmooth31,
	float* d_output){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	for (int i = 0; i < 73; ++i){
		d_output[i + id * 73 + 73 * 73 * 0] =     d_postEdge1[id * 73 + i];
		d_output[i + id * 73 + 73 * 73 * 1] =     d_postEdge2[id * 73 + i];
		d_output[i + id * 73 + 73 * 73 * 2] = d_postGradient1[id * 73 + i];
		d_output[i + id * 73 + 73 * 73 * 3] = d_postGradient2[id * 73 + i];
		d_output[i + id * 73 + 73 * 73 * 4] = d_postGradient3[id * 73 + i];
		d_output[i + id * 73 + 73 * 73 * 5] =  d_postSobel3LR[id * 73 + i];
		d_output[i + id * 73 + 73 * 73 * 6] =  d_postSobel3UD[id * 73 + i];
		d_output[i + id * 73 + 73 * 73 * 7] =  d_postSmooth31[id * 73 + i];
	}
}

#pragma endregion

#pragma region input
//Find what input would be best.
//This is possible with the number of threads and blocks form the ID, ID can be considered 0 based, idk how though but it works
//After that the array is a 1D array anyways that I'm working with shouldn't be hard to just copy pasta into a cuda function

//Might wanna remove input weights. Seems useless I think
using std::vector;

//Store which neurons fired and use that instead of IsPositive to edit the weights. Also store totals for each output neuron.
void DeepLearner::StoreScreen(float* screenBits){
	if (rand.randomFloat() > 0.5f) return;

	for (int i = 0; i < screenSize; ++i){
		inputStorage[i + screenSize * screenStorageCount] = screenBits[i];
	}

	for (int i = 0; i < numFirstHiddenNeurons; ++i){
		FirstLayerStorage[i + numFirstHiddenNeurons * screenStorageCount] = FirstLayerFire[i];
	}

	for (int i = 0; i < numInput; ++i){
		OutputLayerStorage[i + numInput * screenStorageCount] = OutputLayerTotals[i];
	}

	keypressStorage[screenStorageCount] = lastInput;
	//int screenStride = 50 * 50;
	//int rowStride = 50;
	//for (int x = 0; x < 50; ++x){
	//	for (int y = 0; y < 50; ++y){
	//		//Index violation right now, so sad :(
	//		inputStorage[screenStride * screenStorageCount + x * rowStride + y] = reduceScreen[x * rowStride + y];
	//	}
	//}

	//inputStorage[screenStorageCount] = lastInput;

	++screenStorageCount;
	if (screenStorageCount > numScreenHistory - 1){
		screenStorageCount = 0;
		FullStorage = true;
	}
}

int  DeepLearner::GetInput(vector<float*> screengrab){
	if (pause) return 0;
	numCalls++;
	float* screenBits = new float[screenSize];
	GetScreen();

	//update weights if I got higher score
	if (lastScore < *score){
		learn(true);
		lastScore = *score;
	}

	if (numCalls > 3){
#pragma region Random Input
		if (rand.randomInRange(0, 1) < f_RandomChance){
			lastInput = rand.randomInRange(0, numInput -1);
			numCalls = 0;
			//f_RandomChance = 0.00f;
		}
#pragma endregion

		else{

			//Screen maniputlation only works for 800x600 screens currently. Changes to a screen of 400x300 greyscaled
			//Also stores the 8x6 mini pixel set in the variable screenbits
#pragma region screen manipulation
			//Reduce screen is 300x300
			//screenBits is what the AI interacts with
			//Layer sizes, 300x300, 296x296, 148x148, 146x146, 73x73
			//Total Sizes, 90000, 262848, 65712,191844, 47961
			float* d_input;
			float* d_output;
			float* d_matrix;
			int sizeInput;
			int sizeOutput;
			int sizeMatrix;
			float* d_sobelL;
			float* d_sobelR;
			float* d_sobelU;
			float* d_sobelD;
			
#pragma region First Layer
			//Deleted during first layer
#pragma region Matrix Setup
			float sobelL5[25] = {
				+01.0f, +02.0f, -00.0f, -02.0f, -01.0f,
				+04.0f, +08.0f, +00.0f, -08.0f, -04.0f,
				+06.0f, +12.0f, +00.0f, -12.0f, -06.0f,
				+04.0f, +08.0f, +00.0f, -08.0f, -04.0f,
				+08.0f, +02.0f, -00.0f, -02.0f, -08.0f
			};
			float sobelR5[25] = {
				-01.0f, -02.0f, -00.0f, +02.0f, +01.0f,
				-04.0f, -08.0f, +00.0f, +08.0f, +04.0f,
				-06.0f, -12.0f, +00.0f, +12.0f, +06.0f,
				-04.0f, -08.0f, +00.0f, +08.0f, +04.0f,
				-08.0f, -02.0f, -00.0f, +02.0f, +08.0f
			};
			float sobelD5[25] = {
				-01.0f, -04.0f, -06.0f, -04.0f, -01.0f,
				-02.0f, -08.0f, -12.0f, -08.0f, -02.0f,
				+00.0f, +00.0f, +00.0f, -00.0f, -00.0f,
				+02.0f, +08.0f, +12.0f, +08.0f, +02.0f,
				+01.0f, +04.0f, +06.0f, +04.0f, +01.0f
			};
			float sobelU5[25] = {
				+01.0f, +04.0f, +06.0f, +04.0f, +01.0f,
				+02.0f, +08.0f, +12.0f, +08.0f, +02.0f,
				+00.0f, +00.0f, +00.0f, -00.0f, -00.0f,
				-02.0f, -08.0f, -12.0f, -08.0f, -02.0f,
				-01.0f, -04.0f, -06.0f, -04.0f, -01.0f
			};
			float Identity5[25] = {
				+01.0f, +01.0f, +01.0f, +01.0f, +01.0f,
				+01.0f, +01.0f, +01.0f, +01.0f, +01.0f,
				+01.0f, +01.0f, +01.0f, +01.0f, +01.0f,
				+01.0f, +01.0f, +01.0f, +01.0f, +01.0f,
				+01.0f, +01.0f, +01.0f, +01.0f, +01.0f
			};
			float edge5[25] = {
					+0.0f, -1.0f, -1.0f, -1.0f, +0.0f,
					-1.0f, -1.0f, +3.0f, -1.0f, -1.0f,
					-1.0f, +3.0f, +4.0f, +3.0f, -1.0f,
					-1.0f, -1.0f, +3.0f, -1.0f, -1.0f,
					+0.0f, -1.0f, -1.0f, -1.0f, +0.0f
			}; 

#pragma endregion

#pragma region Variables
			float* preSobelL5;
			float* preSobelR5;
			float* preSobelD5;
			float* preSobelU5;
			float* preSmooth5;
			float* preSharp5;

			//Need to delete during second layer
			float* postSobelL5;
			float* postSobelR5;
			float* postSobelD5;
			float* postSobelU5;
			float* postSmooth5;
			float* postSharp5;
#pragma endregion
			
#pragma region Sobel
			preSobelL5 = new float[296 * 296];
			preSobelR5 = new float[296 * 296];
			preSobelD5 = new float[296 * 296];
			preSobelU5 = new float[296 * 296];

			postSobelL5 = new float[296 * 296];
			postSobelR5 = new float[296 * 296];
			postSobelD5 = new float[296 * 296];
			postSobelU5 = new float[296 * 296];

#pragma region Convolution
			sizeInput = 300 * 300 *sizeof(float);
			sizeOutput = 296 * 296 * sizeof(float);
			sizeMatrix = 5 * 5 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);
			cudaMalloc((void**)&d_matrix, sizeMatrix);

			cudaMemcpy(d_input, reduceScreen, sizeInput, cudaMemcpyHostToDevice);

			//Input, output, matrix
			cudaMemcpy(d_matrix, sobelL5, sizeMatrix, cudaMemcpyHostToDevice);
			ApplyMat5 << <1, 296 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preSobelL5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_matrix, sobelR5, sizeMatrix, cudaMemcpyHostToDevice);
			ApplyMat5 << <1, 296 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preSobelR5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_matrix, sobelU5, sizeMatrix, cudaMemcpyHostToDevice);
			ApplyMat5 << <1, 296 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preSobelD5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_matrix, sobelD5, sizeMatrix, cudaMemcpyHostToDevice);
			ApplyMat5 << <1, 296 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preSobelU5, d_output, sizeOutput, cudaMemcpyDeviceToHost);
			
			cudaFree(d_input);
			cudaFree(d_output);
			cudaFree(d_matrix);
#pragma endregion

#pragma region Pooling
			sizeInput = 296 * 296 * sizeof(float);
			sizeOutput = 148 * 148 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_input, preSobelL5, sizeInput, cudaMemcpyHostToDevice);
			ApplyPool5 << < 1, 148 >> > (d_input, d_output);
			cudaMemcpy(postSobelL5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, preSobelR5, sizeInput, cudaMemcpyHostToDevice);
			ApplyPool5 << < 1, 148 >> > (d_input, d_output);
			cudaMemcpy(postSobelR5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, preSobelD5, sizeInput, cudaMemcpyHostToDevice);
			ApplyPool5 << < 1, 148 >> > (d_input, d_output);
			cudaMemcpy(postSobelD5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, preSobelU5, sizeInput, cudaMemcpyHostToDevice);
			ApplyPool5 << < 1, 148 >> > (d_input, d_output);
			cudaMemcpy(postSobelU5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);

#pragma endregion

#pragma endregion

#pragma region sharp5

			float* sharp5;
			sharp5 = new float[5 * 5];
			preSharp5 = new float[296 * 296];
			postSharp5 = new float[148 * 148];
#pragma region Setup Matrix
			for (int r = 0; r < 5; ++r){
				for (int c = 0; c < 5; ++c){
					float total;
					switch (c){
					case 0:
						total = 0.0f;
						break;
					case 1:
						total = -1.0f;
						break;
					case 2:
						total = -1.0f;
						break;
					case 3:
						total = -1.0f;
						break;
					case 4:
						total = 0.0f;
						break;
					}
					switch (r){
					case 0:
						break;
					case 1:
						switch (c)
						{
						case 0:
							total = -1.0f;
							break;
						case 2:
							total = 4.0f;
							break;
						case 4:
							total = -1.0f;
							break;
						default:
							break;
						}
						break;
					case 2:
						switch (c)
						{
						case 0:
							total = -1.0f;
							break;
						case 1:
							total = 4.0f;
							break;
						case 2:
							total = 5.0f;
							break;
						case 3:
							total = 4.0f;
							break;
						case 4:
							total = -1.0f;
							break;
						default:
							break;
						}
						break;
					case 3:
						switch (c)
						{
						case 0:
							total = -1.0f;
							break;
						case 2:
							total = 4.0f;
							break;
						case 4:
							total = -1.0f;
							break;
						default:
							break;
						}
						break;
					case 4:
						break;
					}
					sharp5[r * 5 + c] = total;
				}
			}
#pragma endregion

#pragma region convolution
			d_input;
			d_output;
			d_matrix;

			sizeInput = 300 * 300 * sizeof(float);
			sizeOutput = 296 * 296 * sizeof(float);
			sizeMatrix = 5 * 5 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);
			cudaMalloc((void**)&d_matrix, sizeMatrix);

			cudaMemcpy(d_input, reduceScreen, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrix, sobelD5, sizeMatrix, cudaMemcpyHostToDevice);

			//Input, output, matrix
			ApplyMat5 << <1, 296 >> >(d_input, d_output, d_matrix);

			cudaMemcpy(preSharp5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);
			cudaFree(d_matrix);
#pragma endregion

#pragma region Pooling
			sizeInput = 296 * 296 * sizeof(float);
			sizeOutput = 148 * 148 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_input, preSharp5, sizeInput, cudaMemcpyHostToDevice);

			ApplyPool5 << < 1, 148 >> > (d_input, d_output);

			cudaMemcpy(postSharp5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);

#pragma endregion


#pragma endregion

#pragma region smooth5

			float* smooth5;
			smooth5 = new float[5 * 5];
			preSmooth5 = new float[296 * 296];
			postSmooth5 = new float[148 * 148];

#pragma region setup matrix
			for (int r = 0; r < 5; ++r){
				for (int c = 0; c < 5; ++c){
					float total;
					switch (c){
					case 0:
						total = 0.01f;
						break;
					case 1:
						total = 0.02f;
						break;
					case 2:
						total = 0.04f;
						break;
					case 3:
						total = 0.02f;
						break;
					case 4:
						total = 0.01f;
						break;
					}
					switch (r){
					case 0:
						break;
					case 1:
						total *= 2;
						break;
					case 2:
						total *= 4;
						break;
					case 3:
						total *= 2;
						break;
					case 4:
						break;
					}
					smooth5[r * 5 + c] = total;
				}
			}
#pragma endregion

#pragma region Convolution
			d_input;
			d_output;
			d_matrix;

			sizeInput = 300 * 300 * sizeof(float);
			sizeOutput = 296 * 296 * sizeof(float);
			sizeMatrix = 5 * 5 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);
			cudaMalloc((void**)&d_matrix, sizeMatrix);

			cudaMemcpy(d_input, reduceScreen, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrix, smooth5, sizeMatrix, cudaMemcpyHostToDevice);

			//Input, output, matrix
			ApplyMat5 << <1, 296 >> >(d_input, d_output, d_matrix);

			cudaMemcpy(preSmooth5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);
			cudaFree(d_matrix);
#pragma endregion

#pragma region Pooling
			sizeInput = 296 * 296 * sizeof(float);
			sizeOutput = 148 * 148 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_input, preSmooth5, sizeInput, cudaMemcpyHostToDevice);

			ApplyPool5 << < 1, 148 >> > (d_input, d_output);

			cudaMemcpy(postSmooth5, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);

#pragma endregion

#pragma endregion

			//writeBMP(preEdge5, 296, 296,   "D:/PreEdge5.bmp");
			//writeBMP(preSharp5, 296, 296,  "D:/preSharp5.bmp");
			//writeBMP(preSmooth5, 296, 296, "D:/preSmooth5.bmp");

			//writeBMP(postEdge5, 148, 148,   "D:/PostEdge5.bmp");
			//writeBMP(postSharp5, 148, 148,  "D:/PostSharp5.bmp");
			//writeBMP(postSmooth5, 148, 148, "D:/PostSmooth5.bmp");

#pragma region Cleanup data
			delete[] smooth5;
			delete[] preSmooth5;
			delete[] preSobelL5;
			delete[] preSobelR5;
			delete[] preSobelD5;
			delete[] preSobelU5;
			delete[] preSharp5;
#pragma endregion

#pragma endregion

			//preSobelL5, preSobelR5, preSobelD5, preSobelU5, preSmooth5
			//postedge31-35, postgradient31-34
#pragma region Second Layer
			float* edge3 =     new float[9];
			float* sharp3 =    new float[9];
			float* smooth3 =   new float[9];
			float* gradient3 = new float[9];
			float* sobel3LR =  new float[9];
			float* sobel3UD =  new float[9];

			//All sobel and smooth, output postedge31-35
#pragma region edge3
			float* preEdge31 = new float[146 * 146];
			float* preEdge32 = new float[146 * 146];
			float* postEdge31 = new float[73 * 73];
			float* postEdge32 = new float[73 * 73];
#pragma region setup matrix
			edge3[0] = -1.0f;
			edge3[1] = -1.0f;
			edge3[2] = -1.0f;
			edge3[3] = -1.0f;
			edge3[4] = +8.0f;
			edge3[5] = -1.0f;
			edge3[6] = -1.0f;
			edge3[7] = -1.0f;
			edge3[8] = -1.0f;
#pragma endregion

#pragma region convolution
			sizeInput = 148 * 148 * sizeof(float);
			sizeOutput = 146 * 146 * sizeof(float);
			sizeMatrix = 3 * 3 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);
			cudaMalloc((void**)&d_matrix, sizeMatrix);

			cudaMemcpy(d_matrix, edge3, sizeMatrix, cudaMemcpyHostToDevice);

			cudaMemcpy(d_input, postSobelD5, sizeInput, cudaMemcpyHostToDevice);
			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preEdge31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, postSobelL5, sizeInput, cudaMemcpyHostToDevice);
			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preEdge32, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input); 
			cudaFree(d_output); 
			cudaFree(d_matrix);
#pragma endregion

#pragma region pooling

			sizeInput = 146 * 146 * sizeof(float);
			sizeOutput = 73 * 73 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_input, preEdge31, sizeInput, cudaMemcpyHostToDevice);
			ApplySecondPool << <1, 73 >> >(d_input, d_output);
			cudaMemcpy(postEdge31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, preEdge32, sizeInput, cudaMemcpyHostToDevice);
			ApplySecondPool << <1, 73 >> >(d_input, d_output);
			cudaMemcpy(postEdge32, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input); 
			cudaFree(d_output);
			
#pragma endregion

#pragma endregion

#pragma region sharp3 unused
//			float* preSharp31 = new float[146 * 146];
//			float* postSharp31 = new float[73 * 73];
//#pragma region setup matrix
//			sharp3[0] = 0.0f;
//			sharp3[1] = -1.0f;
//			sharp3[2] = 0.0f;
//			sharp3[3] = -1.0f;
//			sharp3[4] = 5.0f;
//			sharp3[5] = -1.0f;
//			sharp3[6] = 0.0f;
//			sharp3[7] = -1.0f;
//			sharp3[8] = 0.0f;
//#pragma endregion
//
//#pragma region Convolution
//			sizeInput = 148 * 148 * sizeof(float);
//			sizeOutput = 146 * 146 * sizeof(float);
//			sizeMatrix = 3 * 3 * sizeof(float);
//
//			cudaMalloc((void**)&d_input, sizeInput);
//			cudaMalloc((void**)&d_output, sizeOutput);
//			cudaMalloc((void**)&d_matrix, sizeMatrix);
//
//			cudaMemcpy(d_input, postEdge5, sizeInput, cudaMemcpyHostToDevice);
//			cudaMemcpy(d_matrix, sharp3, sizeMatrix, cudaMemcpyHostToDevice);
//
//			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);
//
//			cudaMemcpy(preSharp31, d_output, sizeOutput, cudaMemcpyDeviceToHost);
//
//			cudaFree(d_input); 
//			cudaFree(d_output); 
//			cudaFree(d_matrix);
//
//#pragma endregion
//
//#pragma region pooling
//			sizeInput = 146 * 146 * sizeof(float);
//			sizeOutput = 73 * 73 * sizeof(float);
//
//			cudaMalloc((void**)&d_input, sizeInput);
//			cudaMalloc((void**)&d_output, sizeOutput);
//
//			cudaMemcpy(d_input, preSharp31, sizeInput, cudaMemcpyHostToDevice);
//
//			ApplySecondPool << <1, 73 >> >(d_input, d_output);
//
//			cudaMemcpy(postSharp31, d_output, sizeOutput, cudaMemcpyDeviceToHost);
//
//			cudaFree(d_input); 
//			cudaFree(d_output);
//
//#pragma endregion
//
//
#pragma endregion

			//postSharp5
#pragma region smooth3
			float*  preSmooth31 = new float[146 * 146];
			float* postSmooth31 = new float[73 * 73];

#pragma region setup matrix

			smooth3[0] = -1.0f;
			smooth3[1] = -1.0f;
			smooth3[2] = -1.0f;
			smooth3[3] = -1.0f;
			smooth3[4] = 8.0f;
			smooth3[5] = -1.0f;
			smooth3[6] = -1.0f;
			smooth3[7] = -1.0f;
			smooth3[8] = -1.0f;
#pragma endregion

#pragma region Convolution
			sizeInput = 148 * 148 * sizeof(float);
			sizeOutput = 146 * 146 * sizeof(float);
			sizeMatrix = 3 * 3 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);
			cudaMalloc((void**)&d_matrix, sizeMatrix);

			cudaMemcpy(d_input, postSharp5, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrix, smooth3, sizeMatrix, cudaMemcpyHostToDevice);

			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);

			cudaMemcpy(preSmooth31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input); 
			cudaFree(d_output); 
			cudaFree(d_matrix);

#pragma endregion

#pragma region pooling
			sizeInput = 146 * 146 * sizeof(float);
			sizeOutput = 73 * 73 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_input, preSmooth31, sizeInput, cudaMemcpyHostToDevice);

			ApplySecondPool << <1, 73 >> >(d_input, d_output);

			cudaMemcpy(postSmooth31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input); 
			cudaFree(d_output);
#pragma endregion

#pragma endregion

			//All sobel, output postgradient31-34
#pragma region gradient3
			float*  preGradient31 = new float[146 * 146];
			float*  preGradient32 = new float[146 * 146];
			float*  preGradient33 = new float[146 * 146];
			float* postGradient31 = new float[73 * 73];
			float* postGradient32 = new float[73 * 73];
			float* postGradient33 = new float[73 * 73];

#pragma region setup matrix
			gradient3[0] = 1.0f / 16.0f;
			gradient3[1] = 2.0f / 16.0f;
			gradient3[2] = 1.0f / 16.0f;
			gradient3[3] = 2.0f / 16.0f;
			gradient3[4] = 4.0f / 16.0f;
			gradient3[5] = 2.0f / 16.0f;
			gradient3[6] = 1.0f / 16.0f;
			gradient3[7] = 2.0f / 16.0f;
			gradient3[8] = 1.0f / 16.0f;
#pragma endregion

#pragma region convolution

			sizeInput = 148 * 148 * sizeof(float);
			sizeOutput = 146 * 146 * sizeof(float);
			sizeMatrix = 3 * 3 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);
			cudaMalloc((void**)&d_matrix, sizeMatrix);

			cudaMemcpy(d_matrix, gradient3, sizeMatrix, cudaMemcpyHostToDevice);

			cudaMemcpy(d_input, postSobelD5, sizeInput, cudaMemcpyHostToDevice);
			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preGradient31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, postSobelU5, sizeInput, cudaMemcpyHostToDevice);
			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preGradient32, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, postSobelL5, sizeInput, cudaMemcpyHostToDevice);
			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preGradient33, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input); 
			cudaFree(d_output); 
			cudaFree(d_matrix);
#pragma endregion

#pragma region pooling
			sizeInput = 146 * 146 * sizeof(float);
			sizeOutput = 73 * 73 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_input, preGradient31, sizeInput, cudaMemcpyHostToDevice);
			ApplySecondPool << <1, 73 >> >(d_input, d_output);
			cudaMemcpy(postGradient31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, preGradient32, sizeInput, cudaMemcpyHostToDevice);
			ApplySecondPool << <1, 73 >> >(d_input, d_output);
			cudaMemcpy(postGradient32, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_input, preGradient33, sizeInput, cudaMemcpyHostToDevice);
			ApplySecondPool << <1, 73 >> >(d_input, d_output);
			cudaMemcpy(postGradient33, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input); 
			cudaFree(d_output);

#pragma endregion


#pragma endregion

#pragma region Sobel LR
			float*  preSobelLR31 = new float[146 * 146];
			float* postSobelLR31 = new float[73 * 73];
#pragma region setup matrix

			sobel3LR[0] = +1.0f;
			sobel3LR[1] = +0.0f;
			sobel3LR[2] = -1.0f;
			sobel3LR[3] = +2.0f;
			sobel3LR[4] = +0.0f;
			sobel3LR[5] = -2.0f;
			sobel3LR[6] = +1.0f;
			sobel3LR[7] = +0.0f;
			sobel3LR[8] = -1.0f;
#pragma endregion

#pragma region Convolution
			sizeInput = 148 * 148 * sizeof(float);
			sizeOutput = 146 * 146 * sizeof(float);
			sizeMatrix = 3 * 3 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);
			cudaMalloc((void**)&d_matrix, sizeMatrix);
			cudaMemcpy(d_matrix, sobel3LR, sizeMatrix, cudaMemcpyHostToDevice);

			cudaMemcpy(d_input, postSharp5, sizeInput, cudaMemcpyHostToDevice);
			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preSobelLR31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);
			cudaFree(d_matrix);

#pragma endregion

#pragma region pooling
			sizeInput = 146 * 146 * sizeof(float);
			sizeOutput = 73 * 73 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_input, preSobelLR31, sizeInput, cudaMemcpyHostToDevice);
			ApplySecondPool << <1, 73 >> >(d_input, d_output);
			cudaMemcpy(postSobelLR31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);
#pragma endregion

#pragma endregion

#pragma region Sobel UD
			float*  preSobelUD31 = new float[146 * 146];
			float* postSobelUD31 = new float[73 * 73];
#pragma region setup matrix

			sobel3UD[0] = +1.0f;
			sobel3UD[1] = +2.0f;
			sobel3UD[2] = +1.0f;
			sobel3UD[3] = +0.0f;
			sobel3UD[4] = +0.0f;
			sobel3UD[5] = +0.0f;
			sobel3UD[6] = -1.0f;
			sobel3UD[7] = -2.0f;
			sobel3UD[8] = -1.0f;
#pragma endregion

#pragma region Convolution
			sizeInput = 148 * 148 * sizeof(float);
			sizeOutput = 146 * 146 * sizeof(float);
			sizeMatrix = 3 * 3 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);
			cudaMalloc((void**)&d_matrix, sizeMatrix);
			cudaMemcpy(d_matrix, sobel3UD, sizeMatrix, cudaMemcpyHostToDevice);

			cudaMemcpy(d_input, postSharp5, sizeInput, cudaMemcpyHostToDevice);
			ApplyMat3 << <1, 146 >> >(d_input, d_output, d_matrix);
			cudaMemcpy(preSobelUD31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);
			cudaFree(d_matrix);

#pragma endregion

#pragma region pooling
			sizeInput = 146 * 146 * sizeof(float);
			sizeOutput = 73 * 73 * sizeof(float);

			cudaMalloc((void**)&d_input, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_input, preSobelUD31, sizeInput, cudaMemcpyHostToDevice);
			ApplySecondPool << <1, 73 >> >(d_input, d_output);
			cudaMemcpy(postSobelUD31, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_input);
			cudaFree(d_output);
#pragma endregion

#pragma endregion

#pragma region cleanup
			delete[] postSmooth5;
			delete[] postSobelD5;
			delete[] postSobelL5;
			delete[] postSobelR5;
			delete[] postSobelU5;
			delete[] postSharp5;

			delete[] preEdge31;
			delete[] preEdge32;
			delete[] preGradient31;
			delete[] preGradient32;
			delete[] preGradient33;
			delete[] preSmooth31;
			delete[] preSobelLR31;
			delete[] preSobelUD31;

			delete[] gradient3;
			delete[] edge3;
			delete[] sharp3;
			delete[] smooth3;
#pragma endregion


#pragma endregion

#pragma region Combine into one
			sizeInput = 73 * 73 * sizeof(float);
			sizeOutput = screenSize * sizeof(float);

			float* d_postEdge1;
			float* d_postEdge2;
			float* d_postGradient1;
			float* d_postGradient2;
			float* d_postGradient3;
			float* d_postSobel3LR;
			float* d_postSobel3UD;
			float* d_postSmooth31;


			cudaMalloc((void**)&d_postEdge1, sizeInput);
			cudaMalloc((void**)&d_postEdge2, sizeInput);
			cudaMalloc((void**)&d_postGradient1, sizeInput);
			cudaMalloc((void**)&d_postGradient2, sizeInput);
			cudaMalloc((void**)&d_postGradient3, sizeInput);
			cudaMalloc((void**)&d_postSobel3LR, sizeInput);
			cudaMalloc((void**)&d_postSobel3UD, sizeInput);
			cudaMalloc((void**)&d_postSmooth31, sizeInput);
			cudaMalloc((void**)&d_output, sizeOutput);

			cudaMemcpy(d_postEdge1, postEdge31, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_postEdge2, postEdge32, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_postGradient1, postGradient31, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_postGradient2, postGradient32, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_postGradient3, postGradient33, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_postSobel3LR, postSobelLR31, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_postSobel3UD, postSobelUD31, sizeInput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_postSmooth31, postSmooth31, sizeInput, cudaMemcpyHostToDevice);

			CombineScreen<<<1, 73>>>(d_postEdge1, 
				d_postEdge2,
				d_postGradient1, 
				d_postGradient2, 
				d_postGradient3,
				d_postSobel3LR,
				d_postSobel3UD,
				d_postSmooth31,
				d_output);

			cudaMemcpy(screenBits, d_output, sizeOutput, cudaMemcpyDeviceToHost);

			writeBMP(screenBits, 73, 73 * 8, "D:/ScreenBits.bmp");
			writeBMP(postSobelLR31, 73, 73, "D:/postSobelLR31.bmp");
			writeBMP(postSobelUD31, 73, 73, "D:/postSobelUD31.bmp");
			writeBMP(postSmooth31, 73, 73, "D:/postSmooth31.bmp");

#pragma region cleanup
			cudaFree(d_postEdge1);
			cudaFree(d_postEdge2);
			cudaFree(d_postGradient1);
			cudaFree(d_postGradient2);
			cudaFree(d_postGradient3);
			cudaFree(d_postSobel3LR);
			cudaFree(d_postSobel3UD);
			cudaFree(d_postSmooth31);
			cudaFree(d_output);

			delete[] postEdge31;
			delete[] postEdge32;
			delete[] postGradient31;
			delete[] postGradient32;
			delete[] postGradient33;
			delete[] postSobelLR31;
			delete[] postSobelUD31;
			delete[] postSmooth31;
#pragma endregion



#pragma endregion

#pragma region Old screen manip
			////Seperate the reduce screen into 8x6 chunks.
			////Average the intensity for those pixels.
			//int bitsIndex = 0;

			////Reduce the screen into a 50x50 grid of 8x6 pixels. Average the intensity to get the average brightness of that grid.
			//for (int r = 0; r < 50; r++){
			//	for (int c = 0; c < 50; c++){

			//		float intense = 0.0f;
			//		int numPixels = 0;
			//		for (int row = 0; row < 6; ++row){
			//			for (int col = 0; col < 6; ++col){
			//				float reduxIntense = reduceScreen[(r * 300 * 6) + (row * 300) + (c * 6 + col)];
			//				intense += reduxIntense;
			//				++numPixels;
			//			}
			//		}
			//		screenBits[bitsIndex] = (intense) / ((float)numPixels);
			//		++bitsIndex;
			//	}
			//}
#pragma endregion

#pragma endregion

			//Get all the inputs times their weight change it up into 2x2 squares and store it in the array InputVotes
#pragma region Input Weights
			//float* InputVotes =  float[2304];// 625];//2304];

			////3x3 overlapping implementation

			//for (int r = 0; r < 48; ++r){
			//	for (int c = 0; c < 48; ++c){
			//		float total = 0;
			//		//First row
			//		total += screenBits[r * 50 + c] * inputWeights[r * 50 + c];
			//		//total += screenBits[r * 50 + c + 1] * inputWeights[r * 50 + c + 1];
			//		total += screenBits[r * 50 + c + 2] * inputWeights[r * 50 + c + 2];
			//		//Second row
			//		//total += screenBits[(r + 1) * 50 + c] * inputWeights[(r + 2) * 50 + c];
			//		total += screenBits[(r + 1) * 50 + c + 1] * inputWeights[(r + 2) * 50 + c + 1];
			//		//total += screenBits[(r + 1) * 50 + c + 2] * inputWeights[(r + 2) * 50 + c + 2];
			//		//Third row
			//		total += screenBits[(r + 2) * 50 + c] * inputWeights[(r + 2) * 50 + c];
			//		//total += screenBits[(r + 2) * 50 + c + 1] * inputWeights[(r + 2) * 50 + c + 1];
			//		total += screenBits[(r + 2) * 50 + c + 2] * inputWeights[(r + 2) * 50 + c + 2];
			//		//total += InputBias[r * 48 + c];
			//		InputVotes[r * 48 + c] = total;// (1 / (1 + exp(total)));
			//	}
			//}

			//2x2 no overlap implementation

			//for (int r = 0; r < 50; r += 2){
			//	for (int c = 0; c < 50; c += 2){
			//		float total = 0;
			//		total += screenBits[r * 50 + c] * inputWeights[r * 50 + c];
			//		total += screenBits[r * 50 + c + 1] * inputWeights[r * 50 + c + 1];
			//		total += screenBits[(r + 1) * 50 + c] * inputWeights[(r + 1) * 50 + c];
			//		total += screenBits[(r + 1) * 50 + c + 1] * inputWeights[(r + 1) * 50 + c + 1];
			//		//r and c are based on 50x50 grid. Divide by two to get 25x25 grid
			//		//total += InputBias[(r / 2) * 25 + (c / 2)];
			//		//Sigmoid everything!
			//		InputVotes[(r / 2) * 25 + (c / 2)] = total;// (1 / (1 + exp(total)));
			//	}
			//}

			//Cuda implementation of old input votes

			//float* d_screen;
			//float* d_weights;
			//int* d_numInput;
			//float* d_Votes;
			//float* InputVotes =  float[4 * 625];
			//int sizeInt = sizeof(int);
			//int sizeScreen = (50 * 50) *sizeof(float);
			//int sizeWeights = (50 * 50) *sizeof(int);
			//cudaMal((void**)&d_screen, sizeScreen);
			//cudaMal((void**)&d_weights, sizeWeights);
			//cudaMal((void**)&d_Votes, sizeWeights);
			//cudaMal((void**)&d_numInput, sizeInt);
			//cudaMemcpy(d_screen, screenBits, sizeScreen, cudaMemcpyHostToDevice);
			//cudaMemcpy(d_weights, inputWeights, sizeWeights, cudaMemcpyHostToDevice);
			//cudaMemcpy(d_numInput, &numInput, sizeInt, cudaMemcpyHostToDevice);
			//CalcInput << < 1, 625 >> >(d_screen, d_weights, d_Votes, 50);
			//cudaMemcpy(InputVotes, d_Votes, sizeWeights, cudaMemcpyDeviceToHost);
			//cudaF(d_screen);
			//cudaF(d_weights);
			//cudaF(d_Votes);
			//cudaF(d_numInput);

#pragma endregion

			//Run sigmoid on all input and store the votes or output in the float array FirstHiddenVotes
#pragma region FirstHidden
			float* d_screenBits;
			float* d_FHW;
			float* d_bias;
			float* d_FirstHiddenVotes;
			//1-2 layer
			float* HiddenVotes;//FirstHiddenVotes;

			int sizeFHW = firstHiddenWeightsSize *sizeof(float);
			//1-2 layer
			int sizeHidden = numFirstHiddenNeurons * sizeof(float);
			int sizeInputVotes = screenSize * sizeof(float);
			int sizeBias = numFirstHiddenNeurons * sizeof(float);
			HiddenVotes = new float[numFirstHiddenNeurons];

			cudaMalloc((void**)&d_FHW, sizeFHW);
			cudaMalloc((void**)&d_FirstHiddenVotes, sizeHidden);
			cudaMalloc((void**)&d_screenBits, sizeInputVotes);
			cudaMalloc((void**)&d_bias, sizeBias);
			cudaMemcpy(d_FHW, firstHiddenWeights, sizeFHW, cudaMemcpyHostToDevice);
			cudaMemcpy(d_screenBits, screenBits, sizeInputVotes, cudaMemcpyHostToDevice);
			//for (int c = 0; c < screenSize; ++c){
			//	if (screenBits[c] > 0){
			//		qDebug() << "C: " << c << ": " << screenBits[c];
			//	}
			//}
			cudaMemcpy(d_bias, firstBias, sizeBias, cudaMemcpyHostToDevice);

			//Input votes, Hidden weights, Number of Inputs, Votes array
			FirstHidden << <1, numFirstHiddenNeurons >> >(d_screenBits, d_FHW, d_bias, screenSize, d_FirstHiddenVotes);
			cudaMemcpy(HiddenVotes, d_FirstHiddenVotes, sizeHidden, cudaMemcpyDeviceToHost);
			cudaMemcpy(FirstLayerFire, d_FirstHiddenVotes, sizeHidden, cudaMemcpyDeviceToHost);
			cudaFree(d_FHW);
			cudaFree(d_FirstHiddenVotes);
			cudaFree(d_screenBits);
			cudaFree(d_bias);

			//for (int i = 0; i < 20; ++i){
			//	qDebug() << "First Hidden Votes: " << i << ": " << FirstHiddenVotes[i];
			//}

#pragma endregion

			//qDebug();
			//qDebug() << "Second hidden";
			//Run sigmoid on all input and store the votes or output in the float array SecondHiddenVotes
#pragma region SecondHidden
			//float* d_InputVotes;
			//float* d_SHW;
			//float* d_SecondHiddenVotes;
			//float* HiddenVotes;//SecondHiddenVotes;

			//int sizeSHW = (20 * 15) *sizeof(float);
			//int sizeSecondHidden = 15 * sizeof(float);
			//sizeInputVotes = 20 * sizeof(float);
			//sizeBias = 15 * sizeof(float);

			//cudaMal((void**)&d_SHW, sizeSHW);
			//cudaMal((void**)&d_SecondHiddenVotes, sizeSecondHidden);
			//cudaMal((void**)&d_InputVotes, sizeInputVotes);
			//cudaMal((void**)&d_bias, sizeBias);
			////SecondHiddenVotes
			//HiddenVotes =  float[15];

			////for (int i = 0; i < 15; ++i){
			////	qDebug() << "SHW: " << secondHiddenWeights[i];
			////}

			////for (int i = 0; i < 20; ++i){
			////	qDebug() << "Second Votes " << i << ": " << FirstHiddenVotes[i];
			////}

			//cudaMemcpy(d_SHW, secondHiddenWeights, sizeSHW, cudaMemcpyHostToDevice);
			//cudaMemcpy(d_InputVotes, FirstHiddenVotes, sizeInputVotes, cudaMemcpyHostToDevice);
			//cudaMemcpy(d_bias, secondBias, sizeBias, cudaMemcpyHostToDevice);


			////Input votes, Hidden weights, Number of Inputs, Votes array
			//FirstHidden <<<1, 15 >>>(d_InputVotes, d_SHW, d_bias, 20, d_SecondHiddenVotes);

			////SecondHiddenVotes
			//cudaMemcpy(HiddenVotes, d_SecondHiddenVotes, sizeSecondHidden, cudaMemcpyDeviceToHost);


			//cudaF(d_SHW);
			//cudaF(d_SecondHiddenVotes);
			//cudaF(d_InputVotes);
			//cudaF(d_bias);

#pragma endregion

//
//			//qDebug();
//			//qDebug() << "Third hidden";
#pragma region ThirdHidden
//			float* d_THW;
//			float* d_thirdHiddenVotes;
//			float* HiddenVotes;
//
//			int sizeTHW = (10 * 15) *sizeof(float);
//			int sizethirdHidden = 10 * sizeof(float);
//			sizeInputVotes = 15 * sizeof(float);
//			sizeBias = 10 * sizeof(float);
//
//			cudaMal((void**)&d_THW, sizeTHW);
//			cudaMal((void**)&d_thirdHiddenVotes, sizethirdHidden);
//			cudaMal((void**)&d_InputVotes, sizeInputVotes);
//			cudaMal((void**)&d_bias, sizeBias);
//			HiddenVotes =  float[10];
//
//			cudaMemcpy(d_THW, thirdHiddenWeights, sizeTHW, cudaMemcpyHostToDevice);
//			cudaMemcpy(d_InputVotes, SecondHiddenVotes, sizeInputVotes, cudaMemcpyHostToDevice);
//			cudaMemcpy(d_bias, bias, sizeBias, cudaMemcpyHostToDevice);
//
			//for (int i = 0; i < 15; ++i){
			//	qDebug() << "Third Votes " << i << ": " << /*Second*/HiddenVotes[i];
			//}
//
//			//Input votes, Hidden weights, Number of Inputs, Votes array
//			FirstHidden << <1, 10 >> >(d_InputVotes, d_THW, d_bias, 15, d_thirdHiddenVotes);
//
//			cudaMemcpy(HiddenVotes, d_thirdHiddenVotes, sizethirdHidden, cudaMemcpyDeviceToHost);
//
//
//			cudaF(d_THW);
//			cudaF(d_thirdHiddenVotes);
//			cudaF(d_InputVotes); 
//			cudaF(d_bias);
//
#pragma endregion

			//Connect all hidden nodes to the output nodes. Store values in the array votes
#pragma region output
			float* d_outputHiddenVotes;
			float* d_outputWeights;
			float* d_votes;
			float* votes;

			votes = new float[numInput];

			//1-2 layer
			int sizeHiddenOutput = numFirstHiddenNeurons * sizeof(float);
			//1-2 layer
			int sizeWeightsOutput = numFirstHiddenNeurons * numInput * sizeof(float);
			int sizeVotesOutput = numInput * sizeof(float);

			cudaMalloc((void**)&d_outputHiddenVotes, sizeHiddenOutput);
			cudaMalloc((void**)&d_outputWeights, sizeWeightsOutput);
			cudaMalloc((void**)&d_votes, sizeVotesOutput);

			cudaMemcpy(d_outputHiddenVotes, HiddenVotes, sizeHiddenOutput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_outputWeights, outputWeights, sizeWeightsOutput, cudaMemcpyHostToDevice);

			//The number of threads is the number of inputs possible, so Left or Right
			//The third varaible is the number of hidden layers
			//1-2 layer
			OutputLayer << <1, numInput >> >(d_outputHiddenVotes, d_outputWeights, numFirstHiddenNeurons, d_votes);

			cudaMemcpy(votes, d_votes, sizeVotesOutput, cudaMemcpyDeviceToHost);
			cudaMemcpy(OutputLayerTotals, d_votes, sizeVotesOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_outputHiddenVotes);
			cudaFree(d_outputWeights);
			cudaFree(d_votes);

			//for (int i = 0; i < 10; ++i){
			//	qDebug() << i << " " << HiddenVotes[i];
			//}
#pragma endregion

#pragma region Tally Votes

			float tally = -FLT_MAX;
			for (int i = 0; i < numInput; ++i){
				qDebug() << "Input: " << i << " Tally: " << votes[i];
				if (tally < votes[i]){
					tally = votes[i];
					lastInput = i;
				}
			}

			qDebug();
			qDebug();


			numCalls = 0;
#pragma endregion


#pragma region cleanup
			delete[] votes;
			delete[] HiddenVotes;
#pragma endregion
		}
		StoreScreen(screenBits);

	}

	delete[] screenBits;

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
	for (int i = 0; i < numPixels; ++i){
		greyScreen[i] = 0.144*pixelsR[i] + 0.587*pixelsG[i] + 0.299*pixelsB[i];
		//if (greyScreen[i] > 0.1f) qDebug() << "Huh? " << greyScreen[i];
		//sprintf(rgbdata[i], "%c", greyScreen[i]);
		//if (rgbdata[i] > 0.1f) qDebug() << "Huh? " << rgbdata[i];
	}

	//writeBMP(greyScreen, 800, 600);

	//for (int i = 0; i < numPixels; i++){
	//	if (greyScreen[i] > 0.0f){
	//		qDebug() << "Pixel: " << i << " Intensity: " << greyScreen[i];
	//	}
	//}

	//Shrink the image
	int i = 0;
	for (int r = 0; r < *height; r += 2){
		for (int c = 100; c < ((*width) - 100); c += 2){
			float x1 = greyScreen[c + r * 800];
			float x2 = greyScreen[c + 1 + r * 800];
			float x3 = greyScreen[c + (r + 1) * 800];
			float x4 = greyScreen[c + 1 + (r + 1) * 800];
			
			float avg = (x1 + x2 + x3 + x4) / 4;
			if (avg > 1){
				qDebug() << "Broke";
			}

			reduceScreen[i] = avg;
			++i;
		}
	}

	//writeBMP(reduceScreen, 300, 300, "D:/ReduceScreen.bmp");

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
	//cudaMal((void**)&d_pixelsR, numPixels);
	//cudaMal((void**)&d_pixelsG, numPixels);
	//cudaMal((void**)&d_pixelsB, numPixels);
	//cudaMal((void**)&d_reducePixels, numPixels);

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

	//cudaF(d_reducePixels); 
	//cudaF(d_pixelsR); 
	//cudaF(d_pixelsG); 
	//cudaF(d_pixelsB);

#pragma endregion

	delete[] pixelsR;
	delete[] pixelsG;
	delete[] pixelsB;
	delete[] greyScreen;

	//free(pixelsR); free(pixelsG); free(pixelsB); free(greyScreen);
	//qDebug() << "Got a screen, mebe";
}

void DeepLearner::GameOver(bool isWin){
	//Modify weights to decrease the value of what happened if loss happened
	learn(isWin);
}

void DeepLearner::SwitchAlgorithm(type algoType){
	//Changes the algorith, may remove.
	algo = algoType;
}

void DeepLearner::play(){
	//Play a game with a lr of 0;
}

#pragma endregion

/* Current learning takes the 1 for scoring or -1 for losing and updates the weights
it changes the weights based on a percent of their weight to each thing.

Sigma has a great gradient decent formula though.
Error(+1 or -1) + sigma derived (s( 1 - s)) + xM

Squared error loss
1/2 (sum (func prediction - actual value)^2)
gradient decent sum(error * sigma * actual value(1 or -1 i guess)

update weight by doing w - gradient decent.
*/

#pragma region Learning

#pragma region learning cuda code
__global__ void updateInput(float* screen, float* weight, float* d_Votes){

}

//lr, weights to update, int score increase or decrease, decision neuron array
__global__ void updateHidden(float* input, float* weight, float* bias, int d_numVotes, float* d_votes, float learningRate, float error){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

#pragma region Normal Iteration
	float total = 0.0f;

	//printf("Num Votes: %i", d_numVotes);

	for (int i = 0; i < d_numVotes; ++i){
		//if (weight[id*d_numVotes + i] > 0) printf("Weight higher than 0: %f", weight[id*d_numVotes + i]);
		//if (input[i] > 0) printf("Input: %f ", input[i]);
		//printf("Weight: %f\n", weight[id * d_numVotes + i]);

		//d_numVotes is the stride
		float sig = input[i] * weight[id * d_numVotes + i];
		total += sig;// (1 / (1 + exp(-sig)));
	}

	//Should use sigmoid here. Maybe Could be in for loop though
	//printf("Total: %f\n", total);
	total += bias[id];
	//printf("Bias: %f\n", bias[id]);
	//int finalTotal = total;
	total = (1 / (1 + exp(-total)));
	//if (total < 0.1) printf("Total %i: %f\n", id, total);
	//total = ((int)(total)) % 3;

	//printf("Total: %i:%f\n", id, total);
	d_votes[id] = total;
#pragma endregion

#pragma region update weights
	for (int i = 0; i < d_numVotes; ++i){
		//if ((error * learningRate * ((total) * (1 - total))) <= -0.0001 ||
		//	(error * learningRate * ((total) * (1 - total))) >= +0.0001){
		//	//printf("Different? ");

		//	printf("Start weight %i: %f, New Weight: %f, Change: %f\n", id * d_numVotes + i, 
		//		weight[id * d_numVotes + i], 
		//		weight[id * d_numVotes + i] - (error * learningRate * total * (1 - total)), 
		//		(error * learningRate * (total * (1 - total))));
		//}
		//int modTotal = total * 3.0f - 1.5f;
		//if (modTotal > 0) printf("finalTotal: %i, bias: %i", finalTotal, bias[id]);
		//if (modTotal <  0) printf("finalTotal: %i", finalTotal);
		//if (modTotal == 0) printf("finalTotal: %i", finalTotal);
		weight[id * d_numVotes + i] = weight[id * d_numVotes + i] - (error * learningRate * ((1 / (1 + exp(-total))) * (1 - (1 / (1 + exp(-total)))))) * (id % d_numVotes);

		//Clamp weight to 0 to 1
		weight[id * d_numVotes + i] = max(0.0f, weight[id * d_numVotes + i]);
		weight[id * d_numVotes + i] = min(weight[id * d_numVotes + i], 1.0f);
	}
	
	float biasTotal = total * 2 - 1;
	//printf("Bias Total: %i: %f\n", id, biasTotal);
	//printf("Error: %f\n", error);
	//printf("Bias: %i, %f\n", id, bias[id]);
	//printf("Bias subtraction: %i, %f\n", id, biasTotal * error * learningRate);
	bias[id] = bias[id] - biasTotal * error * learningRate;
	//printf("Bias: %i, %f\n", id, bias[id]);
	//printf("Bias Total: %i: %f\n", id, bias[id]);

#pragma endregion
}

//Number of threads is the number of screens or number of stored inputs
//ID could be lr? maybe?
//float* array of weights, number of nodes, float* of outputweights
__global__ void printUpdateHidden(float* input, float* weight, float* bias, int d_numVotes, float* d_votes, float learningRate, float error){
int id = threadIdx.x + blockDim.x * blockIdx.x;

#pragma region Normal Iteration
float total = 0.0f;

//printf("Num Votes: %i", d_numVotes);

for (int i = 0; i < d_numVotes; ++i){
	//if (weight[id*d_numVotes + i] > 0) printf("Weight higher than 0: %f", weight[id*d_numVotes + i]);
	//if (input[i] > 0) printf("Input: %f ", input[i]);
	//printf("Weight: %f\n", weight[id * d_numVotes + i]);

	//d_numVotes is the stride
	float sig = input[i] * weight[id * d_numVotes + i];
	total += sig;// (1 / (1 + exp(-sig)));
}

//Should use sigmoid here. Maybe Could be in for loop though
//printf("Total: %f\n", total);
total += bias[id];
//printf("Bias: %f\n", bias[id]);
int finalTotal = total;
total = (1 / (1 + exp(-total)));
//if (total < 0.1) printf("Total %i: %f\n", id, total);
//total = ((int)(total)) % 3;

//printf("Total: %i:%f\n", id, total);
d_votes[id] = total;
#pragma endregion

int modTotal = total * 3.0f - 1.5f;
if (modTotal >  0) printf("finalTotal > 0: %i, bias: %i\n", finalTotal, bias[id]);
if (modTotal <  0) printf("finalTotal < 0: %i, bias: %i\n", finalTotal, bias[id]);
if (modTotal == 0) printf("finalTotal = 0: %i, bias: %i\n", finalTotal, bias[id]);
if (bias[id] > 0.1f) printf("Bias above 0");

#pragma region update weights
for (int i = 0; i < d_numVotes; ++i){
	//if ((error * learningRate * ((total) * (1 - total))) <= -0.0001 ||
	//	(error * learningRate * ((total) * (1 - total))) >= +0.0001){
	//	//printf("Different? ");

	//	printf("Start weight %i: %f, New Weight: %f, Change: %f\n", id * d_numVotes + i, 
	//		weight[id * d_numVotes + i], 
	//		weight[id * d_numVotes + i] - (error * learningRate * total * (1 - total)), 
	//		(error * learningRate * (total * (1 - total))));
	//}
	weight[id * d_numVotes + i] = weight[id * d_numVotes + i] - (error * learningRate * ((1 / (1 + exp(-total))) * (1 - (1 / (1 + exp(-total)))))) * (id % d_numVotes);

	//Clamp weight to 0 to 1
	weight[id * d_numVotes + i] = max(0.0f, weight[id * d_numVotes + i]);
	weight[id * d_numVotes + i] = min(weight[id * d_numVotes + i], 1.0f);
}

float biasTotal = total * 2 - 1;
//printf("Bias Total: %i: %f\n", id, biasTotal);
//printf("Error: %f\n", error);
//printf("Bias: %i, %f\n", id, bias[id]);
//printf("Bias subtraction: %i, %f\n", id, biasTotal * error * learningRate);
bias[id] = bias[id] - biasTotal * error * learningRate;
//printf("Bias: %i, %f\n", id, bias[id]);
//printf("Bias Total: %i: %f\n", id, bias[id]);

#pragma endregion

}

__global__ void updateOutputWeights(float* d_weights, float error, float lr, int keypress, 
	int numHiddenNeurons, float* outputTotals, int numInput){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	int index = numHiddenNeurons * keypress + id;

	float certainty = 0.0f;
	for (int i = 0; i < numInput; ++i){
		certainty += outputTotals[i];
	}
	certainty = outputTotals[keypress] / certainty;
	//printf("Certainty: %f\n", certainty);
	
	//int isPositive = 1;// d_weights[index] * 105 - 52.5;
	//isPositive = min(isPositive, 1);
	//isPositive = max(-1, isPositive);
	//if (isPositive == 0){
	//	isPositive = -1;
	//}
	//if(isPositive == 0)	printf("IsPositive: %i", isPositive);

	//TODO test removing weight
	float change = error * lr * d_weights[index] * certainty;

	//printf("Error: %f, LR: %f, Weight: %f Change: %f\n", error, lr, d_weights[index], change);
	d_weights[index] = d_weights[index] + change;

	//Clamp
	d_weights[index] = min(1.0f, d_weights[index]);
	d_weights[index] = max(0.0f, d_weights[index]);
}

/*
For each screen input weight
Get change by algorithm, E * LR * outputweight
update weight by change + weight;
*/
__global__ void updateHiddenWeights(float* d_weights, float error, float lr, int keyPress, float* d_outputweights, 
	int screenSize, int numHiddenNeurons, float* d_bias, float* firstFire){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	float totalChange = 0.0f;
	for (int i = 0; i < screenSize; ++i){
		//Output weights stride is numNeurons, keypress is index into that section
		float change = error * lr *d_outputweights[id * numHiddenNeurons + keyPress] * (firstFire[id] * 2 - 1);
		totalChange += change;

		d_weights[id * screenSize + i] = d_weights[id * screenSize + i] + change;

		d_weights[id * screenSize + i] = min(1.0f, d_weights[id * screenSize + i]);
		d_weights[id * screenSize + i] = max(0.0f, d_weights[id * screenSize + i]);
	}
	float biasChange = totalChange * -0.5f;
	//printf("TotalChange: %f", biasChange);
	d_bias[id] = d_bias[id] + biasChange;
}

#pragma endregion

void DeepLearner::learn(bool isWin){
	if (f_RandomChance > 0.10) {
		f_RandomChance -= 0.001f;
	}

	float increase = (isWin) ? +0.01 : -0.15;
	int NumScreens = (FullStorage) ? numScreenHistory : screenStorageCount;

	for (int i = 0; i <  NumScreens; ++i){
#pragma region Variables
		float error = increase;
		float learnRate = lr;
		float* d_weights;
		float* d_outputWeights;
		float* d_bias;
		float* d_outputTotals;
		float* outputTotals;

		float* d_firstFiring;
		float* firstFiring;

		int sizeFirstFiring;
		int sizeWeights;
		int sizeBias;
		int KeyPress;
		int sizeOutputTotals;

#pragma endregion

#pragma region output weight changes
		sizeWeights = numFirstHiddenNeurons * numInput * sizeof(float);
		sizeOutputTotals = numInput * sizeof(float);
		KeyPress = keypressStorage[i];
		outputTotals = new float[numInput];
		for (int j = 0; j < numInput; ++j){
			outputTotals[j] = OutputLayerStorage[j + i * numInput];
			//qDebug() << "Output Totals: " << outputTotals[j];
		}

		//Initialize and setup device weights
		cudaMalloc((void**)&d_outputWeights, sizeWeights);
		cudaMemcpy(d_outputWeights, outputWeights, sizeWeights, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_outputTotals, sizeOutputTotals);
		cudaMemcpy(d_outputTotals, outputTotals, sizeOutputTotals, cudaMemcpyHostToDevice);

		//Update the weights
		//                                                    d_outputweights, error, learnRate, keypress, numHiddenNeurons
		updateOutputWeights << <1, numFirstHiddenNeurons >> >(d_outputWeights, error, learnRate, KeyPress, 
			numFirstHiddenNeurons, d_outputTotals, numInput);

		//Get device weights to host weights
		cudaMemcpy(outputWeights, d_outputWeights, sizeWeights, cudaMemcpyDeviceToHost);
		//Cleanup
		cudaFree(d_outputWeights);
		cudaFree(d_outputTotals);
		delete[] outputTotals;
#pragma endregion

#pragma region First layer weight changes

		sizeWeights = firstHiddenWeightsSize * sizeof(float);
		//Hidden changes less than output layer
		learnRate = learnRate * learnRate;
		sizeBias = numFirstHiddenNeurons * sizeof(float);
		sizeFirstFiring = numFirstHiddenNeurons * sizeof(float);

		firstFiring = new float[numFirstHiddenNeurons];
		for (int j = 0; j < numFirstHiddenNeurons; ++j){
			firstFiring[j] = FirstLayerStorage[j + i * numFirstHiddenNeurons];
		}

		//Initialize device weights
		cudaMalloc((void**)&d_weights, sizeWeights);
		cudaMemcpy(d_weights, firstHiddenWeights, sizeWeights, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_outputWeights, sizeWeights);
		cudaMemcpy(d_outputWeights, firstHiddenWeights, sizeWeights, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_bias, sizeBias);
		cudaMemcpy(d_bias, firstBias, sizeBias, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_firstFiring, sizeFirstFiring);
		cudaMemcpy(d_firstFiring, firstFiring, sizeFirstFiring, cudaMemcpyHostToDevice);
		//W = W + E * LR * SUM(OW's)
		//  											  d_weights, error, learnRate, keyPress, d_outputweights, screenSize, numNeurons           , d_bias
		updateHiddenWeights<<<1, numFirstHiddenNeurons>>>(d_weights, error, learnRate, KeyPress, d_outputWeights, 
			screenSize, numFirstHiddenNeurons, d_bias, d_firstFiring);

		//Update host weights with device weights
		cudaMemcpy(firstHiddenWeights, d_weights, sizeWeights, cudaMemcpyDeviceToHost);
		cudaMemcpy(firstBias, d_bias, sizeBias, cudaMemcpyDeviceToHost);

		//Cleanup
		cudaFree(d_weights);
		cudaFree(d_outputWeights);
		cudaFree(d_bias);
		cudaFree(d_firstFiring);
		delete[] firstFiring;
#pragma endregion

		//	//float* d_screen;
		//	//float* d_weights;
		//	//int* d_numInput;
		//	//float* d_Votes;
		//	//float* InputVotes = new float[4 * 625];
		//	//int sizeInt = sizeof(int);
		//	//int sizeScreen = (50 * 50) *sizeof(float);
		//	//int sizeWeights = (50 * 50) *sizeof(int);
		//	//cudaMalloc((void**)&d_screen, sizeScreen);
		//	//cudaMalloc((void**)&d_weights, sizeWeights);
		//	//cudaMalloc((void**)&d_Votes, sizeWeights);
		//	//cudaMalloc((void**)&d_numInput, sizeInt);
		//	//cudaMemcpy(d_screen, screenBits, sizeScreen, cudaMemcpyHostToDevice);
		//	//cudaMemcpy(d_weights, inputWeights, sizeWeights, cudaMemcpyHostToDevice);
		//	//cudaMemcpy(d_numInput, &numInput, sizeInt, cudaMemcpyHostToDevice);
		//	//updateInput << < 4, 625 >> >(d_screen, d_weights, d_Votes);
		//	//cudaMemcpy(InputVotes, d_Votes, sizeWeights, cudaMemcpyDeviceToHost);
		//	//cudaMemcpy(inputWeights, d_weights, sizeWeights, cudaMemcpyDeviceToHost);
		//	//cudaFree(d_screen);
		//	//cudaFree(d_weights);
		//	//cudaFree(d_Votes);
		//	//cudaFree(d_numInput);
	}
}

void DeepLearner::ResetScore(){
	lastScore = 0;
	FullStorage = false;
	screenStorageCount = 0;
}

#pragma endregion

#pragma region Holding code, current old learning
//if (f_RandomChance > 0.10) {
//	f_RandomChance -= 0.001f;
//}
////1-2 layer
//float increase = (isWin) ? +0.01 : -0.15;
//int NumScreens = (FullStorage) ? 15 : screenStorageCount;
//
////INTENTIONAL ERROR, starting at the numscreens and decrementing. Isn't right because last screen is at screenStorageCount. Here for simplicity currently.
//
////Increase is error
////Num screens is the amount of input that has been stored
////Iterate over the inputs and times by weight. Update weights by w = w - error * sig_prime
////Iterate over first layer with input layer inputs, use update above
////Iterate over second layer with first layer inputs, use update above
////Iterate over third layer, with second layer inputs, use update above
////Iterate over output layer, 
//for (int i = 0; i < 1;/* NumScreens;*/ ++i){
//	//for (int i = NumScreens - 1; i > 0; --i){
//	int UpdateScreen = i;// rand.randomInRange(0, NumScreens - 2);
//	//float* screenBits = new float[50 * 50];
//	//for (int j = 0; j < 50 * 50; ++j){
//	//	screenBits[j] = inputStorage[UpdateScreen * 50 * 50 + j];
//	//}
//
//#pragma region getInputs
//	float* InputVotes = new float[screenSize];// 625];
//
//	for (int j = 0; j < screenSize; ++j){
//		InputVotes[j] = inputStorage[UpdateScreen * screenSize + j];
//	}
//
//	//writeBMP(InputVotes, 73, 73 * 7, "D:/InputVotes.bmp");
//
//	//3x3 overlapping 48x48
//
//	//for (int r = 0; r < 48; ++r){
//	//	for (int c = 0; c < 48; ++c){
//	//		float total = 0;
//	//		//First row
//	//		total += screenBits[r * 50 + c] * inputWeights[r * 50 + c];
//	//		total += screenBits[r * 50 + c + 1] * inputWeights[r * 50 + c + 1];
//	//		total += screenBits[r * 50 + c + 2] * inputWeights[r * 50 + c + 2];
//	//		//Second row
//	//		total += screenBits[(r + 1) * 50 + c] * inputWeights[(r + 2) * 50 + c];
//	//		total += screenBits[(r + 1) * 50 + c + 1] * inputWeights[(r + 2) * 50 + c + 1];
//	//		total += screenBits[(r + 1) * 50 + c + 2] * inputWeights[(r + 2) * 50 + c + 2];
//	//		//Third row
//	//		total += screenBits[(r + 2) * 50 + c] * inputWeights[(r + 2) * 50 + c];
//	//		total += screenBits[(r + 2) * 50 + c + 1] * inputWeights[(r + 2) * 50 + c + 1];
//	//		total += screenBits[(r + 2) * 50 + c + 2] * inputWeights[(r + 2) * 50 + c + 2];
//	//		//total += InputBias[r * 48 + c];
//	//		InputVotes[r * 48 + c] = total;// (1 / (1 + exp(total)));
//	//	}
//	//}
//
//	//2X2 implementation no overlap
//
//	//for (int r = 0; r < 50; r += 2){
//	//	for (int c = 0; c < 50; c += 2){
//	//		float total = 0;
//	//		total += screenBits[r * 50 + c] * inputWeights[r * 50 + c];
//	//		total += screenBits[r * 50 + c + 1] * inputWeights[r * 50 + c + 1];
//	//		total += screenBits[(r + 1) * 50 + c] * inputWeights[(r + 1) * 50 + c];
//	//		total += screenBits[(r + 1) * 50 + c + 1] * inputWeights[(r + 1) * 50 + c + 1];
//	//		//r and c are based on 50x50 grid. Divide by two to get 25x25 grid
//	//		total += InputBias[(r / 2) * 25 + (c / 2)];
//	//		//Sigmoid everything!
//	//		InputVotes[(r / 2) * 25 + (c / 2)] = (1 / (1 + exp(total)));
//	//	}
//	//}
//
//#pragma region Input biases, old and unused
//
//	//for (int i = 0; i < 25 * 25; ++i){
//	//	if (InputVotes[i] > 0.5){
//	//		InputBias[i] -= increase * lr;
//	//	}
//	//	else{
//	//		InputBias[i] += increase * lr;
//	//	}
//	//}
//
//#pragma endregion
//
//	//float* d_screen;
//	//float* d_weights;
//	//int* d_numInput;
//	//float* d_Votes;
//	//float* InputVotes = new float[4 * 625];
//	//int sizeInt = sizeof(int);
//	//int sizeScreen = (50 * 50) *sizeof(float);
//	//int sizeWeights = (50 * 50) *sizeof(int);
//	//cudaMalloc((void**)&d_screen, sizeScreen);
//	//cudaMalloc((void**)&d_weights, sizeWeights);
//	//cudaMalloc((void**)&d_Votes, sizeWeights);
//	//cudaMalloc((void**)&d_numInput, sizeInt);
//	//cudaMemcpy(d_screen, screenBits, sizeScreen, cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_weights, inputWeights, sizeWeights, cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_numInput, &numInput, sizeInt, cudaMemcpyHostToDevice);
//	//updateInput << < 4, 625 >> >(d_screen, d_weights, d_Votes);
//	//cudaMemcpy(InputVotes, d_Votes, sizeWeights, cudaMemcpyDeviceToHost);
//	//cudaMemcpy(inputWeights, d_weights, sizeWeights, cudaMemcpyDeviceToHost);
//	//cudaFree(d_screen);
//	//cudaFree(d_weights);
//	//cudaFree(d_Votes);
//	//cudaFree(d_numInput);
//#pragma endregion
//
//	//qDebug() << "First Learn";
//#pragma region update first layer
//	float* d_InputVotes;
//	float* d_FHW;
//	float* d_bias;
//	float* d_FirstHiddenVotes;
//	float* HiddenVotes;// FirstHiddenVotes;
//
//	int sizeFHW = (numFirstHiddenNeurons * screenSize) *sizeof(float);
//	int sizeHidden = numFirstHiddenNeurons * sizeof(float);
//	int sizeInputVotes = screenSize * sizeof(float);
//	int sizeBias = numFirstHiddenNeurons * sizeof(float);
//	HiddenVotes = new float[numFirstHiddenNeurons];
//
//	cudaMalloc((void**)&d_FHW, sizeFHW);
//	cudaMalloc((void**)&d_FirstHiddenVotes, sizeHidden);
//	cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
//	cudaMalloc((void**)&d_bias, sizeBias);
//
//	cudaMemcpy(d_FHW, firstHiddenWeights, sizeFHW, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_InputVotes, InputVotes, sizeInputVotes, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_bias, firstBias, sizeBias, cudaMemcpyHostToDevice);
//
//	//Input votes, Hidden weights, Number of Inputs, Votes array, learning rate, error
//	updateHidden << <1, numFirstHiddenNeurons >> >(d_InputVotes, d_FHW, d_bias, screenSize, d_FirstHiddenVotes, lr, increase);
//
//	cudaMemcpy(HiddenVotes, d_FirstHiddenVotes, sizeHidden, cudaMemcpyDeviceToHost);
//	cudaMemcpy(firstHiddenWeights, d_FHW, sizeFHW, cudaMemcpyDeviceToHost);
//	cudaMemcpy(firstBias, d_bias, sizeBias, cudaMemcpyDeviceToHost);
//
//	//for (int j = 0; j < 20; ++j){
//	//	qDebug() << "First bias: " << j << ": " << firstBias[j];
//	//}
//
//	cudaFree(d_FHW);
//	cudaFree(d_FirstHiddenVotes);
//	cudaFree(d_InputVotes);
//	cudaFree(d_bias);
//#pragma endregion
//
//	//qDebug() << "Second Learn";
//#pragma region update second layer
//	//float* d_SHW;
//	//float* d_SecondHiddenVotes;
//	//float* HiddenVotes;// SecondHiddenVotes;
//
//	//int sizeSHW = (20 * 15) *sizeof(float);
//	//int sizeSecondHidden = 15 * sizeof(float);
//	//sizeInputVotes = 20 * sizeof(float);
//	//sizeBias = 15 * sizeof(float);
//
//	//cudaMalloc((void**)&d_SHW, sizeSHW);
//	//cudaMalloc((void**)&d_SecondHiddenVotes, sizeSecondHidden);
//	//cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
//	//cudaMalloc((void**)&d_bias, sizeBias);
//	///*SecondHiddenVotes*/HiddenVotes = new float[15];
//
//	//cudaMemcpy(d_SHW, secondHiddenWeights, sizeSHW, cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_InputVotes, FirstHiddenVotes, sizeInputVotes, cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_bias, secondBias, sizeBias, cudaMemcpyHostToDevice);
//
//	//for (int i = 0; i < 15; ++i){
//	//	if(secondBias[i] < -0.1f) qDebug() << "Bias: " << secondBias[i];
//	//}
//
//	////Input votes, Hidden weights, Number of Inputs, Votes array
//	//printUpdateHidden << <1, 15 >> >(d_InputVotes, d_SHW, d_bias, 20, d_SecondHiddenVotes, lr, increase);
//
//	//cudaMemcpy(/*SecondHiddenVotes*/HiddenVotes, d_SecondHiddenVotes, sizeSecondHidden, cudaMemcpyDeviceToHost);
//	//cudaMemcpy(secondBias, d_bias, sizeBias, cudaMemcpyDeviceToHost);
//	//cudaMemcpy(secondHiddenWeights, d_SHW, sizeSHW, cudaMemcpyDeviceToHost);
//
//	//cudaFree(d_bias);
//	//cudaFree(d_InputVotes); cudaFree(d_SHW); cudaFree(d_SecondHiddenVotes);
//#pragma endregion
//
//#pragma region update third layer
//	//		float* d_THW;
//	//		float* d_thirdHiddenVotes;
//	//		float* HiddenVotes;
//	//
//	//		int sizeTHW = (10 * 15) *sizeof(float);
//	//		int sizethirdHidden = 10 * sizeof(float);
//	//		sizeInputVotes = 15 * sizeof(float);
//	//		sizeBias = 10 * sizeof(float);
//	//
//	//		cudaMalloc((void**)&d_THW, sizeTHW);
//	//		cudaMalloc((void**)&d_thirdHiddenVotes, sizethirdHidden);
//	//		cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
//	//		cudaMalloc((void**)&d_bias, sizeBias);
//	//		HiddenVotes = new float[10];
//	//
//	//		cudaMemcpy(d_THW, thirdHiddenWeights, sizeTHW, cudaMemcpyHostToDevice);
//	//		cudaMemcpy(d_InputVotes, InputVotes, sizeInputVotes, cudaMemcpyHostToDevice);
//	//		cudaMemcpy(d_bias, bias, sizeBias, cudaMemcpyHostToDevice);
//	//
//	//		//Input votes, Hidden weights, Number of Inputs, Votes array
//	//		updateHidden << <1, 10 >> >(d_InputVotes, d_THW, d_bias, 15, d_thirdHiddenVotes, lr, increase);
//	//
//	//		cudaMemcpy(HiddenVotes, d_thirdHiddenVotes, sizethirdHidden, cudaMemcpyDeviceToHost);
//	//		cudaMemcpy(thirdHiddenWeights, d_THW, sizeTHW, cudaMemcpyDeviceToHost);
//	//
//	//		cudaFree(d_bias);
//	//		cudaFree(d_InputVotes); cudaFree(d_THW); cudaFree(d_thirdHiddenVotes);
//#pragma endregion
//
//#pragma region update outputs
//	float* d_outputHiddenVotes;
//	float* d_outputWeights;
//	float* d_votes;
//	float* votes;
//
//	votes = new float[numInput];
//
//	//1-2 layer
//	int sizeHiddenOutput = 15 * sizeof(float);
//	//1-2 layer
//	int sizeWeightsOutput = 15 * numInput * sizeof(float);
//	int sizeVotesOutput = numInput * sizeof(float);
//
//	cudaMalloc((void**)&d_outputHiddenVotes, sizeHiddenOutput);
//	cudaMalloc((void**)&d_outputWeights, sizeWeightsOutput);
//	cudaMalloc((void**)&d_votes, sizeVotesOutput);
//
//	cudaMemcpy(d_outputHiddenVotes, HiddenVotes, sizeHiddenOutput, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_outputWeights, outputWeights, sizeWeightsOutput, cudaMemcpyHostToDevice);
//
//	//The number of threads is the number of inputs possible, so Left or Right
//	//The third varaible is the number of hidden layers
//	//1-2 layer
//	OutputLayer << <1, 3 >> >(d_outputHiddenVotes, d_outputWeights, 15, d_votes);
//
//	cudaMemcpy(votes, d_votes, sizeVotesOutput, cudaMemcpyDeviceToHost);
//
//	cudaFree(d_outputHiddenVotes);
//	cudaFree(d_outputWeights);
//	cudaFree(d_votes);
//
//	//for (int i = 0; i < 10; ++i){
//	//	qDebug() << i << " " << HiddenVotes[i];
//	//}
//
//	float tally = -FLT_MAX;
//	for (int i = 0; i < numInput; ++i){
//		//qDebug() << "Input: " << i << " Tally: " << votes[i];
//		if (tally < votes[i]){
//			tally = votes[i];
//			lastInput = i;
//		}
//	}
//
//	for (int i = 0; i < 15; ++i){
//		//1-2 layer
//		outputWeights[lastInput * 15 + i] = outputWeights[lastInput * 15 + i] + increase * lr;
//	}
//#pragma endregion
//
//#pragma region cleanup
//	delete[] InputVotes;
//	delete[] votes;
//	//delete[] FirstHiddenVotes;
//	delete[] HiddenVotes;
//#pragma endregion

//	//Run sigmoid on all input and store the votes or output in the float array SecondHiddenVotes
//#pragma region SecondHidden
//
//#pragma endregion
//
//#pragma region ThirdHidden
//	
//
//#pragma endregion
//
//	//Connect all hidden nodes to the output nodes. Store values in the array votes
//#pragma region output
//	float* d_outputHiddenVotes;
//	float* d_outputWeights;
//	float* d_votes;
//	float* votes;
//
//	votes = new float[numInput];
//
//	int sizeHiddenOutput = 10 * sizeof(float);
//	int sizeWeightsOutput = 10 * numInput * sizeof(float);
//	int sizeVotesOutput = numInput * sizeof(float);
//
//	cudaMalloc((void**)&d_outputHiddenVotes, sizeHiddenOutput);
//	cudaMalloc((void**)&d_outputWeights, sizeWeightsOutput);
//	cudaMalloc((void**)&d_votes, sizeVotesOutput);
//
//	cudaMemcpy(d_outputHiddenVotes, HiddenVotes, sizeHiddenOutput, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_outputWeights, outputWeights, sizeWeightsOutput, cudaMemcpyHostToDevice);
//
//	//The number of threads is the number of inputs possible, so Left or Right
//	//The third varaible is the number of hidden layers
//	OutputLayer << <1, 3 >> >(d_outputHiddenVotes, d_outputWeights, 10, d_votes);
//
//	cudaMemcpy(votes, d_votes, sizeVotesOutput, cudaMemcpyDeviceToHost);
//
//	cudaFree(d_outputHiddenVotes); cudaFree(d_outputWeights); cudaFree(d_votes);
//
//	//for (int i = 0; i < 10; ++i){
//	//	qDebug() << i << " " << HiddenVotes[i];
//	//}
//#pragma endregion
//

//#pragma region rewards
//	//Iterate over the screens stored so far
//	for (int i = 0; i < NumScreens; ++i){
//		//Reward the neurons weight that gave a positive input to the output neuron
//#pragma region Output Weight reward
//		//Need to change outputWeights size is 10 * numInputs
//		//There are 10 hidden neurons that all connect to the desired output neuron
//		//Reward all 10 based on their contribution
//		for (int weightIndex = 0; weightIndex < 10; ++weightIndex){
//			//Stride would be 10, inputStorage is which one to start at
//			//Increase weight by 1 percent
//			float learningRate = 1 - i * 0.05;
//			if (learningRate < 0) learningRate = 0.05;
//
//			outputWeights[inputStorage[i] * 10 + weightIndex] += increase * learningRate;
//		}
//#pragma endregion
//
//#pragma region Third Hidden Weight Reward
//		//Need to change firstHiddenWeights size is 50 * 50 * 20
//		//There are 2500 input neurons that all connect to each hidden neuron
//		//Rewarding all of them based on their contirubtion, doesn't seem right
//
//		//Reward them by a percentage based on their weight to the desired output neuron
//		//Iterate over all previous inputs
//		for (int i = 0; i < NumScreens; ++i){
//
//			//Iterate over all 20 first hidden neurons
//			for (int hiddenIndex = 0; hiddenIndex < 20; ++hiddenIndex){
//
//				//Get weight to output neuron
//				float weight = outputWeights[20 * inputStorage[i] + hiddenIndex];
//
//				//Iterate over all the input nodes and increase the firstHiddenWeights value
//				for (int inputWeightIndex = 0; inputWeightIndex < 2500; ++inputWeightIndex){
//
//					//Stride is 2500
//					float learningRate = 1 - i * 0.05;
//					if (learningRate < 0) learningRate = 0.05;
//					firstHiddenWeights[hiddenIndex * 2500 + inputWeightIndex] += weight * increase * learningRate;
//				}
//			}
//		}
//
//#pragma endregion
//
//#pragma region Second Hidden Weight Reward
//
//#pragma endregion
//
//#pragma region First Hidden Weight Reward
//
//#pragma endregion
#pragma endregion