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


//Might want to reduce size of initial screen. Will allow for more screens stored. Longer learning time though :(
//Need to cuda-ify the learning algorithm.

//Might wanna remove input weights. Seems useless I think
using std::vector;

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

	reduceScreen = new float[rWidth * rHeight];
	inputWeights = new float[50 * 50];
	for (int i = 0; i < 50 * 50; ++i){
		inputWeights[i] = 1;//rand.randomFloat();
	}

	InputBias = new float[25 * 25];
	for (int i = 0; i < 25 * 25; ++i){
		InputBias[i] = -1.0f *  rand.randomFloat();
	}

	//Number of hidden nodes * number of input nodes
	firstHiddenWeights = new float[20 * (25 * 25)];
	for (int i = 0; i < 20 * 25 * 25; ++i){
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

	firstBias = new float[20];
	//Bias is currently generic numbers
	//Bias is set for specifically for breakout
	for (int i = 0; i < 20; ++i){
		firstBias[i] = -1 * rand.randomInRange(175, 200);
	}

	secondBias = new float[15];
	//Bias is currently generic numbers
	//Bias is set for specifically for breakout
	for (int i = 0; i < 15; ++i){
		secondBias[i] =  -1 * rand.randomInRange(4, 7);
	}

	//1-2 layer
	outputWeights = new float[15 * numInput];
	for (int i = 0; i < 15 * numInput; ++i){
		outputWeights[i] = rand.randomFloat();
	}

	//400x300 is screen size, 225 frames is 15 seconds
	screenStorageCount = 0;
	inputStorage = new int[50 * 50 * 225];
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

	//Should use sigmoid here. Maybe Could be in for loop though
	//printf("Total: %f\n", total);
	total += bias[id];
	//printf("Total: %f\n", total);
	//printf("Bias: %f\n", bias[id]);
	total = (1 / (1 + exp(-total)));
	//total = ((int)(total)) % 3;

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
#pragma endregion

#pragma region input
//Find what input would be best.
//This is possible with the number of threads and blocks form the ID, ID can be considered 0 based, idk how though but it works
//After that the array is a 1D array anyways that I'm working with shouldn't be hard to just copy pasta into a cuda function
void DeepLearner::StoreScreen(){
	int screenStride = 50 * 50;
	int rowStride = 50;

	for (int x = 0; x < 50; ++x){
		for (int y = 0; y < 50; ++y){
			//Index violation right now, so sad :(
			inputStorage[screenStride * screenStorageCount + x * rowStride + y] = reduceScreen[x * rowStride + y];
		}
	}

	//inputStorage[screenStorageCount] = lastInput;

	++screenStorageCount;
	if (screenStorageCount >= 223){
		screenStorageCount = 0;
		FullStorage = true;
	}
}

int  DeepLearner::GetInput(vector<float*> screengrab){
	numCalls++;
	float* screenBits = new float[50 * 50];
	GetScreen();

	//update weights if I got higher score
	if (lastScore < *score){
		learn(true);
		lastScore = *score;
	}

	if (numCalls > 3){
#pragma region Random Input
		if (rand.randomInRange(0, 1) < f_RandomChance){
			lastInput = rand.randomInRange(0, numInput);
			numCalls = 0;
			//f_RandomChance = 0.00f;
		}
#pragma endregion

		else{

			//Screen maniputlation only works for 800x600 screens currently. Changes to a screen of 400x300 greyscaled
			//Also stores the 8x6 mini pixel set in the variable screenbits
#pragma region screen manipulation
			//Seperate the reduce screen into 8x6 chunks.
			//Average the intensity for those pixels.
			int bitsIndex = 0;

			//Reduce the screen into a 50x50 grid of 8x6 pixels. Average the intensity to get the average brightness of that grid.
			for (int r = 0; r < 50; r++){
				for (int c = 0; c < 50; c++){
					float intense = 0.0f;
					int numPixels = 0;
					for (int row = 0; row < 6; ++row){
						for (int col = 0; col < 8; ++col){
							float reduxIntense = reduceScreen[(r * 400 * 6) + (row * 400) + (c * 8 + col)];
							intense += reduxIntense;
							++numPixels;
						}
					}
					if (intense > 0.0f){
						//qDebug() << "Intense value: " << intense;
					}
					screenBits[bitsIndex] = (intense) / ((float)numPixels);
					++bitsIndex;
				}
			}

#pragma endregion

			//Get all the inputs times their weight change it up into 2x2 squares and store it in the array InputVotes
#pragma region Input Weights
			float* InputVotes = new float[625];

			//for (int i = 0; i < 50 * 50; i++){
				//qDebug() << "Screen Bits: " << screenBits[i];
			//}
			
			for (int r = 0; r < 50; r += 2){
				for (int c = 0; c < 50; c += 2){
					float total = 0;

					//qDebug() << "Screen weight 1: " << screenBits[r * 50 + c];
					//qDebug() << "Screen weight 2: " << screenBits[r * 50 + c + 1];
					//qDebug() << "Screen weight 3: " << screenBits[(r + 1) * 50 + c];
					//qDebug() << "Screen weight 4: " << screenBits[(r + 1) * 50 + c + 1];
					
					//qDebug() << "Input weight 1: " << inputWeights[r * 50 + c];
					//qDebug() << "Input weight 2: " << inputWeights[r * 50 + c + 1];
					//qDebug() << "Input weight 3: " << inputWeights[(r + 1) * 50 + c];
					//qDebug() << "Input weight 4: " << inputWeights[(r + 1) * 50 + c + 1];

					total += screenBits[r * 50 + c] * inputWeights[r * 50 + c];
					total += screenBits[r * 50 + c + 1] * inputWeights[r * 50 + c + 1];
					total += screenBits[(r + 1) * 50 + c] * inputWeights[(r + 1) * 50 + c];
					total += screenBits[(r + 1) * 50 + c + 1] * inputWeights[(r + 1) * 50 + c + 1];

					//qDebug() << "Total: " << total;

					//r and c are based on 50x50 grid. Divide by two to get 25x25 grid
					total += InputBias[(r / 2) * 25 + (c / 2)];

					//qDebug() << "Total with bias: " << total;

					//Sigmoid everything!
					InputVotes[(r / 2) * 25 + (c / 2)] =  (1 / (1 + exp(total)));
					//qDebug() << InputVotes[(r / 2) * 25 + (c / 2)];
				}
			}

			//for (int i = 0; i < 25 * 25; ++i){
			//	if (InputVotes[i] > -1)
			//		qDebug() << "Vote: " << i << " " << InputVotes[i];
			//}

			//Cuda implementation of old input votes

			//float* d_screen;
			//float* d_weights;
			//int* d_numInput;
			//float* d_Votes;
			//float* InputVotes = new float[4 * 625];
			//int sizeInt = sizeof(int);
			//int sizeScreen = (50 * 50) *sizeof(float);
			//int sizeWeights = (50 * 50) *sizeof(int);
			//cudaMalloc((void**)&d_screen, sizeScreen);
			//cudaMalloc((void**)&d_weights, sizeWeights);
			//cudaMalloc((void**)&d_Votes, sizeWeights);
			//cudaMalloc((void**)&d_numInput, sizeInt);
			//cudaMemcpy(d_screen, screenBits, sizeScreen, cudaMemcpyHostToDevice);
			//cudaMemcpy(d_weights, inputWeights, sizeWeights, cudaMemcpyHostToDevice);
			//cudaMemcpy(d_numInput, &numInput, sizeInt, cudaMemcpyHostToDevice);
			//CalcInput << < 1, 625 >> >(d_screen, d_weights, d_Votes, 50);
			//cudaMemcpy(InputVotes, d_Votes, sizeWeights, cudaMemcpyDeviceToHost);
			//cudaFree(d_screen);
			//cudaFree(d_weights);
			//cudaFree(d_Votes);
			//cudaFree(d_numInput);

#pragma endregion

			//Run sigmoid on all input and store the votes or output in the float array FirstHiddenVotes
#pragma region FirstHidden
			float* d_InputVotes;
			float* d_FHW;
			float* d_bias;
			float* d_FirstHiddenVotes;
			//1-2 layer
			float* FirstHiddenVotes;

			int sizeFHW = (20 * 25 * 25) *sizeof(float);
			//1-2 layer
			int sizeHidden = 20 * sizeof(float);
			int sizeInputVotes = 25 * 25 * sizeof(float);
			int sizeBias = 20 * sizeof(float);

			FirstHiddenVotes = new float[20];
			cudaMalloc((void**)&d_FHW, sizeFHW);
			cudaMalloc((void**)&d_FirstHiddenVotes, sizeHidden);
			cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
			cudaMalloc((void**)&d_bias, sizeBias);
			cudaMemcpy(d_FHW, firstHiddenWeights, sizeFHW, cudaMemcpyHostToDevice);
			cudaMemcpy(d_InputVotes, InputVotes, sizeInputVotes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bias, firstBias, sizeBias, cudaMemcpyHostToDevice);

			//Input votes, Hidden weights, Number of Inputs, Votes array
			FirstHidden << <1, 20 >> >(d_InputVotes, d_FHW, d_bias, 25 * 25, d_FirstHiddenVotes);
			cudaMemcpy(FirstHiddenVotes, d_FirstHiddenVotes, sizeHidden, cudaMemcpyDeviceToHost);
			cudaFree(d_FHW);
			cudaFree(d_FirstHiddenVotes);
			cudaFree(d_InputVotes);
			cudaFree(d_bias);

			//for (int i = 0; i < 20; ++i){
			//	qDebug() << "First Hidden Votes: " << i << ": " << FirstHiddenVotes[i];
			//}

#pragma endregion

			//qDebug();
			//qDebug() << "Second hidden";
			//Run sigmoid on all input and store the votes or output in the float array SecondHiddenVotes
#pragma region SecondHidden
			float* d_SHW;
			float* d_SecondHiddenVotes;
			float* HiddenVotes;//SecondHiddenVotes;

			int sizeSHW = (20 * 15) *sizeof(float);
			int sizeSecondHidden = 15 * sizeof(float);
			sizeInputVotes = 20 * sizeof(float);
			sizeBias = 15 * sizeof(float);

			cudaMalloc((void**)&d_SHW, sizeSHW);
			cudaMalloc((void**)&d_SecondHiddenVotes, sizeSecondHidden);
			cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
			cudaMalloc((void**)&d_bias, sizeBias);
			//SecondHiddenVotes
			HiddenVotes = new float[15];

			//for (int i = 0; i < 15; ++i){
			//	qDebug() << "SHW: " << secondHiddenWeights[i];
			//}

			//for (int i = 0; i < 20; ++i){
			//	qDebug() << "Second Votes " << i << ": " << FirstHiddenVotes[i];
			//}

			cudaMemcpy(d_SHW, secondHiddenWeights, sizeSHW, cudaMemcpyHostToDevice);
			cudaMemcpy(d_InputVotes, FirstHiddenVotes, sizeInputVotes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bias, secondBias, sizeBias, cudaMemcpyHostToDevice);


			//Input votes, Hidden weights, Number of Inputs, Votes array
			FirstHidden <<<1, 15 >>>(d_InputVotes, d_SHW, d_bias, 20, d_SecondHiddenVotes);

			//SecondHiddenVotes
			cudaMemcpy(HiddenVotes, d_SecondHiddenVotes, sizeSecondHidden, cudaMemcpyDeviceToHost);


			cudaFree(d_SHW);
			cudaFree(d_SecondHiddenVotes);
			cudaFree(d_InputVotes);
			cudaFree(d_bias);

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
//			cudaMalloc((void**)&d_THW, sizeTHW);
//			cudaMalloc((void**)&d_thirdHiddenVotes, sizethirdHidden);
//			cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
//			cudaMalloc((void**)&d_bias, sizeBias);
//			HiddenVotes = new float[10];
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
//			cudaFree(d_THW);
//			cudaFree(d_thirdHiddenVotes);
//			cudaFree(d_InputVotes); 
//			cudaFree(d_bias);
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
			int sizeHiddenOutput = 15 * sizeof(float);
			//1-2 layer
			int sizeWeightsOutput = 15 * numInput * sizeof(float);
			int sizeVotesOutput = numInput * sizeof(float);

			cudaMalloc((void**)&d_outputHiddenVotes, sizeHiddenOutput);
			cudaMalloc((void**)&d_outputWeights, sizeWeightsOutput);
			cudaMalloc((void**)&d_votes, sizeVotesOutput);

			cudaMemcpy(d_outputHiddenVotes, HiddenVotes, sizeHiddenOutput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_outputWeights, outputWeights, sizeWeightsOutput, cudaMemcpyHostToDevice);

			//The number of threads is the number of inputs possible, so Left or Right
			//The third varaible is the number of hidden layers
			//1-2 layer
			OutputLayer << <1, 3 >> >(d_outputHiddenVotes, d_outputWeights, 15, d_votes);

			cudaMemcpy(votes, d_votes, sizeVotesOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_outputHiddenVotes);
			cudaFree(d_outputWeights);
			cudaFree(d_votes);

			//for (int i = 0; i < 10; ++i){
			//	qDebug() << i << " " << HiddenVotes[i];
			//}
#pragma endregion

#pragma region Tally Votes

			//TODO: tally is currently a memory leak. I should fix when I can
			int tally = -INT_MAX;
			for (int i = 0; i < numInput; ++i){
				qDebug() << "Input: " << i << " Tally: " << votes[i];
				if (tally < votes[i]){
					tally = votes[i];
					lastInput = i;
				}
			}

			qDebug();
			qDebug();

			//delete[] tally;
			delete[] screenBits;
			delete[] InputVotes;
			delete[] HiddenVotes;
			//delete[] votes;
			numCalls = 0;
#pragma endregion
		}
	}
	StoreScreen();
	//delete[] screenbits;

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
	}

	//for (int i = 0; i < numPixels; i++){
	//	if (greyScreen[i] > 0.0f){
	//		qDebug() << "Pixel: " << i << " Intensity: " << greyScreen[i];
	//	}
	//}

	//Shrink the image
	int i = 0;
	for (int r = 0; r < *height; r += 2){
		for (int c = 0; c < *width; c += 2){
			float x1 = greyScreen[c + r * 800];
			float x2 = greyScreen[c + 1 + r * 800];
			float x3 = greyScreen[c + (r + 1) * 800];
			float x4 = greyScreen[c + 1 + (r + 1) * 800];

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

#pragma region cuda code
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
		total += (1 / (1 + exp(-sig)));
	}

	//Should use sigmoid here. Maybe Could be in for loop though
	total += bias[id];
	//printf("Total: %f\n", total);
	//printf("Bias: %f\n", bias[id]);
	total = (1 / (1 + exp(-total)));
	//total = ((int)(total)) % 3;

	//printf("Total: %f\n", total);
	d_votes[id] = total;
#pragma endregion

#pragma region update weights
	for (int i = 0; i < d_numVotes; ++i){
		weight[id * d_numVotes + i] = weight[id * d_numVotes + i] - (error * learningRate * ((1 / (1 + exp(-total))) * (1 - (1 / (1 + exp(-total))))));
	}
#pragma endregion
}

//Number of threads is the number of screens or number of stored inputs
//ID could be lr? maybe?
//float* array of weights, number of nodes, float* of outputweights
__global__ void updateWeight(float* weightsInput, int numHiddenNodes, float* weightsOutput, int numInputNodes){

}
#pragma endregion

void DeepLearner::learn(bool isWin){
	if (f_RandomChance > 0.10) {
		f_RandomChance -= 0.0001f;
	}
	//1-2 layer
	float increase = (isWin) ? +0.01 : -0.01;
	int NumScreens = (FullStorage) ? 225 : screenStorageCount;

	//INTENTIONAL ERROR, starting at the numscreens and decrementing. Isn't right because last screen is at screenStorageCount. Here for simplicity currently.

	//Increase is error
	//Num screens is the amount of input that has been stored
	//Iterate over the inputs and times by weight. Update weights by w = w - error * sig_prime
	//Iterate over first layer with input layer inputs, use update above
	//Iterate over second layer with first layer inputs, use update above
	//Iterate over third layer, with second layer inputs, use update above
	//Iterate over output layer, 
	for (int i = 0; i < 30; ++i){
	//for (int i = NumScreens - 1; i > 0; --i){
		int UpdateScreen = rand.randomInRange(0, NumScreens - 2);
		float* screenBits = new float[50 * 50];
		for (int j = 0; j < 50 * 50; ++j){
			screenBits[j] = inputStorage[UpdateScreen * 50 * 50 + j];
		}

#pragma region update inputs
		float* InputVotes = new float[625];

		//for (int i = 0; i < 50 * 50; i++){
		//qDebug() << "Screen Bits: " << screenBits[i];
		//}

		for (int r = 0; r < 50; r += 2){
			for (int c = 0; c < 50; c += 2){
				float total = 0;

				//qDebug() << "Screen weight 1: " << screenBits[r * 50 + c];
				//qDebug() << "Screen weight 2: " << screenBits[r * 50 + c + 1];
				//qDebug() << "Screen weight 3: " << screenBits[(r + 1) * 50 + c];
				//qDebug() << "Screen weight 4: " << screenBits[(r + 1) * 50 + c + 1];

				//qDebug() << "Input weight 1: " << inputWeights[r * 50 + c];
				//qDebug() << "Input weight 2: " << inputWeights[r * 50 + c + 1];
				//qDebug() << "Input weight 3: " << inputWeights[(r + 1) * 50 + c];
				//qDebug() << "Input weight 4: " << inputWeights[(r + 1) * 50 + c + 1];

				total += screenBits[r * 50 + c] * inputWeights[r * 50 + c];
				total += screenBits[r * 50 + c + 1] * inputWeights[r * 50 + c + 1];
				total += screenBits[(r + 1) * 50 + c] * inputWeights[(r + 1) * 50 + c];
				total += screenBits[(r + 1) * 50 + c + 1] * inputWeights[(r + 1) * 50 + c + 1];

				//qDebug() << "Total: " << total;

				//r and c are based on 50x50 grid. Divide by two to get 25x25 grid
				total += InputBias[(r / 2) * 25 + (c / 2)];

				//qDebug() << "Total with bias: " << total;

				//Sigmoid everything!
				InputVotes[(r / 2) * 25 + (c / 2)] = (1 / (1 + exp(total)));
			}
		}

		//float* d_screen;
		//float* d_weights;
		//int* d_numInput;
		//float* d_Votes;
		//float* InputVotes = new float[4 * 625];
		//int sizeInt = sizeof(int);
		//int sizeScreen = (50 * 50) *sizeof(float);
		//int sizeWeights = (50 * 50) *sizeof(int);
		//cudaMalloc((void**)&d_screen, sizeScreen);
		//cudaMalloc((void**)&d_weights, sizeWeights);
		//cudaMalloc((void**)&d_Votes, sizeWeights);
		//cudaMalloc((void**)&d_numInput, sizeInt);
		//cudaMemcpy(d_screen, screenBits, sizeScreen, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_weights, inputWeights, sizeWeights, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_numInput, &numInput, sizeInt, cudaMemcpyHostToDevice);
		//updateInput << < 4, 625 >> >(d_screen, d_weights, d_Votes);
		//cudaMemcpy(InputVotes, d_Votes, sizeWeights, cudaMemcpyDeviceToHost);
		//cudaMemcpy(inputWeights, d_weights, sizeWeights, cudaMemcpyDeviceToHost);
		//cudaFree(d_screen);
		//cudaFree(d_weights);
		//cudaFree(d_Votes);
		//cudaFree(d_numInput);
#pragma endregion

#pragma region update first layer
		float* d_InputVotes;
		float* d_FHW;
		float* d_bias;
		float* d_FirstHiddenVotes;
		float* FirstHiddenVotes;

		int sizeFHW = (20 * 25 * 25) *sizeof(float);
		int sizeHidden = 20 * sizeof(float);
		int sizeInputVotes = 25 * 25 * sizeof(float);
		int sizeBias = 20 * sizeof(float);

		cudaMalloc((void**)&d_FHW, sizeFHW);
		cudaMalloc((void**)&d_FirstHiddenVotes, sizeHidden);
		cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
		cudaMalloc((void**)&d_bias, sizeBias);
		FirstHiddenVotes = new float[20];

		cudaMemcpy(d_FHW, firstHiddenWeights, sizeFHW, cudaMemcpyHostToDevice);
		cudaMemcpy(d_InputVotes, InputVotes, sizeInputVotes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_bias, firstBias, sizeBias, cudaMemcpyHostToDevice);

		//Input votes, Hidden weights, Number of Inputs, Votes array, learning rate, error
		updateHidden << <2, 10 >> >(d_InputVotes, d_FHW, d_bias, 25 * 25, d_FirstHiddenVotes, lr, increase);

		cudaMemcpy(FirstHiddenVotes, d_FirstHiddenVotes, sizeHidden, cudaMemcpyDeviceToHost);
		cudaMemcpy(firstHiddenWeights, d_FHW, sizeFHW, cudaMemcpyDeviceToHost);

		cudaFree(d_FHW);
		cudaFree(d_FirstHiddenVotes);
		cudaFree(d_InputVotes);
		cudaFree(d_bias);
#pragma endregion

#pragma region update second layer
		float* d_SHW;
		float* d_SecondHiddenVotes;
		float* HiddenVotes;// SecondHiddenVotes;

		int sizeSHW = (20 * 15) *sizeof(float);
		int sizeSecondHidden = 15 * sizeof(float);
		sizeInputVotes = 20 * sizeof(float);
		sizeBias = 15 * sizeof(float);

		cudaMalloc((void**)&d_SHW, sizeSHW);
		cudaMalloc((void**)&d_SecondHiddenVotes, sizeSecondHidden);
		cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
		cudaMalloc((void**)&d_bias, sizeBias);
		/*SecondHiddenVotes*/HiddenVotes = new float[15];

		cudaMemcpy(d_SHW, secondHiddenWeights, sizeSHW, cudaMemcpyHostToDevice);
		cudaMemcpy(d_InputVotes, InputVotes, sizeInputVotes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_bias, secondBias, sizeBias, cudaMemcpyHostToDevice);

		//Input votes, Hidden weights, Number of Inputs, Votes array
		updateHidden << <1, 15 >> >(d_InputVotes, d_SHW, d_bias, 20, d_SecondHiddenVotes, lr, increase);

		cudaMemcpy(/*SecondHiddenVotes*/HiddenVotes, d_SecondHiddenVotes, sizeSecondHidden, cudaMemcpyDeviceToHost);
		cudaMemcpy(secondHiddenWeights, d_SHW, sizeSHW, cudaMemcpyDeviceToHost);

		cudaFree(d_bias);
		cudaFree(d_InputVotes); cudaFree(d_SHW); cudaFree(d_SecondHiddenVotes);
#pragma endregion

#pragma region update third layer
//		float* d_THW;
//		float* d_thirdHiddenVotes;
//		float* HiddenVotes;
//
//		int sizeTHW = (10 * 15) *sizeof(float);
//		int sizethirdHidden = 10 * sizeof(float);
//		sizeInputVotes = 15 * sizeof(float);
//		sizeBias = 10 * sizeof(float);
//
//		cudaMalloc((void**)&d_THW, sizeTHW);
//		cudaMalloc((void**)&d_thirdHiddenVotes, sizethirdHidden);
//		cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
//		cudaMalloc((void**)&d_bias, sizeBias);
//		HiddenVotes = new float[10];
//
//		cudaMemcpy(d_THW, thirdHiddenWeights, sizeTHW, cudaMemcpyHostToDevice);
//		cudaMemcpy(d_InputVotes, InputVotes, sizeInputVotes, cudaMemcpyHostToDevice);
//		cudaMemcpy(d_bias, bias, sizeBias, cudaMemcpyHostToDevice);
//
//		//Input votes, Hidden weights, Number of Inputs, Votes array
//		updateHidden << <1, 10 >> >(d_InputVotes, d_THW, d_bias, 15, d_thirdHiddenVotes, lr, increase);
//
//		cudaMemcpy(HiddenVotes, d_thirdHiddenVotes, sizethirdHidden, cudaMemcpyDeviceToHost);
//		cudaMemcpy(thirdHiddenWeights, d_THW, sizeTHW, cudaMemcpyDeviceToHost);
//
//		cudaFree(d_bias);
//		cudaFree(d_InputVotes); cudaFree(d_THW); cudaFree(d_thirdHiddenVotes);
#pragma endregion

#pragma region update outputs
		float* d_outputHiddenVotes;
		float* d_outputWeights;
		float* d_votes;
		float* votes;

		votes = new float[numInput];

		//1-2 layer
		int sizeHiddenOutput = 15 * sizeof(float);
		//1-2 layer
		int sizeWeightsOutput = 15 * numInput * sizeof(float);
		int sizeVotesOutput = numInput * sizeof(float);

		cudaMalloc((void**)&d_outputHiddenVotes, sizeHiddenOutput);
		cudaMalloc((void**)&d_outputWeights, sizeWeightsOutput);
		cudaMalloc((void**)&d_votes, sizeVotesOutput);

		cudaMemcpy(d_outputHiddenVotes, HiddenVotes, sizeHiddenOutput, cudaMemcpyHostToDevice);
		cudaMemcpy(d_outputWeights, outputWeights, sizeWeightsOutput, cudaMemcpyHostToDevice);

		//The number of threads is the number of inputs possible, so Left or Right
		//The third varaible is the number of hidden layers
		//1-2 layer
		OutputLayer << <1, 3 >> >(d_outputHiddenVotes, d_outputWeights, 15, d_votes);

		cudaMemcpy(votes, d_votes, sizeVotesOutput, cudaMemcpyDeviceToHost);

		cudaFree(d_outputHiddenVotes);
		cudaFree(d_outputWeights);
		cudaFree(d_votes);

		//for (int i = 0; i < 10; ++i){
		//	qDebug() << i << " " << HiddenVotes[i];
		//}

		//TODO: tally is currently a memory leak. I should fix when I can
		int tally = 0.0f;
		for (int i = 0; i < numInput; ++i){
			//qDebug() << "Input: " << i << " Tally: " << votes[i];
			if (tally < votes[i]){
				tally = votes[i];
				lastInput = i;
			}
		}

		for (int i = 0; i < 15; ++i){
			//1-2 layer
			outputWeights[lastInput * 15 + i] = outputWeights[lastInput * 15 + i] + increase * lr;
		}
#pragma endregion
		delete[] screenBits;
		delete[] InputVotes;
		delete[] HiddenVotes;

	}


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


	//}
}

void DeepLearner::ResetScore(){
	lastScore = 0;
}

#pragma endregion



#pragma region Holding code, current old learning

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