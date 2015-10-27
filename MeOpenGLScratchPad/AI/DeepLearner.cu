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
		inputWeights[i] = rand.randomFloat();
	}

	//Number of hidden nodes * number of input nodes
	firstHiddenWeights = new float[10 * (50*50)];
	for (int i = 0; i < 10 * 50 * 50; ++i){
		firstHiddenWeights[i] = rand.randomFloat();
	}
	
	bias = new float[10];
	for (int i = 0; i < 10; ++i){
		bias[i] = -1 * rand.randomInRange(0, 40.0f);
	}

	outputWeights = new float[10 * numInput];
	for (int i = 0; i < 10 * numInput; ++i){
		outputWeights[i] = rand.randomFloat();
	}
	
	//400x300 is screen size, 225 frames is 15 seconds
	screenStorageCount = 0;
	inputStorage = new int[225];
	FullStorage = false;
}

DeepLearner::DeepLearner() : f_RandomChance(0.1) 
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
__global__ void CalcInput(float* screen, float* weight, float* d_Votes){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	d_Votes[id] = screen[id] * weight[id];
}

__global__ void FirstHidden(float* input, float* weight, float* bias, int d_numVotes, int* d_votes){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	float total = 0.0f;

	//printf("Num Votes: %i", d_numVotes);

	for (int i = 0; i < d_numVotes; ++i){
		//if (weight[id*d_numVotes + i] > 0) printf("Weight higher than 0: %f", weight[id*d_numVotes + i]);
		//if (input[i] > 0) printf("Input: %f ", input[i]);
		//printf("Weight: %f\n", weight[id * d_numVotes + i]);

		total += input[i] * weight[id * d_numVotes + i];
	}

	//Should use sigmoid here. Maybe Could be in for loop though
	total += *bias;
	//printf("Total: %f\n", total);
	total = (int)(1 / (1 + exp(-total))) % 3;
	//total = ((int)(total)) % 3;

	//printf("Total: %f\n", total);
	d_votes[id] = total;
}

__global__ void OutputLayer(float* hiddenVotes, float* weight, int d_numHiddenNodes, float* d_votes){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	d_votes[id] = 0.0f;

	for (int i = 0; i < d_numHiddenNodes; ++i){
		//printf("Hidden Votes: %f, Weight: %f\n", hiddenVotes[i], weight[id * d_numHiddenNodes + i]);
		d_votes[id] += hiddenVotes[i] * weight[id * d_numHiddenNodes + i];
	}

	printf("Votes: %f\n", d_votes[id]);
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
void DeepLearner::StoreScreen(){
	int screenStride = 400 * 300;
	//int rowStride = 400;
	//for (int x = 0; x < 400; ++x){
	//	for (int y = 0; y < 300; ++y){
	//		//Index violation right now, so sad :(
	//		//screenStorage[screenStride * screenStorageCount + x * rowStride + y] = reduceScreen[x * rowStride + y];
	//	}
	//}
	inputStorage[screenStorageCount] = lastInput;

	++screenStorageCount;
	if (screenStorageCount >= 225){
		screenStorageCount = 0;
		FullStorage = true;
	}
}

int  DeepLearner::GetInput(vector<float*> screengrab){
	numCalls++;

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
		}
#pragma endregion

		else{

//Screen maniputlation only works for 800x600 screens currently. Changes to a screen of 400x300 greyscaled
//Also stores the 8x6 mini pixel set in the variable screenbits
#pragma region screen manipulation
			GetScreen();
			//Seperate the reduce screen into 8x6 chunks.
			//Average the intensity for those pixels.
			float* screenBits = new float[50 * 50];
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

//Get all the inputs times their weight and store it in the array InputVotes
#pragma region Input Weights
			float* d_screen;
			float* d_weights;
			int* d_numInput;
			float* d_Votes;
			float* InputVotes = new float[4 * 625];

			int sizeInt = sizeof(int);
			int sizeScreen = (50 * 50) *sizeof(float);
			int sizeWeights = (50 * 50) *sizeof(int);

			cudaMalloc((void**)&d_screen, sizeScreen);
			cudaMalloc((void**)&d_weights, sizeWeights);
			cudaMalloc((void**)&d_Votes, sizeWeights);
			cudaMalloc((void**)&d_numInput, sizeInt);

			cudaMemcpy(d_screen, screenBits, sizeScreen, cudaMemcpyHostToDevice);
			cudaMemcpy(d_weights, inputWeights, sizeWeights, cudaMemcpyHostToDevice);
			cudaMemcpy(d_numInput, &numInput, sizeInt, cudaMemcpyHostToDevice);

			CalcInput <<< 4, 625 >>>(d_screen, d_weights, d_Votes);

			cudaMemcpy(InputVotes, d_Votes, sizeWeights, cudaMemcpyDeviceToHost);

			cudaFree(d_Votes);
			cudaFree(d_numInput);
			cudaFree(d_screen);
			cudaFree(d_numInput);

#pragma endregion

//Run sigmoid on all input and store the votes or output in the array HiddenVotes
#pragma region FirstHidden
			float* d_InputVotes;
			float* d_FHW;
			float* d_bias;
			int* d_HiddenVotes;
			int* HiddenVotes;

			int sizeFHW = (10 * 50 * 50) *sizeof(float);
			int sizeHidden = 10 * sizeof(float);
			int sizeInputVotes = 50 * 50 * sizeof(float);
			int sizeBias = 10 * sizeof(float);

			cudaMalloc((void**)&d_FHW, sizeFHW);
			cudaMalloc((void**)&d_HiddenVotes, sizeHidden);
			cudaMalloc((void**)&d_InputVotes, sizeInputVotes);
			cudaMalloc((void**)&d_bias, sizeBias);
			HiddenVotes = new int[10];

			cudaMemcpy(d_FHW, firstHiddenWeights, sizeFHW, cudaMemcpyHostToDevice);
			cudaMemcpy(d_InputVotes, InputVotes, sizeInputVotes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bias, bias, sizeBias, cudaMemcpyHostToDevice);

			//Input votes, Hidden weights, Number of Inputs, Votes array
			FirstHidden <<<1, 10 >>>(d_InputVotes, d_FHW, d_bias, 50 * 50, d_HiddenVotes);

			cudaMemcpy(HiddenVotes, d_HiddenVotes, sizeHidden, cudaMemcpyDeviceToHost);
			
			cudaFree(d_InputVotes); cudaFree(d_FHW); cudaFree(d_HiddenVotes);

#pragma endregion

//Connect all hidden nodes to the output nodes. Store values in the array votes
#pragma region output
			float* d_outputHiddenVotes;
			float* d_outputWeights;
			float* d_votes;
			float* votes;

			votes = new float[3];
			
			int sizeHiddenOutput = 10 * sizeof(float);
			int sizeWeightsOutput = 10 * numInput * sizeof(float);
			int sizeVotesOutput = numInput * sizeof(float);

			cudaMalloc((void**)&d_outputHiddenVotes, sizeHiddenOutput);
			cudaMalloc((void**)&d_outputWeights, sizeWeightsOutput);
			cudaMalloc((void**)&d_votes, sizeVotesOutput);

			cudaMemcpy(d_outputHiddenVotes, HiddenVotes, sizeHiddenOutput, cudaMemcpyHostToDevice);
			cudaMemcpy(d_outputWeights, outputWeights, sizeWeightsOutput, cudaMemcpyHostToDevice);

			//The number of threads is the number of inputs possible, so Left or Right
			//The third varaible is the number of hidden layers
			OutputLayer <<<1, 3 >>>(d_outputHiddenVotes, d_outputWeights, 10, d_votes);

			cudaMemcpy(votes, d_votes, sizeVotesOutput, cudaMemcpyDeviceToHost);

			cudaFree(d_outputHiddenVotes); cudaFree(d_outputWeights); cudaFree(d_votes);
#pragma endregion

#pragma region Tally Votes
			//TODO: tally is currently a memory leak. I should fix when I can
			int tally = 0.0f;
			for (int i = 0; i < numInput; ++i){
				qDebug() << "Input: " << i << " Tally: " << votes[i];
				if (tally < votes[i]){
					tally = votes[i];
					lastInput = i;
				}
			}

			qDebug();
			qDebug();

			delete[] screenBits;
			//delete[] tally;
			delete[] InputVotes;
			delete[] HiddenVotes;
			delete[] votes;
			numCalls = 0;
#pragma endregion
		}
	}
	StoreScreen();
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
	//Modify weights to decrease the value of what happened if loss happened
	learn(isWin);
}

void DeepLearner::SwitchAlgorithm(type algoType){
	//Changes the algorith, may remove.
	algo = algoType;
}

void DeepLearner::learn(bool isWin){
	float increase = (isWin) ? 0.01 : -0.01;
	int NumScreens = (FullStorage) ? 225 : screenStorageCount;

#pragma region rewards
	//Iterate over the screens stored so far
	for (int i = 0; i < NumScreens; ++i){
		//Reward the neurons weight that gave a positive input to the output neuron
#pragma region Output Weight reward
		//Need to change outputWeights size is 10 * numInputs
		//There are 10 hidden neurons that all connect to the desired output neuron
		//Reward all 10 based on their contribution
		for (int weightIndex = 0; weightIndex < 10; ++weightIndex){
			//Stride would be 10, inputStorage is which one to start at
			//Increase weight by 1 percent
			outputWeights[inputStorage[i] * 10 + weightIndex] *= (1.0f + increase);
		}
#pragma endregion

#pragma region Hidden Weight Reward
		//Need to change firstHiddenWeights size is 50 * 50 * 10
		//There are 2500 input neurons that all connect to each hidden neuron
		//Rewarding all of them based on their contirubtion, doesn't seem right

		//Reward them by a percentage based on their weight to the desired output neuron
		//Iterate over all previous inputs
		for (int i = 0; i < NumScreens; ++i){
			//Iterate over all 10 hidden neurons
			for (int hiddenIndex = 0; hiddenIndex < 10; ++hiddenIndex){
				//Get weight to output neuron
				float weight = outputWeights[10 * inputStorage[i] + hiddenIndex];

				//Iterate over all the input nodes and increase the firstHiddenWeights value
				for (int inputWeightIndex = 0; inputWeightIndex < 2500; ++inputWeightIndex){
					//Stride is 2500
					firstHiddenWeights[hiddenIndex * 2500 + inputWeightIndex] += weight * increase;
				}
			}
		}

#pragma endregion

#pragma endregion
	}
}

void DeepLearner::play(){
	//Play a game with a lr of 0;
}

#pragma endregion

void DeepLearner::ResetScore(){
	lastScore = 0;
}