#include "DeepLearner.h"
#include <random>
#include <time.h>
#include <algorithm>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
using std::vector;

#pragma region Initialization
void DeepLearner::Initialize(int* scorePoint, int* widthPoint, int* heightPoint, int Number_Of_Inputs, float learningRate, type algoType){
	score = scorePoint;
	width = widthPoint;
	height = heightPoint;
	numInput = Number_Of_Inputs;
	algo = algoType;
	lr = learningRate;
	srand(time(NULL));
	numCalls = 0;
	lastInput = 0;
}

DeepLearner::DeepLearner()
{

}

DeepLearner::~DeepLearner()
{
}
#pragma endregion

#pragma region input
//This is where I get calculations, and pass the screengrab down to the neurons.
__global__ void CalcInput(int* d_Input, int* d_numInput){
	*d_Input = *d_numInput -1;
}


//Find what input would be best.
int  DeepLearner::GetInput(vector<float*> screengrab){
	numCalls++;
	if (numCalls > 3){
		int* d_Input = &lastInput;
		std::cout << *d_Input << std::endl;
		int* d_numInput = &numInput;
		std::cout << *d_numInput << std::endl;
		int size = sizeof(int);
		cudaMalloc((void**)&d_Input, size);
		cudaMalloc((void**)&d_numInput, size);
		cudaMemcpy(d_numInput, &numInput, size, cudaMemcpyHostToDevice);
		CalcInput<<<1, 1>>>(d_Input, d_numInput);
		cudaMemcpy(&lastInput, d_Input, size, cudaMemcpyDeviceToHost);
		//std::cout << *d_Input << std::endl;
		//std::cout << *d_numInput << std::endl;
		cudaFree(d_Input);
		cudaFree(d_numInput);
		numCalls = 0;
		//free(d_Input); free(d_numInput);
	}

	return lastInput;
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