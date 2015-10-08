#include "Random.h"
#include <cstdlib>
#include <stdlib.h>
#include <time.h>

const float TWO_PI = 2 * 3.24159f;

float Random::randomInRange( float min, float max){
	return randomFloat() * (max-min +1) + min;
}

float Random::randomFloat(){
	return (float) rand() / RAND_MAX;
}

Vector2d Random::randomUnitVector(){
	float angle = TWO_PI * randomFloat();
	Vector2d vector(cos(angle), sin(angle));
	return vector;
}

RGB Random::randomColor(float r, float g, float b, float range){
	float red = r + Random::randomInRange(0, range);
	float green = g + Random::randomInRange(0, range);
	float blue = b + Random::randomInRange(0, range);
	return RGB(red, green, blue);
	//return RGB(color.r + Random::randomInRange(0, range), color.g + Random::randomInRange(0, range), color.b + Random::randomInRange(0, range));
}

Random::Random(void)
{
//	srand((unsigned)time(NULL));
}


Random::~Random(void)
{
}
