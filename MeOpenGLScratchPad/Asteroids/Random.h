#ifndef RANDOM_H
#define RANDOM_H
#include "Vector2d.h"
#include "Core.h"
using Core::RGB;

struct Random
{
	float randomInRange( float min, float max);
	float randomFloat();
	Vector2d randomUnitVector();
	RGB Random::randomColor(float r, float g, float b, float range);
	Random(void);
	~Random(void);
};

#endif // !RANDOM_H

