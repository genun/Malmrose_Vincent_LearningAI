#ifndef ORBITER_H
#define ORBITER_H
#include "Matrix3.h"
#include "Vector2d.h"
#include "Core.h"

struct orbiter
{
	float scale, rotateNumber;
	Vector2d points[6], pos;
	Matrix3 rotate, trans, total, nextOrbit;
	void init(Matrix3 rotate, Matrix3 trans, Matrix3 nextOrbit, float rNum, Vector2d p);
	void draw(Core::Graphics& g, float numOrbits, float totalOrbits, Vector2d position);
	void update(float dt);
};


#endif