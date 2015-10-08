#ifndef TURRET_H
#define TURRET_H
#include "Matrix3.h"
#include "Vector2d.h"
#include "Core.h"

struct Turret{
	Vector2d Points[9];
	Matrix3 rotation, translation, transformation;
	void init(Matrix3 mat);
	void update(float dt, Matrix3 ShipTrans, Vector2d MousePoint);
	void draw(Core::Graphics& g);
};

#endif