#ifndef BULLET_H
#define BULLET_H

#include "Vector2d.h"
#include "Matrix3.h"
#include "Core.h"

struct bullet{
	bool alive;
	float time;
	Vector2d points[5];
	Vector2d pos, s;
	Matrix3 rot;
	void init(Vector2d position, Matrix3 rotate, Vector2d speed);
	void update(float dt);
	void draw(Core::Graphics& g);
};

#endif