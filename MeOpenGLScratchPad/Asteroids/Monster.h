#ifndef MONSTER_H
#define MONSTER_H
#include "Shape.h"
#include "Vector2d.h"

struct Monster: public Shape{
	float speed;
	int node;
	Vector2d Lposition;
	void update(float dt);
	void draw(Core::Graphics&);
	Monster();
};

#endif