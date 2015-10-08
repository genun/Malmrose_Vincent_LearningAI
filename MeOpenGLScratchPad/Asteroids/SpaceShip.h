#ifndef SPACESHIP_H
#define SPACESHIP_H
#include "Shape.h"
#include "Core.h"
#include "Matrix3.h"

struct SpaceShip{
	SpaceShip();
	int typeChange;
	float rot;
	Vector2d position;
	Vector2d points[6];
	Vector2d velocity;
	Vector2d Speed;
	Matrix3 rotation, translation, currentTrans;
	void update(float dt, bool& drawBorder, float width, float height);
	void wallBounce();
	void bounce(Vector2d wall);
	void SpaceShip::draw(Core::Graphics& g);


};

#endif

