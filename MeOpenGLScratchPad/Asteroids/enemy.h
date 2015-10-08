#ifndef ENEMY_H
#define ENEMY_H
#include "Vector2d.h"
#include "Core.h"
#include "Matrix3.h"
#include "Random.h"

struct enemy{
	Vector2d pos;
	Matrix3 rotation;
	Vector2d velocity;
	Random rand;
	Vector2d enemNode[15];
	void init();
	void shutdown();
	void update(Vector2d shipPos);
	void draw(Core::Graphics g);
	enemy();
	~enemy();
};

#endif
