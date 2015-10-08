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
	float timeFlying = 0.0f;
	float lifeTime = 6.0f;

	float minLongspeed = 3.0f;
	float maxLongspeed = 5.0f;
	float angleSpeed = 3.0f;

	void init();
	void shutdown();
	void update(Vector2d shipPos, float dt);
	void draw(Core::Graphics g);
	enemy();
	~enemy();
};

#endif
