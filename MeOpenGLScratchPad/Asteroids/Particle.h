#ifndef PARTICLE_H
#define PARTICLE_H
#include "Vector2d.h"
#include "Core.h"
#include "Random.h"
using Core::RGB;

struct Particle
{
	RGB color;
	Vector2d position, velocity;
	void update(float dt);
	void draw(Core::Graphics& g);
	Particle(void);
	Particle(RGB col, Vector2d pos, Vector2d vel);
	~Particle(void);
};  

#endif // !PARTICLE_H


