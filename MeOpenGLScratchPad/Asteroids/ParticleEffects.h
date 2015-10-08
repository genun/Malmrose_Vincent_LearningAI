#ifndef PARTICLE_EFFECT_H
#define PARTICLE_EFFECT_H

#include "Particle.h"
#include "Core.h"
#include "SpaceShip.h"
#include "enemy.h"

struct ParticleEffects
{
	float lifetime;
	Particle parts[10000];
	void update(float dt);
	void draw(Core::Graphics g);
	ParticleEffects(RGB col, SpaceShip ship);
	ParticleEffects(RGB col, enemy en);
	ParticleEffects(void);
	~ParticleEffects(void);
};


#endif // !PARTICLE_EFFECT_H
