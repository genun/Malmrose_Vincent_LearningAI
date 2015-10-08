#ifndef PARTICLEMANAGER_H
#define PARTICLEMANAGER_H

#include "ParticleEffects.h"
#include "Particle.h"
#include "Core.h"
#include <vector>

using namespace std;

struct ParticleManager
{
	vector<ParticleEffects> effects;
	void add(ParticleEffects eff);
	void update(float dt);
	void draw(Core::Graphics g);
};

#endif