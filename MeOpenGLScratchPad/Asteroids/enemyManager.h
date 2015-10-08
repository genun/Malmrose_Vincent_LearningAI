#ifndef ENEMYMANGER_H
#define ENEMYMANGER_H

#include "enemy.h"
#include "Core.h"
#include "Bullet.h"
#include "ParticleManager.h"
#include "Random.h"
#include "SpaceShip.h"
#include <vector>

struct enemyManager{
	std::vector<enemy> enemies;
	float previous, current;
	void add(enemy en);
	void death(int i, ParticleManager& effect);
	void update(float dt, bullet bull, ParticleManager& effect, SpaceShip ship, int& score, int& hp);
	void draw(Core::Graphics g);
	void init();
};

#endif
