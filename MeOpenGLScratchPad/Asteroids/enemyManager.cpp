#include "enemyManager.h"


void enemyManager::add(enemy en){
	enemies.push_back(en);
}

void enemyManager::update(float dt, bullet bull, ParticleManager& effect, SpaceShip ship, int& score, int& hp){
	current += dt;
	if(current - previous > spawnRate){
		enemy enem;
		enem.init();
		add(enem);
		previous = 0;
		current = 0;
	}

	for(unsigned int i = 0; i < enemies.size(); ++i){
		enemies[i].update(ship.position, dt);
		if (enemies[i].timeFlying > enemies[i].lifeTime){
			death(i, effect);
		}
		if(abs(bull.pos.x - enemies[i].pos.x) <20 && abs(bull.pos.y - enemies[i].pos.y) < 20){
			death(i, effect);
			score ++;
		}
		else if(abs(ship.position.x - enemies[i].pos.x) <20 && abs(ship.position.y - enemies[i].pos.y) < 20){
			death(i, effect);
			hp --;
		}
	}
}

void enemyManager::death(int i, ParticleManager& effect){
	Random rand;
	effect.add(ParticleEffects(rand.randomColor(10, 100, 100, 75), enemies[i]));
	enemies[i].shutdown();
	enemies.erase(enemies.begin() + i);
}

void enemyManager::draw(Core::Graphics g){
	for(unsigned int i = 0; i < enemies.size(); ++i){
		enemies[i].draw(g);
	}
}

void enemyManager::init(){
	previous = 0;
	current = 0;
}
