#include "ParticleManager.h"

void ParticleManager::add(ParticleEffects eff){
	effects.push_back(eff);
}

void ParticleManager::update(float dt){
	for(unsigned int i = 0; i < effects.size(); ++i){
		effects[i].update(dt);
		if(effects[i].lifetime < 0){
			effects.erase(effects.begin() + i);
		}
	}
}

void ParticleManager::draw(Core::Graphics g){
	for(unsigned int i = 0; i < effects.size(); ++i){
		effects[i].draw(g);
	}
}

void ParticleManager(){
}