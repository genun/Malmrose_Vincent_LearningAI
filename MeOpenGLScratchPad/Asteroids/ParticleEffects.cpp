#include "ParticleEffects.h"

const int numParticles = 100;

void ParticleEffects::update(float dt){
	lifetime -= dt;
	for(int i = 0; i < numParticles; i++){
		parts[i].update(dt);
	}
}

void ParticleEffects::draw(Core::Graphics g){
	for(int i = 0; i < numParticles; i++){
		parts[i].draw(g);
	}
}

/*
	Currently 1 stands for the explosion and any other is a Flamethrower using given data.
*/
ParticleEffects::ParticleEffects(RGB col, SpaceShip ship)
{
	Random randi;
	randi = Random();
	lifetime = 0.2f;
	for(int i = 0; i < numParticles; i++){
		parts[i] = Particle(col, ship.position, Vector2d(0, randi.randomInRange(0, 0)));
		parts[i].velocity = parts[i].velocity + ship.Speed;
		parts[i].velocity.x = parts[i].velocity.x + (randi.randomInRange(-0.5, 0.5) / 10000);
		parts[i].velocity = ship.rotation * parts[i].velocity;
		Matrix3 cone;
		cone.Rotation(randi.randomInRange(-0.25f, -0.75f));
		parts[i].velocity = cone * parts[i].velocity;
		parts[i].velocity = parts[i].velocity * -25;
		parts[i].position = parts[i].position + (ship.velocity * 3) ;
	}
}

ParticleEffects::ParticleEffects(RGB col, enemy en){
		lifetime = 2;
		for(int i = 0; i < numParticles; i++){
			parts[i] = Particle();
			parts[i].color = col;
			parts[i].position = en.pos;
		}
}


ParticleEffects::ParticleEffects(void){

}

ParticleEffects::~ParticleEffects(void)
{
}
