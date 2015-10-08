#include "Particle.h"


void Particle::update(float dt){
	position = position + velocity * dt;
}

void Particle::draw(Core::Graphics& g){
	g.SetColor(color);
	float offSet = 2;
	Vector2d p1 = position + Vector2d(0, offSet);
	Vector2d p2 = position + Vector2d(offSet, 0);
	Vector2d p3 = position + Vector2d(0, -offSet);
	Vector2d p4 = position + Vector2d(-offSet, 0);
	
	g.DrawLine(position.x, position.y, p1.x, p1.y);
	g.DrawLine(position.x, position.y, p2.x, p2.y);
	g.DrawLine(position.x, position.y, p3.x, p3.y);
	g.DrawLine(position.x, position.y, p4.x, p4.y);
}

Particle::Particle(RGB col, Vector2d pos, Vector2d vel)
{
	color = col;
	position = pos;
	velocity = vel;
	velocity = velocity * 25;
}

Particle::Particle(void)
{
	Random randi;
	randi = Random();
	color = RGB(0, 255, 255);
	position = Vector2d(400, 400);
	velocity = Vector2d(randi.randomUnitVector() * randi.randomInRange(0, 5));
	velocity = velocity * 15;
}

Particle::~Particle(void)
{
}
