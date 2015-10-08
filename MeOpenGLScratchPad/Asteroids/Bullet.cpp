#include "Bullet.h"

bool alive;
float time;

void bullet::init(Vector2d position, Matrix3 rotate, Vector2d speed){
	if(!alive){
		bullet::points[0] = Vector2d(2, 4);	
		bullet::points[1] = Vector2d(-2, 4);	
		bullet::points[2] = Vector2d(-2, -4);	
		bullet::points[3] = Vector2d(2, -4);	
		bullet::points[4] = Vector2d(2, 4);	
		alive = true;
		time = 0.7f;
		pos = position;
		rot = rotate;
		s = speed.Normalized();
	}
}

void bullet::update(float dt){
	if(alive){
		Vector2d speed = rot * Vector2d(0, -12.5f);
		pos = pos + speed;
		time -= dt;
		if(time < 0){
			alive = false;
			pos.x = -500;
			pos.y = -500;
		}
	}
}

void bullet::draw(Core::Graphics& g){
	if(alive == true){
		rot.mat[0][2] = pos.x;
		rot.mat[1][2] = pos.y;
		const unsigned int NUM_LINES = sizeof(points) / sizeof(*points);
		for(unsigned int i = 0; i < NUM_LINES - 1; i++){
			const Vector2d& p1 = rot * (points[i]);
			const Vector2d& p2 = rot * (points[(i+1) % NUM_LINES]);
			g.DrawLine(p1.x, p1.y, p2.x, p2.y);
		}
		rot.mat[0][2] = 0;
		rot.mat[1][2] = 0;
	}
}
