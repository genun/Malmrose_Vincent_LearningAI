#include "Monster.h"



Monster::Monster(){
	speed = 0.07f;
	node = 0;
}

Vector2d myNode[5] = {
	Vector2d(100.0f, 100.0f),
	Vector2d(400.0f, 100.0f),
	Vector2d(100.0f, 600.0f),
	Vector2d(500.0f, 500.0f),
	Vector2d(500.0f, 700.0f)
};

Vector2d mPoints[4] = {
	Vector2d(5, 5),
	Vector2d(-5, 5),
	Vector2d(-5, -5),
	Vector2d(5, -5)
};

void Monster::update(float dt){
	dt;
	float scale;
	bool x = false, y = false;
	if(node != 4){
		Vector2d& n1 = myNode[node];
		Vector2d& n2 = myNode[node + 1];
		scale = (Monster::speed / (n1 - n2).Length()) * 50;
		Vector2d LERP = n1.LERP(scale, n2);
		Lposition = Lposition + (LERP - n1);
		if(Lposition.x >= n2.x * 0.9 && Lposition.x <= n2.x * 1.1){
			x = true;
		}
		if(Lposition.y >=n2.y * 0.9 && Lposition.y <=n2.y * 1.1){
			y = true;
		}
		if(x && y){
			node++;
		}
	}
	else{
		Vector2d& n1 = myNode[4];
		Vector2d& n2 = myNode[0];
		scale = Monster::speed / ((n1 - n2).Length()) *50;
		Vector2d LERP = n1.LERP(scale, n2);
		Lposition = Lposition + (LERP - n1);
		if(Lposition.x >= n2.x * 0.9 && Lposition.x <= n2.x * 1.1){
			x = true;
		}
		if(Lposition.y >=n2.y * 0.9 && Lposition.y <=n2.y * 1.1){
			y = true;
		}
		if(x && y){
			node = 0;
		}
	}
	if(node == 5){
		node = 0;
	}
}

void Monster::draw(Core::Graphics& g){
	const unsigned int NUM_LINES = sizeof(mPoints) / sizeof(*mPoints);
	for(unsigned int i = 0; i < NUM_LINES - 1; i++){
		const Vector2d& mp1 = Lposition + mPoints[i];
		const Vector2d& mp2 = Lposition + mPoints[(i+1) % NUM_LINES];
		g.DrawLine(mp1.x, mp1.y, mp2.x, mp2.y);
	}
}
