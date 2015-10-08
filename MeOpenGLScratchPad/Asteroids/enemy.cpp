#include "enemy.h"


	/*Vector2d pos;
	Vector2d* shipPos;
	Matrix3 rotation;
	Vector2d velocity;
	void init();
	void shutdown();
	void update();
	void draw();*/

void enemy::update(Vector2d shipPos){
	velocity = Vector2d(shipPos - pos).Normalized() * velocity.Length();
	pos = pos + velocity;
	
	Vector2d dir =  shipPos - pos;
	dir = dir.Normalized();
	Vector2d otherDir = dir.PerpCCW();
	rotation.mat[0][0] = -otherDir.x;
	rotation.mat[1][0] = -otherDir.y;
	rotation.mat[0][1] = -dir.x;
	rotation.mat[1][1] = -dir.y;

	if (timeFlying > 8.0f){
		Delete(enemy());
	}
}

void enemy::draw(Core::Graphics g){
	g.SetColor(RGB(0, 255, 100));
	const unsigned int NUM_LINES = sizeof(enemNode) / sizeof(*enemNode);
	for(unsigned int i = 0; i < NUM_LINES - 1; i++){
		const Vector2d& p1 = rotation * enemNode[i] + pos;
		const Vector2d& p2 = rotation * enemNode[(i+1) % NUM_LINES] + pos;
		g.DrawLine(p1.x, p1.y, p2.x, p2.y);
	}
	g.SetColor(RGB(255, 255, 255));
}

void enemy::init(){
	velocity = Vector2d(0, 3.0f);
	float start = rand.randomInRange(0, 3);
	if(start < 1){
		pos = Vector2d(rand.randomInRange(-30, -10), rand.randomInRange(0,799));
	}
	else if(start > 1 && start <= 2){
		pos = Vector2d(rand.randomInRange(0, 799), rand.randomInRange(-40, -10));
	}
	else if(start > 2 && start <= 3){
		pos = Vector2d(rand.randomInRange(810, 840), rand.randomInRange(0,799));
	}
	else if(start > 3 && start <= 4){
		pos = Vector2d(rand.randomInRange(0, 799), rand.randomInRange(810, 840));
	}
	
	enemNode[0] = Vector2d(-4, 0);
	enemNode[1] = Vector2d(-7, 4);
	enemNode[2] = Vector2d(-10, 0);
	enemNode[3] = Vector2d(-7, -4);
	enemNode[4] = Vector2d(-4, 0);
	
	enemNode[5] = Vector2d(-4, -4);
	enemNode[6] = Vector2d(4, -4);
	
	enemNode[7] = Vector2d(4, 0);
	enemNode[8] = Vector2d(7, 4);
	enemNode[9] = Vector2d(10, 0);
	enemNode[10] = Vector2d(7, -4);
	enemNode[11] = Vector2d(4, 0);
	
	enemNode[12] = Vector2d(4, 4);
	enemNode[13] = Vector2d(-4, 4);
	enemNode[14] = Vector2d(-4, 0);
}

void enemy::shutdown(){
}


enemy::enemy(){
}

enemy::~enemy(){
	
}
