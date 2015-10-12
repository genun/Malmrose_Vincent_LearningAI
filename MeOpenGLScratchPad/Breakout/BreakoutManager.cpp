#include <iostream>
#include "BreakoutManager.h"

void BreakoutManager::Update(Paddle::inputType input){
	checkCollision();
	ball->Update();
	paddle->Update(input);
}

BreakoutManager::BreakoutManager()
{
}


BreakoutManager::~BreakoutManager()
{
}

void BreakoutManager::checkCollision(){
	for (int i = 0; i < brickLineWidth; ++i){
		for (int j = 0; j < brickLineHeight; ++j){
			if (bricks[i][j]->destroyed); //Bricks gone, do nothing
			else if (Collide(bricks[i][j]->pos, bricks[i][j]->width, bricks[i][j]->height)){
				ball->Collide(bricks[i][j]->pos, bricks[i][j]->width, bricks[i][j]->height);
			}
		}
	}
	if (Collide(paddle->pos, paddle->width, paddle->height)) ball->Collide(paddle->pos, paddle->width, paddle->height);
}

bool BreakoutManager::Collide(glm::vec3 pos, float width, float height){
	float bx = ball->pos.x;
	float by = ball->pos.y;
	float r = 0.5f;// ball->rad;


	float left1, left2;
	float right1, right2;
	float top1, top2;
	float bottom1, bottom2;

	left1 = bx - r;
	left2 = pos.x - width;
	right1 = bx + r;
	right2 = pos.x + width;
	top1 = by + r;
	top2 = pos.y + height;
	bottom1 = by - r;
	bottom2 = pos.y - height;

	if (bottom1 < top2 && top1 > bottom2 && right1 > left2 && left1 < right2) {
		std::cout << "Collided" << std::endl;
		return(true);
	}
	std::cout << "____" << std::endl;

	return(false);
}