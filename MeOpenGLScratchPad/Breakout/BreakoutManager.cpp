#include <iostream>
#include "BreakoutManager.h"

void BreakoutManager::Update(){
	if (ball->pos.y < -3.8) Fail();
	checkCollision();
	ball->Update();

	Paddle::inputType paddleInput = Paddle::inputType::NONE;

#pragma region AI input
	//Screen Grab
	std::vector<float*> emptyGrab;
	int input = ai.GetInput(emptyGrab);
	switch (input){
	case 0:
		paddleInput = Paddle::inputType::LEFT;
		break;
	case 1:
		paddleInput = Paddle::inputType::LEFT;
		break;
	case 2:
		paddleInput = Paddle::inputType::RIGHT;
		break;
	}
#pragma endregiondd

#pragma region Keyboard Input
	if (GetAsyncKeyState('A'))
		paddleInput = Paddle::inputType::LEFT;
	if (GetAsyncKeyState('D'))
		paddleInput = Paddle::inputType::RIGHT;
#pragma endregion

	paddle->Update(paddleInput);

	if (score == (brickLineHeight * brickLineWidth)) {
		WinGame();
	}
}

//No more failing for you
void BreakoutManager::Fail(){
	//*cont = false;
}

//Turn dat winning off.
void BreakoutManager::WinGame(){
	//*win = true;
}

BreakoutManager::BreakoutManager()
{
	ai.Initialize(0, 0, 0, 3, 0.1f);
}

BreakoutManager::~BreakoutManager()
{
	score = 0;
	//delete ball;
	//delete paddle;
	//delete bricks;
	//for (int i = 0; i < brickLineWidth; i++){
	//	for (int j = 0; j < brickLineHeight; j++){
	//	}
	//}
}

void BreakoutManager::checkCollision(){
	for (int i = 0; i < brickLineWidth; ++i){
		for (int j = 0; j < brickLineHeight; ++j){
			if (bricks[i][j]->destroyed); //Bricks gone, do nothing
			else if (Collide(bricks[i][j]->pos, bricks[i][j]->width, bricks[i][j]->height)){
				ball->Collide(bricks[i][j]->pos, bricks[i][j]->width, bricks[i][j]->height);
				bricks[i][j]->GetHit();
				score++;
			}
		}
	}
	if (Collide(paddle->pos, paddle->width, paddle->height)) ball->PaddleCollide();
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
		//std::cout << "Collided" << std::endl;
		return(true);
	}
	//std::cout << "____" << std::endl;

	return(false);
}