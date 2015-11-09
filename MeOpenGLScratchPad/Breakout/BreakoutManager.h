#pragma once
#include "Breakout\Ball.h"
#include "Breakout\Brick.h"
#include "Breakout\Paddle.h"
#include "AI\DeepLearner.h"


class BreakoutManager
{
public:
	static const int brickLineWidth = 12;
	static const int brickLineHeight = 4;

	DeepLearner ai;
	Ball* ball;
	Brick* bricks[brickLineWidth][brickLineHeight];
	Paddle* paddle;

	int score = 0;
	bool* cont;
	bool* win;


	void Update();
	void checkCollision();
	bool Collide(glm::vec3 pos, float width, float height);
	void Fail();
	void WinGame();
	void init();
	void setAI(DeepLearner &newAI, int* width, int* height);
	void setAI(DeepLearner& newAI);

	BreakoutManager();
	~BreakoutManager();
};
