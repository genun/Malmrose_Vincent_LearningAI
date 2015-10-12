#pragma once
#include "Breakout\Ball.h"
#include "Breakout\Brick.h"
#include "Breakout\Paddle.h"


class BreakoutManager
{
public:

	static const int brickLineWidth = 1;
	static const int brickLineHeight = 1;
	Ball* ball;
	Brick* bricks[brickLineWidth][brickLineHeight];
	Paddle* paddle;

	void Update(Paddle::inputType input);
	void checkCollision();
	bool Collide(glm::vec3 pos, float width, float height);

	BreakoutManager();
	~BreakoutManager();
};


