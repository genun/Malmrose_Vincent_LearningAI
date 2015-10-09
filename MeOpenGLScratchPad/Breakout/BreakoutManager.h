#pragma once
#include "Breakout\Ball.h"
#include "Breakout\Brick.h"
#include "Breakout\Paddle.h"


class BreakoutManager
{
public:

	//enum inputType{
	//	LEFT,
	//	RIGHT
	//};

	static const int brickLineWidth = 8;
	static const int brickLineHeight = 4;
	Ball* ball;
	Brick* bricks[brickLineWidth][brickLineHeight];
	//Paddle* paddle;

	void Update(/*inputType input*/);

	BreakoutManager();
	~BreakoutManager();
};


