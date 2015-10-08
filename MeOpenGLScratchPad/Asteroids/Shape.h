#pragma once
#ifndef SHAPE_H
#define SHAPE_H
#include "Vector2d.h"
#include "Core.h"

struct Shape{
	Vector2d position;
	Vector2d points[6];
	void draw(Core::Graphics&);
};

#endif

