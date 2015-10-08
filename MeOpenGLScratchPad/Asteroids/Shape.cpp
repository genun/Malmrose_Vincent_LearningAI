#include "Shape.h"
#include <iostream>


void Shape::draw(Core::Graphics& g){
	const unsigned int NUM_LINES = sizeof(points) / sizeof(*points);
	for(unsigned int i = 0; i < NUM_LINES - 1; i++){
		const Vector2d& p1 = position + points[i];
		const Vector2d& p2 = position + points[(i+1) % NUM_LINES];
		g.DrawLine(p1.x, p1.y, p2.x, p2.y);
	}
}
