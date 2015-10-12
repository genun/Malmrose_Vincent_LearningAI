#include "Brick.h"


Brick::Brick(void)
{
}


Brick::~Brick(void)
{
}

void Brick::GetHit(){
	img->isVisible = false;
	destroyed = true;
}

void Brick::Init(glm::vec3 position, Renderable* brickImage, float w, float h){
	pos = position;
	img = brickImage;
	width = w;
	height = h;
	destroyed = false;
}
