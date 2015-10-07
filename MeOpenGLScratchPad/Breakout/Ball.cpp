#include "Ball.h"


Ball::Ball(void)
{
}


Ball::~Ball(void)
{
}

void Ball::Init(glm::vec3 position, glm::vec3 velocity){
	pos = position;
	vel = velocity;
}

void Ball::Update(){
	glm::vec3 newPos = pos + vel * (1.0f / 30.0f);
	if (newPos.x > 4.5 || newPos.x < -4.5){
		vel.x = -vel.x;
	}
	if( newPos.y < -4 || newPos.y > 3.2){
		vel.y = -vel.y;
	}
	pos = pos + vel * (1.0f / 30.0f);

	img->whereMatrix = glm::translate(pos) * glm::scale(glm::vec3(0.25f));
}

void Ball::Bounce(){

}

void Ball::Collide(glm::vec3 OtherPos, glm::vec3 width, glm::vec3 height, Renderable ballImage){

}
