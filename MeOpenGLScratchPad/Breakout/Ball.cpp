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
	CheckWallCollision();
	CheckOtherCollision();

	//No DT currently. Assuming 30fps.
	pos = pos + vel * (1.0f / 30.0f);

	//Update the renderable.
	img->whereMatrix = glm::translate(pos) * glm::scale(glm::vec3(0.25f));
}

void Ball::CheckOtherCollision(){

}

void Ball::CheckWallCollision(){
	glm::vec3 CheckCollisionPosition = pos + vel * (1.0f / 30.0f);
	if (CheckCollisionPosition.x > 4.5 || CheckCollisionPosition.x < -4.5){
		vel.x = -vel.x;
	}
	if (CheckCollisionPosition.y < -4 || CheckCollisionPosition.y > 3.2){
		vel.y = -vel.y;
	}
}

void Ball::Bounce(){

}

void Ball::Collide(glm::vec3 OtherPos, glm::vec3 width, glm::vec3 height, Renderable ballImage){

}
