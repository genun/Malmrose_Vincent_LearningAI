#include "Ball.h"
#include <iostream>


Ball::Ball(void)
{
}


Ball::~Ball(void)
{
}

void Ball::Init(glm::vec3 position, glm::vec3 velocity, float radius){
	pos = position;
	vel = velocity;
	rad = radius;
}

void Ball::Update(){
	CheckWallCollision();
	//KeyboardInput();

	//No DT currently. Assuming 30fps.
	pos = pos + vel * (1.0f / 30.0f);

	//Update the renderable.
	img->whereMatrix = glm::translate(pos) * glm::scale(glm::vec3(0.25f));
}

void Ball::KeyboardInput(){
	if (GetAsyncKeyState(VK_UP))
		vel = glm::vec3(0.0f, 1.0f, 0.0f);
	else if (GetAsyncKeyState(VK_DOWN))
		vel = glm::vec3(0.0f, -1.0f, 0.0f);
	else if (GetAsyncKeyState(VK_LEFT))
		vel = glm::vec3(-1.0f, 0.0f, 0.0f);
	else if (GetAsyncKeyState(VK_RIGHT))
		vel = glm::vec3(1.0f, 0.0f, 0.0f);
	else
		vel = glm::vec3(0.0f);
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

void Ball::Collide(glm::vec3 OtherPos, float width, float height){
	//std::cout << "Collided" << std::endl;
	float percentXLeft = (pos.x - rad)/ (OtherPos.x - width);
	float percentXRight = (pos.x + rad) / (OtherPos.x - width);;
	float percentYTop = (pos.y - rad) / (OtherPos.y - height);;
	float percentYBot = (pos.y + rad) / (OtherPos.y - height);;

	float xPercent = (percentXLeft > percentXRight) ? percentXLeft : percentXRight;
	float yPercent = (percentYTop > percentYBot ) ? percentYTop : percentYBot;

	if (xPercent > yPercent) vel.x = vel.x * -1;
	else vel.y = vel.y * -1;
}

void Ball::PaddleCollide(){
	vel.y = vel.y * -1;
}
