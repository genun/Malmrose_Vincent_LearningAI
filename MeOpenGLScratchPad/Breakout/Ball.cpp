#include "Ball.h"
#include <qt\qdebug.h>
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
	if (CheckCollisionPosition.x > 10.5 || CheckCollisionPosition.x < -10.5){
		vel.x = -vel.x;
	}
	if (/*CheckCollisionPosition.y < -8 ||*/ CheckCollisionPosition.y > 8.2){
		vel.y = -vel.y;
	}
}

void Ball::Bounce(){

}

void Ball::Collide(glm::vec3 OtherPos, float width, float height){
	//std::cout << "Collided" << std::endl;


	if (pos.y <= OtherPos.y - (height / 2)){
		qDebug() << "Bounce Y";
		vel.y = vel.y * -1;
	}

	else if (pos.y >= OtherPos.y + (height / 2)){
		qDebug() << "Bounce Y";
		vel.y = vel.y * -1;
	}

	else if (pos.x < OtherPos.x){
		qDebug() << "Bounce X";
		vel.x = vel.x * -1;
	}

	else if(pos.x > OtherPos.x){
		qDebug() << "Bounce X";
		vel.x = vel.x * -1;
	}

/*
	float x = pos.x;
	if (x > 0) x *= -1;
	if (OtherPos.x > 0) OtherPos.x *= -1;
	float percentXLeft = (x - rad) - (OtherPos.x - width);
	float percentXRigh = (x + rad) - (OtherPos.x + width);
	float percentYTop = (pos.y - rad) - (OtherPos.y - height);
	float percentYBot = (pos.y + rad) - (OtherPos.y + height);

	float xPercent = (percentXLeft > percentXRigh) ? percentXLeft : percentXRigh;
	float yPercent = (percentYTop > percentYBot ) ? percentYTop : percentYBot;

	if (xPercent > yPercent){
		qDebug() << "Bounce Y";
		vel.y = vel.y * -1;
	}
	else {
		qDebug() << "Bounce X";
		vel.x = vel.x * -1;
	}*/
}

void Ball::PaddleCollide(){
	vel.y = vel.y * -1;
}
