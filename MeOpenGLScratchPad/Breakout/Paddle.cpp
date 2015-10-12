#include "Paddle.h"
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>


Paddle::Paddle(void)
{
}


Paddle::~Paddle(void)
{
}

void Paddle::Update(inputType input){
	if (input == inputType::LEFT){
		pos = pos + glm::vec3(speed, 0, 0) * 0.1f;
	}
	else if (input == inputType::RIGHT){
		pos = pos - glm::vec3(speed, 0, 0) * 0.1f;
	}
	img->whereMatrix = glm::translate(pos) * scale;
}

void Paddle::Init(glm::vec3 position, float newSpeed, Renderable* paddleImage, glm::mat4 paddleScale, float w, float h){
	pos = position;
	speed = newSpeed;
	img = paddleImage;
	scale = paddleScale;
	width = w;
	height = h;
}
