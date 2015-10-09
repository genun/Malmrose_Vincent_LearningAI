#pragma once

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>
#include <Rendering\Renderer.h>

class Ball
{
public:
	Renderable* img;
	glm::vec3 pos;
	glm::vec3 vel;
	void Update();
	void Init(glm::vec3 position, glm::vec3 velocity);
	void Bounce();
	void Collide(glm::vec3 OtherPos, glm::vec3 width, 
		glm::vec3 height, Renderable ballImage);
	void CheckWallCollision();
	void CheckOtherCollision();

	Ball(void);
	~Ball(void);
};

