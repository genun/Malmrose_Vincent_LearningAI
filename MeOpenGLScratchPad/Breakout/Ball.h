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
	float rad;

	void Update();
	void Init(glm::vec3 position, glm::vec3 velocity, float radius);
	void Bounce();
	void Collide(glm::vec3 OtherPos, float width, 
		float height);
	void CheckWallCollision();
	void CheckOtherCollision();
	void KeyboardInput();

	Ball(void);
	~Ball(void);
};

