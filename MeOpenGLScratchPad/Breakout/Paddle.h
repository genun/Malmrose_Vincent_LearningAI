#pragma once
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>
#include <Rendering\Renderer.h>

class Paddle
{
public:
	Renderable* img;

	glm::vec3 pos;
	float speed;

	void Update();
	void Init(glm::vec3 position, float newSpeed, Renderable* paddleImage);

	Paddle(void);
	~Paddle(void);
};

