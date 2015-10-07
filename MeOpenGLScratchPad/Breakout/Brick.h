#pragma once
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>
#include <Rendering\Renderer.h>

class Brick
{
public:
	Renderable* img;

	glm::vec3 pos;

	void GetHit();
	void Init(glm::vec3 position, Renderable* brickImage);

	Brick(void);
	~Brick(void);
};

