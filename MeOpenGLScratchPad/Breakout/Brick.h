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
	float width, height;
	bool destroyed;

	void GetHit();
	void Init(glm::vec3 position, Renderable* brickImage, float w, float h);

	Brick(void);
	~Brick(void);
};

