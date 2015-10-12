#pragma once
#include <glm\glm.hpp>
//#include <glm\gtc\matrix_transform.hpp>
//#include <glm\gtx\transform.hpp>
//#include <Breakout\BreakoutManager.h>
#include <Rendering\Renderer.h>

class Paddle
{
public:

	enum inputType{
		LEFT,
		RIGHT,
		NONE
	};

	Renderable* img;

	glm::mat4 scale;
	glm::vec3 pos;
	float speed, width, height;

	void Update(inputType input);
	void Init(glm::vec3 position, float newSpeed, Renderable* paddleImage, glm::mat4 paddleScalem, float w, float h);

	Paddle(void);
	~Paddle(void);
};

