#ifndef CAMERA_H
#define CAMERA_H

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>

class Camera
{
	glm::vec3 position;
	glm::vec3 viewDirection;
	const glm::vec3 UP;
	glm::vec2 oldMousePosition;
public:
	Camera();
	glm::mat4 getWorldtoViewMatrix() const;
	void mouseUpdate(const glm::vec2& newMousePosition);
	void transUp(bool goUP);
	void transForward(bool forward);
	void strafe(bool left);
	glm::vec3 getPosition();
};

#endif