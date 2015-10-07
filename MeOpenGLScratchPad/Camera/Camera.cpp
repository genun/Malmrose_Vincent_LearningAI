#include <Camera\Camera.h>

Camera::Camera() : viewDirection(0.0f, 0.0f, -1.0f),
UP(0.0f, 1.0f, 0.0f)
{

}

glm::mat4 Camera::getWorldtoViewMatrix() const{
	return glm::lookAt(position, position + viewDirection, UP);
}

void Camera::transUp(bool goUP){
	if(goUP)
		position.y = position.y + 0.05f;
	else
		position.y = position.y - 0.05f;

}

void Camera::mouseUpdate(const glm::vec2& newMousePosition){
	glm::vec2 mouseDelta = newMousePosition - oldMousePosition;
	if(glm::length(mouseDelta) > 50.0f){
		oldMousePosition = newMousePosition;
		return;
	}
	const float rotationalSpeed = -0.20f;
	glm::vec3 toRotateAround = glm::cross(viewDirection, UP);

	viewDirection = glm::mat3(
		glm::rotate(mouseDelta.x * rotationalSpeed, UP) * 
		glm::rotate(mouseDelta.y * rotationalSpeed, toRotateAround)) * viewDirection;

	viewDirection = glm::mat3() * viewDirection;

	oldMousePosition = newMousePosition;
}

void Camera::transForward(bool forward){
	if(forward)
		position = position + (glm::normalize(viewDirection) * 0.05f);
	else
		position = position - (glm::normalize(viewDirection) * 0.05f);
}

void Camera::strafe(bool left){
	if(left)
		position = position - (glm::cross(glm::normalize(viewDirection), UP) * 0.05f);
	else
		position = position + (glm::cross(glm::normalize(viewDirection), UP) * 0.05f);
}

glm::vec3 Camera::getPosition(){
	return position;
}
