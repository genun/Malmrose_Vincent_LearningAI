#version 330

in layout(location = 0) vec3 position;
in layout(location = 1) vec3 color;

uniform mat4 whereMat;
uniform mat4 projectionMatrix;
uniform mat4 cameraMatrix;

out vec3 vertColor;

void main(){

	vec4 v = vec4(position, 1.0);
	vec4 newPosition = whereMat * v;
	vec4 addCameraTransformMatrix = cameraMatrix * newPosition;
	gl_Position = projectionMatrix * addCameraTransformMatrix;
	vertColor = color;

}
