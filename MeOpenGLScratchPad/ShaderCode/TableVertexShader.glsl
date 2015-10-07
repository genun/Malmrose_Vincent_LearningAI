#version 400

in layout(location=0) vec3 position;
in layout(location=1) vec3 color;
in layout(location=2) vec3 normal;
in layout(location=3) vec2 UV;

uniform mat4 whereMat;
uniform mat4 projectionMatrix;
uniform mat4 cameraMatrix;

out vec2 daUV;
out mat3 where;
out vec3 fragPosition;
out vec3 normals;

void main()
{
	vec4 v = vec4(position, 1.0);
	vec4 newPosition = whereMat * v;
	vec4 addCameraTransformMatrix = cameraMatrix * newPosition;
	gl_Position = projectionMatrix * addCameraTransformMatrix;
	
	daUV = UV;
	where = mat3(whereMat);
	fragPosition = vec3(newPosition);
	normals = normalize(vec3(whereMat * vec4(normal, 0.0f)));
}
