#version 400

in vec2 daUV;

uniform sampler2D textured;

out vec3 FinalColor;


void main()
{
	FinalColor = vec3(texture(textured, daUV));
}