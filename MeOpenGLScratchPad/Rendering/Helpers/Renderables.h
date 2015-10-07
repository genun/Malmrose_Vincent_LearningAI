#ifndef ENGINE_RENDERABLES
#define ENGINE_RENDERABLES

#include <glm/glm.hpp>
#include <Rendering\Helpers\UniformInfo.h>
class Geometry;
class ShaderInfo;
class BufferInfo;
class TextureInfo;
/*
	This is the what where and how all together.
	we need geometry, shader, and uniform what where how
*/

class Renderable{
	UniformInfo* uniformInfo;
	Geometry* geo;
	GLuint progID;

	void*  uniformData[10];

	GLuint whereMatrixUniformLocation;

	TextureInfo* texture;

	bool hasTexture;
	bool isAvailable;
	friend class Renderer;

public:
	glm::mat4 whereMatrix;

	bool isVisible;
	Renderable(): isAvailable(true), hasTexture(false), isVisible(true){}
};

#endif