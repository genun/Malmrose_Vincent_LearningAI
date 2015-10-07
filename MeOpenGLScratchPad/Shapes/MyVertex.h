#ifndef MY_VERTEX
#define MY_VERTEX
#include "TypeDefs.h"
#include "ExportImportHeader.h"
#include "glm\glm.hpp"
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

struct MyVertex
{
	glm::vec3 position;
	glm::vec4 color;
	glm::vec3 normal;
	glm::vec2 uv;
	glm::vec3 tangent;
	static const uint STRIDE = 15 * sizeof(float);
};


#endif