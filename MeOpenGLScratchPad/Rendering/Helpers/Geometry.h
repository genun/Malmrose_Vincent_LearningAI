#ifndef ENGINE_GEOMETRY_H
#define ENGINE_GEOMETRY_H

#include <GL/glew.h>
class VertexLayoutInfo;

class Geometry{
	GLuint vertexAttribID;
	
	GLuint indexBufferID;
	GLuint indexByteOffset;
	GLuint numIndices;

	GLuint indexingMode;

	VertexLayoutInfo* vertexLayoutInfo;

	bool isAvailable;

	friend class Renderer;
	void addVertexBuff(GLuint id, GLuint offset);
	void addIndexBuff(GLuint id, GLuint offset);

public:
	
	Geometry(): isAvailable(true),
		indexBufferID(-1),
		indexByteOffset(-1),
		vertexAttribID(-1),
		vertexLayoutInfo(0)
	{}
};

#endif
