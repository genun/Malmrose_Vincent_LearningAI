#ifndef UNIFORM_LOCATION_INFO
#define UNIFORM_LOCATION_INFO

#include <gl\glew.h>


enum UniformType{
	VEC2,
	VEC3,
	VEC4,
	MAT3,
	MAT4,
	TEXTURE,
	MY_BOOLEAN
};

class UniformInfo{
	static const GLuint MAX_UNIFORM_LOCATIONS = 5;
	GLuint locations[MAX_UNIFORM_LOCATIONS];
	UniformType uniformType[MAX_UNIFORM_LOCATIONS];
	GLuint numLocations;
	bool isAvailable;
	friend class Renderable;
	friend class Renderer;
public:
	UniformInfo(): numLocations(0), isAvailable(true){}
};

#endif