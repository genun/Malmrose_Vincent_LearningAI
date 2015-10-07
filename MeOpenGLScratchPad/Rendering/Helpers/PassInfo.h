#ifndef ENGINE_PASS_INFO_H
#define ENGINE_PASS_INFO_H

#include <Camera\Camera.h>

class Renderable;

class PassInfo{

	static const GLuint MAX_RENDERABLES = 10;
	Renderable* renderables[MAX_RENDERABLES];
	GLuint numRenderables;
	bool isAvailable;
	friend class Renderer;

public:
	PassInfo(): numRenderables(0), isAvailable(true){}

	inline void addRenderable(Renderable* renderable);

};

void PassInfo::addRenderable(Renderable* renderable){
	//assert(numRenderables < MAX_RENDERABLES);
	renderables[numRenderables++]= renderable;
};

#endif
