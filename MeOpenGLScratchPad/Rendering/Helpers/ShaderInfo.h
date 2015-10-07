#ifndef SHADER_INFO
#define SHADER_INFO

class ShaderInfo{
	bool isAvailable;
	GLuint progID;
	friend class Renderer;
public:
	ShaderInfo(): isAvailable(true), progID(-1) {}
};


#endif