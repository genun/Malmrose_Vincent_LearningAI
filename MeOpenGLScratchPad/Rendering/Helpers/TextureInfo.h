#ifndef TEXTURE_INFO
#define TEXTURE_INFO

class TextureInfo{
	GLint uniformLocation;
	GLuint textureID;
	bool isAvailable;

	friend class Renderer;
public:
	TextureInfo(): isAvailable(true), uniformLocation(0){}
};

#endif