#ifndef BUFFER_INFO
#define BUFFER_INFO
#include <GL/glew.h>

class BufferInfo{

public:

	int sizeRemaining;
	GLuint glBufferID;
	GLsizeiptr nextAvailableByte;

	
	//size of a meg in hex
	static const GLuint MAX_BUFFER_SIZE = 0x10000000;
	bool hasBuffer;

	//May remove or add back in later.
	//void genBuffer();

	BufferInfo():
		hasBuffer(false),
		sizeRemaining(-1)
	{}

	GLuint getRemainingSize() const { return(MAX_BUFFER_SIZE - nextAvailableByte);};
	
};

#endif