#ifndef VERTEX_LAYOUT_INFO
#define VETERX_LAYOUT_INFO


class VertexLayoutInfo{

	//Can use this for offset position as well as size of the attrib thing itself.
	static const GLuint MAX_VERTEX_ATTRIBUTES = 5;
	GLuint attributeSizes[MAX_VERTEX_ATTRIBUTES];

	GLuint stride;
	GLuint numAttributes;
	bool isAvailable;
	friend class Renderer;

public:
	VertexLayoutInfo() : stride(0), numAttributes(0), isAvailable(true){}
};

#endif
