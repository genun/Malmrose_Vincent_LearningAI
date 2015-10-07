#ifndef MY_SHAPE_CREATOR
#define MY_SHAPE_CREATOR
#include <GL\glew.h>
#include <ShapeGenerator.h>
#include <Shapes\MyShapeData.h>

struct ShapeMaker{
	static const unsigned int CONE_HEIGHT = 10;
	static Neumont::ShapeData makeCylinder(GLuint tesselation);
	static Neumont::ShapeData makeCone(GLuint tesselation);
	static MyShapeData makeCube();
	static MyShapeData makeInverseCube();
	static MyShapeData copyToShapeData(MyVertex  verts[],
		int numVerts, unsigned short indices[],
		int numIndices);
};

#endif