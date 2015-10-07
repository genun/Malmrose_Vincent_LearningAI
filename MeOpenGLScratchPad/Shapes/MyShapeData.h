#ifndef MY_SHAPE_DATA
#define MY_SHAPE_DATA
#include <Shapes\MyVertex.h>

struct MyShapeData
{
	MyVertex* verts;
	uint numVerts;
	ushort* indices;
	uint numIndices;
	MyShapeData() : verts(0), numVerts(0), indices(0), numIndices(0) {}
	uint vertexBufferSize() const { return numVerts * sizeof(MyVertex); }
	uint indexBufferSize() const { return numIndices * sizeof(ushort); }
	void cleanUp()
	{
		delete[] verts;
		verts = 0;
		delete[] indices;
		indices = 0;
		numVerts = numIndices = 0;
	}
};

#endif