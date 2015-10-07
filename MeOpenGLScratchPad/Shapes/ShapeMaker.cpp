#include <Shapes\ShapeMaker.h>
#include <Rendering\Renderer.h>
#include <glm\glm.hpp>
#include <glm\gtx\rotate_vector.hpp>
#include <Shapes\MyVertex.h>
#include <Shapes\MyShapeData.h>

using Neumont::ShapeData;
using glm::vec3;
using glm::vec4;
using glm::mat4;

ShapeData ShapeMaker::makeCone(uint tesselation)
{
	// Cone head
	vec3 coneTip(0.2f, 0.0f, 0.0f);
	float coneRadius = 0.175f;

	// Rotate thisVert around x axis
	vec4 thisVert(0, coneRadius, 0, 0);
	glm::mat4 rotator = glm::rotate(360.0f / tesselation, glm::vec3(1.0f, 0.0f, 0.0f));

	ShapeData ret;
	// 2 verts at same position around the rim so I can vary the normals
	// + 2 -> one for cone tip and one for cone base center
	ret.numVerts = 2 * tesselation + 2; 
	ret.verts = new Neumont::Vertex[ret.numVerts];

	uint coneTipIndex = tesselation * 2;
	uint coneBaseCenterIndex = coneTipIndex + 1;
	for(uint i = 0; i < tesselation; i++)
	{
		// Side facing triangle vert
		Neumont::Vertex& v = ret.verts[i];
		v.color = vec4(0.0f, 1.0f, 0.0f, 0.0f);
		v.position = vec3(thisVert);
		// Just want its position in the YZ plane, ignore x value;
		v.normal = glm::normalize(glm::vec3(0, v.position.y, v.position.z));
		thisVert = rotator * thisVert;
	}

	for(uint i = tesselation; i < tesselation * 2; i++)
	{
		// Cone bottom facing vert. Everything is same except the normal
		Neumont::Vertex& v = ret.verts[i - tesselation];
		Neumont::Vertex& v2 = ret.verts[i];
		v2.color = v.color;
		v2.position = v.position;
		v2.normal = glm::vec3(-1.0f, 0.0f, 0.0f);
	}

	ret.verts[coneTipIndex].position = coneTip;
	ret.verts[coneTipIndex].color = vec4(1.0f, 0.0f, 0.0f, 0.0f);
	ret.verts[coneTipIndex].normal = glm::vec3(1.0f, 0.0f, 0.0f);

	ret.verts[coneBaseCenterIndex].position = glm::vec3(0, 0.0f, 0.0f);
	ret.verts[coneBaseCenterIndex].color = vec4(0.0f, 0.0f, 1.0f, 0.0f);
	ret.verts[coneBaseCenterIndex].normal = glm::vec3(-1.0f, 0.0f, 0.0f);


	const uint NUM_VERTS_PER_TRI = 3;
	ret.numIndices = NUM_VERTS_PER_TRI * tesselation * 2;
	ret.indices = new ushort[ret.numIndices];

	uint indiceIndex = 0;
	for(uint i = 0; i < tesselation; i++)
	{
		// Side face
		ret.indices[indiceIndex++] = i;
		ret.indices[indiceIndex++] = (i + 1) % tesselation;
		ret.indices[indiceIndex++] = coneTipIndex;
	}
	for(uint i = tesselation; i < tesselation * 2; i++)
	{
		// Bottom face
		ret.indices[indiceIndex++] = i;
		ret.indices[indiceIndex++] = (i + 1) % tesselation;
		ret.indices[indiceIndex++] = coneBaseCenterIndex;
	}
	assert(indiceIndex == ret.numIndices);

	// Cynlindar stem
	return ret;
}

ShapeData ShapeMaker::makeCylinder(uint tesselation)
{
	ShapeData ret;
	ret.numVerts = tesselation * 2 + 2; // + 2 for top and bottom center
	ret.verts = new Neumont::Vertex[ret.numVerts];

	vec4 thisVert(0, 1, 0, 0);
	glm::mat4 rotator = glm::rotate(360.0f / tesselation, glm::vec3(1.0f, 0.0f, 0.0f));

	for(uint i = 0; i < tesselation; i++)
	{
		Neumont::Vertex& v0 = ret.verts[i];
		Neumont::Vertex& v1 = ret.verts[i + tesselation];

		v0.position = vec3(thisVert);
		v0.position.x = 1.0f;
		v1.position = vec3(thisVert);

		v0.color = vec4(1.0f, 1.0f, 0.0f, 1.0f);
		v1.color = vec4(0.0f, 1.0f, 1.0f, 1.0f);
		v0.normal.x = +1.0f;
		v1.normal.x = -1.0f;

		thisVert = rotator * thisVert;
	}
	uint topCenterVertIndex = ret.numVerts - 2;
	uint bottomCenterVertIndex = ret.numVerts - 1;
	//ret.verts[topCenterVertIndex].position.x = 1.0f;
	ret.verts[topCenterVertIndex].normal.x = +1.0f;
	ret.verts[topCenterVertIndex].color = vec4(1.0f, 0.0f, 1.0f, 1.0f);
	ret.verts[bottomCenterVertIndex].normal.x = -1.0f;
	ret.verts[bottomCenterVertIndex].color = vec4(1.0f, 0.0f, 1.0f, 1.0f);

	const uint NUM_TRIS_PER_SIDE_FACE = 2;
	const uint NUM_TRIS_PER_END_FACE = 1;
	const uint NUM_ENDS = 2;
	const uint NUM_VERTS_PER_TRI = 3;
	const uint NUM_SIDE_INDICES = tesselation * NUM_TRIS_PER_SIDE_FACE * NUM_VERTS_PER_TRI;
	const uint NUM_END_INDICES = tesselation * NUM_TRIS_PER_END_FACE * NUM_VERTS_PER_TRI * NUM_ENDS;
	ret.numIndices = NUM_SIDE_INDICES + NUM_END_INDICES;
	ret.indices = new ushort[ret.numIndices];

	uint indiceIndex = 0;
	for(uint i = 0; i < tesselation; i++)
	{
		// Top
		ret.indices[indiceIndex++] = i;
		ret.indices[indiceIndex++] = (i + 1) % tesselation;
		ret.indices[indiceIndex++] = topCenterVertIndex;
	} 
	for(uint i = 0; i < tesselation; i++)
	{
		// Bottom
		ret.indices[indiceIndex++] = i + tesselation;
		ret.indices[indiceIndex++] = (i + 1 < tesselation) ? i + 1 + tesselation : tesselation;
		ret.indices[indiceIndex++] = bottomCenterVertIndex;
	}

	// Side
	for(uint i = 0; i < tesselation; i++)
	{
		// Face 1
		ret.indices[indiceIndex++] = i;
		ret.indices[indiceIndex++] = (i + 1) % tesselation;
		ret.indices[indiceIndex++] = i + tesselation;

		// Face 2
		ret.indices[indiceIndex++] = (i + 1) % tesselation;
		ret.indices[indiceIndex++] = (i + 1 < tesselation) ? (i + 1 + tesselation) : i + 1;
		ret.indices[indiceIndex++] = i + tesselation;
	}
	assert(indiceIndex == ret.numIndices);
	return ret;
}

//MyShapeData ShapeMaker::makeCube(){
//	MyShapeData ret;
//	return ret;
//}

MyShapeData ShapeMaker::makeCube()
{
	using glm::vec2;
	using glm::vec3;
	using glm::vec4;
	MyVertex  stackVerts[] = 
	{
		// Top
		vec3(-1.0f, +1.0f, +1.0f), // 0
		vec4(+1.0f, +0.0f, +0.0f, +1.0f), // Color
		vec3(+0.0f, +1.0f, +0.0f), // Normal
		vec2(+0.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, +1.0f, +1.0f), // 1
		vec4(+0.0f, +1.0f, +0.0f, +1.0f), // Color
		vec3(+0.0f, +1.0f, +0.0f), // Normal
		vec2(+1.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, +1.0f, -1.0f), // 2
		vec4(+0.0f, +0.0f, +1.0f, +1.0f), // Color
		vec3(+0.0f, +1.0f, +0.0f), // Normal
		vec2(+1.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, +1.0f, -1.0f), // 3
		vec4(+1.0f, +1.0f, +1.0f, +1.0f), // Color
		vec3(+0.0f, +1.0f, +0.0f), // Normal
		vec2(+0.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget

		// Front
		vec3(-1.0f, +1.0f, -1.0f), // 4
		vec4(+1.0f, +0.0f, +1.0f, +1.0f), // Color
		vec3(+0.0f, +0.0f, -1.0f), // Normal
		vec2(+0.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, +1.0f, -1.0f), // 5
		vec4(+0.0f, +0.5f, +0.2f, +1.0f), // Color
		vec3(+0.0f, +0.0f, -1.0f), // Normal
		vec2(+1.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, -1.0f, -1.0f), // 6
		vec4(+0.8f, +0.6f, +0.4f, +1.0f), // Color
		vec3(+0.0f, +0.0f, -1.0f), // Normal
		vec2(+1.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, -1.0f, -1.0f), // 7
		vec4(+0.3f, +1.0f, +0.5f, +1.0f), // Color
		vec3(+0.0f, +0.0f, -1.0f), // Normal
		vec2(+0.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget

		// Right
		vec3(+1.0f, +1.0f, -1.0f), // 8
		vec4(+0.2f, +0.5f, +0.2f, +1.0f), // Color
		vec3(+1.0f, +0.0f, +0.0f), // Normal
		vec2(-1.0f, -0.0f), // UV
		vec3(+0.0f, +0.0f, +1.0f), //Tanget
		vec3(+1.0f, +1.0f, +1.0f), // 9
		vec4(+0.9f, +0.3f, +0.7f, +1.0f), // Color
		vec3(+1.0f, +0.0f, +0.0f), // Normal
		vec2(-0.0f, -0.0f), // UV
		vec3(+0.0f, +0.0f, +1.0f), //Tanget
		vec3(+1.0f, -1.0f, +1.0f), // 10
		vec4(+0.3f, +0.7f, +0.5f, +1.0f), // Color
		vec3(+1.0f, +0.0f, +0.0f), // Normal
		vec2(-0.0f, -1.0f), // UV
		vec3(+0.0f, +0.0f, +1.0f), //Tanget
		vec3(+1.0f, -1.0f, -1.0f), // 11
		vec4(+0.5f, +0.7f, +0.5f, +1.0f), // Color
		vec3(+1.0f, +0.0f, +0.0f), // Normal
		vec2(-1.0f, -1.0f), // UV
		vec3(+0.0f, +0.0f, +1.0f), //Tanget

		// Left
		vec3(-1.0f, +1.0f, +1.0f), // 12
		vec4(+0.7f, +0.8f, +0.2f, +1.0f), // Color
		vec3(-1.0f, +0.0f, +0.0f), // Normal
		vec2(-1.0f, -0.0f), // UV
		vec3(+0.0f, +0.0f, -1.0f), //Tanget
		vec3(-1.0f, +1.0f, -1.0f), // 13
		vec4(+0.5f, +0.7f, +0.3f, +1.0f), // Color
		vec3(-1.0f, +0.0f, +0.0f), // Normal
		vec2(-0.0f, -0.0f), // UV
		vec3(+0.0f, +0.0f, -1.0f), //Tanget
		vec3(-1.0f, -1.0f, -1.0f), // 14
		vec4(+0.4f, +0.7f, +0.7f, +1.0f), // Color
		vec3(-1.0f, +0.0f, +0.0f), // Normal
		vec2(-0.0f, -1.0f), // UV
		vec3(+0.0f, +0.0f, -1.0f), //Tanget
		vec3(-1.0f, -1.0f, +1.0f), // 15
		vec4(+0.2f, +0.5f, +1.0f, +1.0f), // Color
		vec3(-1.0f, +0.0f, +0.0f), // Normal
		vec2(-1.0f, -1.0f), // UV
		vec3(+0.0f, +0.0f, -1.0f), //Tanget

		// Back
		vec3(+1.0f, +1.0f, +1.0f), // 16
		vec4(+0.6f, +1.0f, +0.7f, +1.0f), // Color
		vec3(+0.0f, +0.0f, +1.0f), // Normal
		vec2(+1.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, +1.0f, +1.0f), // 17
		vec4(+0.6f, +0.4f, +0.8f, +1.0f), // Color
		vec3(+0.0f, +0.0f, +1.0f), // Normal
		vec2(+0.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, -1.0f, +1.0f), // 18
		vec4(+0.2f, +0.8f, +0.7f, +1.0f), // Color
		vec3(+0.0f, +0.0f, +1.0f), // Normal
		vec2(+0.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, -1.0f, +1.0f), // 19
		vec4(+0.2f, +0.7f, +1.0f, +1.0f), // Color
		vec3(+0.0f, +0.0f, +1.0f), // Normal
		vec2(+1.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget

		// Bottom
		vec3(+1.0f, -1.0f, -1.0f), // 20
		vec4(+0.8f, +0.3f, +0.7f, +1.0f), // Color
		vec3(+0.0f, -1.0f, +0.0f), // Normal
		vec2(+1.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, -1.0f, -1.0f), // 21
		vec4(+0.8f, +0.9f, +0.5f, +1.0f), // Color
		vec3(+0.0f, -1.0f, +0.0f), // Normal
		vec2(+0.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, -1.0f, +1.0f), // 22
		vec4(+0.5f, +0.8f, +0.5f, +1.0f), // Color
		vec3(+0.0f, -1.0f, +0.0f), // Normal
		vec2(+0.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, -1.0f, +1.0f), // 23
		vec4(+0.9f, +1.0f, +0.2f, +1.0f), // Color
		vec3(+0.0f, -1.0f, +0.0f), // Normal
		vec2(+1.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f) //Tanget
	};
	unsigned short stackIndices[] = {
		0, 1, 2, 0, 2, 3, // Top
		4, 5, 6, 4, 6, 7, // Front
		8, 9, 10, 8, 10, 11, // Right 
		12, 13, 14, 12, 14, 15, // Left
		16, 17, 18, 16, 18, 19, // Back
		20, 22, 21, 20, 23, 22, // Bottom
	};

	return copyToShapeData(stackVerts, ARRAYSIZE(stackVerts), stackIndices, ARRAYSIZE(stackIndices));
}

MyShapeData ShapeMaker::makeInverseCube()
{
	using glm::vec2;
	using glm::vec3;
	using glm::vec4;
	MyVertex  stackVerts[] = 
	{
		// Top
		vec3(-1.0f, +1.0f, +1.0f), // 0
		vec4(+1.0f, +0.0f, +0.0f, +1.0f), // Color
		vec3(+0.0f, -1.0f, +0.0f), // Normal
		vec2(+0.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, +1.0f, +1.0f), // 1
		vec4(+0.0f, +1.0f, +0.0f, +1.0f), // Color
		vec3(+0.0f, -1.0f, +0.0f), // Normal
		vec2(+1.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, +1.0f, -1.0f), // 2
		vec4(+0.0f, +0.0f, +1.0f, +1.0f), // Color
		vec3(+0.0f, -1.0f, +0.0f), // Normal
		vec2(+1.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, +1.0f, -1.0f), // 3
		vec4(+1.0f, +1.0f, +1.0f, +1.0f), // Color
		vec3(+0.0f, -1.0f, +0.0f), // Normal
		vec2(+0.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget

		// Front
		vec3(-1.0f, +1.0f, -1.0f), // 4
		vec4(+1.0f, +0.0f, +1.0f, +1.0f), // Color
		vec3(+0.0f, +0.0f, +1.0f), // Normal
		vec2(+0.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, +1.0f, -1.0f), // 5
		vec4(+0.0f, +0.5f, +0.2f, +1.0f), // Color
		vec3(+0.0f, +0.0f, +1.0f), // Normal
		vec2(+1.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, -1.0f, -1.0f), // 6
		vec4(+0.8f, +0.6f, +0.4f, +1.0f), // Color
		vec3(+0.0f, +0.0f, +1.0f), // Normal
		vec2(+1.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, -1.0f, -1.0f), // 7
		vec4(+0.3f, +1.0f, +0.5f, +1.0f), // Color
		vec3(+0.0f, +0.0f, +1.0f), // Normal
		vec2(+0.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget

		// Right
		vec3(+1.0f, +1.0f, -1.0f), // 8
		vec4(+0.2f, +0.5f, +0.2f, +1.0f), // Color
		vec3(-1.0f, +0.0f, +0.0f), // Normal
		vec2(-1.0f, -0.0f), // UV
		vec3(+0.0f, +0.0f, +1.0f), //Tanget
		vec3(+1.0f, +1.0f, +1.0f), // 9
		vec4(+0.9f, +0.3f, +0.7f, +1.0f), // Color
		vec3(-1.0f, +0.0f, +0.0f), // Normal
		vec2(-0.0f, -0.0f), // UV
		vec3(+0.0f, +0.0f, +1.0f), //Tanget
		vec3(+1.0f, -1.0f, +1.0f), // 10
		vec4(+0.3f, +0.7f, +0.5f, +1.0f), // Color
		vec3(-1.0f, +0.0f, +0.0f), // Normal
		vec2(-0.0f, -1.0f), // UV
		vec3(+0.0f, +0.0f, +1.0f), //Tanget
		vec3(+1.0f, -1.0f, -1.0f), // 11
		vec4(+0.5f, +0.7f, +0.5f, +1.0f), // Color
		vec3(-1.0f, +0.0f, +0.0f), // Normal
		vec2(-1.0f, -1.0f), // UV
		vec3(+0.0f, +0.0f, +1.0f), //Tanget

		// Left
		vec3(-1.0f, +1.0f, +1.0f), // 12
		vec4(+0.7f, +0.8f, +0.2f, +1.0f), // Color
		vec3(+1.0f, +0.0f, +0.0f), // Normal
		vec2(-1.0f, -0.0f), // UV
		vec3(+0.0f, +0.0f, -1.0f), //Tanget
		vec3(-1.0f, +1.0f, -1.0f), // 13
		vec4(+0.5f, +0.7f, +0.3f, +1.0f), // Color
		vec3(+1.0f, +0.0f, +0.0f), // Normal
		vec2(-0.0f, -0.0f), // UV
		vec3(+0.0f, +0.0f, -1.0f), //Tanget
		vec3(-1.0f, -1.0f, -1.0f), // 14
		vec4(+0.4f, +0.7f, +0.7f, +1.0f), // Color
		vec3(+1.0f, +0.0f, +0.0f), // Normal
		vec2(-0.0f, -1.0f), // UV
		vec3(+0.0f, +0.0f, -1.0f), //Tanget
		vec3(-1.0f, -1.0f, +1.0f), // 15
		vec4(+0.2f, +0.5f, +1.0f, +1.0f), // Color
		vec3(+1.0f, +0.0f, +0.0f), // Normal
		vec2(-1.0f, -1.0f), // UV
		vec3(+0.0f, +0.0f, -1.0f), //Tanget

		// Back
		vec3(+1.0f, +1.0f, +1.0f), // 16
		vec4(+0.6f, +1.0f, +0.7f, +1.0f), // Color
		vec3(+0.0f, +0.0f, -1.0f), // Normal
		vec2(+1.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, +1.0f, +1.0f), // 17
		vec4(+0.6f, +0.4f, +0.8f, +1.0f), // Color
		vec3(+0.0f, +0.0f, -1.0f), // Normal
		vec2(+0.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, -1.0f, +1.0f), // 18
		vec4(+0.2f, +0.8f, +0.7f, +1.0f), // Color
		vec3(+0.0f, +0.0f, -1.0f), // Normal
		vec2(+0.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, -1.0f, +1.0f), // 19
		vec4(+0.2f, +0.7f, +1.0f, +1.0f), // Color
		vec3(+0.0f, +0.0f, -1.0f), // Normal
		vec2(+1.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget

		// Bottom
		vec3(+1.0f, -1.0f, -1.0f), // 20
		vec4(+0.8f, +0.3f, +0.7f, +1.0f), // Color
		vec3(+0.0f, +1.0f, +0.0f), // Normal
		vec2(+1.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, -1.0f, -1.0f), // 21
		vec4(+0.8f, +0.9f, +0.5f, +1.0f), // Color
		vec3(+0.0f, +1.0f, +0.0f), // Normal
		vec2(+0.0f, +1.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(-1.0f, -1.0f, +1.0f), // 22
		vec4(+0.5f, +0.8f, +0.5f, +1.0f), // Color
		vec3(+0.0f, +1.0f, +0.0f), // Normal
		vec2(+0.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f), //Tanget
		vec3(+1.0f, -1.0f, +1.0f), // 23
		vec4(+0.9f, +1.0f, +0.2f, +1.0f), // Color
		vec3(+0.0f, +1.0f, +0.0f), // Normal
		vec2(+1.0f, +0.0f), // UV
		vec3(1.0f, 0.0f, 0.0f) //Tanget
	};
	unsigned short stackIndices[] = {
		0, 1, 2, 0, 2, 3, // Top
		4, 5, 6, 4, 6, 7, // Front
		8, 9, 10, 8, 10, 11, // Right 
		12, 13, 14, 12, 14, 15, // Left
		16, 17, 18, 16, 18, 19, // Back
		20, 22, 21, 20, 23, 22, // Bottom
	};

	return copyToShapeData(stackVerts, ARRAYSIZE(stackVerts), stackIndices, ARRAYSIZE(stackIndices));
}

MyShapeData ShapeMaker::copyToShapeData(MyVertex verts[],
										  int numVerts, unsigned short indices[],
										  int numIndices){
	MyShapeData newShape = MyShapeData();
	
	newShape.verts = new MyVertex [numVerts];
	memcpy(newShape.verts, verts, numVerts * sizeof(MyVertex));
	newShape.numVerts = numVerts;

	newShape.indices = new unsigned short[numIndices];
	memcpy(newShape.indices, indices, numIndices * sizeof(unsigned short));
	newShape.numIndices = numIndices;

	return newShape;
}
