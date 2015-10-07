#include <DebugTools\Shapes\DebugShape.h>
#include <Rendering\Renderer.h>
#include <ShapeGenerator.h>
#include <glm\gtx\rotate_vector.hpp>

//TODO
//Get it so that the shader program ID is actually stored. Currently the ID changes after I add anything else to my renderer.
//Get the include for orientation
//Get line and crosshair geometries in the getGeometries functions, currently I will pass down a lot of geometries.
//Unit Vector, have origin always be the start
//change build order, scratchpad dependent on engine

DebugShape* DebugShape::instance;

using Neumont::ShapeGenerator;
using Neumont::ShapeData;
using glm::vec3;
using glm::mat4;

bool DebugShape::initialize(){
	if(instance!= 0)
		return false;
	instance = new DebugShape();
	return true;
}

void DebugShape::init(){
	myCube = 0;
	mySphere = 0;
	myPlane = 0;
	myLine = 0;
	GLuint sizes[] = {3, 4, 3, 2};
	vertLayout = renderer.addVertexLayoutInfo(sizes, ARRAYSIZE(sizes), Neumont::Vertex::STRIDE);
	GLuint crosshairSizes[] = {3, 3};
	pointVertexLayoutInfo = renderer.addVertexLayoutInfo(crosshairSizes, ARRAYSIZE(crosshairSizes), 6 * sizeof(float));
	lineVertexLayoutInfo = renderer.addVertexLayoutInfo(crosshairSizes, ARRAYSIZE(crosshairSizes), 6 * sizeof(float));

	shader = renderer.addShader("DebugTools\\Shapes\\VertexShader.glsl", "DebugTools\\Shapes\\FragmentShader.glsl");
}
//
//static DebugShape& DebugShape::getInstance(){
//	return *instance;
//}

//GET A SHAPE

Geometry* DebugShape::getCube(){
	if(myCube != 0){
		return myCube;
	}
	ShapeData cubeData = ShapeGenerator::makeCube();
	myCube = renderer.addGeometry(cubeData.verts, cubeData.vertexBufferSize(), 
		cubeData.indices, cubeData.indexBufferSize(), cubeData.numIndices, 
		GL_TRIANGLES, vertLayout);
	return myCube;
}

Geometry* DebugShape::getSphere(){
	if(mySphere != 0)
		return mySphere;
	ShapeData sphereData = ShapeGenerator::makeSphere(20);
	mySphere = renderer.addGeometry(sphereData.verts, sphereData.vertexBufferSize(),
		sphereData.indices, sphereData.indexBufferSize(), sphereData.numIndices,
		GL_TRIANGLES, vertLayout);
	return mySphere;
}

Geometry* DebugShape::getPlane(){
	if(myPlane != 0)
		return myPlane;
	ShapeData planeData = ShapeGenerator::makePlane(5);
	myPlane = renderer.addGeometry(planeData.verts, planeData.vertexBufferSize(),
		planeData.indices, planeData.indexBufferSize(), planeData.numIndices,
		GL_TRIANGLES, vertLayout);
	return myPlane;
}

Geometry* DebugShape::getLine(){
	if(myLine != 0)
		return myLine;
	ShapeData lineData = ShapeGenerator::makeLine();
	myLine = renderer.addGeometry(lineData.verts, lineData.vertexBufferSize(),
		lineData.indices, lineData.indexBufferSize(), lineData.numIndices,
		GL_TRIANGLES, vertLayout);
	return myLine;
}


//ADD A THE SHAPES TO THE SCREEN


Renderable* DebugShape::cube(vec3 position, vec3 scale, 
							 float rotation, vec3 rotationVector){
								 Renderable* cube;
								 //TODO add the rotation and scale matricies
								 mat4 transformMatrix = glm::translate(position);// * glm::rotate(rotation, rotationVector) * glm::scale(scale);
								 cube = renderer.addRenderable(getCube(), shader, transformMatrix);
								 //ADD A PASS INFO FOR THE DEBUGSHAPES
								 //pass->addRenderable(cube);
								 return cube;
}

Renderable* DebugShape::sphere(vec3 position, float scale){
	Renderable* sphere;
	mat4 transformMatrix = glm::translate(position);
	sphere = renderer.addRenderable(getSphere(), shader, transformMatrix);
	return sphere;
}

Renderable* DebugShape::plane(glm::vec3 position, float scale){
	Renderable* plane;
	mat4 transformMatrix = glm::translate(position) * glm::scale(vec3(scale));
	plane = renderer.addRenderable(getPlane(), shader, transformMatrix);
	return plane;
}

Renderable* DebugShape::vector(vec3 end){
	ShapeData cyl = ShapeMaker::makeCylinder(20);
	ShapeData cone = ShapeMaker::makeCone(20);
	Geometry * cylinder = renderer.addGeometry(cyl.verts, cyl.vertexBufferSize(),
		cyl.indices, cyl.indexBufferSize(), cyl.numIndices,
		GL_TRIANGLES, vertLayout);

	Geometry* theCone = renderer.addGeometry(cone.verts, cone.vertexBufferSize(),
		cone.indices, cone.indexBufferSize(), cone.numIndices,
		GL_TRIANGLES, vertLayout);

	vec3 direction = glm::normalize(end);
	mat4 orient = glm::orientation(direction, vec3(1.0f, 0.0f, 0.0f)) * glm::scale(vec3(1.0f, 0.1f, 0.1f));
	Renderable* myCyl = renderer.addRenderable(cylinder, shader, orient);

	mat4 trans = glm::translate(direction) * glm::orientation(direction, vec3(1.0f, 0.0f, 0.0f));
	Renderable* myCone = renderer.addRenderable(theCone, shader, trans);
	return myCyl;
}

Renderable* DebugShape::line(vec3 start, vec3 end){
	float lineVertex[] = {
		0.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 1.0f,

		0.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 1.0f
	};

	short lineIndices[] = {
		0, 1
	};

	vec3 rotationVector = glm::normalize(end - start);
	float scale = (end.length() + start.length());
	mat4 transformationMatrix = glm::translate(start) * 
		glm::orientation(rotationVector, vec3(0.0f, 1.0f, 0.0f)) * 
		glm::scale(vec3(scale));
	Geometry* line = renderer.addGeometry(lineVertex, sizeof(lineVertex),
		lineIndices, sizeof(lineIndices), sizeof(lineIndices)/ sizeof(short),
		GL_LINES, lineVertexLayoutInfo);
	Renderable* lineRender = renderer.addRenderable(line, shader, transformationMatrix);

	return lineRender;
}

Renderable* DebugShape::point(vec3 position){

	float crosshairsVertex[] = {
		-1.0f, +0.0f, +0.0f,
		+1.0f, +1.0f, +1.0f,

		+1.0f, +0.0f, +0.0f,
		+1.0f, +0.0f, +0.0f,

		+0.0f, -1.0f, +0.0f,
		+1.0f, +1.0f, +1.0f,

		+0.0f, +1.0f, +0.0f,
		+0.0f, +1.0f, +0.0f,

		+0.0f, +0.0f, -1.0f,
		+1.0f, +1.0f, +1.0f,

		+0.0f, +0.0f, +1.0f,
		+0.0f, +0.0f, +1.0f
	};

	short crosshairIndices[] = {
		0, 1,
		2, 3,
		4, 5
	};

	Geometry* crosshair;
	crosshair = renderer.addGeometry(crosshairsVertex, sizeof(crosshairsVertex),
		crosshairIndices, sizeof(crosshairIndices), sizeof(crosshairIndices)/ sizeof(short),
		GL_LINES, pointVertexLayoutInfo);

	mat4 transformMatrix = glm::translate(position);
	Renderable* crosshairRender = renderer.addRenderable(crosshair, shader, transformMatrix);


		return 0;
}

