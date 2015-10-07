#ifndef DEBUG_SHAPES
#define DEBUG_SHAPES

#include <GL\glew.h>
#include <glm\glm.hpp>
#include <Rendering\Helpers\ShaderInfo.h>
#include <Shapes\ShapeMaker.h>

class Geometry;
class Renderable;
class VertexLayoutInfo;
class PassInfo;

/*

THINGS THIS LAB NEEDS

1. add the shapes
2. transformation matrix
3. EASY TO USE



TODO

1. ShaderCode
2. Generate the Renderable

*/

class DebugShape{

	static DebugShape* instance;

	Geometry* myCube;
	Geometry* mySphere;
	Geometry* myPlane;
	Geometry* myLine;

	VertexLayoutInfo* vertLayout;
	VertexLayoutInfo* pointVertexLayoutInfo;
	VertexLayoutInfo* lineVertexLayoutInfo;
	ShaderInfo* shader;
	PassInfo* pass;

	DebugShape(){}
	DebugShape(const DebugShape&);
	DebugShape& operator=(const DebugShape&);
public:
	Geometry* getCube();
	Geometry* getSphere();
	Geometry* getPlane();
	Geometry* getLine();

	Renderable* cube(glm::vec3 position = glm::vec3(), 
		glm::vec3 scale = glm::vec3(), 
		float rotation = 0.0f,
		glm::vec3 rotationVector = glm::vec3());

	Renderable* sphere(glm::vec3 position = glm::vec3(),
		float scale = 1.0f);

	Renderable* plane(glm::vec3 position = glm::vec3(),
		float scale = 1.0f);

	Renderable* vector(glm::vec3 end = glm::vec3(0.0f, 0.0f, -1.0f));
	
	Renderable* line(glm::vec3 start = glm::vec3(), 
		glm::vec3 end = glm::vec3());
	
	Renderable* point(glm::vec3 orig = glm::vec3(0.0f, 0.0f, 0.0f));

	bool initialize();
	void init();
	static DebugShape& getInstance(){return *instance;}
};

#define debugShape DebugShape::getInstance()

#endif