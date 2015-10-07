#include <MyGame.h>
#include <ShapeGenerator.h>
#include <Rendering\Renderer.h>
#include <Rendering\Helpers\Renderables.h>
#include <cassert>
#include <glm/gtc/matrix_transform.hpp>
#include <IO\FileIO.h>
#include <DebugTools\Menu\DebugMenu.h>
#include <DebugTools\Shapes\DebugShape.h>
//#include <DebugTools\fastDelegate.h>
#include <Qt\qgroupbox.h>
#include <Qt\qboxlayout.h>
#include <Qt\qdockwidget.h>
#include <QtGui\qmouseevent>
#include <Qt\qapplication.h>
#include <Qt\qdebug.h>
#include <Shapes\ShapeMaker.h>

//using fastdelegate::MakeDelegate;
using Neumont::ShapeGenerator;
using Neumont::ShapeData;
using glm::mat4;
using glm::vec3;
using glm::vec4;

#pragma region Initialization
bool MyGame::initialize(){
	if (!renderer.initialize())
		return false;

	window = new QWidget();
	masterLayout = new QVBoxLayout();
	debugLayout = new QVBoxLayout();
	debugWidget = new QWidget();

	if (!debugMenu.initialize(debugLayout))
		return false;
	if (!debugShape.initialize())
		return false;

	//debugShape.init();

	//debugWidget->setLayout(debugLayout);
	renderer.setMinimumSize(800, 600);

	//masterLayout->addWidget(debugWidget);
	masterLayout->addWidget(&renderer);

	window->setLayout(masterLayout);
	window->show();

	renderStuff();

	//Run Breakout or Asteroids
	Breakout();
	//Asteroids();
	

	connect(&timer, SIGNAL(timeout()), this, SLOT(newFrame()));
		
	return true;
}

void MyGame::gameLoop(){
	timer.start();
}

bool MyGame::shutdown(){
	bool ret = true;
	ret &= renderer.shutdown();
	timer.stop();
	return ret;
}
#pragma endregion

void MyGame::Asteroids(){

}

void MyGame::Breakout(){
	ball = new Ball();
	ball->Init(vec3(-2.0f, -1.0f, -6.0f), vec3(1.0f, 2.0f, 0.0f));

	positionBricks();
	positionPaddle();
	positionBall();
	isBreakout = true;
}

void MyGame::newFrame(){
	updateCamera();
	debugMenu.update();
	//lightUpdate();
	camPos = camera.getPosition();
	renderer.draw(camera.getWorldtoViewMatrix());
	if (isBreakout){

	}
	else{

	}
	ball->Update();
}

#pragma region Renderer initializtion
void MyGame::renderStuff(){
	debugShape.init();
	generateGeometries();

	return;
}

void MyGame::positionBricks(){
	for (int i = 0; i < brickLineHeight; i++){
		for (int j = 0; j < brickLineWidth; j++){
			bricks[j][i] = new Brick();
			vec3 brickPosition = vec3(-3.75 + j * 1.05f, 2.75 + i * -0.75f, 2 - 8.0f);
			bricks[j][i]->Init(brickPosition, makeBrick(glm::translate(brickPosition) 
				*glm::rotate(180.0f, vec3(0.0f, 1.0f, 0.0f)) 
				*glm::scale(vec3(0.40f, 0.20f, 0.10f))));
		}
	}
}

void MyGame::positionPaddle(){
	vec3 paddlePos = vec3(-2.0f, -3.0f, -6.0f);
	paddle->Init(paddlePos, 1.0f, makeBrick(glm::translate(paddlePos) * glm::scale(vec3(1.5f, 0.20f, 0.10f))));
}

void MyGame::positionBall(){
	ball->img = makeBall(glm::translate(ball->pos) * glm::scale(vec3(0.25f)));
}

void MyGame::generateGeometries(){
	GLuint sizes[] = { 3, 4, 3, 2 };
	GLuint mySizes[] = { 3, 4, 3, 2, 3 };

	//Neumont vertex
	VertexLayoutInfo* vertexLayout =
		renderer.addVertexLayoutInfo(sizes, ARRAYSIZE(sizes), Neumont::Vertex::STRIDE);
	//For my custom vertex
	//VertexLayoutInfo* myVertexLayout =
	//	renderer.addVertexLayoutInfo(mySizes, ARRAYSIZE(mySizes), 15 * sizeof(GLfloat));

	//Light and dominate shaders
	//lightShader = renderer.addShader("ShaderCode\\LightVertexShader.glsl", "ShaderCode\\LightFragmentShader.glsl");
	dominateShader = renderer.addShader("ShaderCode\\DominateVertexShader.glsl", "ShaderCode\\DominateFragmentShader.glsl");

	char* brickNames[3] = {
		"cameraPosition",
		"light",
		"dominateColor"
	};
	UniformType brickTypes[3] = {
		VEC3, VEC3, VEC3
	};
	domColor = vec3(0.0f, 1.0f, 0.75f);

	//noUni = renderer.addUniformInfo(0, 0, 0, lightShader);
	dominateUniform = renderer.addUniformInfo(3, brickNames, brickTypes, dominateShader);

	ShapeData cubeGeoData = ShapeGenerator::makeCube();
	brickGeo =
		renderer.addGeometry(
		cubeGeoData.verts, cubeGeoData.vertexBufferSize(),
		cubeGeoData.indices, cubeGeoData.indexBufferSize(), cubeGeoData.numIndices,
		GL_TRIANGLES, vertexLayout);

	ShapeData sphereGeoData = ShapeGenerator::makeSphere(20);
	ballGeo =
		renderer.addGeometry(
		sphereGeoData.verts, sphereGeoData.vertexBufferSize(),
		sphereGeoData.indices, sphereGeoData.indexBufferSize(), sphereGeoData.numIndices,
		GL_TRIANGLES, vertexLayout);
}
#pragma endregion

#pragma region Make Item Functions
Renderable* MyGame::makeBrick(mat4 translate){
	void* dominateData[3] = {
		&camPos[0], &light[0], &domColor[0]
	};
	return renderer.addRenderable(brickGeo, dominateShader, dominateUniform, dominateData, 0, translate);
}

Renderable* MyGame::makeBall(mat4 translate){
	void* dominateData[3] = {
		&camPos[0], &light[0], &domColor[0]
	};
	return renderer.addRenderable(ballGeo, dominateShader, dominateUniform, dominateData, 0, translate);
}
#pragma endregion

#pragma region Camera updates
void MyGame::updateCamera(){
	updateCameraFromMouse();
	updateCameraFromKeyboard();
}

void MyGame::updateCameraFromMouse(){
	if((QApplication::mouseButtons() & Qt::MouseButton::LeftButton) == 0)
		return;
	QPoint p = renderer.mapFromGlobal(QCursor::pos());

	camera.mouseUpdate(glm::vec2(p.x(), p.y()));
}

void MyGame::updateCameraFromKeyboard(){

	if(GetAsyncKeyState('W'))
		camera.transForward(true);
	if(GetAsyncKeyState('S'))
		camera.transForward(false);
	if(GetAsyncKeyState('A'))
		camera.strafe(true);
	if(GetAsyncKeyState('D'))
		camera.strafe(false);
	if(GetAsyncKeyState('R'))
		camera.transUp(true);
	if(GetAsyncKeyState('F'))
		camera.transUp(false);

}

#pragma endregion

//Old code, might use but will delete soon.
/*
void MyGame::lightUpdate(){
	float lightSpeed = 0.3f * 2;
	lightMove = vec3();
	if (GetAsyncKeyState(VK_UP))
		lightMove.z -= lightSpeed;

	if (GetAsyncKeyState(VK_DOWN))
		lightMove.z += lightSpeed;

	if (GetAsyncKeyState(VK_LEFT))
		lightMove.x -= lightSpeed;

	if (GetAsyncKeyState(VK_RIGHT))
		lightMove.x += lightSpeed;

	if (GetAsyncKeyState(VK_PRIOR))
		lightMove.y += lightSpeed;

	if (GetAsyncKeyState(VK_NEXT))
		lightMove.y -= lightSpeed;

	renderer.updateWhere(lightRender, glm::translate(mat4(), lightMove));
	//TODO find out why light moves ten times faster than the renderer update function.
	light = light + vec3(lightMove.x / 10, lightMove.y / 10, lightMove.z / 10);
}

	//Miku
	MyShapeData mikuData = FileIO::readMyBinaryFile("BinaryData/MyMiku.bin");
	mikuGeo = renderer.addGeometry(
		mikuData.verts, mikuData.vertexBufferSize(),
		mikuData.indices, mikuData.indexBufferSize(), mikuData.numIndices,
		GL_TRIANGLES, myVertexLayout);

	MyShapeData mapData = FileIO::readMyBinaryFile("BinaryData/otherMap_2.bin");
	mapGeo = renderer.addGeometry(
		mapData.verts, mapData.vertexBufferSize(),
		mapData.indices, mapData.indexBufferSize(), mapData.numIndices,
		GL_TRIANGLES, myVertexLayout);

void MyGame::addNode(){
	if ((QApplication::mouseButtons() & Qt::MouseButton::RightButton) == 0 || timeLastNodeClicked < 1.0f)
		return;
	timeLastNodeClicked = 0.0f;

	QPoint p = renderer.mapFromGlobal(QCursor::pos());
	vec3 cursorPos = glm::vec3(p.x(), p.y(), 1.0f);
	
	RayClicked click = tracer.click(cursorPos, camera);
	std::vector<Renderable*> render = renderer.getRenderables();
	
	if(click.type == Render_Type::MAP){
		if(click.position.y < 0.0005) click.position.y = 0.0f;
		if(!nodeAbove(click.position)) {
			scene.addNode(click.position);
			Node* n = &scene.nodes[scene.nodeCount - 1];
			scene.addNodeRenderable(click.position);
			for(unsigned int i = 0; i < scene.nodeCount - 1; i++){
				Node n2 = scene.nodes[i];
				bool visible = tracer.checkVisible(click.position, n2.position);
				if(visible) {
					scene.addConnections(n->index, n2.index);
					if(oneWayConnection){
						scene.addConnections(n2.index, n->index);
					}
				}
			}
		}
	}
}

bool MyGame::nodeAbove(vec3 clicked){
	for(unsigned int i = 0; i < scene.nodeCount; i++){
		float xDif = fabs(clicked.x - scene.nodes[i].position.x);
		float yDif = fabs(clicked.y - scene.nodes[i].position.y);
		float zDif = fabs(clicked.z - scene.nodes[i].position.z);

		if(xDif < 1.0f && yDif < 1.0f && zDif < 1.0f) {
			std::vector<Renderable*> render = renderer.getRenderables();

			for(unsigned int j = 0; j <render.size(); j++){
				Renderable* rend = render[j];
				glm::vec3 pos = glm::vec3(rend->whereMatrix[3][0], rend->whereMatrix[3][1], 
					rend->whereMatrix[3][2]);
				float xRendDif = fabs(pos.x - scene.nodes[i].position.x);
				float yRendDif = fabs(pos.y - scene.nodes[i].position.y);
				float zRendDif = fabs(pos.z - scene.nodes[i].position.z);
				if(xRendDif < 1.0f && yRendDif < 1.0f && zRendDif < 1.0f) {
					scene.selectNode(scene.nodes[i].index, rend);
					return true;
				}
			}
		}
	}
	return false;
}

*/