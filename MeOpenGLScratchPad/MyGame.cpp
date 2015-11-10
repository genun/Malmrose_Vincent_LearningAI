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
#include <random>

//using fastdelegate::MakeDelegate;
using Neumont::ShapeGenerator;
using Neumont::ShapeData;
using glm::mat4;
using glm::vec3;
using glm::vec4;

float zDist = -14.0f;

#pragma region Initialization
bool MyGame::initialize(bool* gameCont){
	if (!renderer.initialize())
		return false;
	isQuitting = false;
	cont = gameCont;

	window = new QWidget();
	masterLayout = new QVBoxLayout();
	debugLayout = new QVBoxLayout();
	debugWidget = new QWidget();

	if (!debugMenu.initialize(debugLayout))
		return false;
	debugShape.initialize();
	//if (!debugShape.initialize())
	//	return false;

	//debugShape.init();

	//debugWidget->setLayout(debugLayout);
	renderer.setMinimumSize(width, height);

	//masterLayout->addWidget(debugWidget);
	masterLayout->addWidget(&renderer);

	window->setLayout(masterLayout);
	window->show();

	renderStuff();

	//Run Breakout, because asteroids is in source
	Breakout();

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
	breakManage.~BreakoutManager();
	return ret;
}
#pragma endregion

void MyGame::Breakout(){
	Random rand;
	win = false;
	breakManage.init();
	breakManage.ball = new Ball();
	vec3 scaleBallSpeed = vec3(1.0f);
	float yRange = glm::normalize(rand.randomInRange(-2.0f, -1.0f)) * rand.randomInRange(1.5f, 3.0f);
	float xRange = rand.randomInRange(-2.0f, 2.0f);
	float ballSpeed = 3.0f;
	vec3 ballVelocity = glm::normalize(vec3(xRange, yRange, 0.0f)) * ballSpeed;
	breakManage.ball->Init(vec3(-2.0f, -1.0f, zDist), ballVelocity * scaleBallSpeed, 0.85f);
	breakManage.paddle = new Paddle();

	positionBricks();
	positionPaddle();
	positionBall();
	isBreakout = true;
	breakManage.cont = cont;
	breakManage.win = &win;
}

void MyGame::newFrame(){
	updateCamera();
	debugMenu.update();
	//lightUpdate();
	camPos = camera.getPosition();
	renderer.draw(camera.getWorldtoViewMatrix());
	if (isBreakout){
		breakManage.Update();
	}
	else{

	}

	if (!*cont) {
		*cont = true;
		isQuitting = true;
		//QApplication::quit();
		QCoreApplication::exit(0);
	}

	if (GetAsyncKeyState(VK_ESCAPE)){
		*cont = false;
		QCoreApplication::exit(0);
	}

	if (win){
		*cont = true;
		QCoreApplication::exit(0);

	}
	//ball->Update();
}

#pragma region Renderer initializtion
void MyGame::renderStuff(){
	debugShape.init();
	generateGeometries();

	return;
}

void MyGame::positionBricks(){
	float width = 2.0 * 0.25f;
	float height = 2.0 * 0.013f;
	for (int i = 0; i < breakManage.brickLineHeight; i++){
		for (int j = 0; j < breakManage.brickLineWidth; j++){
			breakManage.bricks[j][i] = new Brick();
			vec3 brickPosition = vec3(-9.60 + j * 1.75f, 5.75 + i * -0.75f, zDist);
			Renderable* hold = makeBrick(glm::translate(brickPosition)*glm::scale(vec3(0.80f, 0.20f, 0.01f)));
			/*glm::rotate(180.0f, vec3(0.0f, 1.0f, 0.0f))*/
			breakManage.bricks[j][i]->Init(brickPosition, hold, width, height);
		}
	}
}

void MyGame::positionPaddle(){
	float width = 0.95f;
	float height = 0.01f;
	vec3 paddlePos = vec3(-2.0f, -6.5f, zDist);
	mat4 scale = glm::scale(vec3(1.0f, 0.20f, 0.01f));
	breakManage.paddle->Init(paddlePos, 1.0f, makeBrick(glm::translate(paddlePos) * scale), scale, width, height);
}

void MyGame::positionBall(){
	breakManage.ball->img = makeBall(glm::translate(breakManage.ball->pos) * glm::scale(vec3(0.25f)));
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
	//updateCameraFromMouse();
	//updateCameraFromKeyboard();
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

void MyGame::setAI(DeepLearner* newAI){
	breakManage.setAI(newAI, &width, &height);
	newAI->pause = false;
}

void MyGame::AttachAI(DeepLearner* newAI){
	breakManage.setAI(newAI);
	breakManage.ai->pause = false;
}

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