#ifndef RENDERER_H
#define RENDERER_H

#include <GL/glew.h>
#include <QtOpenGL\qglwidget>
#include <Rendering\Helpers\Geometry.h>
#include <Rendering\Helpers\BufferInfo.h>
#include <Rendering\Helpers\ShaderInfo.h>
#include <Rendering\Helpers\Renderables.h>
#include <Rendering\Helpers\VertexLayoutInfo.h>
#include <Rendering\Helpers\PassInfo.h>
#include <Rendering\Helpers\UniformInfo.h>
#include <Rendering\Helpers\TextureInfo.h>

//Put tangent where the color should be

class Renderer: public QGLWidget{
	//VARIABLES, DATA
	static const unsigned int MAX_GEOMETRY = 50;
	Geometry geometries[MAX_GEOMETRY];
	static const unsigned int MAX_BUFFER_INFOS = 10;
	BufferInfo bufferInfos[MAX_BUFFER_INFOS];
	static const unsigned int MAX_SHADER_INFOS = 20;
	ShaderInfo shaderInfos[MAX_SHADER_INFOS];
	static const unsigned int MAX_RENDERABLES = 80;
	Renderable renderables[MAX_RENDERABLES];
	static const unsigned int MAX_VERTEX_LAYOUTS = 40;
	VertexLayoutInfo vertexInfos[MAX_VERTEX_LAYOUTS];
	static const unsigned int MAX_UNIFORM_INFOS = 15;
	UniformInfo uniformInfos[MAX_UNIFORM_INFOS];
	static const unsigned int MAX_TEXTURE_INFOS = 15;
	TextureInfo textureInfos[MAX_TEXTURE_INFOS];

	GLuint numRenderables;
	GLuint numGeometries;
	GLuint numBuffers;
	GLuint numShaders;
	GLuint numPassInfo;

	//GEOMETRIES
	Geometry* findAvailableGeometry();

	//SHADERS
	ShaderInfo* getAvailableShader();
	ShaderInfo* findAvailableShader();
	ShaderInfo* allocateNewShader();
	void checkCompileStatuts(GLuint shaderID);
	GLuint linkProgram(GLuint vertexShaderID, GLuint fragmentShaderID);
	GLuint compileShaders(const char* code, GLenum shaderType);
	void checkLinkStatus(GLuint programID);

	//BUFFER
	BufferInfo* findBufferWithSpace(GLuint neededSize);
	BufferInfo* findUnsuedBufferInfo();
	BufferInfo* allocateNewBuffer();
	BufferInfo* getAvailableBuffer(GLuint vertexDataSize);

	//RENDERABLES
	Renderable* findAvailableRenderable();
	void drawRenderable(Renderable* r, glm::mat4 camera);
	
	//VERTEX LAYOUT
	VertexLayoutInfo* getAvailableVertexLayoutInfo();

	//PASS INFO
	PassInfo* getAvailablePassInfo();

	//TEXTURE INFO
	TextureInfo* findAvailableTexture();


	//OTHER
	void Renderer::passDownUniforms(Renderable* renderable);
	UniformInfo* getAvailableUniformInfo();
	glm::mat4 perspective;
	static Renderer* instance;
	glm::mat4 cameraPosition;

	//SINGLETON
	//All three are needed for a singleton, Disallows any type of copy or assignment
	Renderer(){}
	//Copy Constructor
	//Deep copy vs Shallow copy
	Renderer(const Renderer&);
	Renderer& operator=(const Renderer&);

public:

	PassInfo* addPassInfo();

	Geometry* addGeometry(
		void* verts, GLuint vertexDataSize,
		void* indices, GLuint indexDataSize, GLuint numIndices,
		GLuint indexingMode, VertexLayoutInfo* layoutInfo);

	ShaderInfo* addShader(const char* vertexShaderFilePath,
		const char* fragmentShaderFilePath);

	Renderable* addRenderable(
		Geometry* geometry, ShaderInfo* shader,
		UniformInfo* uniform, void* uniformData[],
		TextureInfo* texture,
		glm::mat4 whereMatrix, 
		char* whereUniformLocation = "whereMat");
	void updateWhere(Renderable* renderable, glm::mat4 update);
	Renderable* addRenderable(
		Geometry* geometry, 
		ShaderInfo* shader, 
		glm::mat4 whereMatrix = glm::mat4(),
		char* whereUniformLocation = "whereMat");

	
	Renderable* setRenderableVisibility(
		Renderable* render, bool visibility);

	VertexLayoutInfo* addVertexLayoutInfo(
		GLuint sizes[], GLuint numAttributes, 
		GLuint stride);
	
	UniformInfo* addUniformInfo(
		GLuint numLoc, GLchar* names[],  
		UniformType type[], ShaderInfo* shader);

	TextureInfo* addTexture(
		const char* name,
		const char* type,
		ShaderInfo* shader);

	static bool initialize();
	static bool shutdown();

	void initializeGL();
	void paintGL();
	void draw(glm::mat4 camera);

	static Renderer& getInstance(){return *instance;}
};

#define renderer Renderer::getInstance()

#endif
