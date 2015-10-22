#include <Rendering\Renderer.h>
#include <glm\glm.hpp>
#include <qt\qdebug.h>
#include <IO\FileIO.h>
#include <string>
#include <cassert>

Renderer* Renderer::instance = 0;

#pragma region GEOMETRIES
Geometry* Renderer::findAvailableGeometry(){
	for(unsigned int i = 0; i < MAX_GEOMETRY; i++){
		Geometry* g = geometries + i;
		if (g->isAvailable){
			return g;
		}
	}

	qDebug() << "Error no geometries available";
	exit(1);
	return 0;
}

Geometry* Renderer::addGeometry(void* verts, GLuint vertexDataSize,
							   void* indices, unsigned int indexDataSize, GLuint numIndices,
							   GLuint indexingMode, VertexLayoutInfo* layoutInfo)
{
	Geometry* ret = findAvailableGeometry();
	BufferInfo* buffer;
	glGenVertexArrays(1, &ret->vertexAttribID);
	glBindVertexArray(ret->vertexAttribID);

	buffer = getAvailableBuffer(vertexDataSize);
	glBindBuffer(GL_ARRAY_BUFFER, buffer->glBufferID);
	glBufferSubData(GL_ARRAY_BUFFER, buffer->nextAvailableByte, vertexDataSize, verts);
	GLuint currentOffset = buffer->nextAvailableByte;
	buffer->nextAvailableByte += vertexDataSize;

	buffer = getAvailableBuffer(indexDataSize);
	glBindBuffer(GL_ARRAY_BUFFER, buffer->glBufferID);
	glBufferSubData(GL_ARRAY_BUFFER, buffer->nextAvailableByte, indexDataSize, indices);
	ret->indexBufferID = buffer->glBufferID;
	ret->indexByteOffset = buffer->nextAvailableByte;
	ret->numIndices = numIndices;
	buffer->nextAvailableByte += indexDataSize;

	ret->isAvailable = false;
	ret->indexingMode = indexingMode;
	numGeometries++;

	for(GLuint i = 0; i < layoutInfo->numAttributes; i++){
		glEnableVertexAttribArray(i);
		glVertexAttribPointer(i, layoutInfo->attributeSizes[i], GL_FLOAT, GL_FALSE, layoutInfo->stride, (void*)currentOffset);
		currentOffset += layoutInfo->attributeSizes[i] * sizeof(GL_FLOAT);
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ret->indexBufferID);
	
	ret->vertexLayoutInfo = layoutInfo;
	return ret;
}
#pragma endregion

#pragma region Shaders
ShaderInfo* Renderer::getAvailableShader(){
	ShaderInfo* ret = findAvailableShader();

	if(ret!=0)
		return ret;
	return 0;
}

ShaderInfo* Renderer::findAvailableShader(){
	for(int i = 0; i < MAX_SHADER_INFOS; i++){
		ShaderInfo* s = shaderInfos + i;
		if(s->isAvailable)
			return s;
	}
	qDebug() << "Error could not find available Shader";
	exit(1);
	return 0;
}

GLuint Renderer::linkProgram(GLuint vertexShaderID, GLuint fragmentShaderID){
	GLuint programID = glCreateProgram();
	
	glAttachShader(programID, vertexShaderID);
	glAttachShader(programID, fragmentShaderID);
	
	glLinkProgram(programID);
	
	//checkLinkStatus(programID);

	return programID;
}

void Renderer::checkLinkStatus(GLuint programID){
	GLint compileStatus;
	glGetProgramiv(programID, GL_COMPILE_STATUS, &compileStatus);
	if(compileStatus == GL_TRUE)
		return;
	GLint loglength;
	glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &loglength);
	char*buffer = new char[loglength];
	GLsizei bitbucket;
	glGetProgramInfoLog(programID, loglength, &bitbucket, buffer);
	qDebug() << "Program failed to link ... "<< buffer;
	delete [] buffer;
	exit(1);
}

GLuint Renderer::compileShaders(const char* code, GLenum shaderType){
	GLuint ret = 0;
	ret = glCreateShader(shaderType);
	FileIO file;
	file;

	const char* adapt[1];
	std::string temp(file.file2String(code));

	adapt[0] = temp.c_str();
	glShaderSource(ret, 1, adapt, 0);

	glCompileShader(ret);
	checkCompileStatuts(ret);

	return ret;
}

ShaderInfo* Renderer::addShader(
		const char* vertexShaderFilePath,
		const char* fragmentShaderFilePath){
	ShaderInfo* ret = getAvailableShader();

	GLuint vertexShaderID = compileShaders(vertexShaderFilePath, GL_VERTEX_SHADER);
	GLuint fragmentShaderID = compileShaders(fragmentShaderFilePath, GL_FRAGMENT_SHADER);

	ret->progID = linkProgram(vertexShaderID, fragmentShaderID);
	ret->isAvailable = false;

	return ret;
}

void Renderer::checkCompileStatuts(GLuint shaderID){
	GLint compileStatus;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compileStatus);
	if(compileStatus == GL_TRUE)
		return;
	GLint loglength;
	glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &loglength);
	GLint shaderType;
	glGetShaderiv(shaderID, GL_SHADER_TYPE, &shaderType);
	char*buffer = new char[loglength];
	GLsizei bitbucket;
	glGetShaderInfoLog(shaderID, loglength, &bitbucket, buffer);
	qDebug() << (shaderType == GL_VERTEX_SHADER ? "vertex" : "fragment") << "shader compile error ... "<< buffer;
	delete [] buffer;
	exit(1);
}
#pragma endregion

#pragma region BUFFERS
BufferInfo* Renderer::getAvailableBuffer(GLuint neededSize){

	//Find the buffer info with enough room
	//if one is not found, make one and allocate 1 meg of graphics card RAM

	BufferInfo* ret = findBufferWithSpace(neededSize);

	if(ret != 0)
		return ret;
	return allocateNewBuffer();
}

BufferInfo* Renderer::findBufferWithSpace(GLuint neededSize){

	for(unsigned int i = 0; i < MAX_BUFFER_INFOS; i++){
		if(bufferInfos[i].hasBuffer && bufferInfos[i].getRemainingSize() > neededSize)
			return bufferInfos + i;
	}/*
	qDebug() <<"Unable to find a buffer with space";
	exit(0);*/
	return 0;
}

BufferInfo* Renderer::allocateNewBuffer(){
	BufferInfo* b = findUnsuedBufferInfo();

	glGenBuffers(1, &b->glBufferID);
	glBindBuffer(GL_ARRAY_BUFFER, b->glBufferID);
	glBufferData(GL_ARRAY_BUFFER, BufferInfo::MAX_BUFFER_SIZE, 0, GL_STATIC_DRAW);
	b->hasBuffer = true;
	b->sizeRemaining = BufferInfo::MAX_BUFFER_SIZE;
	b->nextAvailableByte = 0;
	return b;
}

BufferInfo* Renderer::findUnsuedBufferInfo(){
	for(unsigned int i = 0; i < MAX_BUFFER_INFOS; i++){
		if(!bufferInfos[i].hasBuffer){
			return bufferInfos + i;
		}
	}
	qDebug() << "Error, failed to find available buffer....";
	exit(1);
	return 0;
}
#pragma endregion

#pragma region RENDERABLES
Renderable* Renderer::addRenderable(Geometry* geometry, ShaderInfo* shader, 
									UniformInfo* uniform, void*  uniformData[], 
									TextureInfo* texture, glm::mat4 whereMatrix,
									char* whereUniformLocation){
	Renderable* render = findAvailableRenderable();
	render->progID = shader->progID;
	render->geo = geometry;
	render->isAvailable = false;
	render->uniformInfo = uniform;
	render->whereMatrix = whereMatrix;
	render->whereMatrixUniformLocation = glGetUniformLocation(shader->progID, "whereMat");
	for(GLuint i = 0; i < uniform->numLocations; i++){
		render->uniformData[i] = uniformData[i];
	}
	if(texture != 0){
		render->texture = texture;
		render->hasTexture = true;
	}

	numRenderables++;
	return render;
}

Renderable* Renderer::addRenderable(Geometry* geometry, ShaderInfo* shader, glm::mat4 whereMatrix, char* whereUniformLocation){
	UniformInfo* noUniforms = addUniformInfo(0, 0, 0, shader);
	Renderable* render = addRenderable(geometry, shader, noUniforms, 0, 0, whereMatrix, whereUniformLocation);
	return render;
}

Renderable* Renderer::findAvailableRenderable(){
	for(int i = 0; i < MAX_RENDERABLES; i++){
		if(renderables[i].isAvailable)
			return renderables + i;
	}
	qDebug() << "Error, could not find an available Renderable";
	exit(1);
	return 0;
}

void Renderer::drawRenderable(Renderable* r, glm::mat4 camera){
	if(!r->isVisible)
		return;
	Geometry* g = r->geo;
	VertexLayoutInfo* v = g->vertexLayoutInfo;
	passDownUniforms(r);
	
	//glm::mat4 totalTransform = perspective * camera * r->whereMatrix;
	GLuint cameraUniformLocation = glGetUniformLocation(r->progID, "cameraMatrix");
	GLuint projectionUniformLocation = glGetUniformLocation(r->progID, "projectionMatrix");

	glUniformMatrix4fv(r->whereMatrixUniformLocation, 1, GL_FALSE, &r->whereMatrix[0][0]);
	glUniformMatrix4fv(cameraUniformLocation, 1, GL_FALSE, &camera[0][0]);
	glUniformMatrix4fv(projectionUniformLocation, 1, GL_FALSE, &perspective[0][0]);

	glBindVertexArray(g->vertexAttribID);
	glDrawElements(g->indexingMode, g->numIndices, GL_UNSIGNED_SHORT, (void*)g->indexByteOffset);
}

void Renderer::updateWhere(Renderable* renderable, glm::mat4 update){
	renderable->whereMatrix = renderable->whereMatrix * update;
}

Renderable* Renderer::setRenderableVisibility(
		Renderable* render, bool visibility){
	render->isVisible = visibility;
	return render;
}
#pragma endregion

//TEXTURE INFO
TextureInfo* Renderer::addTexture(const char* texName, const char* texType, ShaderInfo* shader){
	TextureInfo* text = findAvailableTexture();

	text->uniformLocation = glGetUniformLocation(shader->progID, "textured");

	glGenTextures(1, &text->textureID);
	glBindTexture(GL_TEXTURE_2D, text->textureID);
	QImage img = QGLWidget::convertToGLFormat(QImage(texName, texType));
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits());
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	text->isAvailable = false;
	return text;
}

TextureInfo* Renderer::findAvailableTexture(){
	for(int i = 0; i < MAX_TEXTURE_INFOS; i++){
		if (textureInfos[i].isAvailable)
			return textureInfos + i;
	}
	qDebug() << "unable to find an available texture info";
	exit(0);
	return 0;
}


//VERTEX LAYOUT
VertexLayoutInfo* Renderer::addVertexLayoutInfo(
	GLuint sizes[], GLuint numAttributes, GLuint stride){
	assert(numAttributes <= VertexLayoutInfo::MAX_VERTEX_ATTRIBUTES);
	VertexLayoutInfo* ret = getAvailableVertexLayoutInfo();
	for(GLuint i = 0; i < numAttributes; i++){
		ret->attributeSizes[i] = sizes[i];
	}
	ret->numAttributes = numAttributes;
	ret->stride = stride;
	ret->isAvailable = false;
	return ret;
}

VertexLayoutInfo* Renderer::getAvailableVertexLayoutInfo(){
	for(int i = 0; i < MAX_VERTEX_LAYOUTS; i++){
		if(vertexInfos[i].isAvailable)
			return vertexInfos + i;
	}
	assert(false);
	return 0;
}

//PASS INFO
PassInfo* Renderer::addPassInfo(){
	PassInfo* pass;
	pass = &PassInfo();
	return pass;
}


//UNIFORM INFORMATION
void Renderer::passDownUniforms(Renderable* renderable){
	UniformInfo* uni = renderable->uniformInfo;
	for(int i = 0; i < uni->MAX_UNIFORM_LOCATIONS; i++){
		switch(uni->uniformType[i]){
		case VEC2:
			glUniform2fv(uni->locations[i], 1, (GLfloat*)renderable->uniformData[i]);
			break;
		case VEC3:
			glUniform3fv(uni->locations[i], 1, (GLfloat*)renderable->uniformData[i]);
			break;
		case VEC4:
			glUniform4fv(uni->locations[i], 1, (GLfloat*)renderable->uniformData[i]);
			break;
		case MAT3:
			glUniformMatrix3fv(uni->locations[i], 1, GL_FALSE, (GLfloat*)renderable->uniformData[i]);
			break;
		case MAT4:
			glUniformMatrix4fv(uni->locations[i], 1, GL_FALSE, (GLfloat*)renderable->uniformData[i]);
		case MY_BOOLEAN:
			glUniform1i(uni->locations[i], *(bool*)renderable->uniformData[i]);
			break;
		}
	}
	if(renderable->hasTexture){
		glBindTexture(GL_TEXTURE_2D, renderable->texture->textureID);
		glUniform1i(renderable->texture->uniformLocation, 0);
		glActiveTexture(renderable->texture->textureID);
	}
}

UniformInfo* Renderer::addUniformInfo(GLuint numLoc, GLchar* names[], UniformType type[], ShaderInfo* shader){
	UniformInfo* ret = getAvailableUniformInfo();
	ret->isAvailable = false;
	ret->numLocations = numLoc;
	for(GLuint i = 0; i < numLoc; i++){
		ret->locations[i] = glGetUniformLocation(shader->progID, names[i]);
		ret->uniformType[i] = type[i];
	}
	return ret;
}

UniformInfo* Renderer::getAvailableUniformInfo(){
	for(int i = 0; i < MAX_UNIFORM_INFOS; i++){
		if(uniformInfos[i].isAvailable)
			return uniformInfos + i;
	}
	qDebug() << "Couldn't find an available UniformInfo";
	exit(0);
}

//OTHERS
bool Renderer::initialize(){
	if(instance !=0)
		return false;
	instance = new Renderer;
	//Add this line again if it is no longer in a menu or other window.
	//instance->show();
	return true;
}

void Renderer::initializeGL(){
	glewInit();
	glEnable(GL_DEPTH_TEST);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	numRenderables = 0;
	numGeometries = 0;
	numBuffers = 0;
	numShaders = 0;
	numPassInfo = 0;
	perspective = glm::perspective(60.0f, ((float)width()/(float)height()), 0.1f, 100.0f);
}

bool Renderer::shutdown(){
	if(instance == 0)
		return false;
	delete instance;
	instance = 0;
	return true;
}

void Renderer::paintGL(){
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, width(), height());
	GLuint currentProgramID = -1;

	for(GLuint i = 0; i < numRenderables; i++){
		Renderable& r = renderables[i];
		if(r.progID != currentProgramID){
			glUseProgram(r.progID);
			currentProgramID = r.progID;
		}
		drawRenderable(&r, cameraPosition);
	}
	//glReadBuffer(GL_FRONT);
	//int numPixels = width() * height();
	//GLfloat* pixels = new GLfloat[numPixels * 3];
	//glReadPixels(0, 0, width(), height(), GL_RGB, GL_FLOAT, pixels);
	//qDebug() << pixels[0];
}

void Renderer::draw(glm::mat4 camera){
	cameraPosition = camera;
	repaint();
}
	