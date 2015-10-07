#ifndef FILE_IO
#define FILE_IO

#include <GL/glew.h>
#include <string>
#include <ShapeData.h>
struct MyShapeData;

class FileIO{
public:

	static std::string file2String(const char* filePath);
	static Neumont::ShapeData FileIO::readBinaryFile(const char* path);
	static MyShapeData FileIO::readMyBinaryFile(const char* path);

};

#endif