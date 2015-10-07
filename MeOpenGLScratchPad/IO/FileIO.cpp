#include <IO\FileIO.h>
#include <fstream>
#include <Qt\qdebug.h>
#include <Shapes\MyShapeData.h>

using std::string;
using std::ifstream;

string FileIO::file2String(const char* fileName){
	ifstream meinput(fileName);
	if(! meinput.good()){
		qDebug() << "File Failed to Load..." << fileName;
		exit(1);
	}

	return std::string(
		std::istreambuf_iterator<char>(meinput),
		std::istreambuf_iterator<char>());
}


Neumont::ShapeData FileIO::readBinaryFile(const char* path){
     std::ifstream input(path, 
		 std::ios::binary | std::ios::in);
     
     //seekg = seek get pointer
     //Tells us how many bytes we need to read
     //Make sure we return to the start
     input.seekg(0, std::ios::end);
     unsigned int numBytes = input.tellg();
     input.seekg(0, std::ios::beg);
     char* bytes = new char[numBytes];
     input.read(bytes, numBytes);
     input.close();
     
     Neumont::ShapeData ret;
     
     char* metaData = bytes;
     char* vertexBase = bytes + 2* sizeof(unsigned int);
     
     ret.numVerts = *reinterpret_cast<unsigned int*>(metaData);
     ret.numIndices = reinterpret_cast<unsigned int*>(metaData)[1];

     char* indexBase = vertexBase + ret.vertexBufferSize();
     ret.verts = reinterpret_cast<Neumont::Vertex*>(vertexBase);
     ret.indices = reinterpret_cast<unsigned short*> (indexBase);

     return ret;
}


MyShapeData FileIO::readMyBinaryFile(const char* path){
     std::ifstream input(path, 
		 std::ios::binary | std::ios::in);
     
     //seekg = seek get pointer
     //Tells us how many bytes we need to read
     //Make sure we return to the start
     input.seekg(0, std::ios::end);
     unsigned int numBytes = input.tellg();
     input.seekg(0, std::ios::beg);
     char* bytes = new char[numBytes];
     input.read(bytes, numBytes);
     input.close();
     
     MyShapeData ret;
     
     char* metaData = bytes;
     char* vertexBase = bytes + 2* sizeof(unsigned int);
     
     ret.numVerts = *reinterpret_cast<unsigned int*>(metaData);
     ret.numIndices = reinterpret_cast<unsigned int*>(metaData)[1];

     char* indexBase = vertexBase + ret.vertexBufferSize();
     ret.verts = reinterpret_cast<MyVertex*>(vertexBase);
     ret.indices = reinterpret_cast<unsigned short*> (indexBase);

     return ret;
} 