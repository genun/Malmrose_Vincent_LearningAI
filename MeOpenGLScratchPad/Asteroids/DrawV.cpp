#include "DrawV.h"

void DrawValue(Core::Graphics& graphics, int x, int y, float num ) {
	stringstream ss;
	ss.precision(5);
	ss << num;
	graphics.DrawString( x, y, ss.str().c_str());
}

void DrawValue(Core::Graphics& graphics, int x, int y, size_t val) {
	stringstream ss;
	ss.precision(5);
	ss << val;
	graphics.DrawString( x, y, ss.str().c_str());
}

void DrawValue(Core::Graphics& graphics, int x, int y, double num ) {
	stringstream ss;
	ss.precision(5);
	ss << num;
	graphics.DrawString( x, y, ss.str().c_str());
}

void DrawValue(Core::Graphics& graphics, int x, int y, int num ) {
	stringstream ss;
	ss.precision(5);
	ss << num;
	graphics.DrawString( x, y, ss.str().c_str());
}

void DrawValue(Core::Graphics& graphics, int x, int y, Vector2d num ) {
	stringstream ss;
	ss.precision(5);
	ss << "x: " << num.x << " y: " << num.y;
	graphics.DrawString( x, y, ss.str().c_str());
}

void DrawValue(Core::Graphics& graphics, int x, int y, Matrix3 mat) {
	mat;
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			stringstream ss;
			ss.precision(5);
			ss << mat.mat[j][i];
			graphics.DrawString(x + i * 75, y + j * 12, ss.str().c_str());
		}
	}
}