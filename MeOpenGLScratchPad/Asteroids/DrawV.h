#ifndef DRAWV_H
#define DRAWV_H
#include "Core.h"
#include <sstream>
using Core::Graphics;
using std::stringstream;
#include "Vector2d.h"
#include "Matrix3.h"

void DrawValue(Core::Graphics& graphics, int x, int y, float num );
void DrawValue(Core::Graphics& graphics, int x, int y, double num );
void DrawValue(Core::Graphics& graphics, int x, int y, int num );
void DrawValue(Core::Graphics& graphics, int x, int y, Vector2d num );
void DrawValue(Core::Graphics& graphics, int x, int y, Matrix3 mat);
void DrawValue(Core::Graphics& graphics, int x, int y, size_t val);


#endif