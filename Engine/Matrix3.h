#ifndef MATRIX3_H
#define MATRIX3_H

#include "Vector2d.h"

struct Matrix3{
	float mat[3][3];

	Matrix3(){
		mat[0][0] = 1.0f;
		mat[1][0] = 0.0f;
		mat[2][0] = 0.0f;
		mat[0][1] = 0.0f;
		mat[1][1] = 1.0f;
		mat[2][1] = 0.0f;
		mat[0][2] = 0.0f;
		mat[1][2] = 0.0f;
		mat[2][2] = 1.0f;
	}
	Matrix3(Vector2d v1, Vector2d v2, Vector2d v3){
		mat[0][0] = v1.x;
		mat[0][1] = v1.y;
		mat[0][2] = 1;
		mat[1][0] = v2.x;
		mat[1][1] = v2.x;
		mat[1][2] = 1;
		mat[2][0] = v3.x;
		mat[2][1] = v3.x;
		mat[2][2] = 1;
	}
	inline void init(float n1, float n2, float n3, float n4, float n5, float n6, float n7, float n8, float n9){
		mat[0][0] = n1;
		mat[1][0] = n2;
		mat[2][0] = n3;
		mat[0][1] = n4;
		mat[1][1] = n5;
		mat[2][1] = n6;
		mat[0][2] = n7;
		mat[1][2] = n8;
		mat[2][2] = n9;
	}
	
	operator float*(){
		return &mat[0][0];
	}

	inline void Rotation(float angle){
		float c = cos(angle);
		float s = sin(angle);
		
		mat[0][0] = c;
		mat[0][1] = -s;
		mat[1][0] = s;
		mat[1][1] = c;
	}

	inline void Scale(float scale){
		mat[0][0] = scale;
		mat[1][1] = scale;
	}

	inline void ScaleX(float scale){
		if(mat[0][1] == 0 && mat[1][0] == 0){
			mat[0][0] = scale;
		}
	}

	inline void ScaleY(float scale){
		
		if(mat[0][1] == 0 && mat[1][0] == 0){
			mat[1][1] = scale;
		}
	}

	inline void Translation(float x, float y){
		mat[0][2] = x;
		mat[1][2] = y;

	}

	inline void Translation(Vector2d& v){
		mat[0][2] = v.x;
		mat[1][2] = v.y;
	}
};

inline Matrix3 operator*(const Matrix3& m1, const Matrix3& m2){
	Matrix3 myMatrix;
	float n1 = m1.mat[0][0] * m2.mat[0][0] + m1.mat[0][1] * m2.mat[1][0];
	float n2 = m1.mat[1][0] * m2.mat[0][0] + m1.mat[1][1] * m2.mat[1][0];

	float n3 = m1.mat[0][0] * m2.mat[0][1] + m1.mat[0][1] * m2.mat[1][1];
	float n4 = m1.mat[1][0] * m2.mat[0][1] + m1.mat[1][1] * m2.mat[1][1];

	float n5 = m1.mat[0][0] * m2.mat[0][2] + m1.mat[0][1] * m2.mat[1][2] + m1.mat[0][2];
	float n6 = m1.mat[0][1] * m2.mat[0][2] + m1.mat[1][1] * m2.mat[1][2] + m1.mat[1][2];

	myMatrix.init(n1, n2, 0, n3, n4, 0, n5, n6, 1);
	return myMatrix;
}

inline Vector2d operator*(const Matrix3& m1, const Vector2d& v2){
	Vector2d newVect;
	newVect.x = m1.mat[0][0] * v2.x + m1.mat[0][1] * v2.y + m1.mat[0][2];
	newVect.y = m1.mat[1][0] * v2.x + m1.mat[1][1] * v2.y + m1.mat[1][2];
	return newVect;
}

#endif